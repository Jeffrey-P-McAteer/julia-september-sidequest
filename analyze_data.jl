#!/usr/bin/env julia

# HARDWARE MONITORING ANALYSIS TOOL
# Purpose: Analyze relationships between CPU frequency and battery drain
# Assumptions: 
#   - Data is collected from a companion script (record_hardware.jl)
#   - Battery percentage is reported with limited precision (may have many identical readings)
#   - CPU frequency varies dynamically based on system load
#   - Time-windowed analysis provides more accurate results than point-to-point

using Pkg

# DEPENDENCY MANAGEMENT BLOCK
# Purpose: Automatically install required packages if missing to ensure script runs anywhere
# Reasoning: Self-contained execution without manual dependency management
packages = ["SQLite", "DataFrames", "PlotlyJS", "Statistics", "StatsBase", "Dates"]
for pkg in packages
    try
        eval(:(using $(Symbol(pkg))))
    catch
        println("Installing $pkg...")
        Pkg.add(pkg)
        eval(:(using $(Symbol(pkg))))
    end
end

using SQLite      # Database access for hardware data
using DataFrames  # Tabular data manipulation
using PlotlyJS    # Interactive plotting and visualization
using Statistics  # Statistical functions (mean, correlation, etc.)
using StatsBase   # Advanced statistics (quantiles, etc.)
using Dates       # DateTime parsing and manipulation

# BUCKET WIDTH CONFIGURATION
# Purpose: Allow configurable bucket width through environment variable
# Default: 0.25 GHz provides good balance of resolution vs. statistical power
# Environment: Set BUCKET_WIDTH to override (e.g., "0.1", "0.5", "1.0")
# Reasoning: Different use cases may need different granularity levels
function get_bucket_width()
    env_width = get(ENV, "BUCKET_WIDTH", "0.25")
    try
        width = parse(Float64, env_width)
        if width <= 0
            @warn "BUCKET_WIDTH must be positive, using default 0.25"
            return 0.25
        elseif width > 2.0
            @warn "BUCKET_WIDTH seems very large ($(width)), consider smaller values"
        end
        return width
    catch
        @warn "Invalid BUCKET_WIDTH value '$(env_width)', using default 0.25"
        return 0.25
    end
end

# DATA LOADING FUNCTION
# Purpose: Load hardware monitoring data from SQLite database
# Assumptions:
#   - Database exists and contains 'hardware_data' table
#   - Data includes timestamp, cpu_frequency_ghz, battery_percentage columns
#   - Missing/NULL values should be excluded from analysis
# Reasoning: Centralized data loading with validation ensures consistent data quality
function load_data()
    # Validate database file exists
    if !isfile("hw-data.db")
        error("hw-data.db not found. Please run record_hardware.jl first to collect data.")
    end
    
    db = SQLite.DB("hw-data.db")
    
    # SQL query filters out NULL values which would corrupt analysis
    # ORDER BY timestamp ensures chronological data processing
    query = """
        SELECT timestamp, cpu_frequency_ghz, battery_percentage
        FROM hardware_data
        WHERE cpu_frequency_ghz IS NOT NULL 
          AND battery_percentage IS NOT NULL
        ORDER BY timestamp
    """
    
    df = DataFrame(SQLite.DBInterface.execute(db, query))
    close(db)  # Always close database connection
    
    # Validate we have data to analyze
    if nrow(df) == 0
        error("No valid data found in database. Please run record_hardware.jl to collect data.")
    end
    
    println("Loaded $(nrow(df)) data points from hw-data.db")
    return df
end

# BATTERY PRECISION DETECTION AND WINDOW OPTIMIZATION
# Purpose: Automatically determine optimal time window based on battery data characteristics
# Assumptions:
#   - Battery percentage may have limited precision (e.g., integer percentages)
#   - Consecutive readings may be identical even during active drain
#   - Longer windows needed for low-precision data, shorter for high-precision
# Reasoning: Adaptive windowing provides more accurate drain rate calculations
#   than fixed windows across different devices/sampling methods
function detect_battery_precision_and_optimal_window(df)
    """
    Analyze battery data to detect precision and calculate optimal time window.
    Returns recommended window size in minutes based on data characteristics.
    """
    
    println("Analyzing battery data precision...")
    
    # Convert string timestamps to DateTime objects for time difference calculations
    df.datetime = [DateTime(ts, "yyyy-mm-dd HH:MM:SS") for ts in df.timestamp]
    
    # CONSECUTIVE BATTERY CHANGE ANALYSIS
    # Purpose: Calculate differences between consecutive battery readings
    # Reasoning: Understanding change patterns helps determine optimal analysis window
    battery_changes = []
    time_diffs_minutes = []
    
    for i in 2:nrow(df)
        battery_diff = abs(df.battery_percentage[i] - df.battery_percentage[i-1])
        # Convert milliseconds to minutes for human-readable units
        time_diff = Dates.value(df.datetime[i] - df.datetime[i-1]) / 60000.0  # minutes
        
        # Only include valid time differences (eliminate duplicate timestamps)
        if time_diff > 0
            push!(battery_changes, battery_diff)
            push!(time_diffs_minutes, time_diff)
        end
    end
    
    # BATTERY PRECISION DETECTION
    # Purpose: Identify the smallest measurable battery change (device precision)
    # Assumption: Minimum non-zero change indicates battery reporting precision
    non_zero_changes = battery_changes[battery_changes .> 0]
    
    if isempty(non_zero_changes)
        println("Warning: No battery changes detected in data")
        return 6  # Conservative default window for static battery scenarios
    end
    
    # Statistical analysis of battery change patterns
    min_change = minimum(non_zero_changes)  # Device precision indicator
    avg_change = mean(non_zero_changes)     # Average change magnitude
    median_change = median(non_zero_changes) # Typical change size
    # Zero change ratio indicates how often battery appears "stuck"
    zero_change_ratio = sum(battery_changes .== 0) / length(battery_changes)
    
    println("Battery precision analysis:")
    println("- Minimum battery change detected: $(round(min_change, digits=3))%")
    println("- Average battery change: $(round(avg_change, digits=3))%")
    println("- Median battery change: $(round(median_change, digits=3))%")
    println("- Ratio of unchanged readings: $(round(zero_change_ratio * 100, digits=1))%")
    
    # Calculate sampling frequency
    avg_sampling_interval = mean(time_diffs_minutes)
    println("- Average sampling interval: $(round(avg_sampling_interval, digits=1)) minutes")
    
    # ADAPTIVE WINDOW SIZE CALCULATION
    # Purpose: Determine optimal time window based on data characteristics
    # Reasoning: Different devices have different battery reporting precision
    #   - High precision: Can use shorter windows (more responsive)
    #   - Low precision: Need longer windows (avoid noise from quantization)
    if min_change >= 1.0
        # High precision (changes by 1% or more) - use shorter windows
        # Assumption: Frequent measurable changes allow responsive analysis
        optimal_window = max(10, avg_sampling_interval * 2)
        precision_level = "High"
    elseif zero_change_ratio > 0.7
        # Low precision (>70% unchanged readings) - need longer windows
        # Reasoning: Many "stuck" readings require longer windows to see actual drain
        optimal_window = max(20, avg_sampling_interval * 6)
        precision_level = "Low"
    elseif zero_change_ratio > 0.4
        # Medium precision (40-70% unchanged) - moderate windows
        optimal_window = max(15, avg_sampling_interval * 4)
        precision_level = "Medium"
    else
        # Good precision (<40% unchanged) - shorter windows
        optimal_window = max(10, avg_sampling_interval * 2)
        precision_level = "Good"
    end
    
    # WINDOW SIZE BOUNDS
    # Purpose: Prevent extremely short or long windows that would be meaningless
    # Reasoning: Too short = noise sensitive, too long = loses temporal resolution
    optimal_window = clamp(optimal_window, 3.0, 18.0)
    
    println("- Detected precision level: $precision_level")
    println("- Recommended time window: $(round(optimal_window, digits=1)) minutes")
    
    return round(optimal_window, digits=0)
end

# WINDOWED ANALYSIS FUNCTION
# Purpose: Calculate battery drain rates using time windows instead of point-to-point
# Assumptions:
#   - Battery readings may be identical for consecutive measurements
#   - CPU frequency varies within windows (need average)
#   - Only discharge periods are meaningful (ignore charging)
# Reasoning: Windows smooth out measurement noise and quantization effects
#   providing more accurate drain rate estimates than individual point differences
function calculate_windowed_analysis(df, window_minutes=15)
    """
    Analyze battery drain and CPU frequency over time windows to handle low-resolution data.
    Uses sliding windows to calculate meaningful drain rates even when consecutive
    battery readings are identical.
    """
    
    println("Analyzing data using $(window_minutes)-minute time windows...")
    
    # Convert timestamps to DateTime objects for temporal calculations
    df.datetime = [DateTime(ts, "yyyy-mm-dd HH:MM:SS") for ts in df.timestamp]
    
    # SLIDING WINDOW CREATION
    # Purpose: Create overlapping time windows for robust analysis
    # Reasoning: Sliding windows (50% overlap) provide smoother analysis than discrete windows
    windows = []
    start_time = df.datetime[1]
    end_time = df.datetime[end]
    
    current_window_start = start_time
    while current_window_start < end_time
        current_window_end = current_window_start + Minute(window_minutes)
        
        # Extract all data points within current time window
        window_data = df[(df.datetime .>= current_window_start) .& (df.datetime .<= current_window_end), :]
        
        # WINDOW VALIDATION AND ANALYSIS
        # Purpose: Ensure window has sufficient data and represents actual battery drain
        if nrow(window_data) >= 2
            # Use first/last readings in window for drain calculation
            window_start_battery = window_data.battery_percentage[1]
            window_end_battery = window_data.battery_percentage[end]
            window_time_diff = Dates.value(window_data.datetime[end] - window_data.datetime[1]) / 60000.0  # minutes
            
            # DRAIN VALIDATION CRITERIA
            # Purpose: Only analyze periods with actual battery consumption
            # Assumptions: 5 min minimum for meaningful measurement, must be discharging
            if window_time_diff >= 5.0 && window_start_battery > window_end_battery  # At least 5 minutes
                battery_drain = window_start_battery - window_end_battery
                # Convert to standardized rate (% per hour)
                drain_rate_per_hour = battery_drain / window_time_diff * 60.0
                
                # CPU FREQUENCY AVERAGING
                # Purpose: Calculate representative CPU frequency for the window
                # Reasoning: CPU frequency varies dynamically; average represents typical workload
                cpu_freq_data = window_data.cpu_frequency_ghz[.!ismissing.(window_data.cpu_frequency_ghz)]
                if !isempty(cpu_freq_data)
                    avg_cpu_freq = mean(cpu_freq_data)
                    
                    # Store validated window data for analysis
                    push!(windows, (
                        start_time = current_window_start,
                        end_time = current_window_end,
                        avg_cpu_freq = avg_cpu_freq,           # Average frequency during drain
                        battery_drain_rate = drain_rate_per_hour, # Standardized drain rate
                        time_span_minutes = window_time_diff,   # Actual window duration
                        battery_drop = battery_drain,          # Absolute battery consumption
                        data_points = nrow(window_data)        # Sample count for confidence
                    ))
                end
            end
        end
        
        # SLIDING WINDOW PROGRESSION
        # Purpose: 50% overlap provides smoother analysis than discrete windows
        # Reasoning: Reduces sensitivity to window boundary effects
        current_window_start += Minute(div(window_minutes, 2))
    end
    
    # Convert to DataFrame
    if !isempty(windows)
        window_df = DataFrame(windows)
        println("Created $(nrow(window_df)) analysis windows from $(nrow(df)) data points")
        return window_df, df
    else
        println("Warning: No valid analysis windows could be created")
        return nothing, df
    end
end

# LEGACY POINT-TO-POINT ANALYSIS
# Purpose: Calculate drain rates between consecutive measurements (fallback method)
# Assumptions:
#   - Consecutive measurements represent actual drain periods
#   - Point-to-point differences are meaningful despite quantization
# Reasoning: Simpler approach for comparison, but more sensitive to measurement noise
#   Used when windowed analysis fails or for visualization of raw data patterns
function calculate_legacy_drain_rate(df)
    """
    Legacy point-to-point analysis for comparison and visualization.
    """
    # Pre-allocate columns for drain rate calculations
    df.battery_drain_rate = Vector{Union{Float64, Missing}}(undef, nrow(df))
    df.time_diff_minutes = Vector{Union{Float64, Missing}}(undef, nrow(df))
    
    # CONSECUTIVE POINT ANALYSIS
    # Purpose: Calculate drain rate between each pair of consecutive measurements
    for i in 2:nrow(df)
        prev_battery = df.battery_percentage[i-1]
        curr_battery = df.battery_percentage[i]
        
        # Parse string timestamps to DateTime objects
        prev_time = DateTime(df.timestamp[i-1], "yyyy-mm-dd HH:MM:SS")
        curr_time = DateTime(df.timestamp[i], "yyyy-mm-dd HH:MM:SS")
        
        time_diff = Dates.value(curr_time - prev_time) / 60000.0  # Convert to minutes
        
        # DISCHARGE VALIDATION
        # Purpose: Only calculate rates for actual battery consumption periods
        # Reasoning: Charging periods would show negative drain (confusing for analysis)
        if time_diff > 0 && prev_battery >= curr_battery  # Only consider discharge
            battery_drain = prev_battery - curr_battery
            drain_rate = battery_drain / time_diff * 60  # Standardize to % per hour
            df.battery_drain_rate[i] = drain_rate
            df.time_diff_minutes[i] = time_diff
        else
            # Mark invalid periods as missing rather than zero
            df.battery_drain_rate[i] = missing
            df.time_diff_minutes[i] = missing
        end
    end
    
    return df
end

# CORRELATION ANALYSIS FUNCTION
# Purpose: Calculate and interpret correlation between CPU frequency and battery drain
# Assumptions:
#   - Windowed analysis is more accurate than point-to-point when available
#   - Correlation coefficient indicates strength of linear relationship
# Reasoning: Correlation analysis reveals if CPU frequency significantly impacts battery life
function analyze_correlation(window_df, df)
    println("\n=== CORRELATION ANALYSIS ===")
    
    # DATA SOURCE SELECTION
    # Purpose: Use best available analysis method (windowed preferred over point-to-point)
    if window_df === nothing
        println("No windowed analysis data available. Trying legacy point-to-point analysis...")
        
        # FALLBACK TO LEGACY ANALYSIS
        # Reasoning: Point-to-point as last resort when windowed analysis fails
        df = calculate_legacy_drain_rate(df)
        valid_data = df[completecases(df[!, [:cpu_frequency_ghz, :battery_drain_rate]]), :]
        
        if nrow(valid_data) < 2
            println("Warning: Insufficient data for correlation analysis")
            return nothing, nothing
        end
        
        cpu_freq = valid_data.cpu_frequency_ghz
        battery_drain = valid_data.battery_drain_rate
        analysis_method = "Point-to-point (Legacy)"
        analysis_data = valid_data
    else
        # WINDOWED ANALYSIS (PREFERRED METHOD)
        # Purpose: Use more accurate windowed drain rates for correlation
        println("Using windowed analysis for more accurate correlation...")
        cpu_freq = window_df.avg_cpu_freq
        battery_drain = window_df.battery_drain_rate
        window_size = round(mean(window_df.time_span_minutes), digits=1)
        analysis_method = "Time-windowed ($(window_size)-min windows)"
        analysis_data = window_df
    end
    
    # PEARSON CORRELATION CALCULATION
    # Purpose: Measure linear relationship strength between CPU frequency and battery drain
    # Assumption: Linear relationship is meaningful for this analysis
    correlation = cor(cpu_freq, battery_drain)
    
    println("Analysis method: $analysis_method")
    println("Data points used: $(length(cpu_freq))")
    println("CPU Frequency vs Battery Drain Rate")
    println("Correlation coefficient: $(round(correlation, digits=4))")
    
    # CORRELATION INTERPRETATION
    # Purpose: Provide human-readable interpretation of correlation strength
    # Reasoning: Statistical significance thresholds help users understand practical impact
    if correlation > 0.3
        println("Strong positive correlation: Higher CPU frequency -> Higher battery drain")
    elseif correlation > 0.1
        println("Moderate positive correlation: Higher CPU frequency -> Higher battery drain")
    elseif correlation > -0.1
        println("Weak correlation: CPU frequency has little impact on battery drain")
    else
        println("Negative correlation: Unexpected relationship")
    end
    
    # Basic statistics
    println("\nStatistics:")
    println("Average CPU frequency: $(round(mean(cpu_freq), digits=2)) GHz")
    println("Average battery drain rate: $(round(mean(battery_drain), digits=2))% per hour")
    println("Max battery drain rate: $(round(maximum(battery_drain), digits=2))% per hour")
    println("Min battery drain rate: $(round(minimum(battery_drain), digits=2))% per hour")
    
    if window_df !== nothing
        println("\nWindow Analysis Details:")
        println("Average time span per window: $(round(mean(window_df.time_span_minutes), digits=1)) minutes")
        println("Average battery drop per window: $(round(mean(window_df.battery_drop), digits=2))%")
        println("Data points per window: $(round(mean(window_df.data_points), digits=1))")
    end
    
    return correlation, analysis_data
end

# BUCKETED BATTERY DRAIN VISUALIZATION
# Purpose: Create bar chart showing average battery drain by CPU frequency ranges
# Assumptions:
#   - CPU frequencies cluster in ranges (bucketing reduces noise)
#   - Raw point-to-point data captures more outliers than windowed data
#   - Configurable bucket width allows adaptation to different frequency ranges
# Reasoning: Bucketing reveals patterns that might be obscured in scatter plots
#   and provides clear comparison between frequency ranges
function create_bucketed_drain_plot(df, window_df)
    """
    Create a bar chart showing bucketed battery drain percent per hour for each CPU frequency level.
    Uses smaller windows and adaptive bucketing to capture outlier behavior.
    """
    # DATA SOURCE: Raw point-to-point data (more measurements = better outlier capture)
    valid_drain_data = df[completecases(df[!, [:cpu_frequency_ghz, :battery_drain_rate]]), :]
    if nrow(valid_drain_data) == 0
        return plot(Layout(
            title="No Data Available for CPU Frequency Buckets",
            annotations=[attr(text="No valid data points found", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=false)]
        ))
    end
    
    freq_data = valid_drain_data.cpu_frequency_ghz
    drain_data = valid_drain_data.battery_drain_rate
    
    # ADAPTIVE BUCKET RANGE DETECTION
    # Purpose: Create frequency buckets that capture the actual data range including outliers
    # Reasoning: Fixed ranges might miss very low or very high frequencies in the data
    min_freq = minimum(freq_data)
    max_freq = maximum(freq_data)
    println("Frequency range: $(round(min_freq, digits=2)) - $(round(max_freq, digits=2)) GHz")
    
    # BUCKET CONFIGURATION
    # Purpose: Define bucket size and range that captures outliers while maintaining resolution
    # Assumption: Configurable bucket width allows adaptation to different frequency ranges
    bucket_width = get_bucket_width()
    bucket_start = floor(min_freq / bucket_width) * bucket_width  # Round down to bucket boundary
    bucket_end = ceil(max_freq / bucket_width) * bucket_width     # Round up to bucket boundary
    
    # MINIMUM RANGE ENFORCEMENT
    # Purpose: Ensure reasonable bucket range even for limited frequency data
    # Reasoning: Very narrow ranges might not be representative of typical CPU behavior
    bucket_start = min(bucket_start, 0.5)  # Extend to at least 0.5 GHz if needed
    bucket_end = max(bucket_end, 4.0)      # Extend to at least 4.0 GHz if needed
    
    bucket_edges = bucket_start:bucket_width:bucket_end
    bucket_labels = ["$(round(i, digits=2))-$(round(i+bucket_width, digits=2))" for i in bucket_edges[1:end-1]]
    
    println("Created $(length(bucket_labels)) buckets from $(round(bucket_start, digits=2)) to $(round(bucket_end, digits=2)) GHz")
    
    # Initialize arrays for bucket data
    bucket_means = Float64[]
    bucket_counts = Int[]
    bucket_stds = Float64[]
    
    # Calculate statistics for each bucket
    for i in 1:length(bucket_labels)
        bucket_min = bucket_edges[i]
        bucket_max = bucket_edges[i+1]
        
        # Find data points in this bucket
        in_bucket = (freq_data .>= bucket_min) .& (freq_data .< bucket_max)
        bucket_drain_values = drain_data[in_bucket]
        
        if length(bucket_drain_values) > 0
            push!(bucket_means, mean(bucket_drain_values))
            push!(bucket_counts, length(bucket_drain_values))
            push!(bucket_stds, length(bucket_drain_values) > 1 ? std(bucket_drain_values) : 0.0)
        else
            push!(bucket_means, 0.0)
            push!(bucket_counts, 0)
            push!(bucket_stds, 0.0)
        end
    end
    
    # BAR CHART VISUALIZATION WITH ERROR BARS
    # Purpose: Show average drain rates with variability indicators
    # Reasoning: Error bars reveal confidence in each bucket's measurements
    #   Color coding helps identify high-drain frequency ranges quickly
    p_buckets = plot(
        bar(
            x=bucket_labels,
            y=bucket_means,
            error_y=attr(
                type="data",
                array=bucket_stds,  # Standard deviation as error bars
                visible=true
            ),
            # TEXT ANNOTATIONS: Show mean+/-std and sample count
            text=[count > 0 ? "$(round(mean, digits=1))+/-$(round(std, digits=1))%/hr\n(n=$(count))" : "No data" 
                  for (mean, count, std) in zip(bucket_means, bucket_counts, bucket_stds)],
            textposition="auto",
            # COLOR CODING: Diverging color scale highlights high vs low drain rates
            marker=attr(
                color=[count > 0 ? mean : NaN for (mean, count) in zip(bucket_means, bucket_counts)],
                colorscale="RdYlBu_r",  # Red-Yellow-Blue reverse (red=high drain)
                showscale=true,
                colorbar=attr(title="Battery Drain<br>(%/hr)"),
                line=attr(color="black", width=1)  # Black borders for clarity
            ),
            hovertemplate="CPU Frequency: %{x} GHz<br>Avg Drain Rate: %{y:.2f}%/hr<br>Std Dev: %{error_y.array:.2f}<br>Data Points: %{text}<extra></extra>"
        ),
        Layout(
            title="Battery Drain by CPU Frequency Buckets<br><sub>$(bucket_width) GHz buckets with outlier capture (range: $(round(bucket_start, digits=2))-$(round(bucket_end, digits=2)) GHz)</sub>",
            xaxis_title="CPU Frequency Range (GHz)",
            yaxis_title="Average Battery Drain Rate (% per hour)",
            showlegend=false,
            xaxis=attr(tickangle=45, tickfont=attr(size=10))
        )
    )
    
    return p_buckets
end

# VIOLIN PLOT FOR DRAIN RATE DISTRIBUTIONS
# Purpose: Show complete distribution shape of battery drain within each frequency bucket
# Assumptions:
#   - Distribution shapes reveal important patterns (normal, skewed, bimodal, etc.)
#   - Same bucketing as bar chart allows direct comparison
#   - Need >=3 points per bucket for meaningful violin shapes
# Reasoning: Bar charts show averages; violins show full distribution including outliers
#   Reveals whether averages are representative or hiding complex patterns
function create_violin_drain_plot(df, window_df)
    """
    Create a violin plot showing the distribution of battery drain values for each CPU frequency bucket.
    Uses the same bucketing strategy as the bar chart for consistency.
    """
    # CONSISTENT DATA SOURCE: Same as bar chart for direct comparison
    valid_drain_data = df[completecases(df[!, [:cpu_frequency_ghz, :battery_drain_rate]]), :]
    if nrow(valid_drain_data) == 0
        return plot(Layout(
            title="No Data Available for CPU Frequency Violin Plot",
            annotations=[attr(text="No valid data points found", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=false)]
        ))
    end
    
    freq_data = valid_drain_data.cpu_frequency_ghz
    drain_data = valid_drain_data.battery_drain_rate
    
    # Use same adaptive bucketing as the bar chart
    min_freq = minimum(freq_data)
    max_freq = maximum(freq_data)
    bucket_width = get_bucket_width()
    bucket_start = floor(min_freq / bucket_width) * bucket_width
    bucket_end = ceil(max_freq / bucket_width) * bucket_width
    bucket_start = min(bucket_start, 0.5)
    bucket_end = max(bucket_end, 4.0)
    
    bucket_edges = bucket_start:bucket_width:bucket_end
    bucket_labels = ["$(round(i, digits=2))-$(round(i+bucket_width, digits=2))" for i in bucket_edges[1:end-1]]
    
    # Collect data for violin plot - we need all individual values, not just means
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for i in 1:length(bucket_labels)
        bucket_min = bucket_edges[i]
        bucket_max = bucket_edges[i+1]
        
        # Find data points in this bucket
        in_bucket = (freq_data .>= bucket_min) .& (freq_data .< bucket_max)
        bucket_drain_values = drain_data[in_bucket]
        
        if length(bucket_drain_values) > 2  # Need at least 3 points for a meaningful violin
            # Add all values from this bucket
            for val in bucket_drain_values
                push!(violin_data, val)
                push!(violin_labels, bucket_labels[i])
                push!(violin_colors, mean(bucket_drain_values))  # Color by bucket mean
            end
        end
    end
    
    if isempty(violin_data)
        return plot(Layout(
            title="Insufficient Data for Violin Plot",
            annotations=[attr(text="Need at least 3 points per bucket for violin plot", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=false)]
        ))
    end
    
    # Create violin plot
    p_violin = plot(
        violin(
            x=violin_labels,
            y=violin_data,
            box_visible=true,  # Show box plot inside violin
            meanline_visible=true,  # Show mean line
            points="outliers",  # Show outlier points
            marker=attr(
                size=3,
                color=violin_colors,
                colorscale="RdYlBu_r",
                showscale=true,
                colorbar=attr(title="Bucket Mean<br>Drain (%/hr)", x=1.02)
            ),
            line=attr(color="black", width=1),
            fillcolor="rgba(100,100,100,0.3)",
            hovertemplate="CPU Frequency: %{x} GHz<br>Battery Drain: %{y:.2f}%/hr<extra></extra>"
        ),
        Layout(
            title="Battery Drain Distribution by CPU Frequency<br><sub>Violin plot showing drain rate distributions within $(bucket_width) GHz buckets</sub>",
            xaxis_title="CPU Frequency Range (GHz)",
            yaxis_title="Battery Drain Rate (% per hour)",
            showlegend=false,
            xaxis=attr(tickangle=45, tickfont=attr(size=10)),
            plot_bgcolor="white"
        )
    )
    
    return p_violin
end

# CPU INSTRUCTIONS PER BATTERY ANALYSIS
# Purpose: Calculate computational efficiency (instructions per 100% battery) by frequency
# Assumptions:
#   - Modern CPUs execute ~2.5 instructions per clock cycle (average)
#   - 1 GHz = 1 billion cycles per second
#   - Efficiency = total possible instructions / battery drain rate
# Reasoning: Shows optimal frequency ranges for computational work vs. battery consumption
#   Higher frequencies may provide more compute but at diminishing battery efficiency
function create_instructions_per_battery_plot(df, window_df)
    """
    Create a bar chart showing estimated CPU instructions possible per 100% battery drain
    for each CPU frequency bucket. This shows computational efficiency at different frequencies.
    """
    # CONSISTENT DATA SOURCE: Same filtering as other plots
    valid_drain_data = df[completecases(df[!, [:cpu_frequency_ghz, :battery_drain_rate]]), :]
    # ZERO-VALUE FILTERING: Remove periods with no meaningful battery drain
    valid_drain_data = valid_drain_data[valid_drain_data.battery_drain_rate .> 0.01, :]
    
    if nrow(valid_drain_data) == 0
        return plot(Layout(
            title="No Data Available for CPU Instructions Analysis",
            annotations=[attr(text="No valid data points found", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=false)]
        ))
    end
    
    freq_data = valid_drain_data.cpu_frequency_ghz
    drain_data = valid_drain_data.battery_drain_rate
    
    # Use same adaptive bucketing as other plots
    min_freq = minimum(freq_data)
    max_freq = maximum(freq_data)
    bucket_width = get_bucket_width()
    bucket_start = floor(min_freq / bucket_width) * bucket_width
    bucket_end = ceil(max_freq / bucket_width) * bucket_width
    bucket_start = min(bucket_start, 0.5)
    bucket_end = max(bucket_end, 4.0)
    
    bucket_edges = bucket_start:bucket_width:bucket_end
    bucket_labels = ["$(round(i, digits=2))-$(round(i+bucket_width, digits=2))" for i in bucket_edges[1:end-1]]
    
    # Calculate instructions per 100% battery for each bucket
    instructions_per_100_battery = Float64[]
    bucket_counts = Int[]
    bucket_freq_means = Float64[]
    
    for i in 1:length(bucket_labels)
        bucket_min = bucket_edges[i]
        bucket_max = bucket_edges[i+1]
        
        # Find data points in this bucket
        in_bucket = (freq_data .>= bucket_min) .& (freq_data .< bucket_max)
        bucket_drain_values = drain_data[in_bucket]
        bucket_freq_values = freq_data[in_bucket]
        
        if length(bucket_drain_values) > 0
            avg_drain_rate = mean(bucket_drain_values)  # %/hour
            avg_freq = mean(bucket_freq_values)  # GHz
            
            # COMPUTATIONAL EFFICIENCY CALCULATION
            # Purpose: Estimate total CPU instructions possible before 100% battery drain
            # CORE ASSUMPTIONS (based on modern CPU architecture):
            # - Modern CPUs execute ~1-4 instructions per clock cycle (2.5 average for mixed workloads)
            # - 1 GHz = 1,000,000,000 cycles per second
            # - 1 hour = 3,600 seconds
            instructions_per_cycle = 2.5  # Conservative estimate for mixed instruction types
            cycles_per_second = avg_freq * 1e9  # Convert GHz to Hz
            cycles_per_hour = cycles_per_second * 3600  # Total cycles in one hour
            instructions_per_hour = cycles_per_hour * instructions_per_cycle  # Max theoretical instructions
            
            # EFFICIENCY METRIC: Instructions per 100% battery consumption
            # Formula: (instructions/hour) / (battery_%/hour) * 100
            if avg_drain_rate > 0
                instructions_per_100_battery_value = (instructions_per_hour / avg_drain_rate) * 100
                push!(instructions_per_100_battery, instructions_per_100_battery_value)
                push!(bucket_counts, length(bucket_drain_values))
                push!(bucket_freq_means, avg_freq)
            else
                # Handle edge case of zero drain (shouldn't occur after filtering)
                push!(instructions_per_100_battery, 0.0)
                push!(bucket_counts, 0)
                push!(bucket_freq_means, avg_freq)
            end
        else
            push!(instructions_per_100_battery, 0.0)
            push!(bucket_counts, 0)
            push!(bucket_freq_means, 0.0)
        end
    end
    
    # Convert to more readable units (trillions of instructions)
    instructions_trillions = instructions_per_100_battery ./ 1e12
    
    # Create bar chart
    p_instructions = plot(
        bar(
            x=bucket_labels,
            y=instructions_trillions,
            text=[count > 0 ? "$(round(instr, digits=1))T\n(cpu runs at $(round(freq, digits=2)) GHz avg)\n(n=$(count))" : "No data"
                  for (instr, freq, count) in zip(instructions_trillions, bucket_freq_means, bucket_counts)],
            textposition="auto",
            marker=attr(
                color=[count > 0 ? instr : NaN for (instr, count) in zip(instructions_trillions, bucket_counts)],
                colorscale="Viridis",
                showscale=true,
                colorbar=attr(title="Instructions<br>(Trillions)", x=1.02),
                line=attr(color="black", width=1)
            ),
            hovertemplate="CPU Frequency: %{x} GHz<br>Instructions per 100% battery: %{y:.1f} trillion<br>Total Instructions: %{text}<extra></extra>"
        ),
        Layout(
            title="CPU Instructions per 100% Battery Drain<br><sub>Computational efficiency by frequency bucket (assumes 2.5 instructions/cycle)</sub>",
            xaxis_title="CPU Frequency Range (GHz)",
            yaxis_title="Instructions per 100% Battery (Trillions)",
            showlegend=false,
            xaxis=attr(tickangle=45, tickfont=attr(size=10))
        )
    )
    
    return p_instructions
end

# COMPREHENSIVE VISUALIZATION DASHBOARD
# Purpose: Create multiple complementary plots for thorough analysis
# Assumptions:
#   - Multiple view types reveal different aspects of the data
#   - Time series show temporal patterns, correlations show relationships
#   - Bucketed views reduce noise, violin plots show distributions
# Reasoning: Single plot types can miss important patterns; comprehensive dashboard
#   provides complete picture of CPU frequency vs. battery drain relationship
function create_visualizations(df, correlation, valid_data, window_df=nothing)
    # Create a comprehensive dashboard with multiple plots
    
    # PLOT 1: BATTERY LEVEL TIME SERIES
    # Purpose: Show overall battery consumption pattern over time
    # Reasoning: Temporal context helps identify charging cycles, usage patterns
    p1 = plot(
        scatter(
            x=df.timestamp,
            y=df.battery_percentage,
            mode="lines+markers",
            name="Battery Level",
            line=attr(color="green", width=2),  # Green = battery color convention
            marker=attr(size=4)
        ),
        Layout(
            title="Battery Level Over Time",
            xaxis_title="Time",
            yaxis_title="Battery Percentage (%)",
            yaxis=attr(range=[0, 100]),  # Fixed scale for battery percentage
            showlegend=false
        )
    )
    
    # PLOT 2: CPU FREQUENCY TIME SERIES
    # Purpose: Show CPU frequency variation over time (dynamic frequency scaling)
    # Reasoning: Reveals workload patterns, frequency scaling behavior, correlates with battery plot
    p2 = plot(
        scatter(
            x=df.timestamp,
            y=df.cpu_frequency_ghz,
            mode="lines+markers",
            name="CPU Frequency",
            line=attr(color="blue", width=2),  # Blue = CPU color convention
            marker=attr(size=4)
        ),
        Layout(
            title="CPU Frequency Over Time",
            xaxis_title="Time",
            yaxis_title="CPU Frequency (GHz)",
            showlegend=false
        )
    )
    
    # PLOT 3: CPU FREQUENCY vs BATTERY DRAIN CORRELATION
    # Purpose: Show relationship between CPU frequency and battery consumption with polynomial fit
    # Assumptions: Bucketed data reduces noise while preserving relationship patterns
    # Reasoning: 3rd-degree polynomial can capture non-linear efficiency curves
    if !isnothing(correlation) && nrow(valid_data) > 0
        # BUCKETED DATA FOR SCATTER PLOT
        # Purpose: Use same bucketing as bar chart for consistency and noise reduction
        raw_valid_data = df[completecases(df[!, [:cpu_frequency_ghz, :battery_drain_rate]]), :]
        raw_valid_data = raw_valid_data[raw_valid_data.battery_drain_rate .> 0.01, :]  # Filter zeros
        
        if nrow(raw_valid_data) > 0
            freq_data = raw_valid_data.cpu_frequency_ghz
            drain_data = raw_valid_data.battery_drain_rate
            
            # Use same adaptive bucketing as bar chart
            min_freq = minimum(freq_data)
            max_freq = maximum(freq_data)
            bucket_width = get_bucket_width()
            bucket_start = floor(min_freq / bucket_width) * bucket_width
            bucket_end = ceil(max_freq / bucket_width) * bucket_width
            bucket_start = min(bucket_start, 0.5)
            bucket_end = max(bucket_end, 4.0)
            
            bucket_edges = bucket_start:bucket_width:bucket_end
            
            # BUCKETED SCATTER POINT CREATION
            # Purpose: Convert frequency ranges to single points for scatter plot
            # Reasoning: Reduces overplotting while preserving statistical relationships
            x_data = Float64[]  # Bucket center frequencies
            y_data = Float64[]  # Mean drain rates per bucket
            bucket_sizes = Int[]  # Sample sizes for marker scaling
            
            for i in 1:length(bucket_edges)-1
                bucket_min = bucket_edges[i]
                bucket_max = bucket_edges[i+1]
                
                # BUCKET POPULATION
                # Find all measurements within current frequency bucket
                in_bucket = (freq_data .>= bucket_min) .& (freq_data .< bucket_max)
                bucket_drain_values = drain_data[in_bucket]
                bucket_freq_values = freq_data[in_bucket]
                
                if length(bucket_drain_values) > 0
                    # BUCKET AGGREGATION
                    # Use geometric center as x-coordinate, mean drain as y-coordinate
                    bucket_center = (bucket_min + bucket_max) / 2
                    mean_drain = mean(bucket_drain_values)
                    
                    push!(x_data, bucket_center)
                    push!(y_data, mean_drain)
                    push!(bucket_sizes, length(bucket_drain_values))  # For marker sizing
                end
            end
            
            analysis_type = "Bucketed ($(bucket_width) GHz buckets)"
        else
            # Fallback to original data if bucketing fails
            if window_df !== nothing && hasproperty(valid_data, :avg_cpu_freq)
                x_data = valid_data.avg_cpu_freq
                y_data = valid_data.battery_drain_rate
                analysis_type = "Windowed"
                bucket_sizes = fill(1, length(x_data))
            else
                x_data = valid_data.cpu_frequency_ghz
                y_data = valid_data.battery_drain_rate
                analysis_type = "Point-to-Point"
                bucket_sizes = fill(1, length(x_data))
            end
        end
        
        p3 = plot(
            scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                name="Bucketed Data Points",
                marker=attr(
                    size=[min(50, max(8, size/2)) for size in bucket_sizes],  # Size based on bucket sample count
                    color=y_data,
                    colorscale="Viridis",
                    showscale=true,
                    colorbar=attr(title="Battery Drain Rate (%/hr)"),
                    line=attr(color="black", width=1)
                ),
                text=[
                    "CPU Frequency: $(round(x, digits=2)) GHz<br>Avg Drain Rate: $(round(y, digits=2))%/hr<br>Sample Count: $(size)"
                    for (x, y, size) in zip(x_data, y_data, bucket_sizes)
                ],
                hovertemplate="%{text}<extra></extra>"
            ),
            Layout(
                title="CPU Frequency vs Battery Drain Rate<br><sub>$(analysis_type) - 3rd Degree Polynomial Fit</sub>",
                xaxis_title="CPU Frequency (GHz)",
                yaxis_title="Battery Drain Rate (% per hour)",
                showlegend=false
            )
        )
        
        # POLYNOMIAL FIT ANALYSIS
        # Purpose: Fit 3rd-degree polynomial to reveal non-linear efficiency patterns
        # Assumptions: CPU power consumption may have non-linear relationship with frequency
        # Reasoning: Linear fits may miss important efficiency curves or saturation effects
        if length(x_data) >= 4  # Minimum points required for stable cubic fit
            x_vals = x_data
            y_vals = y_data
            
            try
                # CUBIC POLYNOMIAL FITTING
                # Purpose: Capture potential non-linear relationships (efficiency curves)
                degree = 3  # Fixed cubic degree for consistency
                # Create Vandermonde matrix: [1, x, x^2, x^3] for each data point
                A = hcat([x_vals.^i for i in 0:degree]...)
                coeffs = A \ y_vals  # Least squares solution (minimizes squared residuals)
                
                # GOODNESS OF FIT CALCULATION
                # Purpose: Quantify how well polynomial explains the data
                y_pred = A * coeffs
                ss_res = sum((y_vals .- y_pred).^2)      # Sum of squared residuals
                ss_tot = sum((y_vals .- mean(y_vals)).^2) # Total sum of squares
                r_squared = 1 - (ss_res / ss_tot)        # Coefficient of determination
                
                # SMOOTH CURVE GENERATION
                # Purpose: Create high-resolution curve for smooth visualization
                x_min, x_max = extrema(x_vals)
                x_range = x_max - x_min
                x_smooth = range(x_min - 0.1*x_range, x_max + 0.1*x_range, length=100)  # 10% extension
                
                # POLYNOMIAL EVALUATION
                # Explicit 3rd-degree polynomial: a0 + a1*x + a2*x^2 + a3*x^3
                y_smooth = coeffs[1] .+ coeffs[2] .* x_smooth .+ coeffs[3] .* (x_smooth.^2) .+ coeffs[4] .* (x_smooth.^3)
                
                # Determine fit quality description
                fit_quality = if r_squared > 0.7
                    "Strong"
                elseif r_squared > 0.4
                    "Moderate"
                else
                    "Weak"
                end
                
                addtraces!(p3, scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode="lines",
                    name="3rd Degree Polynomial",
                    line=attr(color="red", width=3),
                    hovertemplate="3rd Degree Polynomial<br>R^2: $(round(r_squared, digits=3))<br>Quality: $(fit_quality)<extra></extra>"
                ))
                
                # Update title to include fit information
                relayout!(p3, title="CPU Frequency vs Battery Drain Rate<br><sub>$(analysis_type) - 3rd Degree Polynomial (R^2=$(round(r_squared, digits=3)))</sub>")
                
            catch e
                println("Warning: 3rd degree polynomial fit failed: $e")
                # Fallback to linear trend
                x_mean = mean(x_vals)
                y_mean = mean(y_vals)
                slope = sum((x_vals .- x_mean) .* (y_vals .- y_mean)) / sum((x_vals .- x_mean).^2)
                intercept = y_mean - slope * x_mean
                
                x_trend = [minimum(x_vals), maximum(x_vals)]
                y_trend = slope .* x_trend .+ intercept
                
                addtraces!(p3, scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    name="Linear Fallback",
                    line=attr(color="red", width=3, dash="dash")
                ))
            end
        elseif length(x_data) > 0
            # Linear trend for very small datasets
            x_vals = x_data
            y_vals = y_data
            x_mean = mean(x_vals)
            y_mean = mean(y_vals)
            slope = sum((x_vals .- x_mean) .* (y_vals .- y_mean)) / sum((x_vals .- x_mean).^2)
            intercept = y_mean - slope * x_mean
            
            x_trend = [minimum(x_vals), maximum(x_vals)]
            y_trend = slope .* x_trend .+ intercept
            
            addtraces!(p3, scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Linear Trend",
                line=attr(color="red", width=3, dash="dash")
            ))
        end
    else
        p3 = plot(
            Layout(
                title="Insufficient Data for Correlation Analysis",
                annotations=[
                    attr(
                        text="Please collect more data by running record_hardware.jl",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=false,
                        font=attr(size=16)
                    )
                ]
            )
        )
    end
    
    # PLOT 4: BATTERY DRAIN RATE TIME SERIES
    # Purpose: Show instantaneous battery consumption rates over time
    # Assumptions: Zero drain periods are not meaningful (charging, idle, measurement errors)
    # Reasoning: Filtered drain rates reveal actual consumption patterns without noise
    drain_data = df[completecases(df[!, [:battery_drain_rate]]), :]
    # ZERO-VALUE FILTERING
    # Purpose: Remove non-drain periods that would skew visualization
    drain_data = drain_data[drain_data.battery_drain_rate .> 0.01, :]  # 0.01%/hr threshold
    
    if nrow(drain_data) > 0
        p4 = plot(
            scatter(
                x=drain_data.timestamp,
                y=drain_data.battery_drain_rate,
                mode="lines+markers",
                name="Battery Drain Rate",
                line=attr(color="orange", width=2),
                marker=attr(size=4),
                hovertemplate="Time: %{x}<br>Drain Rate: %{y:.2f}%/hr<extra></extra>"
            ),
            Layout(
                title="Battery Drain Rate Over Time<br><sub>Excluding zero/near-zero values (showing only active drain periods)</sub>",
                xaxis_title="Time",
                yaxis_title="Battery Drain Rate (% per hour)",
                showlegend=false
            )
        )
    else
        p4 = plot(
            Layout(
                title="Battery Drain Rate Not Available",
                annotations=[
                    attr(
                        text="Insufficient data for battery drain rate analysis",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=false,
                        font=attr(size=16)
                    )
                ]
            )
        )
    end
    
    # Plot 5: Bucketed CPU frequency vs battery drain
    p5 = create_bucketed_drain_plot(df, window_df)
    
    # Plot 6: Violin plot of CPU frequency vs battery drain distribution
    p6 = create_violin_drain_plot(df, window_df)
    
    # Plot 7: CPU instructions per 100% battery drain
    p7 = create_instructions_per_battery_plot(df, window_df)
    
    # SILENT PLOT GENERATION
    # Purpose: Generate all visualizations without interrupting workflow with pop-ups
    # Reasoning: Batch generation is more efficient, pop-ups interrupt analysis flow
    println("\nGenerating Hardware Monitoring Dashboard...")
    println("1. Battery Level Over Time - Generated")
    println("2. CPU Frequency Over Time - Generated") 
    println("3. CPU Frequency vs Battery Drain Correlation - Generated")
    println("4. Battery Drain Rate Over Time - Generated")
    println("5. Battery Drain by CPU Frequency Buckets - Generated")
    println("6. Battery Drain Distribution by CPU Frequency (Violin Plot) - Generated")
    println("7. CPU Instructions per 100% Battery Drain - Generated")
    
    # Try to create a simple 2x2 dashboard for the first 4 plots
    dashboard = [p1 p2; p3 p4]
    
    # Save individual plots and dashboard
    savefig(p1, "battery_level_over_time.html")
    savefig(p2, "cpu_frequency_over_time.html") 
    savefig(p3, "cpu_vs_battery_correlation.html")
    savefig(p4, "battery_drain_rate_over_time.html")
    savefig(p5, "cpu_frequency_buckets.html")
    savefig(p6, "cpu_frequency_violin.html")
    savefig(p7, "cpu_instructions_per_battery.html")
    
    try
        savefig(dashboard, "hardware_analysis_dashboard.html")
        println("\nDashboard saved as 'hardware_analysis_dashboard.html'")
    catch
        println("\nIndividual plots saved as separate HTML files")
    end
    
    return dashboard
end

function generate_report(df, correlation, valid_data, window_df=nothing)
    println("\n" * "="^60)
    println("HARDWARE MONITORING ANALYSIS REPORT")
    println("="^60)
    println("Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("Data points collected: $(nrow(df))")
    
    if !isnothing(correlation) && nrow(valid_data) > 0
        println("\nKEY FINDINGS:")
        println("- Correlation between CPU frequency and battery drain: $(round(correlation, digits=3))")
        
        if correlation > 0.3
            println("- STRONG evidence that higher CPU frequencies reduce battery life")
            println("- Recommendation: Monitor and limit max CPU frequency to extend battery life")
        elseif correlation > 0.1
            println("- MODERATE evidence that higher CPU frequencies reduce battery life")
            println("- Recommendation: Monitor CPU frequency during intensive tasks")
        else
            println("- Limited evidence of CPU frequency impact on battery life")
            println("- Other factors may be more significant")
        end
        
        # Performance insights - handle both windowed and legacy data
        if window_df !== nothing && hasproperty(valid_data, :avg_cpu_freq)
            # Windowed analysis data
            cpu_freq_col = valid_data.avg_cpu_freq
            freq_q75 = quantile(cpu_freq_col, 0.75)
            freq_q25 = quantile(cpu_freq_col, 0.25)
            high_freq_data = valid_data[cpu_freq_col .> freq_q75, :]
            low_freq_data = valid_data[cpu_freq_col .<= freq_q25, :]
        else
            # Legacy point-to-point data
            cpu_freq_col = valid_data.cpu_frequency_ghz
            freq_q75 = quantile(cpu_freq_col, 0.75)
            freq_q25 = quantile(cpu_freq_col, 0.25)
            high_freq_data = valid_data[cpu_freq_col .> freq_q75, :]
            low_freq_data = valid_data[cpu_freq_col .<= freq_q25, :]
        end
        
        if nrow(high_freq_data) > 0 && nrow(low_freq_data) > 0
            high_freq_drain = mean(high_freq_data.battery_drain_rate)
            low_freq_drain = mean(low_freq_data.battery_drain_rate)
            
            println("\nPERFORMANCE COMPARISON:")
            println("- High CPU frequency periods: $(round(high_freq_drain, digits=2))% battery drain per hour")
            println("- Low CPU frequency periods: $(round(low_freq_drain, digits=2))% battery drain per hour")
            println("- Difference: $(round(high_freq_drain - low_freq_drain, digits=2))% per hour")
            
            if high_freq_drain > low_freq_drain
                battery_savings = (high_freq_drain - low_freq_drain) / high_freq_drain * 100
                println("- Potential battery life improvement: $(round(battery_savings, digits=1))% by using lower frequencies")
            end
        end
    else
        println("\nINSUFFICIENT DATA:")
        println("- Unable to perform correlation analysis")
        println("- Recommendation: Run record_hardware.jl for longer period")
    end
    
    println("\n" * "="^60)
end

# MAIN ANALYSIS ORCHESTRATION
# Purpose: Coordinate the complete analysis workflow from data loading to report generation
# Assumptions:
#   - Database contains sufficient data for meaningful analysis
#   - Adaptive windowing provides better results than fixed parameters
#   - Multiple analysis methods (windowed + legacy) provide comprehensive view
# Reasoning: Structured workflow ensures consistent analysis while adapting to data characteristics
function main()
    println("Hardware Data Analysis Tool")
    println("Loading data from hw-data.db...")
    
    # PHASE 1: DATA LOADING AND VALIDATION
    df = load_data()
    
    # PHASE 2: ADAPTIVE PARAMETER DETECTION
    # Purpose: Automatically optimize analysis parameters for this dataset
    optimal_window_minutes = detect_battery_precision_and_optimal_window(df)
    println("\nUsing $(optimal_window_minutes)-minute time windows for analysis...")
    
    # PHASE 3: WINDOWED ANALYSIS (PRIMARY METHOD)
    # Purpose: Calculate robust drain rates using time windows
    window_df, df = calculate_windowed_analysis(df, optimal_window_minutes)
    
    # PHASE 4: LEGACY ANALYSIS (BACKUP/COMPARISON METHOD)
    # Purpose: Calculate point-to-point drain rates for comparison and visualization
    df = calculate_legacy_drain_rate(df)
    
    # PHASE 5: STATISTICAL CORRELATION ANALYSIS
    result = analyze_correlation(window_df, df)
    
    # PHASE 6: VISUALIZATION AND REPORTING
    if result !== nothing
        correlation, valid_data = result
        
        # Generate comprehensive visualization dashboard
        println("\nCreating visualizations...")
        dashboard = create_visualizations(df, correlation, valid_data, window_df)
        
        # Generate text-based analysis report
        generate_report(df, correlation, valid_data, window_df)
    else
        println("Unable to perform analysis due to insufficient data.")
    end
end

# SCRIPT EXECUTION GUARD
# Purpose: Only run main() when script is executed directly (not when imported)
# Reasoning: Allows script to be both executable and importable as a module
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
