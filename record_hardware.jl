#!/usr/bin/env julia

# HARDWARE MONITORING DATA COLLECTION TOOL
# Purpose: Record CPU frequency and battery percentage at regular intervals
# Assumptions:
#   - System provides accessible hardware monitoring interfaces
#   - Regular sampling (5-second intervals) captures meaningful patterns
#   - Cross-platform compatibility needed (Windows, Linux, macOS)
#   - Long-term data collection for statistical analysis
# Reasoning: Continuous monitoring reveals relationships between CPU frequency
#   and battery consumption that single measurements cannot show

using Pkg

# DEPENDENCY MANAGEMENT BLOCK
# Purpose: Automatically install required packages for standalone execution
# Reasoning: Ensures script works on fresh systems without manual setup
packages = ["SQLite", "DataFrames", "Dates"]
for pkg in packages
    try
        eval(:(using $(Symbol(pkg))))
    catch
        println("Installing $pkg...")
        Pkg.add(pkg)
        eval(:(using $(Symbol(pkg))))
    end
end

using SQLite    # Database storage for time series data
using DataFrames # Data manipulation (used by SQLite)
using Dates     # Timestamp generation and formatting

# CROSS-PLATFORM CPU FREQUENCY DETECTION
# Purpose: Get current/maximum CPU frequency across different operating systems
# Assumptions:
#   - Each OS provides different interfaces for CPU frequency monitoring
#   - Some systems report max frequency, others current frequency
#   - Frequency may be static (max) or dynamic (current scaling frequency)
# Reasoning: CPU frequency directly affects power consumption and performance
#   Different OSes require different system calls and file access methods
function get_cpu_frequency()
    # WINDOWS IMPLEMENTATION
    # Purpose: Use Windows Management Instrumentation (WMI) via PowerShell
    # Assumption: MaxClockSpeed represents current or maximum achievable frequency
    # Limitation: May not capture dynamic frequency scaling in real-time
    if Sys.iswindows()
        try
            cmd = `powershell -Command "Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty MaxClockSpeed"`
            result = read(cmd, String)
            freq = parse(Float64, strip(result)) / 1000.0  # Convert MHz to GHz
            return freq
        catch
            return nothing  # Return nothing on failure (handled gracefully)
        end
    # LINUX IMPLEMENTATION
    # Purpose: Use /proc/cpuinfo for current frequency or /sys/devices for scaling frequency
    # Assumption: Linux provides real-time frequency information through pseudo-filesystems
    # Reasoning: Captures actual dynamic frequency scaling behavior
    elseif Sys.islinux()
        # PRIMARY METHOD: /proc/cpuinfo (current frequency)
        try
            cpuinfo = read("/proc/cpuinfo", String)
            for line in split(cpuinfo, '\n')
                if startswith(line, "cpu MHz")
                    freq_str = split(line, ':')[2]
                    freq = parse(Float64, strip(freq_str)) / 1000.0  # Convert MHz to GHz
                    return freq
                end
            end
        catch
            # FALLBACK METHOD: cpufreq interface (more accurate for modern systems)
            # Purpose: Get real-time scaling frequency from kernel frequency governor
            # Reasoning: scaling_cur_freq shows actual current frequency including power management
            try
                freq_files = readdir("/sys/devices/system/cpu/cpu0/cpufreq/")
                if "scaling_cur_freq" in freq_files
                    freq_str = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", String)
                    freq = parse(Float64, strip(freq_str)) / 1000000.0  # Convert kHz to GHz
                    return freq
                end
            catch
                return nothing  # Both methods failed
            end
        end
    # MACOS IMPLEMENTATION
    # Purpose: Use sysctl or system_profiler for CPU frequency detection
    # Assumption: Apple systems may report max frequency rather than current scaling
    # Reasoning: Different macOS versions and architectures (Intel vs Apple Silicon) need different approaches
    elseif Sys.isapple()
        # PRIMARY METHOD: sysctl (traditional approach)
        try
            cmd = `sysctl -n hw.cpufrequency_max`
            result = read(cmd, String)
            freq = parse(Float64, strip(result)) / 1000000000.0  # Convert Hz to GHz
            return freq
        catch
            # FALLBACK METHOD: system_profiler (works on Apple Silicon and newer Intel Macs)
            # Purpose: Parse human-readable system information when sysctl fails
            # Reasoning: Apple Silicon and some newer systems don't expose hw.cpufrequency_max
            try
                cmd = `system_profiler SPHardwareDataType`
                result = read(cmd, String)
                for line in split(result, '\n')
                    if contains(line, "Processor Speed")
                        # REGEX PARSING: Extract frequency value from "Processor Speed: X.X GHz" format
                        freq_match = match(r"(\d+\.?\d*)\s*GHz", line)
                        if freq_match !== nothing
                            return parse(Float64, freq_match.captures[1])
                        end
                    end
                end
            catch
                return nothing  # All methods failed
            end
        end
    end
    return nothing  # Unsupported platform
end

# CROSS-PLATFORM BATTERY PERCENTAGE DETECTION
# Purpose: Get precise battery charge level across different operating systems
# Assumptions:
#   - Higher precision battery measurements provide better analysis
#   - Some systems provide calculated percentages, others raw charge values
#   - Battery reporting varies significantly between platforms and hardware
# Reasoning: Battery percentage is the primary dependent variable for power analysis
#   Precision matters for detecting small changes over time
function get_battery_percentage()
    # WINDOWS IMPLEMENTATION
    # Purpose: Use Windows Management Instrumentation (WMI) for battery data
    # Assumption: EstimatedChargeRemaining provides percentage (0-100)
    if Sys.iswindows()
        # PRIMARY METHOD: Precise percentage with rounding
        # Purpose: Get more precise measurements and round to 2 decimal places
        try
            cmd = `powershell -Command "Get-WmiObject -Class Win32_Battery | ForEach-Object { [math]::Round((\$_.EstimatedChargeRemaining), 2) }"`
            result = read(cmd, String)
            percentage = parse(Float64, strip(result))
            if percentage > 0  # Validate reasonable percentage
                return percentage
            end
        catch
            # Silently continue to fallback method
        end
        
        # FALLBACK METHOD: Standard WMI battery query
        # Purpose: Use simpler approach if precise method fails
        try
            cmd = `powershell -Command "Get-WmiObject -Class Win32_Battery | Select-Object -ExpandProperty EstimatedChargeRemaining"`
            result = read(cmd, String)
            return parse(Float64, strip(result))
        catch
            return nothing  # No battery or access denied
        end
    # LINUX IMPLEMENTATION
    # Purpose: Use /sys/class/power_supply for precise battery measurements
    # Assumption: Linux exposes raw battery data through sysfs pseudo-filesystem
    # Reasoning: Calculate precise percentage from raw charge/energy values when possible
    elseif Sys.islinux()
        try
            # BATTERY DISCOVERY
            # Purpose: Find battery devices (usually BAT0, BAT1, etc.)
            battery_dirs = filter(x -> startswith(x, "BAT"), readdir("/sys/class/power_supply/"))
            if !isempty(battery_dirs)
                battery_path = "/sys/class/power_supply/$(battery_dirs[1])"  # Use first battery
                
                # PRECISION METHOD 1: charge_now/charge_full (microampere-hours)
                # Purpose: Calculate precise percentage from raw charge values
                # Reasoning: More accurate than pre-calculated capacity values
                if isfile("$battery_path/charge_now") && isfile("$battery_path/charge_full")
                    charge_now = parse(Float64, strip(read("$battery_path/charge_now", String)))
                    charge_full = parse(Float64, strip(read("$battery_path/charge_full", String)))
                    if charge_full > 0  # Avoid division by zero
                        return (charge_now / charge_full) * 100.0
                    end
                    
                # PRECISION METHOD 2: energy_now/energy_full (microwatt-hours)
                # Purpose: Alternative calculation for batteries that report energy instead of charge
                # Reasoning: Some battery controllers expose energy rather than charge values
                elseif isfile("$battery_path/energy_now") && isfile("$battery_path/energy_full")
                    energy_now = parse(Float64, strip(read("$battery_path/energy_now", String)))
                    energy_full = parse(Float64, strip(read("$battery_path/energy_full", String)))
                    if energy_full > 0
                        return (energy_now / energy_full) * 100.0
                    end
                end
                
                # FALLBACK METHOD: capacity file (pre-calculated integer percentage)
                # Purpose: Use kernel-calculated percentage when raw values unavailable
                # Limitation: Usually integer values only (lower precision)
                if isfile("$battery_path/capacity")
                    capacity_str = read("$battery_path/capacity", String)
                    return parse(Float64, strip(capacity_str))
                end
            end
        catch
            return nothing  # Permission denied or battery access failed
        end
    # MACOS IMPLEMENTATION
    # Purpose: Use pmset (Power Management Set) command for battery information
    # Assumption: pmset provides reliable battery percentage information
    # Limitation: Usually returns integer percentages (lower precision than Linux)
    elseif Sys.isapple()
        try
            cmd = `pmset -g batt`  # Get battery status
            result = read(cmd, String)
            for line in split(result, '\n')
                # REGEX PARSING: Extract percentage from pmset output format
                # Example: "InternalBattery-0 (id=12345) 85%; discharging; 2:34 remaining"
                battery_match = match(r"(\d+)%", line)
                if battery_match !== nothing
                    return parse(Float64, battery_match.captures[1])
                end
            end
        catch
            return nothing  # Command failed or no battery
        end
    end
    return nothing  # Unsupported platform or no battery detected
end


# DATABASE INITIALIZATION
# Purpose: Create SQLite database and table structure for time series data storage
# Assumptions:
#   - SQLite provides sufficient performance for time series data
#   - Local file storage is appropriate (no network/cloud requirements)
#   - Simple schema sufficient for analysis needs
# Reasoning: SQLite is lightweight, serverless, and perfect for single-user data collection
#   TEXT timestamps allow easy human reading, REAL allows precise measurements
function initialize_database()
    db = SQLite.DB("hw-data.db")  # Create/open database file
    
    # TABLE SCHEMA DESIGN
    # Purpose: Store time series measurements with optional missing values
    # timestamp: TEXT format for human readability and sorting
    # cpu_frequency_ghz: REAL (nullable) - may fail to read on some systems
    # battery_percentage: REAL (nullable) - may fail on systems without battery
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS hardware_data (
            timestamp TEXT NOT NULL,
            cpu_frequency_ghz REAL,
            battery_percentage REAL
        )
    """)
    
    return db
end

# DATA POINT COLLECTION AND STORAGE
# Purpose: Collect single measurement of CPU frequency and battery level, store in database
# Assumptions:
#   - Measurements can fail (hardware access issues, permissions, etc.)
#   - NULL values in database are acceptable for failed measurements
#   - Timestamp precision of seconds is sufficient
# Reasoning: Atomic data collection ensures consistent timestamps across measurements
#   Graceful handling of failures allows monitoring to continue despite partial sensor access
function record_data_point(db)
    # TIMESTAMP GENERATION
    # Purpose: Create consistent, sortable timestamp for this measurement
    # Format: "yyyy-mm-dd HH:MM:SS" for human readability and SQL sorting
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    
    # SENSOR READING
    # Purpose: Attempt to read both CPU frequency and battery percentage
    # Assumption: Functions return nothing on failure (handled gracefully)
    cpu_freq = get_cpu_frequency()
    battery_pct = get_battery_percentage()
    
    # DATABASE STORAGE
    # Purpose: Store measurement with consistent timestamp
    # Reasoning: Parameterized query prevents SQL injection and handles NULL values
    SQLite.execute(db, """
        INSERT INTO hardware_data (timestamp, cpu_frequency_ghz, battery_percentage)
        VALUES (?, ?, ?)
    """, (timestamp, cpu_freq, battery_pct))
    
    # USER FEEDBACK
    # Purpose: Provide real-time monitoring feedback with formatted output
    # Reasoning: Shows data collection progress and identifies sensor failures immediately
    println("[$timestamp] CPU: $(cpu_freq !== nothing ? "$(round(cpu_freq, digits=2)) GHz" : "N/A"), " *
            "Battery: $(battery_pct !== nothing ? "$(round(battery_pct, digits=2))%" : "N/A")")
end

# MAIN MONITORING LOOP
# Purpose: Orchestrate continuous data collection with graceful shutdown handling
# Assumptions:
#   - 5-second sampling interval captures meaningful patterns without excessive overhead
#   - User will stop monitoring with Ctrl+C when sufficient data is collected
#   - Database should be properly closed to prevent corruption
# Reasoning: Infinite loop with signal handling allows flexible monitoring duration
#   while ensuring data integrity through proper database closure
function main()
    println("Hardware monitoring started. Press Ctrl+C to stop.")
    println("Data will be saved to hw-data.db")
    println()
    
    # DATABASE INITIALIZATION
    db = initialize_database()
    
    # CONTINUOUS MONITORING LOOP
    # Purpose: Collect data points at regular intervals until user interruption
    # Exception handling ensures graceful shutdown and database closure
    try
        while true
            record_data_point(db)
            # SAMPLING INTERVAL
            # Purpose: 5-second intervals balance temporal resolution with system load
            # Reasoning: Frequent enough to capture dynamic frequency scaling,
            #   not so frequent as to impact system performance or create excessive data
            sleep(5)  # Record every 5 seconds
        end
    catch InterruptException
        # GRACEFUL SHUTDOWN HANDLING
        # Purpose: Ensure database is properly closed when user stops monitoring
        # Reasoning: Prevents database corruption and ensures all data is flushed to disk
        println("\nShutting down gracefully...")
        close(db)
        println("Database closed. Data saved to hw-data.db")
    end
end

# SCRIPT EXECUTION GUARD
# Purpose: Only run main() when script is executed directly (not when imported)
# Reasoning: Allows script to be both executable and importable as a module
#   Prevents accidental monitoring when script is included by other code
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end