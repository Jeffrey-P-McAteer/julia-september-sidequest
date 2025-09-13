# julia-september-sidequest

A Julia script project for cross-platform hardware monitoring and analysis.

## Features

- **Hardware Monitoring**: Cross-platform battery and CPU frequency monitoring
- **Data Storage**: SQLite database for efficient data storage
- **Correlation Analysis**: Statistical analysis of CPU frequency vs battery drain
- **Interactive Visualizations**: Beautiful PlotlyJS dashboards showing hardware relationships

## Requirements

- Julia (version 1.0 or later)
- Automatic package installation (SQLite.jl, DataFrames.jl, PlotlyJS.jl, etc.)

## Scripts

### 1. Hardware Data Recording

Records hardware data (battery level, CPU frequency, power draw) to SQLite database until stopped with Ctrl+C:

```bash
julia record_hardware.jl
```

**Cross-platform support:**
- **Windows**: Uses WMI via PowerShell for hardware data
- **Linux**: Reads from /proc/cpuinfo, /sys/class/power_supply/, and RAPL
- **macOS**: Uses sysctl, pmset, and system_profiler

Data is saved to `hw-data.db` every 5 seconds.

### 2. Data Analysis and Visualization

Analyzes correlation between CPU frequency and battery drain, creates interactive dashboard:

```bash
julia analyze_data.jl
```

**Output:**
- Correlation analysis between CPU frequency and battery drain
- Interactive PlotlyJS dashboard with multiple plots
- HTML report saved as `hardware_analysis_dashboard.html`
- Statistical insights and recommendations

## Configuration Options

### Bucket Width Configuration

Control the granularity of CPU frequency analysis by setting the `BUCKET_WIDTH` environment variable:

```bash
# Default 0.25 GHz buckets (fine-grained analysis)
julia analyze_data.jl

# Use 0.1 GHz buckets (very fine-grained)
BUCKET_WIDTH=0.1 julia analyze_data.jl

# Use 0.5 GHz buckets (coarser analysis)
BUCKET_WIDTH=0.5 julia analyze_data.jl

# Use 1.0 GHz buckets (coarse analysis)
BUCKET_WIDTH=1.0 julia analyze_data.jl
```

**Bucket Width Guidelines:**
- **0.1 GHz**: Very detailed analysis, requires lots of data
- **0.25 GHz**: Default, good balance of detail vs. statistical power
- **0.5 GHz**: Good for systems with limited frequency variation
- **1.0 GHz**: Coarse analysis, suitable for basic trends

## Usage Workflow

1. **Collect Data**: Run `julia record_hardware.jl` for at least 15-30 minutes while using your computer normally
2. **Stop Collection**: Press Ctrl+C to gracefully stop data collection
3. **Analyze Results**: Run `julia analyze_data.jl` to see correlation analysis and visualizations (optionally with `BUCKET_WIDTH` setting)
4. **View Dashboard**: Open `hardware_analysis_dashboard.html` in your web browser

## Make Scripts Executable (Linux/macOS)

```bash
chmod +x *.jl
./record_hardware.jl
./analyze_data.jl
```
