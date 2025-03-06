import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats
import sys

# Set up the visualization directory
vis_path = r'C:\Users\HP\Documents\battery\merged_data\merged_data_visualisation'
os.makedirs(vis_path, exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    try:
        fig.savefig(os.path.join(vis_path, filename), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Successfully saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        plt.close(fig)

# Initialize dataframe
df = None

# Try multiple approaches to load data
file_paths = [
    r'C:\Users\HP\Documents\battery\merged_data\final_merged_dataset.csv',
    r'C:\Users\HP\Documents\battery\merged_data\merged_data.csv'
]

# Try to load the data with different file paths
for file_path in file_paths:
    try:
        print(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Success! Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        break  # Break the loop if successful
    except PermissionError:
        print(f"Permission denied for file: {file_path}")
        print("Possible solutions:")
        print("1. Close any other applications that might be using this file")
        print("2. Run this script with administrator privileges")
        print("3. Check file permissions in Windows Explorer")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")


# Check if we successfully loaded or created data
if df is None or df.empty:
    print("No data available for analysis. Please check the file paths and permissions.")
    sys.exit(1)

# Make sure _time is datetime
try:
    df['_time'] = pd.to_datetime(df['_time'])
    print(f"Date range: {df['_time'].min()} to {df['_time'].max()}")
except Exception as e:
    print(f"Error converting time column: {e}")

# 1. Basic Data Exploration
print("\n===== BASIC DATA EXPLORATION =====")

# Data info
print("\nData Types:")
df_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes,
    'Non-Null Count': df.count(),
    'Null Percentage': round(df.isna().mean() * 100, 2)
})
print(df_info)

# Basic descriptive statistics
df_desc = df.describe(include='all').T
df_desc['missing_pct'] = df.isna().mean() * 100
print("\nDescriptive Statistics (First 10 columns):")
print(df_desc.head(10))

# Save basic stats to CSV
try:
    df_info.to_csv(os.path.join(vis_path, 'data_types_info.csv'), index=False)
    df_desc.to_csv(os.path.join(vis_path, 'descriptive_stats.csv'))
    print("Basic statistics saved to CSV files.")
except Exception as e:
    print(f"Error saving statistics to CSV: {e}")

# 2. Missing Value Analysis
print("\n===== MISSING VALUE ANALYSIS =====")

try:
    # Create a heatmap to visualize missing values
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    plt.tight_layout()
    save_fig(plt.gcf(), 'missing_value_heatmap.png')

    # Calculate and plot missing percentages by column
    missing_percentage = df.isna().mean().sort_values(ascending=False) * 100
    missing_cols = missing_percentage[missing_percentage > 0]
    
    if not missing_cols.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        missing_cols.plot(kind='bar', ax=ax)
        plt.title('Percentage of Missing Values by Column')
        plt.ylabel('Missing Percentage')
        plt.xlabel('Columns')
        plt.xticks(rotation=90)
        plt.tight_layout()
        save_fig(plt.gcf(), 'missing_percentage_by_column.png')
    else:
        print("No missing values found in the dataset.")
except Exception as e:
    print(f"Error in missing value analysis: {e}")

# 3. Numeric Data Analysis
print("\n===== NUMERIC DATA ANALYSIS =====")

try:
    # Select numeric columns (excluding cell_voltage columns for separate analysis)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cell_voltage_cols = [col for col in df.columns if col.startswith('cell_voltage')]
    numeric_non_cell_cols = [col for col in numeric_cols if col not in cell_voltage_cols]

    print(f"Analyzing {len(numeric_non_cell_cols)} numeric columns (excluding cell voltage columns)")

    # Distribution plots for important numeric variables
    for col in numeric_non_cell_cols[:min(10, len(numeric_non_cell_cols))]:  # First 10 cols or fewer
        if df[col].notna().sum() > 0:  # Only plot if there are non-NA values
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Histogram
                sns.histplot(df[col].dropna(), ax=axes[0], kde=True)
                axes[0].set_title(f'Distribution of {col}')
                
                # Box plot
                sns.boxplot(y=df[col].dropna(), ax=axes[1])
                axes[1].set_title(f'Box Plot of {col}')
                
                plt.tight_layout()
                save_fig(plt.gcf(), f'distribution_{col}.png')
            except Exception as e:
                print(f"Error creating distribution plot for {col}: {e}")

    # Correlation Analysis
    corr_cols = [col for col in numeric_non_cell_cols if df[col].notna().sum() > 0]
    if len(corr_cols) > 1:
        try:
            corr_matrix = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                        linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Correlation Matrix of Numeric Variables')
            plt.tight_layout()
            save_fig(plt.gcf(), 'correlation_matrix.png')
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")
except Exception as e:
    print(f"Error in numeric data analysis: {e}")

# 4. Cell Voltage Analysis
print("\n===== CELL VOLTAGE ANALYSIS =====")

try:
    # Create a dataframe with just the cell voltage columns
    cell_df = df[cell_voltage_cols]

    # Summary statistics for cell voltages
    cell_stats = cell_df.describe().T
    cell_stats.to_csv(os.path.join(vis_path, 'cell_voltage_stats.csv'))

    # Distribution of all cell voltages
    all_cell_voltages = cell_df.values.flatten()
    all_cell_voltages = all_cell_voltages[~np.isnan(all_cell_voltages)]

    plt.figure(figsize=(12, 6))
    sns.histplot(all_cell_voltages, kde=True, bins=50)
    plt.title('Distribution of All Cell Voltages')
    plt.xlabel('Voltage')
    plt.ylabel('Frequency')
    save_fig(plt.gcf(), 'all_cell_voltages_distribution.png')

    # Box plot of cell voltages
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=cell_df)
    plt.title('Box Plot of Cell Voltages')
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_fig(plt.gcf(), 'cell_voltages_boxplot.png')

    # Cell voltage variation over time (sample for a few cells)
    sample_cells = cell_voltage_cols[:5] if len(cell_voltage_cols) > 5 else cell_voltage_cols
    fig, ax = plt.subplots(figsize=(12, 6))

    for cell in sample_cells:
        ax.plot(df['_time'], df[cell], label=cell)

    ax.set_title('Cell Voltage Over Time (Sample Cells)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    ax.legend()
    plt.tight_layout()
    save_fig(plt.gcf(), 'cell_voltage_over_time.png')

    # Cell voltage imbalance
    if len(cell_voltage_cols) > 0:
        df['cell_voltage_max'] = cell_df.max(axis=1)
        df['cell_voltage_min'] = cell_df.min(axis=1)
        df['cell_voltage_range'] = df['cell_voltage_max'] - df['cell_voltage_min']
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['_time'], df['cell_voltage_range'])
        plt.title('Cell Voltage Imbalance Over Time')
        plt.xlabel('Time')
        plt.ylabel('Voltage Range (Max - Min)')
        plt.tight_layout()
        save_fig(plt.gcf(), 'cell_voltage_imbalance.png')
except Exception as e:
    print(f"Error in cell voltage analysis: {e}")

# 5. Temperature Analysis
print("\n===== TEMPERATURE ANALYSIS =====")

try:
    temp_cols = [
        'BMS_PCB_board_surface_temperature', 
        'Internal_temperature_of_battery',
        'Surface_temperature_in_the_middle_of_cells'
    ]

    # Plot temperature variables over time
    temp_df = df[['_time'] + [col for col in temp_cols if col in df.columns]]
    temp_df = temp_df.set_index('_time').dropna(how='all')

    if not temp_df.empty and temp_df.shape[1] > 0:
        plt.figure(figsize=(12, 6))
        for col in temp_df.columns:
            plt.plot(temp_df.index, temp_df[col], label=col)
        
        plt.title('Temperature Measurements Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.tight_layout()
        save_fig(plt.gcf(), 'temperature_over_time.png')
        
        # Temperature distribution comparison
        plt.figure(figsize=(12, 6))
        for col in temp_df.columns:
            sns.kdeplot(temp_df[col].dropna(), label=col)
        
        plt.title('Temperature Distribution Comparison')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        save_fig(plt.gcf(), 'temperature_distribution.png')
except Exception as e:
    print(f"Error in temperature analysis: {e}")

# 6. SOC Analysis
print("\n===== SOC (STATE OF CHARGE) ANALYSIS =====")

try:
    if 'SOC' in df.columns and df['SOC'].notna().sum() > 0:
        # SOC distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['SOC'].dropna(), kde=True, bins=20)
        plt.title('Distribution of State of Charge (SOC)')
        plt.xlabel('SOC')
        plt.ylabel('Frequency')
        save_fig(plt.gcf(), 'soc_distribution.png')
        
        # SOC over time
        plt.figure(figsize=(12, 6))
        plt.plot(df['_time'], df['SOC'])
        plt.title('State of Charge (SOC) Over Time')
        plt.xlabel('Time')
        plt.ylabel('SOC')
        plt.tight_layout()
        save_fig(plt.gcf(), 'soc_over_time.png')
        
        # SOC vs Temperature (if both exist)
        for temp_col in temp_cols:
            if temp_col in df.columns and df[temp_col].notna().sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[temp_col], y=df['SOC'])
                plt.title(f'SOC vs {temp_col}')
                plt.xlabel(temp_col)
                plt.ylabel('SOC')
                plt.tight_layout()
                save_fig(plt.gcf(), f'soc_vs_{temp_col}.png')
except Exception as e:
    print(f"Error in SOC analysis: {e}")

# 7. Cycle Analysis
print("\n===== CYCLE ANALYSIS =====")

try:
    cycle_cols = ['Number_of_cycles', 'swap_cycle', 'cycle_duration_hours']
    cycle_cols = [col for col in cycle_cols if col in df.columns]

    for col in cycle_cols:
        if df[col].notna().sum() > 0:
            if df[col].dtype == 'bool':
                # For boolean columns like swap_cycle
                plt.figure(figsize=(8, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                save_fig(plt.gcf(), f'{col}_distribution.png')
            else:
                # For numeric columns
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.tight_layout()
                save_fig(plt.gcf(), f'{col}_distribution.png')

    # Check if we have cycle data to compare with SOC
    if 'Number_of_cycles' in df.columns and 'SOC' in df.columns:
        valid_data = df[df['Number_of_cycles'].notna() & df['SOC'].notna()]
        if not valid_data.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Number_of_cycles', y='SOC', data=valid_data)
            plt.title('SOC vs Number of Cycles')
            plt.tight_layout()
            save_fig(plt.gcf(), 'soc_vs_cycles.png')
except Exception as e:
    print(f"Error in cycle analysis: {e}")

# 8. Battery Status and Control Analysis
print("\n===== BATTERY STATUS AND CONTROL ANALYSIS =====")

try:
    status_cols = ['Battery_status', 'Battery_control', 'BMS_switch_C_FET_state', 'BMS_switch_D_FET_state']
    status_cols = [col for col in status_cols if col in df.columns]

    for col in status_cols:
        if df[col].notna().sum() > 0:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().sort_index().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            save_fig(plt.gcf(), f'{col}_distribution.png')
except Exception as e:
    print(f"Error in battery status analysis: {e}")

# 9. Temporal Patterns Analysis
print("\n===== TEMPORAL PATTERNS ANALYSIS =====")

try:
    if not df.empty and '_time' in df.columns:
        # Add time-based features
        df['hour'] = df['_time'].dt.hour
        df['day'] = df['_time'].dt.day
        df['month'] = df['_time'].dt.month
        df['weekday'] = df['_time'].dt.weekday
        
        # Visualize key metrics by hour of day
        for metric in ['SOC', 'Total_current', 'Total_voltage']:
            if metric in df.columns and df[metric].notna().sum() > 0:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='hour', y=metric, data=df)
                plt.title(f'{metric} by Hour of Day')
                plt.tight_layout()
                save_fig(plt.gcf(), f'{metric}_by_hour.png')
except Exception as e:
    print(f"Error in temporal patterns analysis: {e}")

# 10. Alarm Analysis
print("\n===== ALARM ANALYSIS =====")

try:
    if 'alarmFlag' in df.columns and df['alarmFlag'].notna().sum() > 0:
        plt.figure(figsize=(10, 6))
        df['alarmFlag'].value_counts().plot(kind='bar')
        plt.title('Distribution of Alarm Flags')
        plt.xlabel('Alarm Flag')
        plt.ylabel('Count')
        plt.tight_layout()
        save_fig(plt.gcf(), 'alarm_flag_distribution.png')
        
        # If we have alarm descriptions
        if 'alarmDesc' in df.columns and df['alarmDesc'].notna().sum() > 0:
            plt.figure(figsize=(12, 8))
            df['alarmDesc'].value_counts().plot(kind='bar')
            plt.title('Distribution of Alarm Descriptions')
            plt.xticks(rotation=90)
            plt.tight_layout()
            save_fig(plt.gcf(), 'alarm_description_distribution.png')
except Exception as e:
    print(f"Error in alarm analysis: {e}")

# 11. Battery ID Analysis
print("\n===== BATTERY ID ANALYSIS =====")

try:
    if 'battery_id' in df.columns:
        battery_counts = df['battery_id'].value_counts()
        
        plt.figure(figsize=(12, 6))
        battery_counts.plot(kind='bar')
        plt.title('Number of Records by Battery ID')
        plt.xlabel('Battery ID')
        plt.ylabel('Count')
        plt.tight_layout()
        save_fig(plt.gcf(), 'battery_id_counts.png')
        
        # If there are multiple batteries, compare key metrics
        if len(battery_counts) > 1:
            for metric in ['SOC', 'Internal_temperature_of_battery', 'Total_voltage']:
                if metric in df.columns and df[metric].notna().sum() > 0:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x='battery_id', y=metric, data=df)
                    plt.title(f'{metric} by Battery ID')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    save_fig(plt.gcf(), f'{metric}_by_battery_id.png')
except Exception as e:
    print(f"Error in battery ID analysis: {e}")

# 12. Bivariate Analysis for Key Relationships
print("\n===== BIVARIATE ANALYSIS =====")

try:
    # Define pairs of variables to analyze
    pairs = [
        ('SOC', 'Total_voltage'),
        ('SOC', 'Internal_temperature_of_battery'),
        ('Total_current', 'Total_voltage'),
        ('Internal_temperature_of_battery', 'BMS_PCB_board_surface_temperature')
    ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            valid_data = df[df[x].notna() & df[y].notna()]
            if not valid_data.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=x, y=y, data=valid_data)
                plt.title(f'{y} vs {x}')
                plt.tight_layout()
                save_fig(plt.gcf(), f'{y}_vs_{x}.png')
                
                # Calculate correlation
                corr = valid_data[[x, y]].corr().iloc[0, 1]
                print(f"Correlation between {x} and {y}: {corr:.4f}")
except Exception as e:
    print(f"Error in bivariate analysis: {e}")

# 13. Wrap-up and Summary
print("\n===== EDA SUMMARY =====")

# Create a summary of findings
try:
    summary = f"""
# EDA Summary for Battery Dataset

## Dataset Overview
- Total records: {df.shape[0]}
- Total features: {df.shape[1]}
- Date range: {df['_time'].min()} to {df['_time'].max()}
- Number of unique batteries: {df['battery_id'].nunique()}

## Key Statistics
- Average SOC: {df['SOC'].mean():.2f} (if available)
- Average Internal Temperature: {df['Internal_temperature_of_battery'].mean():.2f}°C (if available)
- Average Total Voltage: {df['Total_voltage'].mean():.2f} (if available)

## Missing Data
- Features with missing data: {len(missing_percentage[missing_percentage > 0])}
- Features with >50% missing: {len(missing_percentage[missing_percentage > 50])}

## All visualizations have been saved to: {vis_path}

## Troubleshooting
If you encountered permission errors during this analysis:
1. Make sure no other applications have the data file open
2. Run the script as administrator
3. Check file permissions in Windows Explorer
4. Try saving the file to a different location with full write permissions
"""

    # Save the summary
    with open(os.path.join(vis_path, 'eda_summary.md'), 'w') as f:
        f.write(summary)

    print(summary)
except Exception as e:
    print(f"Error creating summary: {e}")

print(f"\nEDA complete! All visualizations saved to: {vis_path}")