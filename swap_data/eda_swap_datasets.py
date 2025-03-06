import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import os

# Load the data
def load_data(file_path):
    """Load battery swap dataset from CSV"""
    df = pd.read_csv(file_path, parse_dates=['swap_out_date', 'swap_in_date'])
    return df

# Basic data exploration
def explore_data(df):
    """Print basic information about the dataset"""
    print("Dataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Number of unique batteries: {df['battery_id'].nunique()}")
    print(f"Date range: {df['swap_out_date'].min()} to {df['swap_out_date'].max()}")
    print("\nColumns and data types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe(include='all'))
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check distribution of complete vs incomplete cycles
    print("\nComplete vs incomplete cycles:")
    print(df['has_complete_cycle'].value_counts())
    print(f"Percentage complete cycles: {df['has_complete_cycle'].mean() * 100:.2f}%")
    
    # Sample of the data
    print("\nSample records:")
    print(df.head())

# Battery usage patterns
def analyze_battery_usage(df):
    """Analyze patterns in battery usage"""
    print("\n=== Battery Usage Analysis ===")
    
    # Calculate statistics per battery
    battery_stats = df.groupby('battery_id').agg(
        swap_count=('swap_out_date', 'count'),
        complete_cycles=('has_complete_cycle', 'sum'),
        avg_cycle_duration=('cycle_duration_hours', 'mean'),
        avg_soc_change=('SOC_change', 'mean'),
        first_seen=('swap_out_date', 'min'),
        last_seen=('swap_out_date', 'max')
    ).reset_index()
    
    # Calculate days in service
    battery_stats['days_in_service'] = (battery_stats['last_seen'] - battery_stats['first_seen']).dt.total_seconds() / (24 * 3600)
    
    # Calculate swaps per day
    battery_stats['swaps_per_day'] = battery_stats['swap_count'] / battery_stats['days_in_service']
    
    print("Battery usage statistics:")
    print(battery_stats)
    
    return battery_stats

# Temporal patterns analysis
def analyze_temporal_patterns(df):
    """Analyze temporal patterns in battery swaps"""
    print("\n=== Temporal Analysis ===")
    
    # Create time-based features
    df['swap_out_hour'] = df['swap_out_date'].dt.hour
    df['swap_out_day'] = df['swap_out_date'].dt.day_name()
    df['swap_out_date_only'] = df['swap_out_date'].dt.date
    
    if not df['swap_in_date'].isnull().all():
        df['swap_in_hour'] = df['swap_in_date'].dt.hour
        df['swap_in_day'] = df['swap_in_date'].dt.day_name()
    
    # Hourly distribution of swaps
    print("Hourly distribution of battery swaps:")
    hourly_swaps = df['swap_out_hour'].value_counts().sort_index()
    print(hourly_swaps)
    
    # Daily distribution of swaps
    print("\nDaily distribution of battery swaps:")
    daily_swaps = df['swap_out_day'].value_counts()
    print(daily_swaps)
    
    # Daily swap count over time
    daily_count = df.groupby('swap_out_date_only').size()
    print("\nSwaps per day:")
    print(daily_count)
    
    return hourly_swaps, daily_swaps, daily_count

# SOC analysis
def analyze_soc_patterns(df):
    """Analyze State of Charge patterns"""
    print("\n=== SOC Analysis ===")
    
    # Filter for complete cycles only
    complete_cycles = df[df['has_complete_cycle'] == True].copy()
    
    # SOC distributions
    print("Swap-in SOC statistics:")
    print(complete_cycles['swap_in_SOC'].describe())
    
    print("\nSwap-out SOC statistics:")
    print(complete_cycles['swap_out_SOC'].describe())
    
    print("\nSOC change statistics:")
    print(complete_cycles['SOC_change'].describe())
    
    # Calculate charging rate (SOC points per hour)
    complete_cycles['charging_rate'] = complete_cycles['SOC_change'] / complete_cycles['cycle_duration_hours']
    
    print("\nCharging rate statistics (SOC points per hour):")
    print(complete_cycles['charging_rate'].describe())
    
    return complete_cycles

# Create visualizations
def create_visualizations(df, hourly_swaps, daily_swaps, daily_count, complete_cycles, battery_stats, output_dir):
    """Create visualizations for the EDA"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Plot 1: Hourly distribution of swaps
    plt.subplot(2, 3, 1)
    sns.barplot(x=hourly_swaps.index, y=hourly_swaps.values)
    plt.title('Hourly Distribution of Battery Swaps')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Swaps')
    plt.xticks(rotation=45)
    
    # Plot 2: Daily distribution of swaps
    plt.subplot(2, 3, 2)
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.barplot(x=daily_swaps.index, y=daily_swaps.values, order=order)
    plt.title('Daily Distribution of Battery Swaps')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Swaps')
    plt.xticks(rotation=45)
    
    # Plot 3: Daily swap count over time
    plt.subplot(2, 3, 3)
    daily_count.plot()
    plt.title('Swaps Per Day Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Swaps')
    plt.xticks(rotation=45)
    
    # Plot 4: SOC distribution at swap-in
    plt.subplot(2, 3, 4)
    sns.histplot(complete_cycles['swap_in_SOC'], bins=20, kde=True)
    plt.title('Distribution of SOC at Swap-in')
    plt.xlabel('State of Charge (%)')
    plt.ylabel('Frequency')
    
    # Plot 5: Cycle duration distribution
    plt.subplot(2, 3, 5)
    sns.histplot(complete_cycles['cycle_duration_hours'], bins=20, kde=True)
    plt.title('Distribution of Cycle Duration')
    plt.xlabel('Duration (hours)')
    plt.ylabel('Frequency')
    
    # Plot 6: Cycle duration vs SOC change
    plt.subplot(2, 3, 6)
    sns.scatterplot(x='cycle_duration_hours', y='SOC_change', data=complete_cycles)
    plt.title('Cycle Duration vs. SOC Change')
    plt.xlabel('Duration (hours)')
    plt.ylabel('SOC Change (%)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'battery_swap_eda.png')
    plt.savefig(output_path)
    print(f"\nBasic visualizations saved as '{output_path}'")
    
    # Additional visualizations
    
    # Battery usage comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='battery_id', y='swap_count', data=battery_stats)
    plt.title('Swap Count by Battery')
    plt.xlabel('Battery ID')
    plt.ylabel('Number of Swaps')
    plt.xticks(rotation=90)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'battery_usage_comparison.png')
    plt.savefig(output_path)
    print(f"Battery usage comparison saved as '{output_path}'")
    
    # SOC vs Time of Day
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='swap_out_hour', y='swap_in_SOC', data=complete_cycles)
    plt.title('Swap-in SOC vs Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Swap-in SOC (%)')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'soc_vs_time.png')
    plt.savefig(output_path)
    print(f"SOC vs Time visualization saved as '{output_path}'")
    
    # Create additional visualizations
    
    # Charging rate by hour
    plt.figure(figsize=(12, 6))
    hourly_charging = complete_cycles.groupby('swap_out_hour')['charging_rate'].mean()
    sns.barplot(x=hourly_charging.index, y=hourly_charging.values)
    plt.title('Average Charging Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Avg Charging Rate (SOC points/hour)')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'charging_rate_by_hour.png')
    plt.savefig(output_path)
    print(f"Charging rate by hour saved as '{output_path}'")
    
    # SOC change distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(complete_cycles['SOC_change'], bins=20, kde=True)
    plt.title('Distribution of SOC Change')
    plt.xlabel('SOC Change (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'soc_change_distribution.png')
    plt.savefig(output_path)
    print(f"SOC change distribution saved as '{output_path}'")
    
    # Battery efficiency comparison
    if len(battery_stats) > 1:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='battery_id', y='avg_soc_change', data=battery_stats)
        plt.title('Average SOC Change by Battery')
        plt.xlabel('Battery ID')
        plt.ylabel('Avg SOC Change (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'battery_efficiency.png')
        plt.savefig(output_path)
        print(f"Battery efficiency comparison saved as '{output_path}'")

# Cycle analysis
def analyze_cycles(complete_cycles):
    """Analyze battery charging cycles in detail"""
    print("\n=== Cycle Analysis ===")
    
    # Calculate time between consecutive swaps
    if len(complete_cycles) > 1:
        # Sort by battery_id and swap_out_date
        sorted_cycles = complete_cycles.sort_values(['battery_id', 'swap_out_date'])
        
        # Calculate time between swaps for the same battery
        sorted_cycles['next_swap_out'] = sorted_cycles.groupby('battery_id')['swap_out_date'].shift(-1)
        sorted_cycles['time_to_next_swap'] = (sorted_cycles['next_swap_out'] - sorted_cycles['swap_out_date']).dt.total_seconds() / 3600
        
        print("Time between consecutive swaps (hours):")
        print(sorted_cycles['time_to_next_swap'].describe())
        
        # Identify usage patterns
        print("\nCharging efficiency analysis:")
        sorted_cycles['charging_efficiency'] = sorted_cycles['SOC_change'] / sorted_cycles['cycle_duration_hours']
        print(sorted_cycles['charging_efficiency'].describe())
        
        return sorted_cycles
    else:
        print("Not enough complete cycles for cycle analysis.")
        return complete_cycles

# Main function to run the EDA
def run_battery_swap_eda(file_path, output_dir):
    """Run a comprehensive EDA on battery swap dataset"""
    print("=" * 80)
    print("BATTERY SWAP DATASET: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print(f"Saving visualizations to: {output_dir}")
    
    # Load the data
    df = load_data(file_path)
    
    # Basic exploration
    explore_data(df)
    
    # Battery usage analysis
    battery_stats = analyze_battery_usage(df)
    
    # Temporal patterns
    hourly_swaps, daily_swaps, daily_count = analyze_temporal_patterns(df)
    
    # SOC analysis
    complete_cycles = analyze_soc_patterns(df)
    
    # Cycle analysis
    if 'has_complete_cycle' in df.columns and df['has_complete_cycle'].sum() > 1:
        sorted_cycles = analyze_cycles(complete_cycles)
    
    # Create visualizations
    create_visualizations(df, hourly_swaps, daily_swaps, daily_count, complete_cycles, battery_stats, output_dir)
    
    # Additional insights
    print("\n=== Key Insights ===")
    
    # Calculate average metrics
    avg_cycle_duration = complete_cycles['cycle_duration_hours'].mean()
    avg_soc_change = complete_cycles['SOC_change'].mean()
    avg_charging_rate = complete_cycles['SOC_change'].sum() / complete_cycles['cycle_duration_hours'].sum()
    
    print(f"Average cycle duration: {avg_cycle_duration:.2f} hours")
    print(f"Average SOC change: {avg_soc_change:.2f}%")
    print(f"Overall charging rate: {avg_charging_rate:.2f} SOC points per hour")
    
    # Identify optimal charging times
    if len(complete_cycles) > 0:
        charging_by_hour = complete_cycles.groupby('swap_out_hour')['charging_rate'].mean().sort_values(ascending=False)
        print("\nOptimal charging hours (by charging rate):")
        print(charging_by_hour.head())
    
    # Save insights to text file
    insights_path = os.path.join(output_dir, 'battery_swap_insights.txt')
    with open(insights_path, 'w') as f:
        f.write("BATTERY SWAP DATASET: KEY INSIGHTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total records analyzed: {len(df)}\n")
        f.write(f"Unique batteries: {df['battery_id'].nunique()}\n")
        f.write(f"Date range: {df['swap_out_date'].min().date()} to {df['swap_out_date'].max().date()}\n\n")
        f.write(f"Complete cycles: {df['has_complete_cycle'].sum()} ({df['has_complete_cycle'].mean() * 100:.2f}%)\n\n")
        f.write("Key Metrics:\n")
        f.write(f"- Average cycle duration: {avg_cycle_duration:.2f} hours\n")
        f.write(f"- Average SOC change: {avg_soc_change:.2f}%\n")
        f.write(f"- Overall charging rate: {avg_charging_rate:.2f} SOC points per hour\n\n")
        f.write("Optimal charging hours (by charging rate):\n")
        for hour, rate in charging_by_hour.head().items():
            f.write(f"- Hour {hour}: {rate:.2f} SOC points/hour\n")
    
    print(f"\nKey insights saved to '{insights_path}'")
    print("\nEDA complete!")

# Run the analysis
if __name__ == "__main__":
    # Set your file paths
    file_path = r"C:\Users\HP\Documents\battery\data\swap_data\battery_swap_cycles.csv"
    output_dir = r"C:\Users\HP\Documents\battery\data\swap_data\swap_visualisations"
    
    run_battery_swap_eda(file_path, output_dir)