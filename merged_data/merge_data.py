import pandas as pd
import numpy as np
import re
from datetime import datetime

def merge_battery_swap_data_efficient(battery_data_path, swap_data_path, output_path=None):
    """
    Efficiently merge battery dataset with swap dataset based on battery ID and timing information.
    Handles timezone differences between datasets and filters for battery IDs starting with 'B'.
    
    Parameters:
    -----------
    battery_data_path : str
        Path to the battery dataset CSV file
    swap_data_path : str
        Path to the swap dataset CSV file
    output_path : str, optional
        Path to save the merged dataset
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    # Load datasets
    print(f"Loading battery data from: {battery_data_path}")
    battery_df = pd.read_csv(battery_data_path)
    
    print(f"Loading swap data from: {swap_data_path}")
    swap_df = pd.read_csv(swap_data_path)
    
    # Initial dataset sizes
    print(f"Original battery data shape: {battery_df.shape}")
    print(f"Original swap data shape: {swap_df.shape}")
    
    # Rename devId to match with battery_id for merging
    battery_df = battery_df.rename(columns={'devId': 'battery_id'})
    
    # Filter for battery IDs starting with 'B'
    print("Filtering for battery IDs starting with 'B'...")
    battery_df = battery_df[battery_df['battery_id'].str.startswith('B', na=False)]
    swap_df = swap_df[swap_df['battery_id'].str.startswith('B', na=False)]
    
    # Report on filtered dataset sizes
    print(f"Filtered battery data shape: {battery_df.shape}")
    print(f"Filtered swap data shape: {swap_df.shape}")
    
    # Convert date columns to datetime and standardize timezones
    print("Converting date columns to datetime and standardizing timezones...")
    
    # For battery data - ensure it's timezone-naive by converting to UTC and then removing timezone
    battery_df['_time'] = pd.to_datetime(battery_df['_time'], errors='coerce')
    
    # If it has timezone info, convert to UTC and then make naive
    if hasattr(battery_df['_time'].dt, 'tz') and battery_df['_time'].dt.tz is not None:
        print("Battery data has timezone info. Converting to UTC and then to naive datetime.")
        battery_df['_time'] = battery_df['_time'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # For swap data - make timezone-naive
    swap_df['swap_out_date'] = pd.to_datetime(swap_df['swap_out_date'], errors='coerce')
    swap_df['swap_in_date'] = pd.to_datetime(swap_df['swap_in_date'], errors='coerce')
    
    # Ensure swap data is timezone-naive
    if hasattr(swap_df['swap_out_date'].dt, 'tz') and swap_df['swap_out_date'].dt.tz is not None:
        swap_df['swap_out_date'] = swap_df['swap_out_date'].dt.tz_localize(None)
    
    if hasattr(swap_df['swap_in_date'].dt, 'tz') and swap_df['swap_in_date'].dt.tz is not None:
        swap_df['swap_in_date'] = swap_df['swap_in_date'].dt.tz_localize(None)
    
    # Check for NaT values after conversion
    print(f"NaT values in battery _time: {battery_df['_time'].isna().sum()}")
    print(f"NaT values in swap_out_date: {swap_df['swap_out_date'].isna().sum()}")
    print(f"NaT values in swap_in_date: {swap_df['swap_in_date'].isna().sum()}")
    
    # Filter out rows with NaN swap_in_date (incomplete cycles)
    complete_swap_df = swap_df.dropna(subset=['swap_in_date'])
    print(f"Complete swap cycles: {complete_swap_df.shape[0]} out of {swap_df.shape[0]}")
    
    # Initialize new columns in battery_df
    battery_df['swap_cycle'] = False
    battery_df['swap_out_date'] = None
    battery_df['swap_in_date'] = None
    battery_df['cycle_duration_hours'] = None
    battery_df['SOC_change'] = None
    
    print("Merging datasets - using safe datetime comparison approach...")
    
    # Create a mapping of battery_id -> list of (start_time, end_time, swap_row)
    swap_cycles = {}
    
    for _, swap_row in complete_swap_df.iterrows():
        battery_id = swap_row['battery_id']
        swap_in = swap_row['swap_in_date']
        swap_out = swap_row['swap_out_date']
        
        if pd.isna(swap_in) or pd.isna(swap_out):
            continue
            
        if battery_id not in swap_cycles:
            swap_cycles[battery_id] = []
            
        swap_cycles[battery_id].append({
            'start': swap_in,
            'end': swap_out,
            'duration': swap_row['cycle_duration_hours'],
            'soc_change': swap_row['SOC_change']
        })
    
    # Function to check if a timestamp is within any cycle for a battery
    def find_cycle(battery_id, timestamp):
        if battery_id not in swap_cycles:
            return None
            
        for cycle in swap_cycles[battery_id]:
            if cycle['start'] <= timestamp <= cycle['end']:
                return cycle
                
        return None
    
    # Apply the function to each row
    print("Matching battery readings to swap cycles...")
    matched_count = 0
    
    for idx, row in battery_df.iterrows():
        battery_id = row['battery_id']
        timestamp = row['_time']
        
        if pd.isna(timestamp):
            continue
            
        cycle = find_cycle(battery_id, timestamp)
        if cycle:
            matched_count += 1
            battery_df.at[idx, 'swap_cycle'] = True
            battery_df.at[idx, 'swap_in_date'] = cycle['start']
            battery_df.at[idx, 'swap_out_date'] = cycle['end']
            battery_df.at[idx, 'cycle_duration_hours'] = cycle['duration']
            battery_df.at[idx, 'SOC_change'] = cycle['soc_change']
    
    print(f"Matched {matched_count} battery readings to swap cycles")
    
    # Merged dataset is just the battery dataset with the new columns
    merged_df = battery_df
    
    print(f"Merge complete. Final dataset shape: {merged_df.shape}")
    
    # Save to CSV if output path is provided
    if output_path:
        print(f"Saving merged data to: {output_path}")
        merged_df.to_csv(output_path, index=False)
        
    return merged_df

# Correct file paths
battery_data = r'C:\Users\HP\Documents\battery\battery_data\cleaned_battery_data.csv'
swap_data = r'C:\Users\HP\Documents\battery\data\swap_data\battery_swap_cycles.csv'
output_path = r'C:\Users\HP\Documents\battery\merged_data\final_merged_dataset.csv'  # Updated filename

# Run the merge
try:
    merged_data = merge_battery_swap_data_efficient(
        battery_data_path=battery_data,
        swap_data_path=swap_data,
        output_path=output_path
    )
    
    # Display a sample of the merged data
    print("\nSample of merged data:")
    print(merged_data.head())
    
    # Display statistics about the merge
    merged_in_cycle = merged_data[merged_data['swap_cycle'] == True]
    print(f"\nTotal battery readings: {merged_data.shape[0]}")
    print(f"Readings within swap cycles: {merged_in_cycle.shape[0]}")
    if merged_data.shape[0] > 0:
        print(f"Percentage: {merged_in_cycle.shape[0]/merged_data.shape[0]*100:.2f}%")
    
    # Get unique battery IDs in both datasets
    battery_ids_in_battery = set(merged_data['battery_id'].unique())
    battery_ids_in_swap = set(pd.read_csv(swap_data)['battery_id'].unique())
    battery_ids_in_swap = {id for id in battery_ids_in_swap if isinstance(id, str) and id.startswith('B')}
    
    print(f"\nUnique battery IDs in battery dataset: {len(battery_ids_in_battery)}")
    print(f"Unique battery IDs in swap dataset: {len(battery_ids_in_swap)}")
    print(f"Battery IDs in both datasets: {len(battery_ids_in_battery.intersection(battery_ids_in_swap))}")
    
    print("\nMerge completed successfully!")
    print(f"Final merged dataset saved to: {output_path}")
    
except Exception as e:
    import traceback
    print(f"Error during merge: {str(e)}")
    print(traceback.format_exc())