import pandas as pd
from datetime import datetime

# Read and filter the swap in dataset
def process_swap_in_data(input_file):
    # Read the txt file
    df_in = pd.read_csv(input_file, sep='\t')
    
    # Convert date column to datetime
    df_in['swap_in_date'] = pd.to_datetime(df_in['swap_in_date'])
    
    # Filter data between Feb 2, 2024 and Mar 2, 2024 (inclusive)
    start_date = pd.to_datetime('2024-02-02')
    end_date = pd.to_datetime('2024-03-02 23:59:59')
    
    filtered_df_in = df_in[(df_in['swap_in_date'] >= start_date) & (df_in['swap_in_date'] <= end_date)]
    
    # Save the filtered data as CSV
    filtered_df_in.to_csv("filtered_swap_in_dataset.csv", index=False)
    
    print(f"Processing {input_file}:")
    print(f"  Original records: {len(df_in)}")
    print(f"  Filtered records: {len(filtered_df_in)}")
    print(f"  Saved to: filtered_swap_in_dataset.csv")
    print()
    
    return filtered_df_in

# Read and filter the swap out dataset
def process_swap_out_data(input_file):
    # Read the txt file
    df_out = pd.read_csv(input_file, sep='\t')
    
    # Convert date column to datetime
    df_out['swap_out_date'] = pd.to_datetime(df_out['swap_out_date'])
    
    # Filter data between Feb 2, 2024 and Mar 2, 2024 (inclusive)
    start_date = pd.to_datetime('2024-02-02')
    end_date = pd.to_datetime('2024-03-02 23:59:59')
    
    filtered_df_out = df_out[(df_out['swap_out_date'] >= start_date) & (df_out['swap_out_date'] <= end_date)]
    
    # Save the filtered data as CSV
    filtered_df_out.to_csv("filtered_swap_out_dataset.csv", index=False)
    
    print(f"Processing {input_file}:")
    print(f"  Original records: {len(df_out)}")
    print(f"  Filtered records: {len(filtered_df_out)}")
    print(f"  Saved to: filtered_swap_out_dataset.csv")
    print()
    
    return filtered_df_out

# Properly match swap-in and swap-out events for each battery
def merge_and_analyze(df_in, df_out):
    # Create copies to avoid modifying the original dataframes
    swap_in = df_in.copy()
    swap_out = df_out.copy()
    
    # Sort both dataframes by date
    swap_in = swap_in.sort_values('swap_in_date')
    swap_out = swap_out.sort_values('swap_out_date')
    
    # Initialize empty list to store matched pairs
    matched_cycles = []
    
    # Get unique battery IDs from both datasets
    battery_ids = set(swap_in['battery_id']).union(set(swap_out['battery_id']))
    
    # For each battery, find logical swap-in and swap-out pairs
    for battery_id in battery_ids:
        # Get all swap-in events for this battery
        battery_swaps_in = swap_in[swap_in['battery_id'] == battery_id].copy()
        # Get all swap-out events for this battery
        battery_swaps_out = swap_out[swap_out['battery_id'] == battery_id].copy()
        
        # If we have both swap-in and swap-out events for this battery
        if not battery_swaps_in.empty and not battery_swaps_out.empty:
            # Initialize indices for tracking position in both dataframes
            in_idx = 0
            out_idx = 0
            
            while in_idx < len(battery_swaps_in) and out_idx < len(battery_swaps_out):
                current_in = battery_swaps_in.iloc[in_idx]
                current_out = battery_swaps_out.iloc[out_idx]
                
                # If swap-out is after swap-in, we have a valid cycle
                if current_out['swap_out_date'] > current_in['swap_in_date']:
                    # Create a record for this cycle
                    cycle = {
                        'battery_id': battery_id,
                        'swap_in_date': current_in['swap_in_date'],
                        'swap_in_SOC': current_in['swap_in_SOC'],
                        'swap_out_date': current_out['swap_out_date'],
                        'swap_out_SOC': current_out['swap_out_SOC'],
                        'has_complete_cycle': True,
                        'cycle_duration_hours': (current_out['swap_out_date'] - current_in['swap_in_date']).total_seconds() / 3600,
                        'SOC_change': current_out['swap_out_SOC'] - current_in['swap_in_SOC']
                    }
                    matched_cycles.append(cycle)
                    
                    # Move to next swap-in and swap-out
                    in_idx += 1
                    out_idx += 1
                # If swap-out is before swap-in, this swap-out can't match any future swap-ins
                elif current_out['swap_out_date'] < current_in['swap_in_date']:
                    # Record this swap-out with no matching swap-in
                    cycle = {
                        'battery_id': battery_id,
                        'swap_out_date': current_out['swap_out_date'],
                        'swap_out_SOC': current_out['swap_out_SOC'],
                        'has_complete_cycle': False
                    }
                    matched_cycles.append(cycle)
                    out_idx += 1
            
            # Add any remaining swap-ins that have no matching swap-outs
            while in_idx < len(battery_swaps_in):
                current_in = battery_swaps_in.iloc[in_idx]
                cycle = {
                    'battery_id': battery_id,
                    'swap_in_date': current_in['swap_in_date'],
                    'swap_in_SOC': current_in['swap_in_SOC'],
                    'has_complete_cycle': False
                }
                matched_cycles.append(cycle)
                in_idx += 1
                
            # Add any remaining swap-outs that have no matching swap-ins
            while out_idx < len(battery_swaps_out):
                current_out = battery_swaps_out.iloc[out_idx]
                cycle = {
                    'battery_id': battery_id,
                    'swap_out_date': current_out['swap_out_date'],
                    'swap_out_SOC': current_out['swap_out_SOC'],
                    'has_complete_cycle': False
                }
                matched_cycles.append(cycle)
                out_idx += 1
        
        # If we only have swap-in events for this battery
        elif not battery_swaps_in.empty:
            for _, swap in battery_swaps_in.iterrows():
                cycle = {
                    'battery_id': battery_id,
                    'swap_in_date': swap['swap_in_date'],
                    'swap_in_SOC': swap['swap_in_SOC'],
                    'has_complete_cycle': False
                }
                matched_cycles.append(cycle)
        
        # If we only have swap-out events for this battery
        elif not battery_swaps_out.empty:
            for _, swap in battery_swaps_out.iterrows():
                cycle = {
                    'battery_id': battery_id,
                    'swap_out_date': swap['swap_out_date'],
                    'swap_out_SOC': swap['swap_out_SOC'],
                    'has_complete_cycle': False
                }
                matched_cycles.append(cycle)
    
    # Convert list of dictionaries to DataFrame
    cycles_df = pd.DataFrame(matched_cycles)
    
    # Save the matched cycles dataset
    cycles_df.to_csv(r"C:\Users\HP\Documents\battery\data\swap\battery_swap_cycles.csv", index=False)
    
    # Count complete and incomplete cycles
    complete_cycles = cycles_df[cycles_df['has_complete_cycle'] == True]
    incomplete_cycles = cycles_df[cycles_df['has_complete_cycle'] == False]
    swap_in_only = incomplete_cycles[incomplete_cycles['swap_in_date'].notna()]
    swap_out_only = incomplete_cycles[incomplete_cycles['swap_out_date'].notna()]
    
    print("Matched cycles dataset:")
    print(f"  Total records: {len(cycles_df)}")
    print(f"  Complete cycles: {len(complete_cycles)}")
    print(f"  Batteries only swapped in: {len(swap_in_only)}")
    print(f"  Batteries only swapped out: {len(swap_out_only)}")
    print(f"  Saved to: battery_swap_cycles.csv")
    
    return cycles_df

# Main process
df_in = process_swap_in_data(r"C:\Users\HP\Documents\battery\data\swap\swap_in.txt")
df_out = process_swap_out_data(r"C:\Users\HP\Documents\battery\data\swap\swap_out.txt")
cycles_df = merge_and_analyze(df_in, df_out)

print("\nData processing complete!")