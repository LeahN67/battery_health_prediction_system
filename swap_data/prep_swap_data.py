import pandas as pd
from datetime import datetime

# Function to read, filter, and save data
def process_battery_data(input_file, output_file, date_column):
    # Read the txt file
    df = pd.read_csv(input_file, sep='\t')
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Filter data between Feb 2, 2024 and Mar 2, 2024 (inclusive)
    start_date = pd.to_datetime('2024-02-02')
    end_date = pd.to_datetime('2024-03-02 23:59:59')
    
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    
    # Save the filtered data as CSV
    filtered_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Processing {input_file}:")
    print(f"  Original records: {len(df)}")
    print(f"  Filtered records: {len(filtered_df)}")
    print(f"  Saved to: {output_file}")
    print()

# Process swap in dataset
process_battery_data(
    input_file=r"C:\Users\HP\Documents\battery\data\swap\swap_in.txt", 
    output_file=r"C:\Users\HP\Documents\battery\data\swap\filtered_swap_in_dataset.csv", 
    date_column="swap_in_date"
)

# Process swap out dataset
process_battery_data(
    input_file=r"C:\Users\HP\Documents\battery\data\swap\swap_out.txt", 
    output_file=r"C:\Users\HP\Documents\battery\data\swap\filtered_swap_out_dataset.csv", 
    date_column="swap_out_date"
)

print("Data processing complete!")