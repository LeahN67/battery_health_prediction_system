import pandas as pd
import numpy as np
from datetime import datetime

def engineer_battery_features(df):
    """
    Performs feature engineering on battery dataset and drops the id column.
    
    Parameters:
    df (pandas.DataFrame): The input battery dataset
    
    Returns:
    pandas.DataFrame: Enhanced dataset with engineered features
    """
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Drop the id column
    if 'id' in enhanced_df.columns:
        enhanced_df = enhanced_df.drop('id', axis=1)
    
    # Extract time-based features
    enhanced_df['hour'] = pd.to_datetime(enhanced_df['_time']).dt.hour
    enhanced_df['dayOfWeek'] = pd.to_datetime(enhanced_df['_time']).dt.dayofweek
    enhanced_df['month'] = pd.to_datetime(enhanced_df['_time']).dt.month
    
    # Calculate cell voltage statistics
    voltage_cols = [col for col in enhanced_df.columns if col.startswith('cell_voltage_')]
    
    if voltage_cols:
        enhanced_df['cellVoltageMin'] = enhanced_df[voltage_cols].min(axis=1)
        enhanced_df['cellVoltageMax'] = enhanced_df[voltage_cols].max(axis=1)
        enhanced_df['cellVoltageAvg'] = enhanced_df[voltage_cols].mean(axis=1)
        enhanced_df['cellVoltageRange'] = enhanced_df['cellVoltageMax'] - enhanced_df['cellVoltageMin']
        enhanced_df['cellVoltageStd'] = enhanced_df[voltage_cols].std(axis=1)
    
    # Temperature difference
    if 'BMS_PCB_board_surface_temperature' in enhanced_df.columns and 'Internal_temperature_of_battery' in enhanced_df.columns:
        enhanced_df['tempDiff'] = enhanced_df['BMS_PCB_board_surface_temperature'] - enhanced_df['Internal_temperature_of_battery']
    
    # Capacity utilization
    if 'SOC' in enhanced_df.columns and 'times_100_of_battery_capacity' in enhanced_df.columns:
        enhanced_df['capacityUtilization'] = enhanced_df['SOC'] / enhanced_df['times_100_of_battery_capacity']
    
    # Is charging indicator (based on current)
    if 'Total_current' in enhanced_df.columns:
        enhanced_df['isCharging'] = (enhanced_df['Total_current'] > 0).astype(int)
    
    # FET state combination (C and D FETs)
    if 'BMS_switch_C_FET_state' in enhanced_df.columns and 'BMS_switch_D_FET_state' in enhanced_df.columns:
        enhanced_df['fetStateCombo'] = enhanced_df['BMS_switch_C_FET_state'].astype(str) + '_' + enhanced_df['BMS_switch_D_FET_state'].astype(str)
    
    # Normalize total voltage by number of cells
    if 'Total_voltage' in enhanced_df.columns and 'Cell_series_numbe_23_series' in enhanced_df.columns:
        # Use the number of cells from the data if available, otherwise default to 24
        cell_count = enhanced_df['Cell_series_numbe_23_series'].fillna(24)
        enhanced_df['normalizedTotalVoltage'] = enhanced_df['Total_voltage'] / cell_count
    
    # Convert boolean swap_cycle to int for easier processing if needed
    if 'swap_cycle' in enhanced_df.columns:
        enhanced_df['swap_cycle'] = enhanced_df['swap_cycle'].map({'True': 1, 'False': 0})
    
    return enhanced_df


df = pd.read_csv(r'C:\Users\HP\Documents\battery\merged_data\final_merged_dataset.csv')
enhanced_df = engineer_battery_features(df)
enhanced_df.to_csv(r'C:\Users\HP\Documents\battery\merged_data\enhanced_battery_data.csv', index=False)