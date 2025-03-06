import dask.dataframe as dd
import pandas as pd
from ydata_profiling import ProfileReport
import webbrowser

# Define the file path
file_path = "C:/Users/HP/Documents/battery/merged_data/cleaned_battery_data.csv"  # Update path

# Load dataset
df = dd.read_csv(file_path, low_memory=False, assume_missing=True, header=0, dtype=str)
df = df.compute()  # Convert to Pandas

# Ensure first row is not duplicated as data
if df.iloc[0, 0] == "BMS_PCB_board_surface_temperature":
    df = df[1:].reset_index(drop=True)

# Convert numeric columns safely
numeric_cols = [
    "BMS_PCB_board_surface_temperature", "BMS_switch_C_FET_state", "BMS_switch_D_FET_state",
    "Battery_control", "Battery_status", "Cell_series_numbe_23_series",
    "Internal_temperature_of_battery", "Mileage", "Number_of_cycles", "SOC",
    "Speed_information", "Surface_temperature_in_the_middle_of_cells", "Total_charged_capacity",
    "Total_current", "Total_discharged_capacity", "Total_voltage", "alarmFlag",
    "cell_voltage_1", "cell_voltage_2", "cell_voltage_3", "cell_voltage_4", "cell_voltage_5",
    "cell_voltage_6", "cell_voltage_7", "cell_voltage_8", "cell_voltage_9", "cell_voltage_10",
    "cell_voltage_11", "cell_voltage_12", "cell_voltage_13", "cell_voltage_14",
    "cell_voltage_15", "cell_voltage_16", "cell_voltage_17", "cell_voltage_18",
    "cell_voltage_19", "cell_voltage_20", "cell_voltage_21", "cell_voltage_22",
    "cell_voltage_23", "cell_voltage_24", "times_100_of_battery_capacity"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert invalid values to NaN

# Fill missing values
'''
df.fillna({
    "BMS_switch_C_FET_state": 0, "BMS_switch_D_FET_state": 0,
    "Battery_control": 0, "Battery_status": 0,
    "Total_current": df["Total_current"].median(),
    "alarmFlag": df["alarmFlag"].median()
}, inplace=True)

'''

# Save cleaned dataset
cleaned_file = "C:/Users/HP/Documents/battery/merged_data/cleaned_battery_data.csv"
df.to_csv(cleaned_file, index=False)
print(f"✅ Data cleaned and saved as {cleaned_file}")

# **🔍 Optimize EDA to Reduce Memory Usage**
#df_sample = df.sample(frac=0.1, random_state=42)  # Use 10% of data

profile = ProfileReport(
    df,
    title="Battery Data EDA Report",
    minimal=True,
    explorative=True,
    progress_bar=True
)

# Save the report
eda_report_file = "C:/Users/HP/Documents/battery/merged_data/battery_data_eda_report.html"
profile.to_file(eda_report_file)

# Open report in browser
webbrowser.open(eda_report_file)

print(f"✅ EDA report generated and saved as {eda_report_file}")
