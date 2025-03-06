"""
Battery Data Missing Values Exploration

This script performs an exploratory data analysis (EDA) focused on identifying 
and reporting missing values in a battery dataset.

Key functionalities:
- Reads a CSV file containing battery data
- Calculates the number and percentage of missing values per column
- Generates a comprehensive missing values report
- Provides insights into data completeness

Dependencies:
- pandas

Usage:
1. Ensure the correct file path is set for the input CSV
2. Run the script to generate the missing values report
3. Review the console output and generated CSV report

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def analyze_missing_values(file_path):
    """
    Analyze missing values in a DataFrame.
    
    Args:
        file_path (str): Path to the input CSV file
    
    Returns:
        pandas.DataFrame: Summary of missing values
    """
    try:
        # Read the CSV with low memory option to handle large files
        df = pd.read_csv(file_path, low_memory=True)
        
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100

        # Create a summary DataFrame
        missing_summary = pd.DataFrame({
            "Total Missing": missing_values, 
            "Percentage Missing (%)": missing_percentage.round(2)
        })
        
        # Filter to show only columns with missing values
        missing_summary = missing_summary[missing_summary["Total Missing"] > 0]
        
        # Sort by percentage of missing values in descending order
        missing_summary = missing_summary.sort_values("Percentage Missing (%)", ascending=False)
        
        return missing_summary
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

def save_missing_values_report(missing_summary, output_path):
    """
    Save the missing values summary to a CSV file.
    
    Args:
        missing_summary (pandas.DataFrame): Missing values summary
        output_path (str): Path to save the output CSV
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the report
        missing_summary.to_csv(output_path)
        logging.info(f"Missing values report saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")

def main():
    """
    Main function to run the missing values analysis.
    """
    # Input file path (consider using a configuration file or command-line argument)
    input_file = "C:/Users/HP/Documents/battery/merged_data/merged_battery_data.csv"
    
    # Output file path for the report
    output_file = "C:/Users/HP/Documents/battery/merged_data/missing_values_report.csv"
    
    # Analyze missing values
    missing_summary = analyze_missing_values(input_file)
    
    if missing_summary is not None and not missing_summary.empty:
        # Display summary in console
        print("\n🔍 Missing Values Summary:")
        print(missing_summary)
        
        # Save the report
        save_missing_values_report(missing_summary, output_file)
    else:
        logging.warning("No missing values report could be generated.")

if __name__ == "__main__":
    main()