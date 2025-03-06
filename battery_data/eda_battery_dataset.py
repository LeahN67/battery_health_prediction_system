# Battery Data Exploratory Data Analysis (EDA) Script
# This script performs a comprehensive analysis of battery performance data
# Key functionalities:
# 1. Data loading and preprocessing
# 2. Temporal analysis
# 3. Distribution of key variables
# 4. Cell voltage analysis
# 5. Temperature analysis
# 6. Correlation studies
# 7. Battery health metrics visualization
# 8. Outlier and missing data detection

# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib as mpl

# Configuration for matplotlib to handle large datasets
mpl.rcParams['agg.path.chunksize'] = 10000  # Increase chunk size for rendering
mpl.rcParams['path.simplify_threshold'] = 0.2  # Improve plot simplification

# Set visualization style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

class BatteryDataEDA:
    def __init__(self, csv_path, output_dir=None):
        """
        Initialize the EDA analysis with the battery dataset
        
        Parameters:
        -----------
        csv_path : str
            Path to the battery data CSV file
        output_dir : str, optional
            Directory to save output visualizations and reports
        """
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(csv_path), 'battery_visualisations')
        os.makedirs(output_dir, exist_ok=True)
        
        self.csv_path = csv_path
        self.output_dir = output_dir
        
        # Load the dataset
        self.df = self._load_and_preprocess_data()
    
    def _load_and_preprocess_data(self):
        """
        Load CSV data and perform initial preprocessing
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed battery data
        """
        # Read the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Convert _time column to datetime with error handling
        try:
            df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
            
            # Remove rows with invalid datetime
            na_count = df['_time'].isna().sum()
            if na_count > 0:
                original_len = len(df)
                df = df.dropna(subset=['_time'])
                print(f"Removed {original_len - len(df)} rows with invalid datetime values")
        except Exception as e:
            print(f"Error processing time column: {e}")
            raise
        
        return df
    
    def save_figure(self, fig, filename):
        """
        Save figure to output directory with high resolution
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Name of the output file
        """
        fig.savefig(os.path.join(self.output_dir, filename), 
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def plot_time_series(self, column, title, filename):
        """
        Create a time series plot for a specific column
        
        Parameters:
        -----------
        column : str
            Column to plot
        title : str
            Plot title
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        self.df.sort_values('_time').plot(x='_time', y=column, 
                                           ax=ax, marker='o', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        self.save_figure(fig, filename)
    
    def plot_distribution(self, column, title, filename, bins=30):
        """
        Plot distribution of a numeric column
        
        Parameters:
        -----------
        column : str
            Column to analyze
        title : str
            Plot title
        filename : str
            Output filename
        bins : int, optional
            Number of bins for histogram
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(self.df[column].dropna(), kde=True, bins=bins, ax=ax)
        ax.set_title(f'Distribution of {title}')
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        self.save_figure(fig, filename)
    
    def analyze_cell_voltages(self):
        """
        Perform comprehensive cell voltage analysis
        - Heatmap of cell voltages
        - Violin plot of cell voltage distributions
        - Cell voltage imbalance over time
        """
        # Identify cell voltage columns
        cell_voltage_cols = [col for col in self.df.columns if 'cell_voltage_' in col]
        
        if not cell_voltage_cols:
            print("No cell voltage columns found.")
            return
        
        # Cell voltage heatmap (sampled data)
        sample_size = min(100, len(self.df))
        sample_df = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
        
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(sample_df[cell_voltage_cols].transpose(), 
                    cmap='viridis', yticklabels=True, ax=ax)
        ax.set_title('Cell Voltage Heatmap (Sample)')
        self.save_figure(fig, 'cell_voltage_heatmap.png')
        
        # Cell voltage distribution violin plot
        fig, ax = plt.subplots(figsize=(15, 8))
        df_melt = pd.melt(self.df, value_vars=cell_voltage_cols, 
                           var_name='Cell', value_name='Voltage')
        sns.violinplot(x='Cell', y='Voltage', data=df_melt, ax=ax)
        ax.set_title('Distribution of Cell Voltages')
        ax.set_xlabel('Cell')
        ax.set_ylabel('Voltage (mV)')
        plt.xticks(rotation=90)
        self.save_figure(fig, 'cell_voltage_violinplot.png')
        
        # Calculate and plot cell voltage imbalance
        self.df['max_cell_voltage'] = self.df[cell_voltage_cols].max(axis=1)
        self.df['min_cell_voltage'] = self.df[cell_voltage_cols].min(axis=1)
        self.df['cell_voltage_imbalance'] = (
            self.df['max_cell_voltage'] - self.df['min_cell_voltage']
        )
        
        self.plot_time_series('cell_voltage_imbalance', 
                               'Cell Voltage Imbalance Over Time', 
                               'cell_voltage_imbalance.png')
    
    def correlation_analysis(self):
        """
        Perform correlation analysis of numeric variables
        - Generate correlation matrix
        - Create heatmap visualization
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title('Correlation Matrix of Numeric Variables')
        self.save_figure(fig, 'correlation_matrix.png')
    
    def hourly_patterns_analysis(self):
        """
        Analyze patterns across different hours of the day
        - Aggregate key metrics by hour
        - Create multi-panel plot of hourly variations
        """
        # Extract hour from timestamp
        self.df['hour'] = self.df['_time'].dt.hour
        
        # Key metrics to analyze by hour
        hourly_metrics = {
            'SOC': 'Average SOC by Hour',
            'Total_voltage': 'Average Voltage by Hour',
            'Total_current': 'Average Current by Hour',
            'Internal_temperature_of_battery': 'Average Temperature by Hour'
        }
        
        # Compute hourly aggregations
        hourly_stats = self.df.groupby('hour').agg({
            metric: 'mean' for metric in hourly_metrics.keys()
        }).reset_index()
        
        # Create multi-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(hourly_metrics.items()):
            if metric in hourly_stats.columns:
                sns.barplot(x='hour', y=metric, data=hourly_stats, ax=axes[i])
                axes[i].set_title(title)
                axes[i].set_xlabel('Hour of Day')
        
        plt.tight_layout()
        self.save_figure(fig, 'hourly_patterns.png')
    
    def outlier_detection(self, key_variables):
        """
        Detect and visualize outliers in key variables
        
        Parameters:
        -----------
        key_variables : dict
            Dictionary of variables to analyze
        """
        def detect_outliers(column):
            """
            Detect outliers using Interquartile Range (IQR) method
            
            Returns:
            --------
            pd.Series
                Series of outlier values
            """
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return column[(column < lower_bound) | (column > upper_bound)]
        
        # Analyze outliers
        outlier_analysis = {}
        for col, title in key_variables.items():
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                outliers = detect_outliers(self.df[col])
                outlier_analysis[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.df) * 100,
                    'min': outliers.min() if not outliers.empty else None,
                    'max': outliers.max() if not outliers.empty else None
                }
        
        # Save outlier analysis to file
        outlier_file = os.path.join(self.output_dir, "outlier_analysis.txt")
        with open(outlier_file, 'w') as f:
            f.write("Outlier Analysis\n")
            f.write("-" * 50 + "\n\n")
            for col, stats in outlier_analysis.items():
                f.write(f"{col}:\n")
                f.write(f"  Outlier Count: {stats['count']}\n")
                f.write(f"  Percentage: {stats['percentage']:.2f}%\n")
                f.write(f"  Min: {stats['min']}\n")
                f.write(f"  Max: {stats['max']}\n\n")
        
        # Create box plots for outlier visualization
        fig, axes = plt.subplots(len(key_variables), 1, 
                                  figsize=(12, 4*len(key_variables)))
        if len(key_variables) == 1:
            axes = [axes]
        
        for i, (col, title) in enumerate(key_variables.items()):
            if col in self.df.columns:
                sns.boxplot(x=self.df[col], ax=axes[i])
                axes[i].set_title(f'Box Plot of {title}')
                axes[i].set_xlabel(title)
        
        plt.tight_layout()
        self.save_figure(fig, 'outlier_boxplots.png')
    
    def perform_full_eda(self):
        """
        Perform comprehensive Exploratory Data Analysis
        """
        # Define key variables for analysis
        key_variables = {
            'Total_voltage': 'Total Voltage',
            'Total_current': 'Total Current',
            'SOC': 'State of Charge',
            'Internal_temperature_of_battery': 'Internal Battery Temperature',
            'Surface_temperature_in_the_middle_of_cells': 'Cell Surface Temperature',
            'BMS_PCB_board_surface_temperature': 'BMS PCB Temperature',
            'times_100_of_battery_capacity': 'Battery Capacity (x100)'
        }
        
        # Perform various analyses
        print("Starting Exploratory Data Analysis...")
        
        # 1. Basic data exploration and statistics
        print("Basic Data Exploration...")
        print("Dataset Shape:", self.df.shape)
        print("\nData Types:")
        print(self.df.dtypes)
        
        # 2. Distribution of key variables
        print("\nPlotting distributions...")
        for col, title in key_variables.items():
            if col in self.df.columns:
                self.plot_distribution(col, title, f'{col}_distribution.png')
        
        # 3. Time series analysis for key metrics
        print("\nCreating time series plots...")
        self.plot_time_series('SOC', 'State of Charge (SOC) Over Time', 'soc_over_time.png')
        
        # 4. Cell voltage analysis
        print("\nAnalyzing cell voltages...")
        self.analyze_cell_voltages()
        
        # 5. Correlation analysis
        print("\nPerforming correlation analysis...")
        self.correlation_analysis()
        
        # 6. Hourly patterns
        print("\nAnalyzing hourly patterns...")
        self.hourly_patterns_analysis()
        
        # 7. Outlier detection
        print("\nDetecting outliers...")
        self.outlier_detection(key_variables)
        
        print("\nExploratory Data Analysis completed successfully!")
        print(f"All visualizations have been saved to: {self.output_dir}")

# Example usage
if __name__ == "__main__":
    # Path to your battery CSV file
    CSV_PATH = r'path/to/your/cleaned_battery_data.csv'
    
    # Create EDA instance and run analysis
    battery_eda = BatteryDataEDA(CSV_PATH)
    battery_eda.perform_full_eda()