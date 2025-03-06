# Battery Health Prediction System

This project provides a comprehensive battery health monitoring and prediction system. It includes data preparation, exploratory data analysis (EDA), feature engineering, model training, and a predictive maintenance dashboard.

## Project Overview

The system analyzes battery performance metrics to predict battery health and provide maintenance alerts. The workflow includes:

1. Processing battery performance data
2. Analyzing battery swap events (in and out)
3. Merging datasets for comprehensive analysis
4. Feature engineering for predictive modeling
5. Implementing a dashboard for real-time monitoring

## Table of Contents

- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Dashboard](#dashboard)
- [Directory Structure](#directory-structure)

## Setup

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/LeahN67/battery_health_prediction_system.git
   cd battery_health_prediction_system
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv <environment_name>
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     <environment_name>\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source <environment_name>/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Battery Dataset

1. Navigate to the battery data  scripts folder:
   ```bash
   cd battery_data
   ```

2. Run the data preparation script:
   ```bash
   python prep_battery_data.py
   ```
   This script processes raw battery data, performs cleaning, and creates a structured dataset for analysis.

3. Run the EDA script:
   ```bash
   python eda_battery_dataset.py
   ```
   This script performs exploratory data analysis on the battery dataset, generating visualizations for battery performance metrics.

### Swap Dataset

1. Navigate to the swap data processing scripts folder:
   ```bash
   cd swap_data
   ```

2. Run the data preparation script:
   ```bash
   python prep_swap_data.py
   ```
   This script processes the swap in and swap out event data, aligns timestamps, and structures the data for analysis.

3. Run the EDA script:
   ```bash
   python eda_swap_datasets.py
   ```
   This script performs exploratory data analysis on the swap dataset, generating visualizations for swap patterns and related metrics.

### Merged Dataset

1. Run the dataset merging script:
   ```bash
   python merge_data.py
   ```
   This script merges the battery and swap datasets based on battery IDs and timestamps.

2. Run the merged EDA script:
   ```bash
   python eda_merged_dataset.py
   ```
   This script performs exploratory data analysis on the merged dataset, generating visualizations for comprehensive battery performance analysis.

3. Run the feature engineering script:
   ```bash
   python feature_engineering.py
   ```
   This script creates advanced features from the merged dataset for predictive modeling, such as:
   - Voltage statistics (standard deviation, range)
   - Temperature differentials
   - Capacity utilization metrics
   - Temporal features (hour, day of week, month)
   - Cycle-based metrics

## Model Training

The model training is integrated directly into the app.py file. When you run the dashboard application, it:
- Loads and processes the merged dataset
- Prepares features and target variables
- Splits data into training and testing sets
- Trains a CatBoost regressor model
- Evaluates model performance
- Saves the trained model and metadata to the model_artifacts directory

## Dashboard

1. Launch the dashboard application:
   ```bash
   python app.py
   ```
   This will:
   - Load the enhanced battery dataset
   - Train the CatBoost model if not already trained
   - Save the model and metadata to the model_artifacts directory
   - Start the Dash application on http://127.0.0.1:8050/

2. The dashboard includes:
   - Battery cycle distribution visualization
   - Temperature vs. cycles analysis
   - Voltage balance analysis
   - State of charge distribution
   - Predicted vs. actual cycle comparison
   - Feature importance visualization
   - Alert system for battery health issues
   - PDF report generation for detailed alerts

## Directory Structure

```
battery-health-prediction/
├── app.py                     # Main dashboard application with integrated model training
├── requirements.txt           # Project dependencies
├── model_artifacts/           # Saved models and metadata
│   ├── battery_health_model.joblib
│   └── model_metadata.json
├── battery_data/              # Battery data processing scripts
│   ├── prep_battery_dataset.py
│   ├── eda_battery_dataset.py
│   └── battery_visualisations/                  # Visualisations of the battery data
├── swap_data/                 # Swap data processing scripts
│   ├── prep_swap_data.py
│   ├── eda_swap_datasets.py
│   └── swap_visualisations/                  # Visualisations of the swap data
└── merged_data/               # Merged dataset and processing scripts
    ├── eda_merge_dataset.py
    ├── merge_eda.py
    ├── feature_engineering.py
    
```

## Data Description

### Battery Dataset
The battery dataset contains various metrics from battery operations, including:
- Internal temperature
- PCB board surface temperature
- State of charge (SOC)
- Current and voltage readings
- Cell-level voltage measurements
- Battery capacity metrics

### Swap Dataset
The swap dataset tracks battery swap events, including:
- Swap-in events (when a battery is installed)
- Swap-out events (when a battery is removed)
- Timestamps for each event
- Location information
- Battery status before and after swap

### Merged Dataset
The merged dataset combines battery operational data with swap event information to create a comprehensive view of each battery's lifecycle and performance.

## Model Details

The system uses a CatBoost regressor model to predict the number of battery cycles. Key features include:
- Temperature metrics
- Voltage characteristics
- State of charge patterns
- Capacity utilization
- Temporal factors

## Alert System

The dashboard includes an alert system that monitors:
- Excessive battery temperature
- High voltage imbalance between cells
- Low state of charge conditions

When these conditions exceed predefined thresholds, alerts are generated and can be exported as a detailed PDF report.