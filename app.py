import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Load the dataset
file_path = r'C:\Users\HP\Desktop\Leah\battery\merged_data\enhanced_battery_data.csv'
df = pd.read_csv(file_path)

# Convert time column to datetime if needed
if '_time' in df.columns:
    df['_time'] = pd.to_datetime(df['_time'])

# Handle sparse data: Focus on non-zero values or aggregate
# Example: Aggregate cell voltages into summary statistics
voltage_cols = [col for col in df.columns if col.startswith('cell_voltage_')]
df['voltage_std'] = df[voltage_cols].std(axis=1)
df['voltage_range'] = df[voltage_cols].max(axis=1) - df[voltage_cols].min(axis=1)

# Encode categorical features
categorical_features = ['Location_type', 'Battery_status', 'BMS_switch_C_FET_state', 'BMS_switch_D_FET_state', 'alarmDesc', 'fetStateCombo']
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Prepare features and target
features = ['Internal_temperature_of_battery', 'BMS_PCB_board_surface_temperature', 'SOC', 'Total_current', 
            'voltage_std', 'voltage_range', 'tempDiff', 'capacityUtilization', 'cellVoltageRange', 
            'cellVoltageStd', 'Mileage', 'Speed_information', 'Total_charged_capacity', 'Total_discharged_capacity', 
            'Total_voltage', 'times_100_of_battery_capacity', 'cycle_duration_hours', 'SOC_change', 
            'hour', 'dayOfWeek', 'month', 'cellVoltageMin', 'cellVoltageMax', 'cellVoltageAvg', 
            'normalizedTotalVoltage'] + categorical_features

target = 'Number_of_cycles'

# Drop rows with missing target values
df = df.dropna(subset=[target])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostRegressor(cat_features=categorical_features, verbose=0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance: MSE = {mse}, R2 = {r2}")

# Add predictions to the dataframe
df['predicted_cycles'] = model.predict(df[features])

import os
import json
import joblib

# Add model saving functionality
def save_model_and_metadata(model, features, categorical_features, target, alert_thresholds, save_dir='model_artifacts'):
    """
    Save the trained model, features, and associated metadata.
    
    Args:
    - model: Trained CatBoost model
    - features: List of feature names used in training
    - categorical_features: List of categorical feature names
    - target: Target variable name
    - alert_thresholds: Dictionary of alert thresholds
    - save_dir: Directory to save model artifacts
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(save_dir, 'battery_health_model.joblib')
    joblib.dump(model, model_path)
    
    # Prepare and save metadata
    metadata = {
        'features': features,
        'categorical_features': categorical_features,
        'target_variable': target,
        'alert_thresholds': alert_thresholds,
        'model_performance': {
            'mean_squared_error': float(mse),
            'r2_score': float(r2)
        }
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(save_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

# Function to load the saved model and metadata
def load_saved_model(save_dir='model_artifacts'):
    """
    Load the saved model and its metadata.
    
    Args:
    - save_dir: Directory where model artifacts are saved
    
    Returns:
    - Tuple of (model, metadata)
    """
    try:
        # Load the model
        model_path = os.path.join(save_dir, r'C:\Users\HP\Documents\battery\merged_data\model\battery_health_model.joblib')
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except FileNotFoundError:
        print("No saved model found. Please train and save a model first.")
        return None, None

# Save model and metadata after training
save_model_and_metadata(
    model, 
    features, 
    categorical_features, 
    target, 
    alert_thresholds={
        'temperature': {
            'threshold': 50,
            'description': 'Exceeded maximum safe temperature'
        },
        'voltage_std': {
            'threshold': 0.1,
            'description': 'High voltage standard deviation indicates potential cell imbalance'
        },
        'soc': {
            'threshold': 20,
            'description': 'Low state of charge might indicate battery performance issues'
        }
    }
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to generate PDF of alerts
def generate_alerts_pdf(alerts_data):
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create story (content) for PDF
    story = []
    
    # Add title
    story.append(Paragraph("Battery Alerts Detailed Report", styles['Title']))
    
    # Add alerts
    if alerts_data:
        story.append(Paragraph("Detailed Battery Alerts:", styles['Heading2']))
        
        # Create header for alert table
        for battery_id, alerts in alerts_data.items():
            # Battery ID section
            story.append(Paragraph(f"Battery ID: {battery_id}", styles['Heading3']))
            
            # Create table data with Paragraphs to handle text wrapping
            table_data = [
                [
                    Paragraph('Alert Type', styles['Heading4']), 
                    Paragraph('Details', styles['Heading4']), 
                    Paragraph('Current Value', styles['Heading4']), 
                    Paragraph('Threshold', styles['Heading4'])
                ]
            ]
            
            # Add rows for each alert
            for alert in alerts:
                table_data.append([
                    Paragraph(str(alert['type']), styles['Normal']), 
                    Paragraph(str(alert['description']), styles['Normal']), 
                    Paragraph(str(alert['current_value']), styles['Normal']), 
                    Paragraph(str(alert['threshold']), styles['Normal'])
                ])
            
            # Create table with adjusted column widths and text wrapping
            table = Table(table_data, colWidths=[75, 250, 75, 75])
            
            # Add table style
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('WORDWRAP', (0,0), (-1,-1), 1)
            ]))
            
            story.append(table)
            story.append(Paragraph(" ", styles['Normal']))  # Add some space between battery sections
    else:
        story.append(Paragraph("No alerts triggered.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF content
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Battery Health Dashboard"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='cycle-distribution'),
            dcc.Graph(id='temperature-vs-cycles'),
            dcc.Graph(id='voltage-balance-vs-cycles'),
            dcc.Graph(id='soc-distribution'),
            dcc.Graph(id='predicted-vs-actual-cycles'),
            dcc.Graph(id='feature-importance')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Alerts"),
            html.Div(id='alerts'),
            html.A(
                dbc.Button("Download Alerts PDF", color="primary", id='download-alerts-btn'),
                id='download-alerts-link',
                download='battery_alerts.pdf',
                href='',
                target="_blank"
            )
        ], width=12)
    ])
])

# Define callback to update graphs and alerts
@app.callback(
    [Output('cycle-distribution', 'figure'),
     Output('temperature-vs-cycles', 'figure'),
     Output('voltage-balance-vs-cycles', 'figure'),
     Output('soc-distribution', 'figure'),
     Output('predicted-vs-actual-cycles', 'figure'),
     Output('feature-importance', 'figure'),
     Output('alerts', 'children'),
     Output('download-alerts-link', 'href')],
    [Input('cycle-distribution', 'relayoutData')]
)
def update_dashboard(_):
    # Cycle distribution
    cycle_fig = px.histogram(df, x='Number_of_cycles', nbins=30, title='Distribution of Battery Cycle Counts')

    # Temperature analysis
    temp_fig = px.line(df, x='Number_of_cycles', y='Internal_temperature_of_battery', title='Internal Temperature vs Cycle Count')

    # Voltage analysis
    voltage_fig = px.scatter(df, x='Number_of_cycles', y='voltage_std', title='Cell Voltage Standard Deviation vs Cycle Count')

    # SOC distribution
    soc_fig = px.histogram(df, x='SOC', nbins=20, title='State of Charge Distribution')

    # Predicted vs Actual Cycles
    pred_vs_actual_fig = px.scatter(df, x='Number_of_cycles', y='predicted_cycles', 
                                    title='Predicted vs Actual Number of Cycles', 
                                    labels={'predicted_cycles': 'Predicted Cycles', 'Number_of_cycles': 'Actual Cycles'})

    # Feature Importance
    feature_importance = model.get_feature_importance()
    feature_importance_fig = px.bar(x=features, y=feature_importance, 
                                    title='Feature Importance for Predicting Number of Cycles', 
                                    labels={'x': 'Features', 'y': 'Importance'})

    # Alerts with detailed information for each battery
    alert_thresholds = {
        'temperature': {
            'threshold': 50,
            'description': 'Exceeded maximum safe temperature'
        },
        'voltage_std': {
            'threshold': 0.1,
            'description': 'High voltage standard deviation indicates potential cell imbalance'
        },
        'soc': {
            'threshold': 20,
            'description': 'Low state of charge might indicate battery performance issues'
        }
    }
    
    # Grouped alerts by battery_id
    alerts_data = {}
    
    for battery_id, group in df.groupby('battery_id'):
        battery_alerts = []
        
        # Temperature alert
        max_temp = group['Internal_temperature_of_battery'].max()
        if max_temp > alert_thresholds['temperature']['threshold']:
            battery_alerts.append({
                'type': 'Temperature',
                'description': alert_thresholds['temperature']['description'],
                'current_value': max_temp,
                'threshold': alert_thresholds['temperature']['threshold']
            })
        
        # Voltage standard deviation alert
        max_voltage_std = group['voltage_std'].max()
        if max_voltage_std > alert_thresholds['voltage_std']['threshold']:
            battery_alerts.append({
                'type': 'Voltage Deviation',
                'description': alert_thresholds['voltage_std']['description'],
                'current_value': max_voltage_std,
                'threshold': alert_thresholds['voltage_std']['threshold']
            })
        
        # State of Charge alert
        min_soc = group['SOC'].min()
        if min_soc < alert_thresholds['soc']['threshold']:
            battery_alerts.append({
                'type': 'State of Charge',
                'description': alert_thresholds['soc']['description'],
                'current_value': min_soc,
                'threshold': alert_thresholds['soc']['threshold']
            })
        
        # Store alerts if any for this battery
        if battery_alerts:
            alerts_data[battery_id] = battery_alerts

    # Create HTML for alerts display
    if alerts_data:
        alerts_html = []
        for battery_id, alerts in alerts_data.items():
            alerts_html.append(html.H4(f"Battery {battery_id} Alerts:"))
            for alert in alerts:
                alerts_html.append(html.P(
                    f"{alert['type']} Alert: {alert['description']} "
                    f"(Current: {alert['current_value']}, Threshold: {alert['threshold']})"
                ))
    else:
        alerts_html = [html.P("No alerts triggered.")]

    # Generate PDF and create downloadable link
    if alerts_data:
        pdf_bytes = generate_alerts_pdf(alerts_data)
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        download_link = f"data:application/pdf;base64,{pdf_base64}"
    else:
        download_link = ''

    return cycle_fig, temp_fig, voltage_fig, soc_fig, pred_vs_actual_fig, feature_importance_fig, alerts_html, download_link

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)