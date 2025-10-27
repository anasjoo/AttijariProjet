from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import sys

# Add Backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from anomaly_detector_complete import BankingAnomalyDetector
from banking_llm_simple import BankingLLM
from manual_anomaly_detector import ManualAnomalyDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize components
detector = BankingAnomalyDetector()
llm = None  # Lazy initialization
manual_detector = ManualAnomalyDetector()

def get_llm():
    """Lazy initialization of LLM"""
    global llm
    if llm is None:
        llm = BankingLLM()
    return llm

# Load data
clients_df = pd.read_csv('../datasets/fake_clients.csv')
transactions_df = pd.read_csv('../datasets/fake_transactions.csv')
full_df = transactions_df.merge(clients_df, on='client_id', how='left')

# Initialize manual detector with training data
def initialize_manual_detector():
    """Initialize the manual anomaly detector with training data"""
    try:
        # Prepare training data for manual detector
        training_data = manual_detector.prepare_training_data(transactions_df, clients_df)
        
        # Evaluate model and print metrics to terminal
        metrics = manual_detector.evaluate_model(training_data)
        
        if metrics:
            print("\n" + "="*60)
            print("📊 MANUAL ANOMALY DETECTOR METRICS (Terminal Output Only)")
            print("="*60)
            print(f"Model Accuracy: {metrics['accuracy']:.2%}")
            print(f"Total Training Samples: {metrics['total_samples']}")
            print(f"Anomalies Detected in Training: {metrics['anomalies_detected']}")
            print(f"Training Anomaly Rate: {metrics['anomaly_rate']:.2%}")
            print("="*60)
        
        return True
    except Exception as e:
        print(f"Error initializing manual detector: {e}")
        return False

@app.route('/')
def dashboard():
    """Main dashboard page"""
    # Get basic statistics
    total_clients = len(clients_df)
    total_transactions = len(transactions_df)
    
    # Detect anomalies for overview
    anomalies = detector.detect_all_anomalies(full_df)
    total_anomalies = len(anomalies)
    anomaly_rate = (total_anomalies / total_transactions * 100) if total_transactions > 0 else 0
    
    # Get recent anomalies
    recent_anomalies = anomalies.head(10).to_dict('records') if not anomalies.empty else []
    
    # Convert NaN to None for JSON serialization
    for anomaly in recent_anomalies:
        for key, value in anomaly.items():
            if pd.isna(value):
                anomaly[key] = None
    
    return render_template('dashboard.html',
                         total_clients=total_clients,
                         total_transactions=total_transactions,
                         total_anomalies=total_anomalies,
                         anomaly_rate=round(anomaly_rate, 2),
                         recent_anomalies=recent_anomalies)

@app.route('/clients')
def clients():
    """Client listing page"""
    clients_list = clients_df.to_dict('records')
    return render_template('clients.html', clients=clients_list)

@app.route('/client/<client_id>')
def client_detail(client_id):
    """Individual client detail page"""
    client_data = clients_df[clients_df['client_id'] == client_id]
    if client_data.empty:
        return "Client not found", 404
    
    client_info = client_data.iloc[0].to_dict()
    
    # Get client transactions
    client_transactions = full_df[full_df['client_id'] == client_id]
    
    # Detect anomalies for this client
    anomalies = detector.detect_all_anomalies(client_transactions)
    anomalies_list = anomalies.to_dict('records') if not anomalies.empty else []
    
    # Convert NaN to None
    for anomaly in anomalies_list:
        for key, value in anomaly.items():
            if pd.isna(value):
                anomaly[key] = None
    
    return render_template('client_detail.html',
                         client=client_info,
                         transactions=client_transactions.to_dict('records'),
                         anomalies=anomalies_list)

@app.route('/anomalies')
def anomalies():
    """All anomalies page"""
    anomalies = detector.detect_all_anomalies(full_df)
    anomalies_list = anomalies.to_dict('records') if not anomalies.empty else []
    
    # Convert NaN to None
    for anomaly in anomalies_list:
        for key, value in anomaly.items():
            if pd.isna(value):
                anomaly[key] = None
    
    return render_template('anomalies.html', anomalies=anomalies_list)

@app.route('/api/search_client/<query>')
def search_client(query):
    """API endpoint for client search"""
    results = clients_df[
        clients_df['client_id'].str.contains(query, case=False, na=False) |
        clients_df['nom'].str.contains(query, case=False, na=False) |
        clients_df['prenom'].str.contains(query, case=False, na=False)
    ]
    return jsonify(results.to_dict('records'))

@app.route('/manual-entry', methods=['GET'])
def manual_entry_form():
    """Render manual entry form"""
    return render_template('manual_entry.html')

@app.route('/manual-entry', methods=['POST'])
def manual_entry_submit():
    """Handle manual entry form submission and anomaly detection"""
    try:
        # Ensure model is trained
        manual_detector.ensure_model_trained()
        
        # Get form data with proper defaults
        client_data = {
            'client_id': request.form.get('client_id', 'N/A'),
            'montant': float(request.form.get('montant', 0)),
            'revenu_mensuel': float(request.form.get('revenu_mensuel', 3000)),
            'age': int(request.form.get('age', 30)),
            'nb_transactions_7j': int(request.form.get('nb_transactions_7j', 3))
        }
        
        # Detect anomaly
        result = manual_detector.detect_manual_anomaly(client_data)
        
        if result and 'error' not in result:
            print(f"✅ Manual entry processed: {result['message']}")
            # Display result on the same page
            return render_template('manual_entry.html', 
                                 result=result, 
                                 form_data=client_data)
        else:
            return render_template('manual_entry.html', 
                                 error=result.get('message', 'Erreur inconnue'),
                                 form_data=client_data)
            
    except ValueError as e:
        return render_template('manual_entry.html', 
                             error=f"Erreur de saisie: {str(e)}",
                             form_data=request.form)
    except Exception as e:
        print(f"❌ Error processing manual entry: {e}")
        return render_template('manual_entry.html', 
                             error=f"Erreur: {str(e)}",
                             form_data=request.form)

@app.route('/api/explain_anomaly/<client_id>')
def explain_anomaly(client_id):
    """API endpoint for LLM explanation"""
    try:
        client_data = full_df[full_df['client_id'] == client_id]
        if client_data.empty:
            return jsonify({"error": "Client not found"}), 404
        
        anomalies = detector.detect_all_anomalies(client_data)
        if anomalies.empty:
            return jsonify({"explanation": "No anomalies found for this client"})
        
        # Get the most recent anomaly
        anomaly = anomalies.iloc[0]
        client_info = client_data.iloc[0]
        
        transaction_desc = f"Transaction de {anomaly['montant']} TND"
        
        llm_instance = get_llm()
        explanation = llm_instance.expliquer_transaction(
            client_id=client_id,
            profession=client_info['profession'],
            revenu=client_info['revenu_mensuel'],
            transaction=transaction_desc
        )
        
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """API endpoint for dashboard statistics"""
    anomalies = detector.detect_all_anomalies(full_df)
    
    # Anomaly types distribution
    anomaly_types = anomalies['type_anomalie'].value_counts().to_dict() if not anomalies.empty else {}
    
    # Monthly anomaly trend
    anomalies['date'] = pd.to_datetime(anomalies['date'])
    monthly_trend = anomalies.groupby(anomalies['date'].dt.to_period('M')).size().to_dict()
    
    # Risk level distribution
    risk_levels = anomalies['niveau_risque'].value_counts().to_dict() if not anomalies.empty else {}
    
    return jsonify({
        "anomaly_types": anomaly_types,
        "monthly_trend": {str(k): v for k, v in monthly_trend.items()},
        "risk_levels": risk_levels
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050, use_reloader=False)
