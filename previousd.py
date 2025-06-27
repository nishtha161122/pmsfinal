import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os 

class EnhancedLSTMAutoencoder(nn.Module): 
    """  
    Same architecture as in training - needed for loading weights 
    """ 

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2): 
        super(EnhancedLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first = True) 
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size = input_size, batch_first=True) 
    def forward (self,x):
        _, (h_n,_ ) = self.encoder(x) 
        h_n = h_n.repeat(x.size(1), 1, 2).transpose(0,1) 
        decoder = self.decoder(h_n)
        return decoder  

def load_trained_models():
    """ 
    Load all trained models and metadata
    """ 

    print("Loading trained models...") 
    
    # Load metadata
    with open('preprocessed_data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Load training info
    with open('models/training_info.pkl', 'rb') as f:
        training_info = pickle.load(f)
    
    # Load autoencoder
    model = EnhancedLSTMAutoencoder(
        input_size=training_info['input_size'],
        hidden_size=training_info['hidden_size']
    )
    model.load_state_dict(torch.load('models/best_autoencoder.pth'))
    model.eval()
    
    # Load failure classifier
    with open('models/failure_classifier.pkl', 'rb') as f:
        failure_classifier = pickle.load(f)
    
    # Load scaler
    with open('preprocessed_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("All models loaded successfully!") 
    return model, failure_classifier, scaler, metadata, training_info

def compute_reconstruction_errors(model, sequences, batch_size=32):
    """
    Compute reconstruction errors for all sequences
    """

    print(f"Computing reconstruction errors for {len(sequences)} sequences...")
    
    all_tensor = torch.tensor(sequences, dtype=torch.float32)
    reconstruction_errors = []
    feature_wise_errors = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_tensor), batch_size):
            batch = all_tensor[i:i+batch_size]
            reconstructed = model(batch)
            
            # Overall MSE per sequence
            mse = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())
            
            # Feature-wise errors for detailed analysis
            feature_mse = torch.mean((reconstructed - batch) ** 2, dim=1)
            feature_wise_errors.extend(feature_mse.cpu().numpy())
    
    return np.array(reconstruction_errors), np.array(feature_wise_errors)

def calculate_threshold(reconstruction_errors, method='statistical', k=3, percentile=95):
    """
    Calculate anomaly detection threshold using different methods
    """

    if method == 'statistical':
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + k * std_error
        print(f"Statistical threshold (mean + {k}*std): {threshold:.6f}")
    
    elif method == 'percentile':
        threshold = np.percentile(reconstruction_errors, percentile)
        print(f"Percentile threshold ({percentile}th): {threshold:.6f}")
    
    else:
        raise ValueError("Method must be 'statistical' or 'percentile'")
    
    return threshold

def detect_anomalies(reconstruction_errors, threshold):
    """
    Detect anomalies based on reconstruction errors and threshold
    """

    anomalies = reconstruction_errors > threshold
    num_anomalies = np.sum(anomalies)
    anomaly_rate = (num_anomalies / len(reconstruction_errors)) * 100
    
    print(f"\nAnomaly Detection Results:")
    print(f"Total sequences: {len(reconstruction_errors)}")
    print(f"Anomalies detected: {num_anomalies} ({anomaly_rate:.2f}%)")
    print(f"Threshold: {threshold:.6f}")
    print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    print(f"Max reconstruction error: {np.max(reconstruction_errors):.6f}")
    print (f"Min reconstruction error: {np.min(reconstruction_errors)}")
    print(f"Anomaly rate: {anomaly_rate:.2f}")
    
    return anomalies

def classify_failures(sequences, failure_classifier, anomaly_indices):
    """
    Classify failure types for detected anomalies
    """

    if len(anomaly_indices) == 0:
        print("No anomalies detected for failure classification.")
        print(f"No anomalies detected:")
        return None, None
    
    print(f"\nClassifying failure types for {len(anomaly_indices)} anomalies...")
    
    # Flatten sequences for classification
    anomaly_sequences = sequences[anomaly_indices]
    X_anomalies = anomaly_sequences.reshape(len(anomaly_sequences), -1)
    
    # Predict failure types
    failure_predictions = failure_classifier.predict(X_anomalies)
    failure_probabilities = failure_classifier.predict_proba(X_anomalies)
    
    return failure_predictions, failure_probabilities

def analyze_results(sequences, labels, reconstruction_errors, anomalies, 
                   failure_predictions, metadata):
    """
    Comprehensive analysis of detection results
    """

    print("\n" + "-"*20)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("-"*20)
    
    failure_columns = metadata['failure_columns']
    sensor_columns = metadata['sensor_columns']
    
    # Ground truth vs predictions analysis
    if labels is not None:
        print("\nGround Truth vs Anomaly Detection:")
        actual_failures = labels.sum(axis=1) > 0  # Any failure
        
        # Confusion matrix for anomaly detection
        from sklearn.metrics import confusion_matrix, classification_report
        
        print("\nAnomaly Detection Performance:")
        print(classification_report(actual_failures, anomalies, 
                                  target_names=['Normal', 'Anomaly']))
        
        # Confusion matrix
        cm = confusion_matrix(actual_failures, anomalies)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Anomaly Detection Confusion Matrix')
        plt.ylabel('Actual') 
        plt.xlabel('Predicted')
        plt.savefig('plots/anomaly_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Failure type analysis 
    if failure_predictions is not None: 
        print(f"\nFailure Type Distribution in Detected") 