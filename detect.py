import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

class LSTMAutoencoder(nn.Module):
    """Streamlined LSTM Autoencoder - Compatible with saved weights"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder - single layer to match saved weights
        self.encoder = nn.LSTM(
            input_size, hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=0  # No dropout for single layer
        )
        
        # Decoder - single layer to match saved weights
        self.decoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # No dropout for single layer
        )
        
        # Output layer - simple Linear layer to match saved weights
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x): 
        batch_size, seq_len, _ = x.size() 
        
        # Encoding 
        encoded_output, (hidden_state, cell_state) = self.encoder(x)
        context = encoded_output[:, -1, :].unsqueeze(1)
        decoder_input = context.repeat(1, seq_len, 1)
        
        # Decoding
        decoded_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        reconstructed = self.output_layer(decoded_output)
        
        return reconstructed

def inspect_saved_model():
    """Inspect saved model to determine correct architecture"""
    try:
        state_dict = torch.load('models/my_autoencoder.pth', map_location='cpu')
        print("Saved model keys:")
        for key in sorted(state_dict.keys()):
            print(f"  {key}: {state_dict[key].shape}")
        
        # Determine architecture from keys
        has_multi_layer = any('_l1' in key for key in state_dict.keys())
        has_sequential = 'output_layer.0.weight' in state_dict
        
        print(f"\nArchitecture analysis:")
        print(f"  Multi-layer LSTM: {has_multi_layer}")
        print(f"  Sequential output: {has_sequential}")
        
        return state_dict, has_multi_layer, has_sequential
        
    except Exception as e:
        print(f"Could not inspect model: {e}")
        return None, False, False

def load_models_and_data():
    """Load models and data efficiently"""
    print("Loading models and data...")
    
    try:
        # Load metadata
        with open('preprocessed_data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Inspect saved model first
        print("Inspecting saved model architecture...")
        state_dict, has_multi_layer, has_sequential = inspect_saved_model()
        
        if state_dict is None:
            raise Exception("Could not load saved model")
        
        # Create model with correct architecture
        if has_multi_layer:
            if has_sequential:
                # Multi-layer with Sequential output
                class LSTMAutoencoderV2(nn.Module):
                    def __init__(self, input_size, hidden_size=64):
                        super().__init__()
                        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                        self.output_layer = nn.Sequential(
                            nn.Linear(hidden_size, input_size),
                            nn.Tanh()
                        )
                    
                    def forward(self, x):
                        batch_size, seq_len, _ = x.size()
                        encoded_output, (hidden_state, cell_state) = self.encoder(x)
                        context = encoded_output[:, -1, :].unsqueeze(1)
                        decoder_input = context.repeat(1, seq_len, 1)
                        decoded_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
                        reconstructed = self.output_layer(decoded_output)
                        return reconstructed
                
                model = LSTMAutoencoderV2(metadata['num_features'])
            else:
                # Multi-layer with simple Linear output
                model = LSTMAutoencoder(metadata['num_features'], num_layers=2, dropout=0.2)
        else:
            # Single layer 
            model = LSTMAutoencoder(metadata['num_features'], num_layers=1, dropout=0)
    
        # Load the weights 
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded with correct architecture")
        
        # Load scaler
        with open('preprocessed_data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load test data
        sequences = np.load('preprocessed_data/all_sequences.npy')
        labels = np.load('preprocessed_data/all_labels.npy')
        
        print(f"✅ Loaded: {len(sequences)} sequences, {sequences.shape[1:]} shape")
        return model, scaler, sequences, labels, metadata
        
    except Exception as e:
        print(f"❌ Loading error: {e}")
        return None, None, None, None, None

def compute_reconstruction_errors(model, sequences, scaler, batch_size=64):
    """Efficient reconstruction error computation"""
    print(f"Computing reconstruction errors for {len(sequences)} sequences...")
    
    # Scale sequences
    if scaler is not None:
        original_shape = sequences.shape
        sequences_scaled = scaler.transform(sequences.reshape(-1, sequences.shape[-1]))
        sequences = sequences_scaled.reshape(original_shape)
    
    # Convert to tensor
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    reconstruction_errors = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i+batch_size]
            reconstructed = model(batch)
            
            # MSE per sequence
            mse = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())
    
    return np.array(reconstruction_errors)

def detect_anomalies(reconstruction_errors, labels=None, method='percentile'):
    """Efficient anomaly detection"""
    print(f"Detecting anomalies using {method} method...")
    
    if method == 'percentile':
        # Use 95th percentile as threshold
        threshold = np.percentile(reconstruction_errors, 95)
    
    elif method == 'statistical':
        # Mean + 2*std
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + 2 * std_error
    
    elif method == 'adaptive':
        # Find optimal threshold using ground truth
        if labels is not None:
            actual_failures = labels.sum(axis=1) > 0
            best_threshold = np.percentile(reconstruction_errors, 95)
            best_f1 = 0
            
            # Try different percentiles
            for percentile in range(80, 99):
                test_threshold = np.percentile(reconstruction_errors, percentile)
                predictions = reconstruction_errors > test_threshold
                
                if np.sum(predictions) > 0:
                    tp = np.sum(predictions & actual_failures)
                    fp = np.sum(predictions & ~actual_failures)
                    fn = np.sum(~predictions & actual_failures)
                    
                    if tp + fp > 0 and tp + fn > 0:
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_threshold = test_threshold
            
            threshold = best_threshold
            print(f"Best F1-score: {best_f1:.4f}")
        else:
            # Fallback to percentile 
            threshold = np.percentile(reconstruction_errors, 95) 
    
    else:
        # Force detection of top 10% 
        threshold = np.percentile(reconstruction_errors, 90) 
    
    anomalies = reconstruction_errors > threshold 
    num_anomalies = np.sum(anomalies) 
    anomaly_rate = (num_anomalies / len(reconstruction_errors)) * 100 
    
    print(f"Results: {num_anomalies} anomalies ({anomaly_rate:.1f}%) with threshold {threshold:.6f}") 
    
    return anomalies, threshold

def evaluate_performance(reconstruction_errors, anomalies, labels, threshold):
    """Evaluate detection performance"""
    if labels is None:
        print("No ground truth available for evaluation")
        return
    
    print("\n" + "="*20)
    print("PERFORMANCE EVALUATION")
    print("="*20)
    
    actual_failures = labels.sum(axis=1) > 0
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(actual_failures, anomalies, 
                              target_names=['Normal', 'Anomaly'], digits=4))
    
    # ROC AUC
    try:
        auc_score = roc_auc_score(actual_failures, reconstruction_errors)
        print(f"ROC AUC Score: {auc_score:.4f}")
    except:
        print("Could not calculate ROC AUC")

def create_visualizations(reconstruction_errors, anomalies, labels=None, threshold=None):
    """Create essential visualizations"""
    print("Creating visualizations...")
    
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error distribution
    axes[0, 0].hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue')
    if threshold is not None:
        axes[0, 0].axvline(threshold, color='red', linestyle='--', 
                          label=f'Threshold: {threshold:.4f}')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    axes[0, 1].plot(reconstruction_errors, alpha=0.7, color='blue')
    if threshold is not None:
        axes[0, 1].axhline(threshold, color='red', linestyle='--')
    anomaly_indices = np.where(anomalies)[0]
    if len(anomaly_indices) > 0:
        axes[0, 1].scatter(anomaly_indices, reconstruction_errors[anomaly_indices], 
                          color='red', alpha=0.8, s=20)
    axes[0, 1].set_xlabel('Sequence Index')
    axes[0, 1].set_ylabel('Reconstruction Error')
    axes[0, 1].set_title('Errors Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot
    normal_errors = reconstruction_errors[~anomalies]
    anomaly_errors = reconstruction_errors[anomalies]
    axes[1, 0].boxplot([normal_errors, anomaly_errors], labels=['Normal', 'Anomalies'])
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Normal vs Anomalies')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistics
    stats_text = f"""
    Total Sequences: {len(reconstruction_errors)}
    Anomalies: {np.sum(anomalies)} ({np.sum(anomalies)/len(reconstruction_errors)*100:.1f}%)
    Threshold: {threshold:.6f}
    
    Error Statistics:
    Mean: {np.mean(reconstruction_errors):.6f}
    Std: {np.std(reconstruction_errors):.6f}
    Min: {np.min(reconstruction_errors):.6f}
    Max: {np.max(reconstruction_errors):.6f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_anomaly_detection(method='adaptive'):
    """Main function to run efficient anomaly detection"""
    print("="*20)
    print("EFFICIENT ANOMALY DETECTION")
    print("="*20)
    
    # Load everything
    model, scaler, sequences, labels, metadata = load_models_and_data()
    if model is None:
        return None
    
    # Compute reconstruction errors
    reconstruction_errors = compute_reconstruction_errors(model, sequences, scaler)
    
    # Detect anomalies
    anomalies, threshold = detect_anomalies(reconstruction_errors, labels, method)
    
    # Evaluate performance
    evaluate_performance(reconstruction_errors, anomalies, labels, threshold)
    
    # Create visualizations
    create_visualizations(reconstruction_errors, anomalies, labels, threshold)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETED!")
    print("="*20)
    
    return {
        'reconstruction_errors': reconstruction_errors,
        'anomalies': anomalies,
        'threshold': threshold,
        'anomaly_indices': np.where(anomalies)[0],
        'model': model,
        'scaler': scaler
    }

# Quick analysis function for different methods
def compare_methods():
    """Compare different anomaly detection methods"""
    print("Comparing anomaly detection methods...")
    
    model, scaler, sequences, labels, metadata = load_models_and_data()
    if model is None:
        return
    
    reconstruction_errors = compute_reconstruction_errors(model, sequences, scaler)
    
    methods = ['percentile', 'statistical', 'adaptive']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")
        anomalies, threshold = detect_anomalies(reconstruction_errors, labels, method)
        
        if labels is not None:
            actual_failures = labels.sum(axis=1) > 0
            tp = np.sum(anomalies & actual_failures)
            fp = np.sum(anomalies & ~actual_failures)
            fn = np.sum(~anomalies & actual_failures)
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            results[method] = {
                'anomalies': np.sum(anomalies),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold
            }
            
            print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Show comparison
    if results:
        print("\n" + "="*20)
        print("METHOD COMPARISON")
        print("="*20)
        for method, metrics in results.items():
            print(f"{method.upper():<12}: F1={metrics['f1']:.4f}, "
                  f"Anomalies={metrics['anomalies']}, "
                  f"Threshold={metrics['threshold']:.6f}")

if __name__ == "__main__":
    # Run main analysis
    results = run_anomaly_detection(method='adaptive')
    
    # Optionally compare methods
    # compare_methods()