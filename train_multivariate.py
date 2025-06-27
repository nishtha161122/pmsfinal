import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os

# LSTM Autoencoder model
class LSTMAutoencoder(nn.Module):
    """
    Same architecture used during training
    """
    def __init__(self, input_size, hidden_size=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded_output, (hidden_state, cell_state) = self.encoder(x)

        batch_size = x.size(0)
        seq_len = x.size(1)

        decoder_input = hidden_state[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded_output, _ = self.decoder(decoder_input)

        # Map decoder output to input dimension
        output = self.output_layer(decoded_output)
        return output

def load_my_data():
    """Load the preprocessed data files"""
    print("Loading my preprocessed data...") 
    
    # load numpy arrays
    normal_sequences = np.load('preprocessed_data/normal_sequences.npy')
    all_sequences = np.load('preprocessed_data/all_sequences.npy') 
    all_labels = np.load('preprocessed_data/all_labels.npy')
    
    # load metadata
    with open('preprocessed_data/metadata.pkl', 'rb') as file:
        metadata = pickle.load(file)
    
    print(f"Normal sequences shape: {normal_sequences.shape}")
    print(f"All sequences shape: {all_sequences.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return normal_sequences, all_sequences, all_labels, metadata

def train_my_autoencoder(normal_data, num_features, epochs=20):
    """Train the autoencoder on normal data only"""
    print(f"\nStarting autoencoder training...")
    print(f"Number of features: {num_features}")
    print(f"Training samples: {len(normal_data)}")
    
    # create model 
    model = LSTMAutoencoder(input_size=num_features, hidden_size=64) 
    
    # loss function and optimizer  
    loss_function = nn.MSELoss()  # mean squared error 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # convert numpy to pytorch tensor 
    train_data = torch.tensor(normal_data, dtype=torch.float32)
    
    # create data loader for batches 
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 
    
    # training loop 
    model.train()
    print("Training started...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            input_sequences = batch_data[0]
            
            # forward pass
            reconstructed = model(input_sequences)
            loss = loss_function(reconstructed, input_sequences)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        average_loss = epoch_loss / num_batches 
        
        # print progress every few epochs
    print(f"Epoch [{epoch+1}/{epochs}] completed - Average Loss: {average_loss:.6f}") 
    # save the trained model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    torch.save(model.state_dict(), 'models/my_autoencoder.pth')
    print("Autoencoder training completed and saved!")
    
    return model

def train_failure_predictor(sequences, labels, metadata):
    """Train random forest to predict failure types"""
    print(f"\nTraining failure prediction model...")
    
    # flatten the sequences for sklearn (it needs 2D data)
    X = sequences.reshape(len(sequences), -1)  # flatten each sequence 
    y = labels
    
    print(f"Input shape for classifier: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"anomalies detected:")

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42 
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # create and train random forest
    classifier = RandomForestClassifier(
        n_estimators=100,  # number of trees
        random_state=42,
        n_jobs=-1  # use all CPU cores 
    )
    
    print("Training classifier...")
    classifier.fit(X_train, y_train) 
    
    # test the classifier 
    predictions = classifier.predict(X_test) 
    
    # calculate accuracy for each failure type 
    failure_types = metadata['failure_columns'] 
    print("\nTesting results:")
    
    for i, failure_name in enumerate(failure_types):
        true_labels = y_test[:, i]
        pred_labels = predictions[:, i]
        
        # only calculate if we have some failure cases
        if np.sum(true_labels) > 0:
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            
            print(f"{failure_name}:")
            print(f"  Accuracy: {accuracy:.3f}") 
            print(f"  Precision: {precision:.3f}") 
            print(f"  Recall: {recall:.3f}") 
        else:
            print(f"{failure_name}: No failure samples in test set")
    
    # save the classifier
    with open('models/my_classifier.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    
    print("Failure predictor training completed and saved!") 
    return classifier 

def main():
    print("-" * 20)
    print("MY MACHINE FAILURE DETECTION PROJECT")
    print("-" * 20)
    
    try:
        # load my data
        normal_seq, all_seq, all_labels, metadata = load_my_data()
        
        # train autoencoder for anomaly detection
        num_features = metadata['num_features']
        autoencoder_model = train_my_autoencoder(normal_seq, num_features)
        
        # train classifier for failure type prediction
        failure_classifier = train_failure_predictor(all_seq, all_labels, metadata)
        
        print("\n" + "-" * 20)
        print("SUCCESS! Both models trained and saved")
        print("Files created:")
        print("- models/my_autoencoder.pth")
        print("- models/my_classifier.pkl")
        print("-" * 20)
        
    except FileNotFoundError:
        print("ERROR: Can't find preprocessed data files!")
        print("Please run the preprocessing script first.")
    except Exception as error:
        print(f"Something went wrong: {error}")

# run the training
if __name__ == "__main__":
    main() 