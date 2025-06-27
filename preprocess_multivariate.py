import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SENSOR_COLS = ['air_temperature', 'process_temperature', 'rotational_speed', 'Torque', 'tool_wear']
FAILURE_COLS = ['machine_failure', 'tool_wear_failure', 'heat_dissipation_failure', 'PWF_power_failure', 'OSF_overstrain_failure', 'RNF_random_failure']

def load_data(path='datetime.csv'):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}") #Display dataset dimensions (rows, columns)
    print("\nFailure distribution:") #Print header for failure analysis
    for col in FAILURE_COLS:
        count = df[col].sum() #Count total failures (sum of 1s in binary column)
        rate = df[col].mean() * 100 #Calculate failure percentage
        print(f"  {col}: {count} failures ({rate:.2f}%)") #Print failure statistics for each type

        print("-"*20) #(just a code distinguisher)

    return df

def scale_data(df): 
    print("\nScaling sensor data...") #status message
    scaler = MinMaxScaler() #Create MinMaxScaler instance (scales to 0-1 range)
    scaled = scaler.fit_transform(df[SENSOR_COLS]) #Fit scaler to sensor data and transform it
    scaled_df = pd.DataFrame(scaled, columns=SENSOR_COLS) #Create new DataFrame with scaled sensor values

    if 'date' in df.columns: #check if date column exists
        scaled_df.insert(0, 'date', df['date']) #put it in the first column

    scaled_df[FAILURE_COLS] = df[FAILURE_COLS] #Sum failures across columns, keep rows with 0 failures
    normal_df = scaled_df[df[FAILURE_COLS].sum(axis=1) == 0].reset_index(drop=True) #reset row indices

    total = len(scaled_df) #count total 
    normal = len(normal_df) #count normal data points 
    print(f"Total data points: {total}") 
    print(f"Normal operations: {normal} ({normal/total:.2%})") #calc percentages (upto 2 deci)
    print(f"Failures: {total - normal} ({(total - normal)/total:.2%})")

    return scaled_df, normal_df, scaler 

def create_sequences(df, time_steps=200): #functions with deafult windo size of 200 timesteps
    X, y = [], [] #Initialize empty lists for features (X) and labels (y)
    for i in range(len(df) - time_steps + 1):  #Ensures we don't go beyond data end
        X.append(df[SENSOR_COLS].iloc[i:i+time_steps].values) #Get 200 consecutive sensor reading
        y.append(df[FAILURE_COLS].iloc[i+time_steps-1].values) #convert to numpy array
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 

def save_data(normal_seq, all_seq, all_labels, scaler, output_dir='preprocessed_data'): 
    #Create output directory (default: 'preprocessed_data') 
    os.makedirs(output_dir, exist_ok=True) 
    np.save(f'{output_dir}/normal_sequences.npy', normal_seq) #save numpy arrays for normal data
    np.save(f'{output_dir}/all_sequences.npy', all_seq) #for all sequence 
    np.save(f'{output_dir}/all_labels.npy', all_labels) #for all labels 
    with open(f'{output_dir}/scaler.pkl', 'wb') as f: #save fitted scaled useing pickle 
        pickle.dump(scaler, f)

    metadata = { #create metadata dictionary 
        'sensor_columns': SENSOR_COLS, #column names for sensor and failures 
        'failure_columns': FAILURE_COLS,
        'sequence_length': normal_seq.shape[1], #sequence dimension for model configuration 
        'num_features': normal_seq.shape[2] 
    } 
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f) 

    print("\nSaved preprocessed data:") 
    print(f"  - normal_sequences.npy {normal_seq.shape}") 
    print(f"  - all_sequences.npy    {all_seq.shape}") 
    print(f"  - all_labels.npy       {all_labels.shape}") 
    print("  - scaler.pkl")
    print("  - metadata.pkl")

def main():
    print("-"*20)
    print("MULTIVARIATE ANOMALY DETECTION - DATA PREPROCESSING") #print header 
    print("-"*20)

    try: #try block for error handling 
        df = load_data() 
        scaled_df, normal_df, scaler = scale_data(df) 

        normal_seq, _ = create_sequences(normal_df) 
        all_seq, all_labels = create_sequences(scaled_df) 

        save_data(normal_seq, all_seq, all_labels, scaler) 

        print("\nPREPROCESSING COMPLETED SUCCESSFULLY!") 
        print("-"*20) 

    except FileNotFoundError: 
        print("Error: 'datetime.csv' not found.") 
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 