import pandas as pd
import paho.mqtt.client as mqtt
import json
import time

# Read Excel file
df = pd.read_csv('datetime.csv') 

# MQTT broker details
broker = "localhost"
port = 1883
topic = "machine_failure=0"

# Initialize MQTT client and connect
client = mqtt.Client()
client.connect(broker, port, 60)

# Loop through rows and publish as JSON 
for _, row in df.iterrows():
    data = row.to_dict()  # Convert to dict 
    payload = json.dumps(data)  # Convert to JSON string
    client.publish(topic, payload)
    print(f"Published: {payload}")
    time.sleep(1)  # Simulate time delay (1 second per row)

client.disconnect()