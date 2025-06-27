import pandas as pd
from datetime import datetime, timedelta

# Load your data
df = pd.read_csv("Dataset_Machine.csv")

# Compute start and end dates
end_date = datetime.today().date()
start_date = end_date - timedelta(days=len(df) - 1)

# Create the date range
dates = pd.date_range(start=start_date, end=end_date)

# Assign and format as DD/MM/YYYY
df['date'] = dates.strftime("%d/%m/%Y")

# Save to a new file with Windows path
df.to_csv("datetime.csv", index=False) 