import pandas as pd
import random

# Load your dataset
df = pd.read_csv("admission_data_updated.csv")  # Replace with your actual filename

# Create a list of sample university names
universities = [f"University {i}" for i in range(1, 51)]  # You can customize this list

# Assign a random university to each row
df["University Name"] = [random.choice(universities) for _ in range(len(df))]

# Save updated dataset
df.to_csv("admission_data_universities.csv", index=False)
