import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random data
num_points = 1000
random_data = {
    'x': np.random.rand(num_points),
    'y': np.random.rand(num_points),
    'z': np.random.rand(num_points),
    'binary': np.random.choice([0, 1], size=num_points)
}

# Create a DataFrame
df = pd.DataFrame(random_data)

# Save the DataFrame to a CSV file
df.to_csv('my_points.csv', index=False)

print("Random points saved to my_points.csv")
