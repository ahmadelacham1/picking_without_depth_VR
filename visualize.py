import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = "my_points.csv"
df = pd.read_csv(file_path)

# Calculate speeds and accelerations using finite differences
df['speed_x'] = np.gradient(df['x'])
df['speed_y'] = np.gradient(df['y'])
df['speed_z'] = np.gradient(df['z'])

df['acceleration_x'] = np.gradient(df['speed_x'])
df['acceleration_y'] = np.gradient(df['speed_y'])
df['acceleration_z'] = np.gradient(df['speed_z'])

# Create separate DataFrames for picking and non-picking data
picked_df = df[df['binary'] == 1]
not_picked_df = df[df['binary'] == 0]

# Plot speeds with color based on picking status
plt.figure(figsize=(15, 10))

# Speed X
plt.subplot(2, 3, 1)
plt.scatter(df.index, df['speed_x'], c=df['binary'], cmap='bwr', label='Speed X')
plt.xlabel('Samples')
plt.ylabel('Speed X')

# Speed Y
plt.subplot(2, 3, 2)
plt.scatter(df.index, df['speed_y'], c=df['binary'], cmap='bwr', label='Speed Y')
plt.xlabel('Samples')
plt.ylabel('Speed Y')

# Speed Z
plt.subplot(2, 3, 3)
plt.scatter(df.index, df['speed_z'], c=df['binary'], cmap='bwr', label='Speed Z')
plt.xlabel('Samples')
plt.ylabel('Speed Z')


# Plot accelerations with color based on picking status
# Acceleration X
plt.subplot(2, 3, 4)
plt.scatter(df.index, df['acceleration_x'], c=df['binary'], cmap='bwr', label='Acceleration X')
plt.xlabel('Samples')
plt.ylabel('Acceleration X')

# Acceleration Y
plt.subplot(2, 3, 5)
plt.scatter(df.index, df['acceleration_y'], c=df['binary'], cmap='bwr', label='Acceleration Y')
plt.xlabel('Samples')
plt.ylabel('Acceleration Y')

# Acceleration Z
plt.subplot(2, 3, 6)
plt.scatter(df.index, df['acceleration_z'], c=df['binary'], cmap='bwr', label='Acceleration Z')
plt.xlabel('Samples')
plt.ylabel('Acceleration Z')

plt.tight_layout()
plt.show()
