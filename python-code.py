import json
import matplotlib.pyplot as plt
import numpy as np

idx = np.linspace(0,6.28,num=100)
x = np.sin(idx) + np.random.uniform(size=100)/10.0 # query
y = np.cos(idx) # reference
# Read the JSON file
with open('alignment_path.json', 'r') as file:
    json_data = file.read()

# Parse the JSON data
alignment_path_data = json.loads(json_data)

# Convert the data to a list of tuples
alignment_path = [(item[0], item[1]) for item in alignment_path_data]

# Print the result
print(alignment_path)


plt.figure()
for x_i, y_j in alignment_path:
    plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
plt.axis("off")

plt.show()