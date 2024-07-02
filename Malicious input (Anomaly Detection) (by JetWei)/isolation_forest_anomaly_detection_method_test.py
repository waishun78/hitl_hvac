import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Sample dataset
np.random.seed(42)                                                 # The .seed() function is used to initialize the random number generator in NumPy.
normal_temperatures = np.random.normal(loc=22, scale=2, size=100)  # Normal temperatures
anomalous_temperatures = np.array([10, 35, 40])                    # Anomalies temperatures
temperatures = np.concatenate([normal_temperatures, anomalous_temperatures]).reshape(-1, 1)

# Fit the model
clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(temperatures)

# Predict anomalies
scores = clf.decision_function(temperatures)
anomalies = clf.predict(temperatures)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(range(len(temperatures)), temperatures, c=anomalies, cmap='coolwarm')
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Sample Index")
plt.ylabel("Temperature (°C)")

# More positive scores indicate a higher chance the data point is a normal data points, while more negative 
# score indicate the data point differ from the majority, signalling a higher chance it is an anomaly
print("Anomaly scores:\n", scores)
# Predictions are -1 for anomalies (outliers) and 1 for normal data points.
print("Predictions (1: normal, -1: anomaly):\n", anomalies)

plt.savefig('visualisation_of_the_isolation_forest_anomaly_detection_method.png')
plt.show()