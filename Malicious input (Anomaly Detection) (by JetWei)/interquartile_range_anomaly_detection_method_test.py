import numpy as np
import matplotlib.pyplot as plt

# Sample dataset representing aircon temperatures
np.random.seed(42)                                                 # The .seed() function is used to initialize the random number generator in NumPy.  
normal_temperatures = np.random.normal(loc=22, scale=2, size=100)  # Normal temperatures
anomalous_temperatures = np.array([10, 35, 40])                    # Anomalies temperatures
temperatures = np.concatenate([normal_temperatures, anomalous_temperatures])
print(temperatures)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = np.percentile(temperatures, 25)
Q3 = np.percentile(temperatures, 75)
IQR = Q3 - Q1

# Define the bounds for anomalies
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify anomalies
anomalies = temperatures[(temperatures < lower_bound) | (temperatures > upper_bound)]
normal_values = temperatures[(temperatures >= lower_bound) & (temperatures <= upper_bound)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(normal_values, 'bo', label='Normal')
plt.plot(np.where((temperatures < lower_bound) | (temperatures > upper_bound))[0], anomalies, 'ro', label='Anomaly')
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
plt.title("Anomaly Detection in Air Conditioning Temperatures")
plt.xlabel("Sample Index")
plt.ylabel("Temperature (°C)")
plt.legend()

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Anomalies detected:", anomalies)

plt.savefig('visualisation_of_the_interquartile_range_anomaly_detection_method.png', dpi=100)
plt.show()