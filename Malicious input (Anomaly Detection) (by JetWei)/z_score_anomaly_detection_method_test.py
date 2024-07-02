import numpy as np

# Pre-training the Z-Score anomaly detection method by the finding the pre-trained mean and standard deviation
aircon_data = [22, 24, 23, 25, 28, 29, 26, 21, 25, 24, 26]

# Calculate pre-trained mean and standard deviation
mean = np.mean(aircon_data)
std_dev = np.std(aircon_data)
print(mean)
print(std_dev)

# Define threshold (selecting the threshold is crucial, and varies between contexts! Requires trial and error. In this case,
# the threshold of 1 seems good)
threshold = 1


# New human user input of an aircon temperature
new_user_input = 37
# Calculate Z-score of new human user input
calculated_z_score = (new_user_input - mean)/std_dev
print(calculated_z_score)


# Testing if the new human user input of an aircon temperature is an anomaly or not
if abs(calculated_z_score) > threshold:
    print(f"The new user input, of aircon temperature {new_user_input}, is an anomaly")
else:
    print(f"The new user input, of aircon temperature {new_user_input}, is not an anomaly")

# Output:
# The new user input, of aircon temperature 37, is an anomaly