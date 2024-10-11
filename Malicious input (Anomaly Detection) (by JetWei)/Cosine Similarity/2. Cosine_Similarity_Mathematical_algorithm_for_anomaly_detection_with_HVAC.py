# Application of Cosine Similarity in Anomaly Detection as a Mathematical algorithm:
# In the context of anomaly detection, cosine similarity can be used to measure the similarity of 
# a data point to the normal data points. If a data point has a low cosine similarity to the cluster 
# of normal points, it can be considered an anomaly.

# However, in order to detect anomalies, the Cosine Similarity Mathematical algorithm will need to be trained
# with a prior dataset in order to compute a reference vector to compare new vectors to in order to 
# determine if the new vectors are anomalies or not.

# Then a threshold will also be needed to be manually set to set the bar to determine which data points are anomalous 
# or not.


# ///////////////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Random note on the terms 'independent variables/features' and 'dimensions' in Machine Learning (ML):                         /
# These 2 terms are synonymous in the realm of Machine Learning! When a ML model or dataset or problem is said to be of        /
# multi-dimensionality (aka multiple dimensions), what it basically means is that this ML model or dataset or problem considers/
# multiple independent variables/features!                                                                                     /
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# The beauty of mathematics is that even though you cannot visualise high dimensions/independent variables/features, but the math 
# will continue to work


# The 'vector' that will be used in the HVAC HITL RL model, that stores a list of independent variables/features that we will 
# be using in the HVAC HITL aircon simulation (e.g. '[time_of_day, user_type, temperature_preference, metabolic_rate, clothing_insulation]') 

# (flexible to changes, since we can just change the independent variables/features in this 'vector' list)
vector = ['Thermal comfort', 'Age', 'Sex', 'Thermal preference', 'Thermal sensation', 'Air temperature (C)']



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Pretraining to Find the Reference Vector using a prior dataset
training_dataset = pd.read_csv('thermal_comfort_dataset.csv')

reference_vector = training_dataset.mean().values.reshape(1, -1)

print("Reference Vector:", reference_vector)


# Cosine Similarity Mathematical algorithm for Anomaly Detection on a new dataset
test_dataset = pd.read_csv('thermal_comfort_dataset.csv')

similarity_scores = cosine_similarity(test_dataset.values, reference_vector)

test_dataset['similarity_score'] = similarity_scores

threshold = 0.90

test_dataset['anomaly'] = test_dataset['similarity_score'] < threshold


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


# Filter out any rows with a value of 'False' in the 'anomaly' column (since those are anomalies)
test_dataset_filtered = test_dataset[test_dataset['anomaly'] != -1]

print(f'Initial datatset:\n{test_dataset}')
print(f'Filtered datatset:\n{test_dataset_filtered}')
