# What is Cosine Similarity?
# Cosine Similarity is a mathematical measure of similarity between two non-zero vectors. It uses the size of the angle between 
# them to determine how 'similar' these two vectors are. To make a better measurement of how 'similar' two vectors are, the cosine
# mathematical operator is used to convert these angles to a value between -1 to 1, where:
#       - 1 indicates that the two vectors are identical (similar in concept)
#       - 0 indicates that the vectors are orthogonal (no similarity)
#       - -1 indicates that the vectors are diametrically opposed (opposite in concept)

# Mathematical formula for cosine similarity:
#                           Cosine Similarity =  (A⋅B)/(∥A∥∥B∥)

# Source(s): https://www.youtube.com/watch?v=m_CooIRM3UI (codebasics)


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Showing that cosine similarity and cosine distances are opposites
print(cosine_similarity([[3,1]], [[6,2]]))          # Output: 1
print(cosine_distances([[3,1]], [[6,2]]))           # Output: 1.11022302e-16 (basically 0)

# Showing that these 2 vectors have a 'similarity'/closeness of 0.96476382
print(cosine_similarity([[3,1]], [[3,2]]))          # Output: 0.96476382


# ////////////////////////////////////////////////////////////////////////////////////////////////////


# Using cosine similarity/cosine distance in a context
# Context: You have some texts, and you want to figure out which mobile phone company each of them are
# focusing on, as well as checking how similar they are with each other based on the frequency of the 
# names of a mobile phone company appearing in each text

# Some data that we will be using in the dataset
doc1 = """
iphone sales contributed to 70% of revenue. iphone demand is increasing by 20% yoy. 
the main competitor phone galaxy recorded 5% less growth compared to iphone"
"""

doc2 = """
The upside pressure on volumes for the iPhone 12 series, historical outperformance 
in the July-September time period heading into launch event, and further catalysts in relation
to outperformance for iPhone 13 volumes relative to lowered investor expectations implies a 
very attractive set up for the shares.
"""

doc3 = """
samsung's flagship product galaxy is able to penetrate more into asian markets compared to
iphone. galaxy is redesigned with new look that appeals young demographics. 60% of samsung revenues
are coming from galaxy phone sales
"""

doc4 = """
Samsung Electronics unveils its Galaxy S21 flagship, with modest spec improvements 
and a significantly lower price point. Galaxy S21 price is lower by ~20% (much like the iPhone 12A), 
which highlights Samsung's focus on boosting shipments and regaining market share.
"""

# Dataset showing the frequency of a mobile phone company appearing in each text
dataset = pd.DataFrame([
        {'iPhone': 3,'galaxy': 1},
        {'iPhone': 2,'galaxy': 0},
        {'iPhone': 1,'galaxy': 3},
        {'iPhone': 1,'galaxy': 2},
    ],
    index=[
        "doc1",
        "doc2",
        "doc3",
        "doc4"
    ])

print(dataset)
print(dataset.loc["doc1":"doc1"])


# Using cosine similarity to show the 'similarity'/closeness of text 1 and text 2 based on the frequency 
# of the names of a mobile phone company appearing in each text
print(cosine_similarity(dataset.loc["doc1":"doc1"],dataset.loc["doc2":"doc2"]))          # Output: 0.9486833

# Using cosine similarity to show the 'similarity'/closeness of text 1 and text 3 based on the frequency 
# of the names of a mobile phone company appearing in each text
print(cosine_similarity(dataset.loc["doc1":"doc1"],dataset.loc["doc3":"doc3"]))          # Output: 0.6

# Testing for the cosine similarity of other comparisons:
print(cosine_similarity(dataset.loc["doc3":"doc3"],dataset.loc["doc4":"doc4"]))          # Output: 0.98994949
print(cosine_similarity(dataset.loc["doc1":"doc1"],dataset.loc["doc4":"doc4"]))          # Output: 0.70710678


# Using cosine distance to show the 'similarity'/closeness of text 1 and text 2 based on the frequency 
# of the names of a mobile phone company appearing in each text
print(cosine_distances(dataset.loc["doc1":"doc1"],dataset.loc["doc4":"doc4"]))           # Output: 0.29289322
