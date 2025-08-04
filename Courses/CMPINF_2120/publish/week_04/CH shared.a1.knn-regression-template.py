# University of Pittsburgh - School of Computing and Information 
# Master of Data Science
# Applied Predictive Modeling
# Assignment #1
# KNN-Regression / Code TEMPLATE
# Cesar Hernandez

# Use the following code to compute the answer to the KNN-Regression questions
# You should only modify the code within START/END comments

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

###########################################################################
# Step 1: Create the DataFrame
data = pd.DataFrame({
    'square_feet': [1500, 2100, 1650, 2500, 1850, 1400, 2200, 1700, 1950, 2300],
    'year_built': [1998, 2005, 1980, 2010, 1995, 1987, 2015, 2000, 1992, 2008],
    'price': [350, 480, 320, 610, 400, 290, 650, 370, 410, 590]
})

###########################################################################
# Step 2: Standardize the features
features = ['square_feet', 'year_built']
data_x = data[features]

stdscale = StandardScaler()

### PUT YOUR CODE HERE TO PERFORM STANDARDIZATION, as in Module 2 / START ###
data_x_std = pd.DataFrame(stdscale.fit_transform(data_x), columns=[col+"_std" for col in features], index=data.index)
## Adds the two dataframes together by index
data_std_df = pd.concat([data, data_x_std], axis=1)
### PUT YOUR CODE HERE TO PERFORM STANDARDIZATION, as in Module 2 / END #####

###########################################################################
# Step 3: Standardize the query point
query_point = pd.DataFrame([[1800, 1990]], columns=features)

### PUT YOUR CODE HERE TO PERFORM STANDARDIZATION for the query point / START ###
query_std = stdscale.transform(query_point)
# query_std
### PUT YOUR CODE HERE TO PERFORM STANDARDIZATION for the query point / END #####

###########################################################################
# Step 4: Compute Euclidean distances

### PUT YOUR CODE HERE TO COMPUTE ALL THE EUCLIDEAN DISTANCES / START ###
# We take the difference between the query_std with all the other points
# Then, we calculate the norm of the difference vectors for each row
distances = np.linalg.norm(data_std_df[[col+"_std" for col in features]].values - query_std, axis=1)
### PUT YOUR CODE HERE TO COMPUTE ALL THE EUCLIDEAN DISTANCES / END #####

###########################################################################
# Step 5: Get 3 nearest neighbors

### PUT YOUR CODE HERE TO IDENTIFY AND PRINT THE CLOSEST 3 NEIGHBORS / START ###
# This organizes the distances from lowest to highest
## Then, we keep only the first three
## Then, we find those three prices in data_std_df using the indices found in closet_indices
closest_indices = distances.argsort()[:3]
distances[closest_indices]
closest_prices = data_std_df.loc[closest_indices, "price"]
print(closest_prices)
### PUT YOUR CODE HERE TO IDENTIFY AND PRINT THE CLOSEST 3 NEIGHBORS / END #####

###########################################################################
# Step 6: Predict price

### PUT YOUR CODE HERE TO COMPUTE AND PRINT THE PREDICTED PRICE / START ###
# We average the three closest prices
avg = closest_prices.mean()
print(avg)
### PUT YOUR CODE HERE TO COMPUTE AND PRINT THE PREDICTED PRICE / END #####
