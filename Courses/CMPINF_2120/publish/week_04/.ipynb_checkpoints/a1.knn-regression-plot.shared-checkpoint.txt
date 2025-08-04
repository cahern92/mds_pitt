# University of Pittsburgh - School of Computing and Information 
# Master of Data Science
# Applied Predictive Modeling
# Assignment #1
# KNN-Regression / Plot Data

import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create the DataFrame
data = pd.DataFrame({
    'Square_Feet': [1500, 2100, 1650, 2500, 1850, 1400, 2200, 1700, 1950, 2300],
    'Year_Built': [1998, 2005, 1980, 2010, 1995, 1987, 2015, 2000, 1992, 2008],
    'Price': [350, 480, 320, 610, 400, 290, 650, 370, 410, 590]
})

# Create a scatter plot with Year Built on x-axis, Square Feet on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(data['Year_Built'], data['Square_Feet'], color='blue', s=100)

# Annotate each point with its Price
for i, row in data.iterrows():
    plt.text(row['Year_Built'] + 0.5, row['Square_Feet'] + 10, f"${row['Price']}", fontsize=9)

# Labels and title
plt.xlabel('Year Built')
plt.ylabel('Square Feet')
plt.title('House Prices')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

