#Write a program to implement Min-Max algorithm.

def minmax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
        best = float('-inf')
        for i in range(2):
            val = minmax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
        return best
    else:
        best = float('inf')
        for i in range(2):
            val = minmax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
        return best

# Example tree with depth 3 and 8 terminal nodes
values = [3, 5, 2, 9, 12, 5, 23, 23]

# Start the Min-Max algorithm
result = minmax(0, 0, True, values, float('-inf'), float('inf'))
print("The optimal value is:", result)

#3
#Visualize the n-dimensional data using heat-map.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("ToyotaCorolla.csv")

#heat map
sns.heatmap(data[["Price","KM","Doors", "Weight"]].corr(),cmap='jet',annot=True)
plt.show()
