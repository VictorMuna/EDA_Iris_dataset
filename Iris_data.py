# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:24:41 2021

@author: MOGE
"""

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Reading data and Loading data Frame
data = pd.read_csv("Iris.csv")

# Data Exploration
data.head()
data.info()
data.describe()

data['Species'].value_counts()


# Droping 'Id'
data_1 = data.drop(['Id'], axis=1)

# Verifying axis drop
data_1.head()


# Data Visualization
plot = sns.pairplot(data_1, hue='Species', markers='*')
plt.show()


# Checking feature Correlations with Pearson Correlation matrix

num_data = data_1.select_dtypes(exclude="object")
corr_numeric = num_data.corr()
sns.heatmap(corr_numeric, annot=True, cmap="BrBG")
plt.title("Correlation Matrix")
plt.show()

# Scatter plot of most correlated features
scatter = data_1.plot(kind='scatter',x = 'PetalLengthCm',y = 'PetalWidthCm')
scatter

# Visualising in sweetviz

# importing sweetviz
import sweetviz as sv

# analyzing the dataset
report = sv.analyze(data_1)

# display the report
report.show_html('Iris.html')


