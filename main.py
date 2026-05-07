import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SequentialFeatureSelector

RANDOM_STATE = 17342
np.random.seed(RANDOM_STATE)


column_names = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothead",
    "backbone", 
    "breathes",
    "venomous", 
    "fins", 
    "legs", 
    "tail", 
    "domestic",
    "catsize",
    "type"
]


# 1) Load the dataset (e.g., UCI Wine)
df = pd.read_csv('zoo.data', header = None, names = column_names)
print("Raw shape:", df.shape)
df.head()

# 2) Drop rows containing any missing values
df = df.dropna()
print("After dropna:", df.shape)

# 3) Drop irrelevant columns (e.g., ID)
df = df.drop(columns=['animal_name'], errors='ignore')
df.head()

# 4) Separate target and features
y = df['type']
X = df.drop(columns=['type'])
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Classes:", sorted(y.unique()))

# 5) Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])
X = X.astype(int)
y = y.astype(int)

# 6) Encode the target if it is a string
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled feature shape:", X_scaled.shape)

# 7) Scale numerical features (important for KNN and MLP)
X_scaled = StandardScaler().fit_transform(X)
print('Shape:', X_scaled.shape, 'Classes:', set(y))