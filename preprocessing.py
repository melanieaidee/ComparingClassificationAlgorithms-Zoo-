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
from copy import deepcopy

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
    "toothed",
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
print(df.head())

# 2) Drop rows containing any missing values
df = df.dropna()
print("After dropna:", df.shape)

# 3) Drop irrelevant columns (e.g., ID)
df = df.drop(columns=['animal_name'], errors='ignore')
print(df.head())

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

#Part 2
models = {
    'Linear Classifier' : Perceptron(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN' : KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB' : GaussianNB(),
    'Neural Network' : MLPClassifier(hidden_layer_sizes=(64,),
    max_iter=500, random_state=42),
}


rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)
results_part2 = {}

for name, model in models.items():
    scores = cross_val_score(
        model, 
        X_scaled, 
        y, 
        cv=rkf,
        scoring='accuracy', 
        n_jobs=-1)

    mean_acc = scores.mean()
    std_acc = scores.std()
    results_part2[name] = {"mean": mean_acc, "std": std_acc}
    print(f'{name:20s} mean={mean_acc:.4f} std={std_acc:.4f}')

part2_df = pd.DataFrame.from_dict(results_part2, orient = "index")
part2_df = part2_df.rename(columns={"mean": "Mean Accuracy", "std": "Std"})
print(part2_df)


#Part 3
feature_names = X.columns.tolist()
print(feature_names)

def feature_selection_and_eval(base_model, X_scaled, y, feature_names,
                               direction = "forward",
                               n_splits = 10,
                               n_repeats = 20,
                               random_state = RANDOM_STATE):
    rfk_fs = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = random_state)
    sfs = SequentialFeatureSelector(
        estimator = deepcopy(base_model),
        n_features_to_select="auto",
        direction = direction,
        scoring = "accuracy",
        cv = rfk_fs,
        n_jobs = -1
    )

    sfs.fit(X_scaled, y)
    support_mask = sfs.get_support()
    selected_features = np.array(feature_names)[support_mask]

    X_selected = X_scaled[:, support_mask]

    scores = cross_val_score(
        deepcopy(base_model),
        X_selected,
        y,
        cv = rfk_fs,
        scoring = "accuracy",
        n_jobs = -1
    )

    return list(selected_features), scores.mean(), scores.std()

results_part3 = {}

for name, model, in models.items():
    print(f"Running feature selection for: {name}")
    best_subset, mean_acc,std_acc = feature_selection_and_eval(
        base_model = model,
        X_scaled=X_scaled,
        y = y,
        feature_names=feature_names,
        direction="forward"
    )
    results_part3[name] = {
        "Best Feature Subset": best_subset,
        "Mean Accuracy": mean_acc,
        "Std": std_acc
    }
    print(f" Selected feayures ({len(best_subset)}): {best_subset}")
    print(f" mean = {mean_acc:.4f} std = {std_acc:.4f}\n")

part3_df = pd.DataFrame(results_part3).T
print(part3_df)

