from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
models = {
'Linear Classifier' : Perceptron(max_iter=1000, random_state=42),
'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
'KNN' : KNeighborsClassifier(n_neighbors=5),
'Gaussian NB' : GaussianNB(),
'Neural Network' : MLPClassifier(hidden_layer_sizes=(64,),
max_iter=500, random_state=42),
}
