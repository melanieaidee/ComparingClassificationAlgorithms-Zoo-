# CSCI 3329 — Homework 3 Report
## 1. Dataset
Name: Zoo
Source: ICI Machine Learning Repository
Number of Samples: 101
Number of Features: 16
Number of Classes: 7
Class Distribution: Table

Class             Count
1                  41
2                  20
3                  5
4                  13
5                  4
6                  8
7                  10


## 2. Preprocessing
No missing values
All features were in 0/1 integer counts
No caterogical encoding was required
The target variable (type) was cast to integer


## 3. Part 2 — Algorithm Comparison
| Algorithm |            Mean Accuracy       | Std |
|Linear Classifier|        |0.9576|          |.0613|
|Logistic Regression|      |0.9762|          |.0498|
|KNN|                      |0.9377|          |.0737|
|Gaussian NB|              |0.9900|          |.0300|
|Neural Network|           |0.9801|          |.0434|


## 4. Part 3 — Feature Selection
- Search method and justification
| Algorithm |                           Best Feature Subset                          | Mean Accuracy |         | Std |
|Linear Classifier|      |feathers, milk, airborne, aquatic, backbone,...|               |0.957591|           |.061269|
|Logistic Regression|    |hair, feathers, eggs, milk, aquatic, backbone,...|             |0.976182|           |.049806|
|KNN|                    |feathers, milk, airborne, backbone, breathes, ...|             |0.937682|           |.073717|
|Gaussian NB|            |hair, feathers, eggs, milk, aquatic, backbone,...|               |0.99|               |.03|
|Neural Network|         |hair, feathers, eggs, milk, aquatic, backbone,...|             |.980091|            |.043431|


## 5. Discussion
Part 2 and part 3 seem to share the same results only that part 3 is more specific with the decimals. 
Per-algorithm observations: 
  Linear Classifier: Selected features related to locomotion, respiration, and habitat. They separated mammals, birds, and fish.
  Logistic Regression: Selected a balanced mix of mammal, bird, and aquatic indicators, giving it the highest linear-model accuracy.
  KNN: Included domestic, which helps cluster animals with similar human-intercation traits.
  Gaussian NB: Got the highest overall accuracy with .099, showing that the dataset aligns well with NB's independence assumptions after feature selection.
  Neural Network: Selected almost the same subsets as Logistic Regression and Gaussian NB, confirming that these features capture the core biological structure of the dataset.
Limitations and ideas for improvement:
  This dataset only had 101 instances which limited the generalization.
  Some classes had few samples
  The MLP required many iterations and produced concergence warnings. Increasing max_iter might improve the stability.

  
## 6. Reproduction
- Python version: Python 3.14
- key library versions: numpy: 2.4.2
                        pandas: 3.0.2
                        sklearn: 1.8
                        matplotlib: 3.10..9
- Run command: python preprocessing.py
