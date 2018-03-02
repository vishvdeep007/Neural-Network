#Datasets: iris
#classifier: MLPClassifier
#develop by: vD
#date: 2-1-2018

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np

iris = load_iris()

X, y = iris.data, iris.target

X_trian, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)


print("---Display Shape of the Dataset---")
print("\niris.data:", X.shape)
print("\niris.target:", y.shape)


print("\n---Display data training and test---")
print("\n---Training Data---")
print("\nX_trian:", X_trian)
print("\ny_trian:", y_train)
print("\n---Test Data---")
print("\nX_test:", X_test)
print("\ny_test:", y_test)


clf = MLPClassifier(activation='relu', batch_size='auto', learning_rate_init=0.001, solver='lbfgs', hidden_layer_sizes=(15, 25), random_state=1)

clf.fit(X_trian, y_train)

i=0
for i in range[1, 20]:
    predict[i] = clf.predict(X_test, y_test)

j=0
print("\n--Display predict result---")
for j in range(len(predict)):
    print("result {}",predict[j]).format(j)


