#Neural Netowork MLP classification of the supervised learning
#date 2-1-2018
#simple neural network

from sklearn.neural_network import MLPClassifier

X = [[0, 0], [1., 1.]]
y = [0, 1]

clf = MLPClassifier(activation='relu', batch_size='auto', learning_rate_init=0.001,  solver='lbfgs', hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

predict = clf.predict([[2., 2.], [-1., -2.]])

print("---Display Prediction Result---")
print("\n Prediction:", predict)
        
print("\n---Display Accuracy Rate---")
print("\naccuracy:", clf.score(X, y))

