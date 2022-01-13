import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Ir_utils import load_dataset
from public_tests import *

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# reshape training set and test set data 
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten/ 255


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zero(dim):
    """
    Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias) of type float
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

# Implement the cost function and its gradient for the propagation explained above
def propagate(w, b, X, Y):
    # Implement the cost function and its gradient for the propagation explained above
    
    a = sigmoid(np.dot(w.T, X) + b)
    m = X.shape[1]
    cost = -1/m * np.sum(Y * np.log(a) + (1 - Y)*(np.log(1 - a)))

    dw = 1/m * np.dot(X, np.transpose(a - Y))
    db = 1/m * np.sum(a - Y)

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}

    return grads, cost

# Gradient descent
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    # This function optimizes w and b by running a gradient descent algorithm

        w = copy.deepcopy(w)
        b = copy.deepcopy(b)

        costs = []

        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db
        
            # record the costs
            if i % 100 == 0:
                costs.append(cost)

                if print_cost:
                    print("cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                  "b": b}
        
        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

def predict (W, b, X):
    # Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(W.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction = 1
        else:
            Y_prediction = 0
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    # Initialize parameters with zeros
    w, b = initialize_with_zero(X_train.shape[0])

    # Gradient Descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]

    # predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train -Y_train))))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
