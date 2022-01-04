# Exercise 1

test = "Hello world"
print(test)
print("test: " + test)

"""
1. Building basic functions with numpy
    Numpy is the main package for scientific computing in Python. It is maintained by a large community.
    In this exercise you will learn several key numpy functions such as np.exp, np.log and np.reshape.
    You will need to know how to use these functions for future assignments.

    1.1 Sigmoid function, np.exp()
        Before using np.exp(), you will use math.exp() to implement the sigmoid function.
        You will then see why np.exp() is preferable to math.exp().

        Exercise 2 - Basic sigmoid
        Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the
        exponential function.
"""

import math
from public_tests import *

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1 + math.exp(-x))
    return s

print("Basic_sigmoid(1) = " + str(basic_sigmoid(1)))

basic_sigmoid_test(basic_sigmoid)

print("\n-----------------------------------------\n")


### One reason why we use "numpy" instead of "math" in Deep Learning ###

#x = [1, 2, 3] # x becomes a python list object
#basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.

"""
In fact, if  𝑥=(𝑥1,𝑥2,...,𝑥𝑛)  is a row vector then np.exp(x) will apply the exponential function 
to every element of x. The output will thus be: np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})

"""

import numpy as np

# example of np.exp
t_x = np.array([1, 2, 3])

print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

# Futhermore, if x is a vector, then a Python operation such as s = x + 3 or s = 1/x
# will output s as a vector of the same size as x

# example of vector operation
t_x = np.array([1, 2, 3])
print(t_x + 3)  # [4, 5, 6]

"""
Exercise 3 - sigmoid
    Implement the sigmoid function using numpy

    Instruction: x could now be either a real number, a vector, or a matrix. The data structures 
    we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays. You 
    don't need to know more for now.
"""
def sigmoid(x):
    """
    Compute the sigmoid of x
    
    Arguments:
    x -- A scalar or numpy array of any size
    
    Return:
    s -- sigmoid(x)
    """

    s = 1/(1 + np.exp(-x))
    return s

t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

sigmoid_test(sigmoid) # sigmoid(t_x) = [0.73105858 0.88079708 0.95257413]

print("\n-----------------------------------------\n")

"""
1.2 Sigmoid Gradient
    As you;ve seen in lecture, you wwill need to compute gradients to optimize loss functions
    using backpropagation. Let's code your first gradient function.

    Exercise 4 - sigmoid_derivative
        Implement the function sigmoid_grad() to compute the gradient of the sigmoid function
        with respect to its input x. The formula is:

            𝑠𝑖𝑔𝑚𝑜𝑖𝑑_𝑑𝑒𝑟𝑖𝑣𝑎𝑡𝑖𝑣𝑒(𝑥)=𝜎′(𝑥)=𝜎(𝑥)(1−𝜎(𝑥))(2)
        
        You often code this function in two steps:
            1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
            2. Compute 𝜎′(𝑥)=𝑠(1−𝑠)
"""

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    s = 1/(1 + np.exp(-x))

    ds = s*(1 - s)

    return ds

t_x = np.array([1, 2, 3])
print("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))

sigmoid_derivative_test(sigmoid_derivative) # sigmoid_derivative(t_x) = [0.19661193 0.10499359 0.04517666]

print("\n-----------------------------------------\n")


"""
1.3 - Reshaping arrays

    Two common numpy functions used in deep learning are  np.shape and np.reshape().
        - X.shape is used to get the shape (dimension) of a matrix/vector X.
        - X.reshape(...) is used to reshape X into some other dimension.
    For example, in computer science, an image is represented by a 3D array of shape  (𝑙𝑒𝑛𝑔𝑡ℎ,ℎ𝑒𝑖𝑔ℎ𝑡,𝑑𝑒𝑝𝑡ℎ=3) . 
    However, when you read an image as the input of an algorithm you convert it to a vector of shape  (𝑙𝑒𝑛𝑔𝑡ℎ∗ℎ𝑒𝑖𝑔ℎ𝑡∗3,1) . 
    In other words, you "unroll", or reshape, the 3D array into a 1D vector.

    Exercise 5 - Image2vector

        Implement image2vector() that takes an imput of shape (length, height, 3) and returns a vector of shape (length*height*3, 1).
        For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b, c) you would do:

            v = v.reshape((v.shape[0] * v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c

        - Please don't hardcode the dimensions of image as a constant. Instead look up the quantities you need with image.shape[0], etc.
        - You can use v = v.reshape(-1, 1). Just make sure you understand why it works.
"""

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

t_image = np.array([[[ 0.67826139,  0.29380381],
                     [ 0.90714982,  0.52835647],
                     [ 0.4215251 ,  0.45017551]],

                   [[ 0.92814219,  0.96677647],
                    [ 0.85304703,  0.52351845],
                    [ 0.19981397,  0.27417313]],

                   [[ 0.60659855,  0.00533165],
                    [ 0.10820313,  0.49978937],
                    [ 0.34144279,  0.94630077]]])

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values

print ("image2vector(image) = " + str(image2vector(t_image)))

image2vector_test(image2vector)

print("\n-----------------------------------------\n")

