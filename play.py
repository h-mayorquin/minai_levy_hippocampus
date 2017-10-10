import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Structure paramters
N = 200
n = 200
v = 10.0   # Input - C3 connection
b = 10.0   # Input - C1 connection
Kr = 1.0   # Recurrent self-inhibition gain
Ki = 1.0   # Input - C3 inhibition
p = 0.4

# Dynamical parameters
theta = 0.0
phi = 0


# Learning rate
epsilon = 0.1

# Dynamical quantities
m = 0.0  # Number of neurons active on C3
s = 0.0  # Number of active inputs

# First we need to create the input
input = np.zeros(N)

# Let's build the matrix of connections c_ij
c1 = np.random.choice(2, size=(n, n), replace=True, p=[1 - p, p])
c2 = np.random.choice(2, size=(n, n), replace=True, p=[1 - p, p])

# Let's build the initial patterns
number_of_patterns = 20
sparsity = 10

patterns_dictionary = {}
for pattern_number in range(number_of_patterns):
    # Initialize the pattern with zero
    pattern = np.zeros(N)
    # Chose some indexes and set them to 1
    indexes = [pattern_number * sparsity + i for i in range(sparsity)]
    pattern[indexes] = 1
    # Creat the pattern entry in the dictionary
    patterns_dictionary[pattern_number] = pattern

# Now I need the dynamics equations, let's 




