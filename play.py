import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from network import MinaNetwork

### Structure paramters
N_input = 200  # Inputs size
N_recurrent = 200  # C3 size
v = 21.0   # Input - C3 connection
b = 21.0   # Input - C1 connection
Kr = 0.5   # Recurrent self-inhibition gain
Ki = 1.0  # Input - C3 inhibition
Ci = 1.0  # Inhibition from the input to C1
Cr = 0.5  # Inhibition from C3 to C1
p = 1.0   # Sparness parameter

# Dynamical parameters
theta = 0.0
phi = 0

# Training parameters
training_time = 100
epsilon = 0.1

# Patterns
number_of_patterns = 20
sparsity = 10


nn = MinaNetwork(N_input=N_input, N_recurrent=N_recurrent, p=p, v=v, b=b, Ki=Ki, Kr=Kr, Ci=Ci, Cr=Cr,
                 theta=theta, phi=phi, uniform_w=False)

nn.build_patterns_dictionary()

sequence1 = [0, 1, 2, 3, 4]
sequence2 = [8, 9, 10, 11, 12]
epsilon = 0.1
training_time = 200

nn.train_network(epsilon=epsilon, training_time=training_time, sequence=sequence1)
nn.train_network(epsilon=epsilon, training_time=training_time, sequence=sequence2)

fig = plt.figure(figsize=(16, 12))

fig.suptitle('Connectivities (w left, a right)')

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(nn.w, aspect='auto')
# ax1.grid()

ax2 = fig.add_subplot(122)
im2 = ax2.imshow(nn.a, aspect='auto')
# ax2.grid()

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.show()