import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from network import MinaNetwork

N_input = 200 # Inputs size
N_recurrent = 200  # C3 size
v = 25.0   # Input - C3 connection
b = 25.0   # Input - C1 connection
Kr = 1.0   # Recurrent self-inhibition gain
Ki = 0.0 # Input - C3 inhibition
Ci = 1.0  # Inhibition from the input to C1
Cr = 1.0  # Inhibition from C3 to C1
p = 1.0   # Sparness parameter

# Dynamical parameters
theta = 0.0
phi = 0

# Patternsnp
number_of_patterns = 10
sparsity = 10.0


nn = MinaNetwork(n_input=N_input, n_recurrent=N_recurrent, p=p, v=v, b=b, Ki=Ki, Kr=Kr, Ci=Ci, Cr=Cr,
                 theta=theta, phi=phi, uniform_w=False)

nn.build_patterns_dictionary(sparsity=sparsity, number_of_patterns=number_of_patterns)


sequence1 = [0, 1, 2, 3, 4]
sequence1 = [0, 1, 2]
sequence2 = [5, 6, 7, 8, 9]
epsilon = 0.1
training_time = 100
verbose = False

quantities = nn.train_network(epsilon=epsilon, training_time=training_time, pre_synaptic_rule=True,
                              sequence=sequence1, verbose=verbose, save_quantities=True)

z_r_end = np.mean(quantities['z_r'], axis=0)
print(z_r_end)
success = nn.test_recall(sequence=sequence1)
print('success', success)

# quantities = nn.train_network(epsilon=epsilon, training_time=training_time, sequence=sequence2, save_quantities=True)


if False:
    # Recall
    cue = sequence1[0]
    recall_time = 2
    z_recall = nn.recall(recall_time=recall_time, cue=cue, verbose=True)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(z_recall, aspect='auto')

    plt.show()

if False:
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(quantities['m'], label='m')
    ax.plot(quantities['inhibition_r'], label='inhibition_r')
    ax.plot(quantities['input_r'][0], label='input')

    ax.legend()
    plt.show()


if True:
    fig = plt.figure(figsize=(16, 12))

fig.suptitle('Connectivities (w left, a right)')

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(nn.w, aspect='auto', vmin=0, vmax=1)
# ax1.grid()

ax2 = fig.add_subplot(122)
im2 = ax2.imshow(nn.a, aspect='auto', vmin=0, vmax=1)
# ax2.grid()

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.show()