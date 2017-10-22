import numpy as np
import matplotlib.pyplot as plt


def pre_synaptic_simple(epsilon, w, z_post, z_pre):
    increase = np.zeros_like(w)

    n = w.shape[0]
    for i in range(n):
        for j in range(n):
            increase[i, j] = z_pre[j] * z_post[i] - z_pre[j] * w[i, j]

    return epsilon * increase


def post_synaptic_simple(epsilon, w, z_post, z_pre):
    increase = np.zeros_like(w)

    n = w.shape[0]
    for i in range(n):
        for j in range(n):
            increase[i, j] = z_post[i] * z_pre[j] - z_post[i] * w[i, j]


def pre_synaptic(epsilon, w, z_post, z_pre):
    return epsilon * (np.outer(z_post, z_pre) - w * z_pre)


def post_synaptic(epsilon, w, z_post, z_pre):
    return epsilon * (np.outer(z_pre, z_post) - z_post * w).T


def bernoulli_mask(size_from, size_to, p, binomial=True):
    if binomial:
        return np.random.binomial(n=1, p=p, size=(size_to, size_from))
    else:
        return np.random.choice(2, size=(size_to, size_from), replace=True, p=[1 - p, p])


def update_activity(k, z, x_i, c, weight, Is, Ir, G, s, m):
    inhibition = Is * s + Ir * m
    recurrent_excitation = G * np.dot(c * weight, z)
    input_excitation = k * x_i

    return input_excitation, recurrent_excitation, inhibition


class MinaNetwork:

    def __init__(self, n_input=200, n_recurrent=200, p=1.0, v=21, b=21, Ki=1.0, Kr=0.5, Ci=1.0, Cr=0.5, theta=0, phi=0,
                 w=None, a=None, uniform_w=True, w0=0.5):
        """

        :param N_input: Size of the input and output layer (Ethorinal Cortex, C1)
        :param N_recurrent: Size of the recurrent layer (C3)
        :param p:  sparsness of connectivity
        :param v:  input strength to C3
        :param b:  input strength to C1
        :param Ki: Input feed-forward inhibition to C3
        :param Kr: recurrent inhibition in C3
        :param Ci: Input feed-forward inhibition to C1
        :param Cr: Recurrent inhibition in C1
        :param theta: threshold in C3
        :param phi: threshold in phi
        :param w: recurrent connectivity in C3
        :param a: connectivity from C3-C1
        :param uniform_w:  whether we want w to be uniform or not
        """

        self.N_input = n_input
        self.N_recurrent = n_recurrent
        self.p = p
        self.v = v
        self.b = b
        self.Ki = Ki
        self.Kr = Kr
        self.GC3 = 1.0
        self.Ci = Ci
        self.Cr = Cr
        self.GC1 = 1.0
        self.theta = theta
        self.phi = phi

        # Patterns parameters
        self.patterns_dictionary = {}
        self.sparsity = None
        self.number_of_patterns = None
        self.neurons_per_pattern = None

        # Create the masks
        self.c1 = bernoulli_mask(size_from=self.N_recurrent, size_to=self.N_recurrent, p=p, binomial=True)
        self.c2 = bernoulli_mask(size_from=self.N_recurrent, size_to=self.N_input, p=p, binomial=True)

        # Initialize the weight matrices
        self.w0 = w0
        if w is None:
            if uniform_w:
                self.w = np.ones((n_recurrent, n_recurrent)) * self.w0
            else:
                self.w = np.random.rand(n_recurrent, n_recurrent)
        else:
            self.w = w

        if a is None:
            self.a = np.zeros((n_input, n_recurrent))
        else:
            self.a = a

    def print_parameters(self):
        print('Ki', self.Ki)

    def build_patterns_dictionary(self, sparsity=10, number_of_patterns=20):
        patterns_dictionary = {}
        self.sparsity = sparsity
        self.number_of_patterns = number_of_patterns

        neurons_per_pattern = int((sparsity / 100) * self.N_input)

        for pattern_number in range(number_of_patterns):
            # Initialize the pattern with zero
            pattern = np.zeros(self.N_input)
            # Chose some indexes and set them to 1
            indexes = [pattern_number * neurons_per_pattern + i for i in range(neurons_per_pattern)]
            pattern[indexes] = 1
            # Create the pattern entry in the dictionary
            patterns_dictionary[pattern_number] = pattern

        # Scale constants
        self.GC3 /= neurons_per_pattern
        self.GC1 /= neurons_per_pattern
        self.Kr /= neurons_per_pattern
        self.Ki /= neurons_per_pattern
        self.Ci /= neurons_per_pattern
        self.Cr /= neurons_per_pattern

        # Store the patterns
        self.patterns_dictionary = patterns_dictionary
        self.neurons_per_pattern = neurons_per_pattern

    def train_network(self, epsilon, training_time, sequence, verbose=False,
                      pre_synaptic_rule=True, save_quantities=False):
        """
        Train the network
        :param epsilon: The learning rate
        :param training_time:  How many time steps
        :param sequence:  The sequence in which you want to train
        :param: verbose: print the evolution for debugging purposes
        :param pre_synaptic_rule: If True use the pre-synaptic rule for w otherwise use the post-synaptic one
        :param save_quantities: Whether to save quantities
        :return:  A dictionary with the quantities saved
        """

        save_dictionary = {}

        m_history = []
        w_history = []
        a_history = []
        excitation_r_history = []
        excitation_out_history = []
        inhibition_r_history = []
        inhibition_out_history = []
        input_r_history = []
        input_out_history = []
        z_r_history = []
        z_out_history = []

        for _ in range(training_time):
            y_r = np.zeros(self.N_recurrent)
            z_r = np.zeros(self.N_recurrent)
            m = 0.0

            y_out = np.zeros(self.N_input)
            z_out = np.zeros(self.N_input)

            for sequence_number in sequence:
                # Input variables
                x = self.patterns_dictionary[sequence_number]
                s = np.sum(x)
                modified_input = np.zeros(self.N_recurrent)
                modified_input[np.where(x == 1)[0]] = 1.0

                if verbose:
                    print('sequence', sequence_number)
                    print('------')
                    print(_)
                    print('----')
                    print('s')
                    print(s)
                    print('m')
                    print(m)
                # Update values for the C3
                aux = update_activity(self.v, z_r, modified_input, self.c1, self.w, self.Ki, self.Kr, self.GC3, s, m)
                input_excitation_r, recurrent_excitation_r, inhibition_r = aux
                y_r = input_excitation_r + recurrent_excitation_r - inhibition_r
                z_r_pre = np.copy(z_r)
                z_r = (y_r > self.theta).astype('float')

                # Count the neurons that we activated in C3
                m = np.sum(z_r)

                # Update values for C1
                aux = update_activity(self.b, z_r, x, self.c2, self.a, self.Ci, self.Cr, self.GC1, s, m)
                input_excitation_out, recurrent_excitation_out, inhibition_out = aux
                y_out = input_excitation_out + recurrent_excitation_out - inhibition_out
                z_out = (y_out > self.phi).astype('float')

                # Update the weights
                if pre_synaptic_rule:
                    aux = pre_synaptic(epsilon=epsilon, w=self.w, z_post=z_r, z_pre=z_r_pre)
                    self.w += aux
                else:
                    aux = post_synaptic(epsilon=epsilon, w=self.w, z_post=z_r, z_pre=z_r_pre)
                    self.w += aux

                increment_a = pre_synaptic(epsilon=epsilon, w=self.a, z_post=z_out, z_pre=z_r)
                self.a += increment_a

                # Save history
                if save_quantities:
                    m_history.append(m)
                    w_history.append(np.copy(self.w))
                    a_history.append(np.copy(self.a))
                    excitation_r_history.append(recurrent_excitation_r)
                    excitation_out_history.append(recurrent_excitation_out)
                    inhibition_r_history.append(inhibition_r)
                    inhibition_out_history.append(inhibition_out)
                    input_r_history.append(input_excitation_r)
                    input_out_history.append(input_excitation_out)
                    z_r_history.append(z_r)
                    z_out_history.append(z_out)

                if verbose:
                    print('C3 layer')
                    print('recurrent excitation')
                    print(recurrent_excitation_r.astype('int'))
                    print('---- inhibition')
                    print(inhibition_r)
                    print('excitation input')
                    print(input_excitation_r)
                    print('y_r')
                    print(y_r)
                    print('z_r')
                    print(z_r)
                    print('w increase')
                    print(aux)

                    print('C1 layer')
                    print('recurrent excitation_out')
                    print(recurrent_excitation_out.astype('int'))
                    print('inhibition_out')
                    print(inhibition_out)
                    print('excitation input_out')
                    print(input_excitation_out)
                    print('y_out')
                    print(y_out)
                    print('z_out')
                    print(z_out)
                    print('a increment')
                    print(increment_a)

        # Let's store the saved values and return the weight matrixes
        if save_quantities:
            save_dictionary['m'] = m_history
            save_dictionary['a'] = a_history
            save_dictionary['w'] = w_history
            save_dictionary['excitation_r'] = excitation_r_history
            save_dictionary['excitation_out'] = excitation_out_history
            save_dictionary['inhibition_r'] = inhibition_r_history
            save_dictionary['inhibition_out'] = inhibition_out_history
            save_dictionary['input_r'] = input_r_history
            save_dictionary['input_out'] = input_out_history
            save_dictionary['z_r'] = z_r_history
            save_dictionary['z_out'] = z_out_history

        return save_dictionary

    def recall(self, recall_time, cue, verbose=False):

        x = self.patterns_dictionary[cue]
        recall_history = np.zeros((recall_time, self.N_input))

        # Initialize the variables
        y_r = np.zeros(self.N_recurrent)
        z_r = np.zeros(self.N_recurrent)
        y_out = np.zeros(self.N_input)
        z_out = np.zeros(self.N_input)
        m = 0

        for _ in range(recall_time):
            s = np.sum(x)
            modified_input = np.zeros(self.N_recurrent)
            modified_input[np.where(x == 1)[0]] = 1.0

            if verbose:
                print('------')
                print(_)
                print('----')
                print('s')
                print(s)
                print('m')
                print(m)

            # Update values for the C3
            aux = update_activity(self.v, z_r, modified_input, self.c1, self.w, self.Ki, self.Kr, self.GC3, s, m)
            input_excitation_r, recurrent_excitation_r, inhibition_r = aux
            y_r = input_excitation_r + recurrent_excitation_r - inhibition_r
            z_r_pre = np.copy(z_r)
            z_r = (y_r > self.theta).astype('float')

            # Count the neurons that we activated in C3
            m = np.sum(z_r)

            # Update values for C1
            aux = update_activity(self.b, z_r, x, self.c2, self.a, self.Ci, self.Cr, self.GC1, s, m)
            input_excitation_out, recurrent_excitation_out, inhibition_out = aux
            y_out = input_excitation_out + recurrent_excitation_out - inhibition_out
            z_out = (y_out > self.phi).astype('float')

            # History
            recall_history[_, ...] = z_out
            # Eliminate the input
            x = np.zeros(self.N_input)

            if verbose:
                print('C3 layer')
                print('recurrent excitation')
                print(recurrent_excitation_r.astype('int'))
                print('---- inhibition')
                print(inhibition_r)
                print('excitation input')
                print(input_excitation_r)
                print('y_r')
                print(y_r)
                print('z_r')
                print(z_r)

                print('C1 layer')
                print('recurrent excitation_out')
                print(recurrent_excitation_out.astype('int'))
                print('inhibition_out')
                print(inhibition_out)
                print('excitation input_out')
                print(input_excitation_out)
                print('y_out')
                print(y_out)
                print('z_out')
                print(z_out)

        return recall_history

    def test_recall_bolean(self, sequence):
        # Recall
        cue = sequence[0]
        recall_time = len(sequence)
        z_recall = self.recall(recall_time=recall_time, cue=cue, verbose=False)

        # Test equality
        success = True
        for sequence_number, z in zip(sequence, z_recall):
            x = self.patterns_dictionary[sequence_number]
            if not np.allclose(x, z):
                success = False
                break

        return success

    def test_recall(self, sequence):
        # Recall
        cue = sequence[0]
        recall_time = len(sequence)
        z_recall = self.recall(recall_time=recall_time, cue=cue, verbose=False)

        # Test equality
        success = 0.0
        for sequence_number, z in zip(sequence, z_recall):
            x = self.patterns_dictionary[sequence_number]
            if np.allclose(x, z):
                success += 1.0

        success /= recall_time
        return success * 100.0, z_recall

    def plot_weight_matrices(self, switch_grid=True):

        fig = plt.figure(figsize=(16, 12))

        fig.suptitle('Connectivities (w left, a right)')

        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(self.w, aspect='auto', vmin=0, vmax=1)
        if switch_grid:
            ax1.grid()

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(self.a, aspect='auto', vmin=0, vmax=1)
        if switch_grid:
            ax2.grid()

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
