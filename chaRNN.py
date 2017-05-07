import theano
import theano.tensor as T

import numpy as np

class LSTM:
    def __init__(self, inp_size, hidden_size):
        self.inp_size = inp_size
        self.hidden_size = hidden_size

        self.weights = {
                            "f:x" : init_weights([self.inp_size, self.hidden_size]),
                            "f:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "f:b" : init_weights([self.hidden_size]),
                            "i:x" : init_weights([self.inp_size, self.hidden_size]),
                            "i:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "i:b" : init_weights([self.hidden_size]),
                            "o:x" : init_weights([self.inp_size, self.hidden_size]),
                            "o:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "o:b" : init_weights([self.hidden_size]),
                            "c:x" : init_weights([self.inp_size, self.hidden_size]),
                            "c:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "c:b" : init_weights([self.hidden_size]),
                       }

    def get_weights(self):
        return [self.weights[key] for key in self.weights.keys()]

    def recurrence(self, inp, prev_hidden, prev_cell):
        forget = T.nnet.sigmoid(T.dot(inp, self.weights["f:x"]) +\
                                T.dot(prev_hidden, self.weights["f:h"]) +\
                                self.weights["f:b"])

        input_ = T.nnet.sigmoid(T.dot(inp, self.weights["i:x"]) +\
                                T.dot(prev_hidden, self.weights["i:h"]) +\
                                self.weights["i:b"])

        output = T.nnet.sigmoid(T.dot(inp, self.weights["o:x"]) +\
                                T.dot(prev_hidden, self.weights["o:h"]) +\
                                self.weights["o:b"])

        cell = T.mul(forget, prev_cell) + T.mul(input_, T.tanh(T.dot(inp, self.weights["c:x"]) +\
        T.dot(prev_hidden, self.weights["c:h"]) +\
        self.weights["c:b"]))

        hidden = T.mul(output, cell)

        return hidden, cell

def init_weights(shape):
    return theano.shared(np.array(np.random.randn(*shape) * 0.01))

def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2

        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling

        updates.append((acc, acc_new))
        updates.append((p, p - T.clip(lr * g, -0.01, 0.01)))

    return updates

def hard_round(x, digits):
    x = str(round(x, digits))
    x += '0' * (digits - len(x.split('.')[1]))
    return x

def pad(x, length):
    x = str(x)
    return x + ' ' * (length - len(x))

def main():
    truncate = 50
    epochs = 100

    text = open("file.txt", 'rb').read()
    text += ' ' * (len(text) % truncate)
    char_to_index = {}
    index_to_char = {}

    idx = 0
    for letter in text:
        get = char_to_index.get(letter)

        if get == None:
            char_to_index[letter] = idx
            index_to_char[idx] = letter
            idx += 1

    vocab_size = len(char_to_index.keys())

    print char_to_index, index_to_char

    # ------------------- MODEL -------------------

    lstm = LSTM(vocab_size, 512)
    inp = T.matrix()                        # timesteps x vocab_size
    targets = T.matrix()                    # timesteps x vocab_size
    lr = T.scalar()

    fc_param = [init_weights([512, vocab_size])]

    init_hidden, init_cell = T.vector(), T.vector()

    ([hidden, cell], updates) = theano.scan(fn = lstm.recurrence,
                                             sequences = inp,
                                             outputs_info = [init_hidden, init_cell])

    distributions = T.clip(T.nnet.softmax(T.dot(hidden, fc_param[0])), 1e-5, 1-1e-5)
    loss = T.mean(T.nnet.categorical_crossentropy(distributions, targets))

    updates = RMSprop(cost = loss, params = fc_param + lstm.get_weights(), lr = lr)

    train = theano.function(inputs = [inp, targets, init_hidden, init_cell, lr], outputs = [loss, distributions, hidden[-1], cell[-1]], updates = updates)

    # ------------------- MODEL -------------------

    learning_rate = 0.1

    for epoch in range (epochs):
        hidden = np.zeros([512])
        cell = np.zeros([512])

        #if epoch == 1:
        #    learning_rate *= 0.8

        #elif epoch == 4:
        #    learning_rate *= 0.7

        for timestep in range (0, len(text), truncate):
            train_text = text[timestep : timestep + truncate]

            inp = []
            targets = []

            #print train_text

            for letter in train_text[:-1]:
                onehot = np.zeros(shape = [vocab_size])
                onehot[char_to_index[letter]] = 1
                inp.append(onehot)

            for letter in train_text[1:]:
                onehot = np.zeros(shape = [vocab_size])
                onehot[char_to_index[letter]] = 1
                targets.append(onehot)

            loss, generated_text, hidden, cell = train(inp, targets, hidden, cell, learning_rate)

            reconstruction = ''
            #print generated_text
            for l in generated_text:
                reconstruction += index_to_char[np.argmax(l)]

            if timestep % (truncate * 40) == 0:
                hidden = np.zeros([512])
                cell = np.zeros([512])

            if timestep % (truncate * 20) == 0:
                print "epoch " + str(pad(epoch, 3)) + ' | batch - ' + pad(timestep/truncate, 5) + ' / ' + pad(len(text) / truncate, 5) + ' | loss - ' + hard_round(loss, 3) + ' | generated_text - ' + reconstruction

main()
