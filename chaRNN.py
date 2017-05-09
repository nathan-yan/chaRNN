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
                            "f:b" : init_weights([self.hidden_size], type = 'ones'),
                            "i:x" : init_weights([self.inp_size, self.hidden_size]),
                            "i:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "i:b" : init_weights([self.hidden_size], type = 'zeros'),
                            "o:x" : init_weights([self.inp_size, self.hidden_size]),
                            "o:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "o:b" : init_weights([self.hidden_size], type = 'zeros'),
                            "c:x" : init_weights([self.inp_size, self.hidden_size]),
                            "c:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "c:b" : init_weights([self.hidden_size], type = 'zeros'),
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

def init_weights(shape, type = 'regular'):
    if type == 'ones':
        return theano.shared(np.array(np.ones(shape), dtype=np.float32))
    elif type == 'zeros':
        return theano.shared(np.array(np.zeros(shape), dtype=np.float32))

    else:
        return theano.shared(np.array(np.random.randn(*shape) * 0.01, dtype=np.float32))

def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        g = T.clip(g, -1, 1)
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2

        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = T.clip(g / gradient_scaling, -1, 1)

        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

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
    text += '*' * (len(text) % truncate)
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

    # ------------------- MODEL -------------------

    lstm_1 = LSTM(vocab_size, 512)
    lstm_2 = LSTM(512, 512)

    inp = T.matrix()                        # timesteps x vocab_size
    targets = T.matrix()                    # timesteps x vocab_size
    lr = T.scalar()

    fc_param = [init_weights([512, vocab_size])]

    init_hidden_1, init_cell_1 = T.vector(), T.vector()
    init_hidden_2, init_cell_2 = T.vector(), T.vector()

    ([hidden_1, cell_1], updates) = theano.scan(fn = lstm_1.recurrence,
                                             sequences = inp,
                                             outputs_info = [init_hidden_1, init_cell_1])

    ([hidden_2, cell_2], updates) = theano.scan(fn = lstm_2.recurrence,
                                            sequences = hidden_1,
                                            outputs_info = [init_hidden_2, init_cell_2])

    t = T.dot(hidden_2, fc_param[0])
    distributions = T.nnet.softmax(t)
    loss = T.mean(T.nnet.categorical_crossentropy(distributions, targets))

    updates = RMSprop(cost = loss, params = fc_param + lstm_1.get_weights() + lstm_2.get_weights(), lr = lr)

    train = theano.function(inputs = [inp, targets, init_hidden_1, init_cell_1, init_hidden_2, init_cell_2, lr], outputs = [t, loss, distributions, hidden_1[-1], cell_1[-1], hidden_2[-1], cell_2[-1]], updates = updates)

    sample = theano.function(inputs = [inp, init_hidden_1, init_cell_1, init_hidden_2, init_cell_2], outputs = [distributions, hidden_1[-1], cell_1[-1], hidden_2[-1], cell_2[-1]])

    # ------------------- MODEL -------------------

    learning_rate = 2e-3

    from tqdm import *

    for epoch in range (epochs):
        hidden_1 = np.zeros([512]).astype(np.float32)
        cell_1 = np.zeros([512]).astype(np.float32)
        hidden_2 = np.zeros([512]).astype(np.float32)
        cell_2 = np.zeros([512]).astype(np.float32)
        f = None
        for timestep in range (0, len(text), truncate):
            if timestep % (100 * truncate) == 0:
                hidden_1 = np.zeros([512]).astype(np.float32)
                cell_1 = np.zeros([512]).astype(np.float32)
                hidden_2 = np.zeros([512]).astype(np.float32)
                cell_2 = np.zeros([512]).astype(np.float32)

            if timestep % (1000 * truncate) == 0:
                f = open("Record_" + str(timestep), 'wb')
                seed = np.zeros([vocab_size]).astype(np.float32)
                seed[np.random.randint(0, vocab_size)] = 1

                h1 = np.zeros([512]).astype(np.float32)
                c1 = np.zeros([512]).astype(np.float32)
                h2 = np.zeros([512]).astype(np.float32)
                c2 = np.zeros([512]).astype(np.float32)

                for i in tqdm(range (10000)):
                    d, h1, c1, h2, c2 = sample(seed.reshape([1, vocab_size]), h1, c1, h2, c2)
                    seed_ = np.random.choice([_ for _ in range (vocab_size)], p = d[0])
                    seed = np.zeros([vocab_size]).astype(np.float32)
                    seed[seed_] = 1

                    f.write(index_to_char[seed_])
                f.close()

            train_text = text[timestep : timestep + truncate]

            inp = []
            targets = []

            for letter in train_text[:-1]:
                onehot = np.zeros(shape = [vocab_size])
                onehot[char_to_index[letter]] = 1
                inp.append(onehot)

            for letter in train_text[1:]:
                onehot = np.zeros(shape = [vocab_size])
                onehot[char_to_index[letter]] = 1
                targets.append(onehot)

            if timestep < (truncate * 40):
                t, loss, generated_text, hidden_1, cell_1, hidden_2, cell_2 = train(inp, targets, hidden_1, cell_1, hidden_2, cell_2, learning_rate/10.)
            else:
                t, loss, generated_text, hidden_1, cell_1, hidden_2, cell_2 = train(inp, targets, hidden_1, cell_1, hidden_2, cell_2, learning_rate)

            reconstruction = ''
            #print generated_text
            for l in generated_text:
                reconstruction += index_to_char[np.argmax(l)]

            if timestep % (truncate * 5) == 0:
                print np.round(t, 3)
                print "epoch " + str(pad(epoch, 3)) + ' | batch - ' + pad(timestep/truncate, 5) + ' / ' + pad(len(text) / truncate, 5) + ' | loss - ' + hard_round(loss, 3) + " | generated_text - " + reconstruction
main()
