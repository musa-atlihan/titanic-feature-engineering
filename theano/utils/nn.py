try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

"""
Include desired neural network classes.

"""

class LogisticRegression(object):
    """Multi-class logistic regression."""

    def __init__(self, input, n_in, n_out):

        self.n_in = n_in
        self.n_out = n_out

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
        )

        # regularization term 1
        self.reg1 = (
            abs(self.W).sum()
            + abs(self.W).sum()
        )

        # regularization term 2
        self.reg2 = (
            (self.W ** 2).sum()
            + (self.W ** 2).sum()
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def reset(self):
        """Reset W and b values.
        """

        self.W.set_value(np.zeros((self.n_in, self.n_out), 
                                    dtype=theano.config.floatX), 
                                    borrow=True)
        self.b.set_value(np.zeros((self.n_out,), 
                                    dtype=theano.config.floatX), 
                                    borrow=True)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in a batch."""

        # Check if y has same dimension of y_pred.
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    """Hidden layer for Multilayer perceptron."""

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                  activation=T.tanh):

        self.input = input

        if W is None:
            W_vals = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_vals = W_vals * 4.

            W = theano.shared(value=W_vals, name='W', borrow=True)

        if b is None:
            b_vals = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_vals, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

        self.n_in = n_in
        self.n_out = n_out

        self.rng = rng
        self.activation = activation


    def reset(self):
        W_vals = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (self.n_in + self.n_out)),
                    high=np.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )

        if self.activation == T.nnet.sigmoid:
            W_vals = W_vals * 4.

        b_vals = np.zeros((self.n_out,), dtype=theano.config.floatX)

        self.W.set_value(W_vals, borrow=True)
        self.b.set_value(b_vals, borrow=True)




class FNN(object):
    """
    Fully connected neural network with 1 hidden layer.

    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegLayer.negative_log_likelihood
        )

        self.errors = self.logRegLayer.errors

        self.params = (
            self.hiddenLayer.params
            + self.logRegLayer.params
        )

        self.y_pred = self.logRegLayer.y_pred

        self.input = input

    def reset(self):
        self.hiddenLayer.reset()
        self.logRegLayer.reset()


def predict(dataset_x, classifier):
    """Predict labels."""

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    predicted_values = predict_model(dataset_x)

    return predicted_values