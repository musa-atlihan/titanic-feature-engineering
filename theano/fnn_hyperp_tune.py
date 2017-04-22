import os
import numpy as np
import sys
import timeit
import theano
import theano.tensor as T
import pandas as pd
import utils.data as utld
import utils.nn as nn
import matplotlib.pyplot as plt

"""
Optimize a fully connected neural network with 1 hidden layer.

Plot learning curves for different hyperparameters with k-fold
cross validation.
"""


def halt():
    sys.exit(0)

train_set_path = '../data/train_f_engineered_norm.csv'
test_set_path = '../data/test_f_engineered_norm.csv'

plot_dir = 'fnn_plots'
plot_name = 'fnn_curve'

n_in = 37
n_out = 2

# Hyperparameters for num
n_hidden = 50
learning_rate = 0.1
reg1_rate = 0.00 # regularization term 1 rate
reg2_rate = 0.05 # regularization term 2 rate

n_epochs = 60000
curve_folds = 10

# get dataset
ds = utld.get_dataset(train_set_path, randomize=True)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print('... building model')

x = T.matrix('x')
y = T.ivector('y')

rng = np.random.RandomState(4321)

classifier = nn.FNN(
    rng=rng,
    input=x,
    n_in=n_in,
    n_hidden=n_hidden,
    n_out=n_out
)

start_time = timeit.default_timer()


for hyperp_iter in range(1):
    # Accumulate any hyperparameter in this loop if necessary.
    #reg1_rate = reg1_rate + 0.05

    cost = (
        classifier.negative_log_likelihood(y)
        + reg1_rate * classifier.L1
        + reg2_rate * classifier.L2_sqr
    )
    
    test_model = theano.function(
        inputs=[x, y],
        outputs=classifier.errors(y)
    )
    
    validate_model = theano.function(
        inputs=[x, y],
        outputs=classifier.errors(y)
    )
    
    # gradients
    gparams = [T.grad(cost=cost, wrt=param) for param in classifier.params]
    #gparams = T.grad(cost=cost, wrt=classifier.params)
    
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    train_model = theano.function(
        inputs=[x, y],
        outputs=[cost, classifier.errors(y)],
        updates=updates
    )

    # k-fold cross validation
    valid_folds = np.zeros((ds.n_cv_folds, curve_folds))
    train_folds = np.zeros((ds.n_cv_folds, curve_folds))
    for i in range(ds.n_cv_folds):
        print('... training model for cross validation iteration %i' % i)
        classifier.reset() # set W and b to zero
        # split kaggle train set into test, validation and train sets.
        test_set_xy, valid_set_xy, train_set_xy = ds.cv_split()
    
        test_set_x, test_set_y = test_set_xy
        valid_set_x, valid_set_y = valid_set_xy
        train_set_x, train_set_y = train_set_xy

        n_samples = train_set_x.shape[0] // curve_folds
    
        # learning curves
        valid_curve = np.zeros((curve_folds))
        train_curve = np.zeros((curve_folds))
        train_set_size = np.zeros((curve_folds))
        for j in range(curve_folds):
    
            # Train Model:
            print('... Curve iteration %i' % j)
            epoch = 0
            while (epoch < n_epochs):
                epoch = epoch + 1
                train_cost, train_curve[j] = train_model(train_set_x[:(j + 1) * n_samples,:], 
                                                            train_set_y[:(j + 1) * n_samples])
            
            # compute validation
            valid_curve[j] = validate_model(valid_set_x[:(j + 1) * n_samples,:], 
                                                valid_set_y[:(j + 1) * n_samples])
            train_set_size[j] = (j + 1) * n_samples
    
        valid_folds[i,:] = valid_curve
        train_folds[i,:] = train_curve
    
    test_score = test_model(test_set_x, test_set_y)
    
    
    valid_folds_mean = np.mean(valid_folds, 0) * 100.
    train_folds_mean = np.mean(train_folds, 0) * 100.
    
    # tune the coordinates of plot texts
    text_x = []
    text_y = []
    for i in range(3):
        text_x.append(
            train_set_size.max() - ((train_set_size.max() 
                - train_set_size.min()) * 0.4)
        )
        
        text_y.append(
            valid_folds_mean.max() - ((valid_folds_mean.max() 
                - valid_folds_mean.min()) * ((i + 2) * 0.1))
        )
    
    # plot curves
    plt.plot(train_set_size, valid_folds_mean)
    plt.plot(train_set_size, train_folds_mean)
    plt.ylabel('Error Rate (%)')
    plt.xlabel('Training Set Size')
    plt.text(text_x[0], text_y[0], 'Training Error = %.2f %%' % train_folds_mean[curve_folds - 1])
    plt.text(text_x[1], text_y[1], 'Validation Error = %.2f %%' % valid_folds_mean[curve_folds - 1])
    plt.text(text_x[2], text_y[2], 'Test Error = %.2f %%' % (test_score * 100.))
    plt.grid(True)
    plt.savefig(plot_dir
                + '/'
                + plot_name
                + '_lrate_' + str(learning_rate)
                + '_reg1_' + str(reg1_rate)
                + '_reg2_' + str(reg2_rate)
                + '_n_hidden_' + str(n_hidden)
                + '.png', bbox_inches='tight'
    )
    plt.clf()


end_time = timeit.default_timer()

print(
    'Code run for %d epochs, with %.1f minutes.' % (
        epoch, ((end_time - start_time) / 60))
)