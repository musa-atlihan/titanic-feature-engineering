import numpy as np
import timeit
import theano
import theano.tensor as T
import pandas as pd
import utils.data as utld
import utils.nn as nn

"""
Optimize a fully connected neural network and predict for Kaggle test set.

"""

train_set_path = '../data/train_f_engineered_norm.csv'
test_set_path = '../data/test_f_engineered_norm.csv'


# Hyperparameters
n_hidden = 50
learning_rate = 0.1
reg1_rate = 0.00 # regularization term 1 rate
reg2_rate = 0.05 # regularization term 2 rate

n_epochs = 20000

n_in = 37
n_out = 2

# get training dataset
ds = utld.get_dataset(train_set_path)

# the Kaggle test set we will finally use to predict for kaggle contest.
kaggle_test_set = utld.get_dataset(test_set_path)


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

cost = (
    classifier.negative_log_likelihood(y)
    + reg1_rate * classifier.L1
    + reg2_rate * classifier.L2_sqr
)

# gradients
gparams = [T.grad(cost=cost, wrt=param) for param in classifier.params]

updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]

train_model = theano.function(
    inputs=[x, y],
    outputs=[cost, classifier.errors(y)],
    updates=updates
)

start_time = timeit.default_timer()

# Train Model:
epoch = 0
while (epoch < n_epochs):
    epoch = epoch + 1
    train_cost, train_error = train_model(ds.x, ds.y)

predicted_vals = nn.predict(kaggle_test_set.x, classifier)

predicted_vals = np.array([np.arange(892, 1310), predicted_vals])
predicted_vals = np.transpose(predicted_vals)

df_pred = pd.DataFrame(data=predicted_vals, 
             columns=['PassengerId', 'Survived']
)

# Save predictions to file
df_pred.to_csv('pred_fnn.csv', sep=',', index=False)

end_time = timeit.default_timer()

print(
    'Code run for %d epochs, with %.1f minutes.' % (
        epoch, ((end_time - start_time) / 60))
)