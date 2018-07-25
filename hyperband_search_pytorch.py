import pandas as pd
import numpy as np
import feather
import torch
import pickle
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from random_samplers import randomSampler

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

"""Purpose: run hyperband parameter search
"""

# 1. Load test and train datax_train
x_train = feather.read_dataframe('x_train_hyperband.feather')
y_train = feather.read_dataframe('y_train_hyperband.feather')
x_test = feather.read_dataframe('x_test_hyperband.feather')
y_test = feather.read_dataframe('y_test_hyperband.feather')

# convert to PyTorch Variables
x_build_nn = Variable(torch.from_numpy(x_train.as_matrix()).float())
y_build_nn = Variable(torch.from_numpy(y_train['target'].as_matrix()).float(),
                      requires_grad=False)

x_test_nn = Variable(torch.from_numpy(x_test.as_matrix()).float())
y_test_nn = Variable(torch.from_numpy(y_test['target'].as_matrix()).float(),
                     requires_grad=False)

# get input dim for the network
input_dim = x_build_nn.size()[1]

# 2. Define basic Neural Network Structure
class NeuralNetwork(torch.nn.Module):
    """Neural Network Base Structure.
    It will stay the same but the parameters
    will be chosen by the hyperband.
    """

    def __init__(self, input_dim,
                 hidden_1, hidden_2, hidden_3,
                 dropout_2, dropout_3, output_dim=1):

        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_1)
        self.linear2 = torch.nn.Linear(hidden_1, hidden_2)
        self.linear3 = torch.nn.Linear(hidden_2, hidden_3)
        self.linear4 = torch.nn.Linear(hidden_3, output_dim)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.batch_norm_2 = torch.nn.BatchNorm1d(hidden_2)
        self.batch_norm_3 = torch.nn.BatchNorm1d(hidden_3)
        self.dropout_2 = torch.nn.Dropout(dropout_2)
        self.dropout_3 = torch.nn.Dropout(dropout_3)
        self.sigmoid = torch.nn.Sigmoid()

        return None

    def forward(self, x):
        """Forward pass of the Neural Network
        """

        y_pred = self.relu(self.linear1(x))
        y_pred = self.batch_norm_2(self.tanh(self.dropout_2(self.linear2(y_pred))))
        y_pred = self.batch_norm_3(self.tanh(self.dropout_3(self.linear3(y_pred))))
        y_pred = self.linear4(y_pred)
        y_pred = self.sigmoid(y_pred)
        return y_pred

def init_weights(layer):
    """Purpose: initialize weights in each
    LINEAR layer.
    Input: pytorch layer
    """

    if isinstance(layer, torch.nn.Linear):
        np.random.seed(42)
        size = layer.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        # initialize weights
        layer.weight.data.normal_(0.0, variance)

    return None

# 3. Define function that takes neural network, parameters,
# runs everything and returns validation loss

params_def = {}
params_def['layer_1'] = [int, (3, 50)]
params_def['layer_2'] = [int, (3, 40)]
params_def['layer_3'] = [int, (3, 30)]
params_def['dropout_2'] = [float, (0, 1)]
params_def['dropout_3'] = [float, (0, 1)]
params_def['batch_size'] = [int, (32, 512)]
params_def['L2_reg'] = [float, (1e-6, 1e-1)]

def sample_parameters(params_def):
    """Purpose: given the params_def dictionary,
    retturn sampled values of the parameters
    """

    if not isinstance(params_def, dict):
        raise TypeError('Parameters Definition must be dict!')

    sampled = {}

    for k, v in params_def.items():
        # sample float
        if v[0] == float:
            sampled[k] = randomSampler.sampleFloat(v[1][0], v[1][1])
        elif v[0] == int:
            sampled[k] = randomSampler.sampleInteger(v[1][0], v[1][1])

    return sampled

def bind_params_to_model(sampled, input_dim, model_instance):
    """Purpose: take parameters and a model instance
    and bind them together
    """

    # bind model to provided set of parameters
    model_ = model_instance(
        input_dim=input_dim,
        hidden_1=sampled['layer_1'],
        hidden_2=sampled['layer_2'],
        hidden_3=sampled['layer_3'],
        dropout_2=sampled['dropout_2'],
        dropout_3=sampled['dropout_3'])

    # initialize weights
    model_.apply(init_weights)

    return model_

def train_network(sampled, epochs,
                  input_dim, model,
                  x_build_nn, y_build_nn,
                  x_test_nn, y_test_nn):
    """Purpose: given parameters (JSON)
    and model_class - instance of Neural Network class
    Train the network and return test loss
    """

    # 1. Variable "model" is already bound to the set of parameters

    # 2. Specify details for training
    num_epochs = epochs
    batch_size = sampled['batch_size']

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-5,
                                 weight_decay=sampled['L2_reg'])
    epoch_loss = []
    roc_score = []
    roc_score_train = []
    brier_score = []
    brier_score_train = []
    epoch_loss_list = []

    # train the network
    for t in range(num_epochs):

        permutation = torch.randperm(x_build_nn.size()[0])
        epoch_loss = 0

        # for each batch
        for idx, i in enumerate(range(0, x_build_nn.size()[0], batch_size)):

            indices = permutation[i: i+batch_size]
            batch_x, batch_y = x_build_nn[indices], y_build_nn[indices]

            # forward pass
            y_pred_batch = model(batch_x)

            # compute loss for the batch
            loss = criterion(y_pred_batch, batch_y)

            # get epoch loss
            epoch_loss += loss.data[0]

            # zeroing gradients, backward pass and weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute accuracy for epoch
        test_preds = model(x_test_nn)
        train_preds = model(x_build_nn)

        # test set
        roc_auc = compute_roc_score(y_test, test_preds.data.numpy())
        roc_score.append(roc_auc)

        # train set
        roc_train = compute_roc_score(y_build_nn, train_preds.data.numpy())
        roc_score_train.append(roc_train)

        # epoch loss
        epoch_loss_list.append(epoch_loss/idx)

    # return final AUC score
    roc_fin = compute_roc_score(y_test, model(x_test_nn).data.numpy())

    # return AUC score for testing data
    return roc_fin

def compute_roc_score(y_test, y_pred):
    """Purpose: compute ROC score given
    actual and predicted values
    make sure that y_pred.data.numpy()
    """

    # confusion matrix
    false_positive_rate, true_positive_rate, _ = roc_curve(
    y_test, y_pred)

    # compute auc score
    auc_score = auc(false_positive_rate, true_positive_rate)

    return auc_score

### HYPERBAND Optimization ###

# code below taken from http://people.eecs.berkeley.edu/~kjamieson/hyperband.html


max_iter = 81  # maximum iterations/epochs per configuration
eta = 3        # defines downsampling rate (default=3)
logeta = lambda x: np.log(x)/np.log(eta)
s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
B = (s_max+1)*max_iter         # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max+1)):
    n = int(np.ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for
    print("Outer Loop s:", s)

    #### Begin Finite Horizon Successive Halving with (n,r)
    T = [sample_parameters(params_def) for i in range(n)]

    print(len(T), '<<<length of T \n')

    for i in range(s+1):
        print('Inner Loop:', i, ' out of', s+1)
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        n_i = n*eta**(-i)
        r_i = r*eta**(i)
        test_auc_ = [] # container to store AUCs

        for idx, t in enumerate(T):
            print("Inner inner length of T:", len(T), '\n')

            # bind model to the new set of params
            binded_model = bind_params_to_model(sampled=t,
                                                input_dim=input_dim,
                                                model_instance=NeuralNetwork)
            print("Running Config:", t, '\n')
            print("Number of Epochs:", int(r_i), '\n')
            print("{0} out of {1} iteration".format(idx, len(T)), '\n')
            # train the model and return the loss
            test_auc = train_network(sampled=t, epochs=int(r_i),
                                     input_dim=input_dim, model=binded_model,
                                     x_build_nn=x_build_nn, y_build_nn=y_build_nn,
                                     x_test_nn=x_test_nn, y_test_nn=y_test_nn)
            # print for reference
            print("test AUC:", test_auc)
            print("_____ \n")
            test_auc_.append(test_auc)
            # add AUC score for reference
            T[idx]["test_auc_{0}_{1}".format(s, i)] = test_auc

        T = [T[i] for i in np.argsort(test_auc_)[int(n_i/eta):]]

    print('+++++++++++++++ \n')
    print('Best Configurations found from {} iteration:'.format(s))
    print("Configurations:", T, '\n')
    pickle.dump(T, open("configs_found_{0}.p".format(s), "wb"))
    print("pickled to current directory!")
    print('+++++++++++++++ \n')