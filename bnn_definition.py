import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LinearCustomDropout(torch.nn.Linear):
    """
    Custom Linear Dropout function,
    inspired by
    https://discuss.pytorch.org/t/making-a-custom-dropout-function/14053
    """

    def __init__(self, in_feats, out_feats,
                 dropout_rate, bias=True):
        """
        Constructor, takes input/output dimensions and
        dropout rate
        """

        super(LinearCustomDropout, self).__init__(
            in_feats, out_feats, bias)
        self.dropout_mask = torch.nn.Dropout(p=dropout_rate)

        return None

    def forward(self, input):
        """
        Forward with custom dropout
        """

        masked_weight = self.dropout_mask(self.weight)
        
        return F.linear(input, masked_weight, self.bias)

class NeuralNetwork(torch.nn.Module):
    """Neural Network Base Structure.
    """

    def __init__(self, input_dim, hidden_1, output_dim=1):

        super(NeuralNetwork, self).__init__()
        
        self.linear_1 = torch.nn.Linear(input_dim, hidden_1)
        self.linear_2 = LinearCustomDropout(hidden_1, output_dim, 0.05)
        self.relu = torch.nn.ReLU()
        
        return None

    def forward(self, x):
        """Forward pass of the Neural Network
        """

        y_pred = self.relu(self.linear_1(x))
        y_pred = self.linear_2(y_pred)
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
    
    def predict_(model, model_input, n_preds=100):
        """
        Purpose: take fitted network and x block
        return 100 predictions
        """

        preds_out = np.concatenate([model(model_input).data.numpy() for _ in range(n_preds)], axis=1)

        # return mean prediction and raw_predictions

        return np.mean(preds_out, axis=1), preds_out