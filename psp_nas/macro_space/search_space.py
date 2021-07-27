import torch
from constant import *

class MacroSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            # Define operators in search space
            self.search_space = [
                {"name": "attention_type", "value": ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear']}, # 7
                {"name": 'aggregator_type', "value": ["sum", "mean", "max", "mlp"]}, # 4
                {"name": 'activate_function', "value": ["sigmoid", "tanh", "relu", "linear", "softplus", "leaky_relu", "relu6", "elu"]}, # 8
                {"name": 'number_of_heads', "value": [1, 2, 4, 6, 8, 16]}, # 6
                {"name": 'hidden_units', "value": [4, 8, 16, 32, 64, 128, 256]}, # 7

                {"name": "learning_rate", "value": [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]}, # 5
                {"name": "dropout", "value": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, # 10
                {"name": "weight_decay", "value": [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]}, # 6
                
                {"name": "feature_engine", "value": FEATURE_ENGINE_LIST} # 7
            ]

    def get_search_space(self, num_of_layers):
        actual_actions = self.search_space[: 5] * (num_of_layers - 1)
        actual_actions.extend(self.search_space[: 4])
        actual_actions.extend(self.search_space[5: -1])
        actual_actions.append(self.search_space[-1])
        return actual_actions

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    # def generate_action_list(self, num_of_layers=2):
    #     action_names = list(self.search_space.keys())
    #     action_list = action_names * num_of_layers
    #     return action_list


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")