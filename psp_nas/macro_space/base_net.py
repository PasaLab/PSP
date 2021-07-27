import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNet(torch.nn.Module):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, residual=True,
                 state_num=5):
        '''
        :param actions:
        :param multi_label:
        '''
        super(GraphNet, self).__init__()
        # args

        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.residual = residual
        # check structure of GNN
        self.layer_nums = self.evalate_actions(actions, state_num)

        # layer module
        self.build_model(actions, batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.prediction = None
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def evalate_actions(self, actions, state_num):
        state_length = len(actions)
        if state_length % state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        layer_nums = state_length // state_num
        if self.evaluate_structure(actions, layer_nums, state_num=state_num):
            pass
        else:
            print(f"wrong structure: {actions}")
            raise RuntimeError("wrong structure")
        return layer_nums

    def evaluate_structure(self, actions, layer_nums, state_num=6):
        hidden_units_list = []
        out_channels_list = []
        for i in range(layer_nums):
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            hidden_units_list.append(head_num * out_channels)
            out_channels_list.append(out_channels)

        return out_channels_list[-1] == self.num_label

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):

        # build hidden layer
        for i in range(layer_nums):

            # compute input
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract operator types from action
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            # Multi-head used in GAT.
            # "concat" is True, concat output of each head;
            # "concat" is False, get average of each head output;
            concat = True
            if i == layer_nums - 1:
                concat = False  # The last layer get average
            else:
                pass

            if i == 0:
                residual = False and self.residual  # special setting of dgl
            else:
                residual = True and self.residual
            self.layers.append(
                NASLayer(attention_type, aggregator_type, act, head_num, in_channels, out_channels, dropout=drop_out,
                         concat=concat, residual=residual, batch_normal=batch_normal))

    def forward(self, feat, g):

        output = feat
        for i, layer in enumerate(self.layers):
            output = layer(output, g)

        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    # map GNN's parameters into dict
    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = NASLayer.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        return result

    # load parameters from parameter dict
    def load_param(self, param):
        if param is None:
            return
        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])
