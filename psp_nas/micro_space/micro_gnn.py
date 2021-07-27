import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.models import JumpingKnowledge

import math

from .gbdt_utils import get_logger
from .micro_search_space import gnn_map, act_map, LinearConv

logger = get_logger()

class MicroGNN(nn.Module):
    def __init__(self, action, num_feat, num_classes, num_hidden, dropout=0.6, layers=2, stem_multiplier=2, bias=True):
        super(MicroGNN, self).__init__()
        self.num_classes = num_classes
        self._layers = layers
        self.dropout = dropout
        self.bias = bias
        
        assert len(action[:-1]) % 3 == 0
        self._steps = len(action[:-1]) // 3

        his_dim, cur_dim, hidden_dim, out_dim = num_feat, num_feat, num_hidden, num_hidden
        self.his_dim = his_dim
        self.hidden_dim = hidden_dim
    
        # 表示每个前置层的输出维度应该是多少
        each_prev_out_dim = [0] * (self._steps + 1)
        each_prev_out_dim[0] = num_feat
        each_prev_out_dim[1] = num_hidden
        each_prev_out_dim[2] = -1
    
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(his_dim, hidden_dim))

        self.cells = nn.ModuleList()
        self.jk_cells = nn.ModuleList()
        self.transform_dim_module = nn.ModuleList()
        self.need_transform_dim = [False] * (layers - 1)

        for i in range(layers):
            cell = Cell(action, each_prev_out_dim, hidden_dim, out_dim, num_feat=num_feat, num_classes=num_classes, layer_index=i, dropout=self.dropout, concat=True, bias=bias)
            self.cells += [cell]
            his_dim = cur_dim
            cur_dim = cell.multiplier * out_dim if action[-1] == "concat" else out_dim

            if i < layers - 1:
                if cur_dim != hidden_dim:
                    self.fcs.append(nn.Linear(cur_dim, hidden_dim))
                    self.need_transform_dim[i] = True

            if action[-1] == 'lstm':
                self.jk_cells.append(cell.jk_func)

            # 每一层结束后，把每个前置节点的输出维度更新一下
            each_prev_out_dim = cell.each_prev_out_dim
            each_prev_out_dim[0] = each_prev_out_dim[1]
            each_prev_out_dim[1] = hidden_dim
            for j in range(2, len(each_prev_out_dim)):
                each_prev_out_dim[j] = -1

        self.fcs.append(nn.Linear(cur_dim, num_classes))

        self.params_fc = list(self.fcs.parameters())
        self.params_gnn = list(self.cells.parameters()) + list(self.jk_cells.parameters())

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        s0 = x
        s1 = self.fcs[0](x)

        h0 = s1

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, edge_index, h0)
            # 这里是把当前cell输出的维度变换成hidden dim，为了防止后面一个单元的ppnp算子，需要维度统一
            if i < len(self.cells) - 1 and self.need_transform_dim[i]:
                s1 = self.fcs[i + 1](s1)
        out = s1
        logits = self.fcs[-1](out.view(out.size(0), -1))
        return logits

class Cell(nn.Module):
    
    def __init__(self, action_list, each_prev_out_dim, hidden_dim, out_dim, num_feat, num_classes, layer_index, dropout, concat, bias=True):

        assert hidden_dim == out_dim  # current version only support this situation
        super(Cell, self).__init__()
        self.num_classes = num_classes
        self.layer_index = layer_index
        self.num_feat = num_feat

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.concat_of_multihead = concat
        self.bias = bias

        self.dropout = dropout

        self._indices = []
        self._act_list = []

        self._transform_op = nn.ModuleList()
        self._gnn_aggrs = nn.ModuleList()
        self._enhanced_gnn = nn.ModuleList()

        self._gnn_aggrs_name = []

        self._hop_selected = []

        self.each_prev_out_dim = each_prev_out_dim

        self._compile(action_list)
        # self.reset_parameters()

    def reset_parameters(self):
        for aggr in self._gnn_aggrs:
            aggr.reset_parameters()

    def _compile(self, action_list):
        # 把concat_type排掉
        cells_info = action_list[:-1]
        assert len(cells_info) % 3 == 0
        self._steps = len(cells_info) // 3
        self.multiplier = 0
        self._concat = action_list[-1]

        for i, action in enumerate(cells_info):
            if i % 3 == 0:
                self._indices.append(action)
            elif i % 3 == 2:
                self._act_list.append(act_map(action))
            else:
                if action == 'identity' and i < len(cells_info) - 3:
                    self.each_prev_out_dim[i // 3 + 2] = self.each_prev_out_dim[self._indices[-1]]
                elif i < len(cells_info) - 3:
                    self.each_prev_out_dim[i // 3 + 2] = self.hidden_dim

                if action == 'identity':
                    premiliar_gnn = gnn_map(action, self.each_prev_out_dim[self._indices[-1]], self.each_prev_out_dim[self._indices[-1]], self.num_feat, self.dropout, self.concat_of_multihead, self.bias)    
                else:
                    premiliar_gnn = gnn_map(action, self.each_prev_out_dim[self._indices[-1]], self.hidden_dim, self.num_feat, self.dropout, self.concat_of_multihead, self.bias)
                enhanced_gnn = GraphEhanced(self.hidden_dim)

                self._gnn_aggrs.append(premiliar_gnn)
                self._enhanced_gnn.append(enhanced_gnn)

                self._gnn_aggrs_name.append(action)

        for x in self._gnn_aggrs_name:
            if x != 'identity':
                self.multiplier += 1
        
        if self._concat == 'lstm':
            self.jk_func = JumpingKnowledge(mode='lstm', channels=self.hidden_dim ,
                                            num_layers=self.multiplier).cuda()

    def forward(self, s0, s1, edge_index, h0):
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[i]]

            if self._gnn_aggrs_name[i] == 'identity':
                op1 = states[self._indices[i]]
                states += [op1]
                continue
            else:
                op1 = self._gnn_aggrs[i]
                op2 = self._enhanced_gnn[i]

            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            s_premiliar = op1(h1, edge_index)
            s_enhanced = self._act_list[i](op2(s_premiliar, h0, i + 1))

            states += [s_enhanced]

        if self._concat != 'lstm':
            start_index = 0
            for i in range(len(self._gnn_aggrs_name)):
                if self._gnn_aggrs_name[i] != 'identity':
                    out = states[i + 2]
                    start_index = i + 2
                    break
            for i in range(start_index + 1, len(states)):
                if self._gnn_aggrs_name[i - 2] != 'identity':
                    if self._concat == 'concat':
                        out = torch.cat([out, states[i]], dim=1)
                    if self._concat == "add":
                        out = torch.add(out, states[i])
                    elif self._concat == "product":
                        out = torch.mul(out, states[i])
                    elif self._concat == "max":
                        out = torch.max(out, states[i])
        else:
            _out = []
            start_index = 0
            for i in range(len(self._gnn_aggrs_name)):
                if self._gnn_aggrs_name[i] != 'identity':
                    _out.append(states[i + 2])
            out = self.jk_func(_out)
        return out


class GraphEhanced(nn.Module):
    def __init__(self, out_dim, residual=False):
        super(GraphEhanced, self).__init__()

        self.out_dim = out_dim
        self.residual = residual
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_1.data.uniform_(-stdv, stdv)

    def forward(self, last_input, h0, layer_index):
        assert last_input.size() == h0.size()

        mixed = last_input + h0
        output = mixed
        
        return output
        