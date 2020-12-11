from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
import torch
from torch.nn import Linear
from torch import Tensor, tanh
import torch.nn.functional as F

class MyModel_1(MessagePassing):
    def __init__(self, mlp: Callable, aggr: str = 'add', **kwargs):
        super(MyModel_1, self).__init__(aggr=aggr, **kwargs)
        self.mlp = mlp
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)
        
    def forward(self, data):#x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """"""
        # print(x)
        # propagate_type: (x: PairTensor)
        return self.propagate(data.edge_index, x=data.x, edge_attr = data.edge_attr)
    
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class MyModel_2(MessagePassing):
    has_states = False # Does model have states that should be saved in the dataset?
    def __init__(self, aggr: str = 'add', **kwargs):
        super(MyModel_2, self).__init__(aggr=aggr, **kwargs)
        self.dec = Fc2(128,128,27)
        self.msg = Fc1(3,128)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg)
        reset(self.dec)

    def forward(self, data, actor_node = None):  # x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """"""
        # print(x)
        # propagate_type: (x: PairTensor)
        out = self.propagate(data.edge_index, x=data.x, edge_attr=data.edge_attr)
        return self.dec(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.msg(torch.cat([x_j, edge_attr], dim=-1))
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

class MyModel_3(MessagePassing):
    def __init__(self, loc: Callable, dec: Callable, msg:Callable,
                 final:Callable, aggr: str = 'add', **kwargs):
        super(MyModel_3, self).__init__(aggr=aggr, **kwargs)
        self.dec = dec
        self.msg = msg
        self.loc = loc
        self.final = final
        self.mtype = "msg1"
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg)
        reset(self.dec)

    def forward(self, data):  # x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """"""
        # print(x)
        # propagate_type: (x: PairTensor)
        self.mtype = "msg1"
        out = self.propagate(data.edge_index, x=data.x, edge_attr=data.edge_attr)
        out = self.dec(out)
        edge_index0 = data.edge_index[0][(data.edge_index[0][..., None] == torch.where(data.x==1)[0]).any(-1)]
        edge_index1 = data.edge_index[1][(data.edge_index[0][..., None] == torch.where(data.x==1)[0]).any(-1)]
        self.mtype = "msg2"
        out = self.propagate(torch.stack((edge_index0,edge_index1)), x=out)

        return self.final(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor = None) -> Tensor:
        if self.mtype == "msg1":
            return self.loc(torch.cat([x_j, edge_attr], dim=-1))
        if self.mtype == "msg2":
            return self.msg(x_i + x_j)
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor = None) -> Tensor:
        if self.mtype == "msg1":
            return self.loc(torch.cat([x_j, edge_attr], dim=-1))
        if self.mtype == "msg2":
            return self.msg(x_i + x_j)
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

#
class MyModel_4(MessagePassing):
    def __init__(self, isize=32, hsize=32,
                 edge_attr_size = 3, node_attr_size = 1,
                 aggr = None, **kwargs):
        super(MyModel_4, self).__init__(aggr=aggr, **kwargs)
        # self.dec = dec
        # curret: batch_size 1 only. todo nodes_belong
        self.has_states = True
        self.hsize = hsize
        self.rnn = torch.nn.LSTM(isize, hsize, 1)
        self.msg = Fc1(node_attr_size+edge_attr_size ,isize)
        self.dec = FcLin(hsize, 27)

    def get_empty_states(self, n_agents):
        # dims of rnn states = (num_layers, batch_size, features)
        # dims of torch_geometric.data.Batch = (batch_size, ...)
        # so it concats at dim=0. Therefore we provide rnn dims switched:
        # (batch,num_layers,features). We switch back when it reaches rnns.
        # This way reshape operation is avoided.
        # num_layers is 1, that dim is used to concat (hstate,cstate) of lstm
        return torch.zeros(n_agents,2,self.hsize)
    def forward(self, data, actor = None):  # x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """"""
        # if actor:
        #     actor = torch.where(self.env_ids == int(next(iter(actor))))[0].item()

            # data must provide states if it is sampled from a dataset. In simulation,
            # states are not provided since the model keeps track of states already
            # h,c = zip(*[data.states[node.item()] for node in self.env_ids])
            # self.h, self.c = torch.stack(h,axis=1), torch.stack(c,axis=1)
        out, (hn,cn) = self.propagate(data.edge_index, x=data.x,
                             edge_attr=data.edge_attr, actor = actor,
                             states = getattr(data,self._get_name()))
        # torch.cat((hn,cn),axis=1) for single
        next_states = torch.cat((hn.permute(1, 0, 2), cn.permute(1, 0, 2)), axis=1)
        # next_states = torch.stack((hn.squeeze(),cn.squeeze()),axis=1)
        return self.dec(out[-1]).squeeze(), next_states

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.msg(torch.cat([x_j, edge_attr], dim=-1))
        # print(x_j)
        # print(edge_attr)
        # for i, Id in enumerate(ids):
        # self.out,self.h,self.c= self.rnn(out[ids],(self.h,self.c))
        return out

    def aggregate(self, inputs: Tensor, index: Tensor, states: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        # TODO: look into batches with different sequence lengths
        # below assumes:
        # each node receives same number of messages
        # target nodes (index) are sorted: [0,1,2.. or [0,0,1,1,2,2...
        # BATCH
        # index.view(80, 2).permute(1, 0).shape
        # print(inputs)
        # print(states)
        out, (hn, cn) = self.rnn(inputs.view(states.shape[0], -1, self.hsize).permute(1, 0, 2),
                                 (states[:,0:1,:].permute(1, 0, 2),states[:,1:2,:].permute(1, 0, 2)))
        return out,(hn,cn)


class Fc2(torch.nn.Module):
    def __init__(self,a, b, c):
        super(Fc2, self).__init__()
        self.fc1 = torch.nn.Linear(a, b)
        self.fc2 = torch.nn.Linear(b, c)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

class Fc1(torch.nn.Module):
    def __init__(self,a, b):
        super(Fc1, self).__init__()
        self.fc1 = torch.nn.Linear(a, b)
    def forward(self, x):
        return torch.tanh(self.fc1(x))

class FcLin(torch.nn.Module):
    def __init__(self,a, b):
        super(FcLin, self).__init__()
        self.fc1 = torch.nn.Linear(a, b)
    def forward(self, x):
        return self.fc1(x)
