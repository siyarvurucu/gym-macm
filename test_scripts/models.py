from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
import torch
from torch import Tensor
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
    def __init__(self, msg: Callable, dec: Callable, aggr: str = 'max', **kwargs):
        super(MyModel_2, self).__init__(aggr=aggr, **kwargs)
        self.dec = dec
        self.msg = msg
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg)
        reset(self.dec)

    def forward(self, data):  # x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
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

