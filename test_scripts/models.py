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
   
        
class Fc(torch.nn.Module):
    def __init__(self,a, b, c, d):
        super(Fc, self).__init__()
        self.fc1 = torch.nn.Linear(a, b)
        self.fc2 = torch.nn.Linear(b, c)
        self.fc3 = torch.nn.Linear(c, d)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
