import gym
import bots
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def simulate(n_agents=[3], actors=bots.flock,
             time_limit=2):
    if not isinstance(actors, list):
        actors = [actors] * n_agents[0]

    render = True
    env = gym.make("gym_macm:cm-flock-v0", render=render,
                   n_agents=n_agents, actors=actors,
                   time_limit=time_limit)
    env.framework.run()


def collect_data(model,
                 n_agents=[3],
                 time_limit=15,
                 epsilon=None,
                 device='cpu'):
    render = False
    env = gym.make("gym_macm:cm-flock-v0", render=render,
                   n_agents=n_agents, actors=None,
                   time_limit=time_limit)

    actions = []
    observations = []
    # obs_graphs = []
    rewards = []
    obs = obs_to_graph(env.obs, device=device)
    observations.append(obs)

    while not env.done:
        with torch.no_grad():
            acts = model(obs)[:n_agents[0]].max(1)[1]
            if epsilon:
                r = torch.rand(n_agents[0], device=device).le(epsilon)
                acts[r] = torch.randint(0, 26, (n_agents[0],), device=device)[r]
            actions.append(acts)
            acts = acts.cpu()
            acts = flock_action_map(acts)
            acts = {str(ID): acts[ID] for ID in range(n_agents[0])}
            obs, rews = env.step(acts)
            obs = obs_to_graph(env.obs, device=device)
            observations.append(obs)
            rewards.append(torch.tensor([rews[ID] for ID in rews], dtype=torch.float32,
                                        device=device))

    return observations, actions, rewards


def obs_to_graph(obs, complete=True, device='cpu'):
    "mode: closest, every node is connected to closest node of each type"
    "IDs: collect data for nodes only in IDs"
    edge_source = []
    edge_target = []
    edge_attr = []
    x = {}

    for node_target in obs:
        closest_dist = np.inf
        closest_node0 = None
        x[node_target] = 0
        for node_source in obs[node_target]["nodes"]:
            if node_source["type"] == 0:
                if node_source["position"][0] < closest_dist:
                    closest_dist = node_source["position"][0]
                    closest_node0 = node_source
            else:
                edge_source.append(int(node_source["id"]))
                edge_target.append(int(node_target))
                edge_attr.append(node_source["position"])
                x[node_source["id"]] = node_source["type"]

        edge_source.append(int(closest_node0["id"]))
        edge_target.append(int(node_target))
        edge_attr.append(closest_node0["position"])
        x[closest_node0["id"]] = closest_node0["type"]

    if complete:
        x = [[0]] * len(obs) + [[1]]
    else:
        edge_source, edge_target, graph_to_ext = reduce_node_indices(edge_source,
                                                                     edge_target)
        x = [[x[graph_to_ext[i]]] for i in graph_to_ext]

    edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_attr, dtype=torch.float))

    if device != 'cpu':
        data.to(device)

    if complete:
        return data  # ,graph_to_ext
    else:
        return data, graph_to_ext


def reduce_node_indices(u, v):
    uv = np.array(u + v)
    u = np.array(u)
    v = np.array(v)
    c = 0
    uuv = np.unique(uv)
    graph_to_external = {}
    for i in uuv:
        while c in uuv:
            graph_to_external[c] = str(c)
            c += 1
        if c > max(uuv):
            break
        if i >= len(uuv):
            graph_to_external[c] = str(i)
            u[u == i] = c
            v[v == i] = c
            c += 1
        else:
            graph_to_external[i] = str(i)
    return u, v, graph_to_external


class GnnActor:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        data, graph_to_env = obs_to_graph(obs, complete=False)
        env_to_graph = {graph_to_env[k]: k for k in graph_to_env}
        with torch.no_grad():
            out = self.model(data)
        out = flock_action_map((out.max(1)[1]))
        return out[env_to_graph[next(iter(obs))]]


def flock_action_map(actions, nn_to_env=True):
    map = np.array([np.array([0, 0, 0]), np.array([1, 0, 0]),
                    np.array([2, 0, 0]), np.array([0, 1, 0]),
                    np.array([1, 1, 0]), np.array([2, 1, 0]),
                    np.array([0, 2, 0]), np.array([1, 2, 0]),
                    np.array([2, 2, 0]), np.array([0, 0, 1]),
                    np.array([1, 0, 1]), np.array([2, 0, 1]),
                    np.array([0, 1, 1]), np.array([1, 1, 1]),
                    np.array([2, 1, 1]), np.array([0, 2, 1]),
                    np.array([1, 2, 1]), np.array([2, 2, 1]),
                    np.array([0, 0, 2]), np.array([1, 0, 2]),
                    np.array([2, 0, 2]), np.array([0, 1, 2]),
                    np.array([1, 1, 2]), np.array([2, 1, 2]),
                    np.array([0, 2, 2]), np.array([1, 2, 2]),
                    np.array([2, 2, 2])])
    if nn_to_env:
        return map[actions]
    else:
        pass


import random


class ReplayMemory(object):

    def __init__(self, capacity, normalize_rews=False,
                 device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.normalize = normalize_rews

    def push(self, data):
        self.memory.extend(data)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def sample(self, batch_size):
        s0, a, s1, r = zip(*random.sample(self.memory, batch_size))
        s0 = Batch.from_data_list(s0)
        s1 = Batch.from_data_list(s1)
        a = torch.cat(a)
        r = torch.cat(r)
        if self.normalize:
            r -= torch.mean(r)
            r /= torch.std(r)
        return s0, a, s1, r

    def __len__(self):
        return len(self.memory)

# class logger:
#     def __init__(self, *args):
#         self.logger = {str(arg):{"values":[],"marks":[]}
#                        for arg in args}
#     def __call__(self, *args):
#         for arg in args:

# def collect_data(n_agents=[3], actors=bots.flock):
#     if not isinstance(actors, list):
#         actors = [actors] * n_agents[0]
