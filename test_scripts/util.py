import gym
import bots
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def simulate(n_agents=[3], actors=bots.flock, colors = None,
             time_limit=2, **kwargs):
    if not isinstance(actors, list):
        actors = [actors] * n_agents[0]

    render = True
    env = gym.make("gym_macm:cm-flock-v0", render=render,
                   n_agents=n_agents, actors=actors, colors = colors,
                   time_limit=time_limit)
    env.framework.Print(env.name)
    env.framework.updateProjection()
    for kw in kwargs:
        env.framework.Print(str(kw)+": "+str(kwargs[kw]))
    env.framework.run()


def collect_data(model,
                 n_agents=[3],
                 time_limit=15,
                 epsilon=None,
                 teacher_bot=None,
                 device='cpu'):
    ''' data info:
                Graph data is created using obs from environment. Node states are internal information
                 of agents. Agents may or may not have states, so during data collection, each model
                 write the state of their agents as <model_name> attr of data, inside of this
                 function. This is required for the training, if the data is
                 being collected for later use and sampled randomly.

                 During inference, actors keep track of their states already so no need to write it
                 into data.
            '''
    render = False
    env = gym.make("gym_macm:cm-flock-v0", render=render,
                   n_agents=n_agents, actors=None,
                   time_limit=time_limit, hz = 15)

    # if machines:
    #     assert(len(machines)==(n_agents[0]-1+int(teacher_bot==None)))
    #     for i,agent in enumerate(env.agents):
    #         agent.machine = machines[i]

    actions = []
    observations = []
    # obs_graphs = []
    rewards = []
    obs = obs_to_graph(env.obs, device=device)
    # TODO: for model in models:
    if model.has_states:
        setattr(obs, model._get_name(), model.get_empty_states(n_agents[0]))
    observations.append(obs)


    while not env.done:
        with torch.no_grad():
            if model.has_states:
                acts, next_states = model(obs)
                acts = acts[:n_agents[0]].max(1)[1]
            else:
                acts = model(obs)[:n_agents[0]].max(1)[1]

            if epsilon:
                r = torch.rand(n_agents[0], device=device).le(epsilon)
                acts[r] = torch.randint(0, 26, (n_agents[0],), device=device)[r]
            if teacher_bot:
                acts[0] = torch.tensor(flock_action_map(teacher_bot(env.obs),
                                           nn_to_env=False), device=device)
            actions.append(acts)
            acts = acts.cpu()
            acts = flock_action_map(acts)
            agent_ids = obs.ids[obs.ids.lt(n_agents[0])]
            acts = {ID.item(): acts[i] for i, ID in enumerate(agent_ids)}
            # acts = {ID: acts[i] for i,ID in enumerate(obs.ids)}
            obs, rews = env.step(acts)
            obs = obs_to_graph(env.obs, device=device)
            new_agent_ids = obs.ids[obs.ids.lt(n_agents[0])]
            if model.has_states:
                setattr(obs, model._get_name(), next_states)
            observations.append(obs)
            rewards.append(torch.tensor([rews[ID] for ID in rews], dtype=torch.float32,
                                        device=device))


    return observations, actions, rewards


def obs_to_graph(obs, complete=True, device='cpu'):
    "mode: closest, every node is connected to closest node of each type"

    edge_source = []
    edge_target = []
    edge_attr = []
    id_to_graph_ind = {i:i for i in range(len(obs))} if complete else {}
    x = [0]*len(obs) if complete else []

    for node_target_id in obs:
        if node_target_id not in id_to_graph_ind:
            id_to_graph_ind[node_target_id] = len(id_to_graph_ind)
            x.append(0)

        for node_source in obs[node_target_id]["nodes"]:
            if node_source["id"] not in id_to_graph_ind:
                id_to_graph_ind[node_source["id"]] = len(id_to_graph_ind)
                x.append(node_source["type"])

            edge_source.append(id_to_graph_ind[node_source["id"]])
            edge_target.append(id_to_graph_ind[node_target_id])
            edge_attr.append(node_source["position"])

        # edge_source.append(int(closest_node0["id"]))
        # edge_target.append(int(node_target))
        # edge_attr.append(closest_node0["position"])
        # x[closest_node0["id"]] = closest_node0["type"]
        # ids[closest_node0["id"]] = int(closest_node0["id"])

    # if complete:
    #     # hardcoded. # of agents + # of targets. TODO
    #     ids = [i for i in range(len(obs)+1)]
    #     x = [0]*len(obs) + [1]
    # else:
    #     edge_source, edge_target, graph_to_ext = reduce_node_indices(edge_source,
    #                                                                  edge_target)
    #     x = [x[graph_to_ext[i]] for i in graph_to_ext]
    #     ids = [ids[graph_to_ext[i]] for i in graph_to_ext]

    edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float).view(-1,1)
    ids = torch.tensor([i for i in id_to_graph_ind], dtype=torch.int16)

    data = Data(x=x, edge_index=edge_index, ids=ids, edge_attr=torch.tensor(edge_attr, dtype=torch.float))

    if device != 'cpu':
        data.to(device)

    return data

def reduce_node_indices(u, v):
    '''
    :param u: source node indices
    :param v: target node indices
    :return: reduced node indices (u, v) and dict to map them back to environment

    Models receive partial observation if they do not control all agents.
    This function reduces the indices. (1,3,4,6,9) --> (1,2,3,4,5), {2:'6', 5:'9'}
    '''

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
    def __init__(self, model, epsilon = None,
                 device = 'cpu'):
        self.model = model
        if model.has_states:
            self.state = model.get_empty_states(1)
            self.state_shape = self.state.shape
        self.epsilon = epsilon
        self.device = device
        self.action_size = 27

    def __call__(self, obs):
        data = obs_to_graph(obs, complete=False)
        if self.model.has_states:
            setattr(data, self.model._get_name(), self.state)
            out, self.state = self.model(data)
            # self.state = new_states.permute(1,0).unsqueeze(0)
        # env_to_graph = {graph_to_env[k]: k for k in graph_to_env}
        # my_id = next(iter(obs))
        else:
            out = self.model(data)[0]
        out = out.max(0)[1]
        if self.epsilon and random.random() < self.epsilon:
            # even if the action is random, should the state be updated normally
            out = torch.randint(0, self.action_size-1, (1,), device=self.device)
        out = flock_action_map(out)
        return out


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
        return np.where((map==actions).all(axis=1))[0]


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
        if batch_size != 1:
            s0 = Batch.from_data_list(s0)
            s1 = Batch.from_data_list(s1)
            a = torch.cat(a)
            r = torch.cat(r)
        else:
            s0 = s0[0]
            s1 = s1[0]
            a = a[0]
            r = r[0]
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

