import gym
import bots
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def simulate(n_agents=[3], actors=bots.flock,
             colors = None, **kwargs):
    if not isinstance(actors, list):
        actors = [actors] * n_agents[0]

    pr_kwargs = {}
    for kw in list(kwargs):
        if kw.startswith("print"):
            pr_kwargs[kw[5:]] = kwargs[kw]
            kwargs.pop(kw)
    env = gym.make("gym_macm:cm-flock-v0", render=True,
                   n_agents=n_agents, actors=actors, colors = colors,
                   **kwargs)
    env.framework.Print(env.name)
    env.framework.updateProjection()
    for kw in pr_kwargs:
        env.framework.Print(str(kw)+": "+str(pr_kwargs[kw]), static = True)
    env.framework.run()


def collect_data(model,
                 n_agents=[3],
                 epsilon=None,
                 teacher_bot=None,
                 device='cpu', **kwargs):
    ''' data info:
                Graph data is created using obs from environment. Node states are internal information
                 of agents. Agents may or may not have states, so during data collection, each model
                 write the state of their agents as <model_name> attr of data, inside of this
                 function. This is required for the training, if the data is
                 being collected for later use and sampled randomly.

                 During inference, actors keep track of their states already so no need to write it
                 into data.
            '''
    env = gym.make("gym_macm:cm-flock-v0",
                   n_agents=n_agents, actors=None, **kwargs)


    actions = []
    observations = []
    rewards = []
    obs = obs_to_graph(env.obs, device=device)
    # TODO: for model in models: multiple model training
    if model.has_states:
        setattr(obs, model._get_name(), model.get_empty_states(n_agents[0]).to(device))
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
            rewards.append(torch.tensor([rews[ID.item()] for ID in agent_ids], dtype=torch.float32,
                                        device=device))

            obs = obs_to_graph(env.obs, device=device)
            new_agent_ids = obs.ids[obs.ids.lt(n_agents[0])]
            if model.has_states:
                setattr(obs, model._get_name(), next_states)
            observations.append(obs)



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

    edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float).view(-1,1)
    ids = torch.tensor([i for i in id_to_graph_ind], dtype=torch.int16)

    data = Data(x=x, edge_index=edge_index, ids=ids, edge_attr=torch.tensor(edge_attr, dtype=torch.float))

    if device != 'cpu':
        data.to(device)

    return data

class GnnActor:
    def __init__(self, model, epsilon = None,
                 device = 'cpu', p2c = False):
        self.model = model
        if model.has_states:
            self.state = model.get_empty_states(1)
            self.state_shape = self.state.shape
        self.epsilon = epsilon
        self.device = device
        self.action_size = 27
        self.polar2cart = p2c

    def __call__(self, obs):
        if self.polar2cart:
            for node in obs[next(iter(obs))]["nodes"]:
                r,t = node["position"]
                node["position"] = np.array([r,np.cos(t),np.sin(t)])
        data = obs_to_graph(obs, complete=False, device=self.device)
        if self.model.has_states:
            setattr(data, self.model._get_name(), self.state)
            out, self.state = self.model(data)
            # self.state = new_states.permute(1,0).unsqueeze(0)
        # env_to_graph = {graph_to_env[k]: k for k in graph_to_env}
        # my_id = next(iter(obs))
        else:
            out = self.model(data)[0]
        out = out.squeeze().max(0)[1]
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

    def __init__(self, capacity, normalize_rews = False,
                 sampling = "single steps", device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.normalize = normalize_rews
        self.ep_idx = [0]
        self.sampling = sampling

    def push(self, data):
        self.memory.extend(data)
        overflow = len(self.memory) - self.capacity
        if overflow > 0:
            self.ep_idx = [idx-overflow for idx in self.ep_idx if idx>=overflow]
            self.memory = self.memory[-self.capacity:]
        self.ep_idx.append(len(self.memory))

    def sample(self, batch_size):
        if self.sampling == "single steps":
            s0, a, s1, r = zip(*random.sample(self.memory, batch_size))
            if batch_size != 1:
                Bs0 = Batch.from_data_list(s0)
                Bs1 = Batch.from_data_list(s1)
                Ba = torch.cat(a)
                Br = torch.cat(r)
            else:
                s0 = s0[0]
                s1 = s1[0]
                a = a[0]
                r = r[0]
            if self.normalize:
                r -= torch.mean(r)
                r /= torch.std(r)
        if self.sampling == "episodes":
            idx = random.randint(1, len(self.ep_idx)-1)
            ep_b, ep_e = self.ep_idx[idx-1], self.ep_idx[idx]
            s0, a, s1, r = zip(*self.memory[ep_b:ep_e])
            Bs0 = Batch.from_data_list(s0)
            Bs1 = Batch.from_data_list(s1)
            Ba = torch.cat(a)
            Br = torch.cat(r)

        # if testing:
        #     return Bs0, Ba, Bs1, Br, s0, a, s1, r
        # else:
        # return Bs0, Ba, Bs1, Br, s0, a, s1, r
        return Bs0, Ba, Bs1, Br

    def __len__(self):
        return len(self.memory)

def pick_hz():
    return random.choices([8, 16, 32, 64], [8, 4, 2, 1])[0]
