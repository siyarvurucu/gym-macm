import numpy as np

def bot0(obs):
    # print(obs)
    # Attacks closest enemy
    enemies = [agent for agent in obs["agents"] if (agent["type"]==0)]
    if enemies:
        closest = enemies[0]
        for agent in enemies:
            if agent["position"][0] < closest["position"][0]:
                closest = agent
        rotation = np.sign(closest["position"][1]) + 1
        forward = int(np.abs(closest["position"][1])<(np.pi/5)) + 1
        attack = int(closest["position"][0] < 3)
        return np.array([forward, 1, rotation, attack])
    else:
        return idle()


def idle(obs=None):
    return np.array([1, 1, 1, 0])

def forward(obs=None):
    return np.array([2, 1, 1, 0])

def rotate(obs=None):
    return np.array([1, 1, 2, 0])

def diag(obs=None):
    return np.array([2, 2, 1, 0])

def circle(obs=None):
    if np.random.rand() <0.5:
        return np.array([2, 1, 2, 0])
    else:
        return np.array([2, 1, 1, 0])

def flock(obs):
    # print(obs)
    mode = "cartesian" # radial
    obs = list(obs.values())[0]
    targets = [node for node in obs["nodes"] if (node["type"]==1)]
    if targets:
        closest = targets[0]
        for target in targets:
            if target["position"][0] < target["position"][0]:
                closest = target
        if closest["position"][0]<1:
            return idle()[:3]
        else:
            if mode=="cartesian":
                rotation = np.sign(closest["position"][2]) + 1
                forward = int(closest["position"][1] > np.cos(np.pi/4)) + 1
            if mode=="raidal":
                rotation = np.sign(closest["position"][1]) + 1
                forward = int(np.abs(closest["position"][1]) < (np.pi / 4)) + 1
            return np.array([forward, 1, rotation])
    else:
        return idle()[:3]