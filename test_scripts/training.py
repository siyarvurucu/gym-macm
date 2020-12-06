import torch
from util import *
import bots
from models import MyModel_1, Fc1, Fc2, MyModel_2
import time

# MODEL
device = 'cpu'
fc1 = Fc1(3,32)
fc2 = Fc2(32,64,27)
Qnet = MyModel_2(msg=fc1, dec=fc2)
Tnet = MyModel_2(msg=fc1, dec=fc2)
Tnet.eval()
Qnet.to(device)
Tnet.to(device)
optimizer = torch.optim.Adam(Qnet.parameters())
gnn_actor = GnnActor(Qnet)

# COLLECT
dataloader = ReplayMemory(60000)
N_AGENTS = 8
N_TARGETS = 1
TIME_LIMIT = 25
HZ = 15 # fps of physics. same with 'hz' of gym.make
obs, actions, rewards = collect_data(model = Qnet,
                                     n_agents = [N_AGENTS],
                                     time_limit = TIME_LIMIT,
                                     epsilon = 1,
                                     teacher_bot = bots.flock,
                                     device = device)
sasr = list(zip(obs[:-1], actions, obs[1:], rewards))
dataloader.push(sasr)

# iteration
from torch.nn import functional as F
batch_size = 30
mask = torch.ones(batch_size*(N_AGENTS+N_TARGETS)).bool()
mask[N_AGENTS::N_AGENTS+1] = False # TODO: this works only for 1 target

# loader = iter(DataLoader(sasr, batch_size=batch_size, shuffle=False))

logger = {"pred_q":[], "rewards":[], "loss":[]}

train_steps = 5000
collect_x = int(20 * (TIME_LIMIT*N_AGENTS*HZ) / batch_size)
update_target_x = 100
simulate_x = 500
eps_st, eps_end = 1, 0
gamma_st, gamma_end = 0.9, 0.9

st = time.time()
for i in range(train_steps):
    gamma = ((gamma_end-gamma_st)*i/train_steps) + gamma_st
    if i < 1000:
        gamma = 0
    states0, actions, states1, rewards = dataloader.sample(batch_size)
    state_action_values = Qnet(states0)[mask].gather(1, actions.view(-1,1))
    next_state_values = Tnet(states1)[mask].max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values*gamma) + rewards
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in Qnet.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if (i % collect_x) == 0:
        print("Step: %d"%i)
        print("collecting...")
        epsilon = ((eps_end - eps_st) * i / train_steps) + eps_st
        obs, acts, rews = collect_data(model=Qnet,
                                             n_agents=[N_AGENTS],
                                             epsilon=epsilon,
                                             device = device)
        # TODO: sample from collected data
        sasr = list(zip(obs[:-1], acts, obs[1:], rews))
        dataloader.push(sasr)

    if (i % update_target_x) == 0:
        Tnet.load_state_dict(Qnet.state_dict())

    if (i % simulate_x) == 0:
        print("Step: %d" % i)
        print(time.time()-st)
        gnn_actor = GnnActor(Qnet)
        print("Simulating at iter %d"%i)
        simulate(n_agents = [N_AGENTS],
                 actors = [bots.flock]+ [gnn_actor]*(N_AGENTS-1),
                 time_limit=10)

    logger["pred_q"].append(torch.mean(next_state_values).detach())
    logger["rewards"].append(torch.mean(rewards).detach())
    logger["loss"].append(loss.detach())




