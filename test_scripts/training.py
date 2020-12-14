import torch
from util import *
import bots
from models import *
import time
from plott import training_plotter

N_AGENTS = 8
N_TARGETS = 1
TIME_LIMIT = 40
HZ = 30 # fps of physics
COORD = "cartesian"
REW = "binary"
# COORD = "polar"
batch_size = 256

# MODEL
if COORD == "cartesian":
    edge_attr_size = 3
if COORD == "polar":
    edge_attr_size = 2
device = 'cpu'
Qnet = MyModel_4(edge_attr_size = edge_attr_size) #MyModel_1(mlp=fc1)
# Qnet.load_state_dict(torch.load("saved_models/M2_dqn",map_location=torch.device('cpu')))
Tnet = MyModel_4(edge_attr_size = edge_attr_size)
Tnet.load_state_dict(Qnet.state_dict())


Tnet.eval()
Qnet.to(device)
Tnet.to(device)
optimizer = torch.optim.Adam(Qnet.parameters())
# optimizer.load_state_dict(torch.load("saved_models/opt_M2_dqn"))

# COLLECT

dataloader = ReplayMemory(200000)
while len(dataloader)<10*batch_size:
    obs, actions, rewards = collect_data(model = Qnet,
                                         n_agents = [N_AGENTS],
                                         time_limit = TIME_LIMIT,
                                         epsilon = 1,
                                         teacher_bot = bots.flock,
                                         device = device,
                                         hz = HZ,
                                         coord = COORD, reward_mode = REW,
                                         verbose_display = True)
    sasr = list(zip(obs[:-1], actions, obs[1:], rewards))
    dataloader.push(sasr)

# iteration
from torch.nn import functional as F
mask = torch.ones(batch_size*(N_AGENTS+N_TARGETS)).bool()
mask[N_AGENTS::N_AGENTS+1] = False # TODO: this works only for 1 target
if Qnet.has_states:
    mask = 0 # uncomment if model has states
# loader = iter(DataLoader(sasr, batch_size=batch_size, shuffle=False))
#
logger = {"pred_q":[0],"rewards":[0], "loss":[]}
plotter = training_plotter(logger)
eps_st, eps_end = 0.3, 0.3
gamma = 0.999
train_steps = round(1e+7 / batch_size)
# time_limit x hz determines the amount of collected data.
# the constant at collect_x is the ratio (collected_sample / trained_sample)
collect_x = round(10 * (TIME_LIMIT*HZ) / batch_size)
update_target_x = int(3e+3 / batch_size)
simulate_x = round(3e+5 / batch_size)
plot_x = round(0.5*1e+5 / batch_size)
st = time.time()
for i in range(train_steps):
    # gamma = ((gamma_end-gamma_st)*i/train_steps) + gamma_st
    # if i < 1000:
    #     gamma = 0
    Qnet.train()
    states0, actions, states1, rewards = dataloader.sample(batch_size)
    state_action_values = Qnet(states0)[mask].gather(1, actions.view(-1,1))
    next_state_values = Tnet(states1)[mask].max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_values*gamma)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in Qnet.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    Qnet.eval()
    # v_collect_x = int(collect_x + 10 * np.cos(2 * i * 2 * np.pi / 1000))
    if (i % collect_x) == 0:
        # print("Step: %d"%i)
        # print("collecting...")
        # epsilon = ((eps_end - eps_st) * i / train_steps) + eps_st
        # epsilon = 1-logger["rewards"][-1]
        epsilon = 0.3
        # vepsilon = epsilon + 0.3 * np.sin(i * 2 * np.pi / 1000)
        obs, acts, rews = collect_data(model=Qnet,
                                             n_agents=[N_AGENTS],
                                             epsilon=epsilon,
                                             time_limit = TIME_LIMIT,
                                             device = device,
                                             hz = HZ,
                                             coord = COORD,
                                             reward_mode = REW)
        # TODO: sample from collected data
        sasr = list(zip(obs[:-1], acts, obs[1:], rews))
        dataloader.push(sasr)
    #
    if (i % update_target_x) == 0:
        Tnet.load_state_dict(Qnet.state_dict())

    if (i % simulate_x) == 0:
        print("Step: %d" % i)
        print(time.time()-st)
        print("Simulating at iter %d" % i)
        torch.save(Qnet.state_dict(), "saved_models/recent")
        simulate(n_agents = [N_AGENTS],
                 actors = [bots.flock]+ [GnnActor(Qnet, epsilon = 0.05) for i in range(N_AGENTS-1)] ,
                 time_limit=15, hz = HZ, coord=COORD,
                 printModel="GNN v3", printIteration=i,
                 )


    logger["pred_q"].append(torch.mean(next_state_values).item())
    logger["rewards"].append(torch.mean(rewards).item())
    logger["loss"].append(loss.item())


    if (i % plot_x) == 0:
        plotter(logger)

print("end of training")
print("%d seconds passed" %(time.time()-st))
print("Simulating at iter %d" % i)
Qnet.eval()
simulate(n_agents = [N_AGENTS],
         actors = [bots.flock]+ [GnnActor(Qnet, epsilon = 0.05) for i in range(N_AGENTS-1)] ,
         time_limit=10, Model = "GNN LSTM", Iteration = i)
