import torch
from util import *
import bots
from models import *
import time
from plott import training_plotter


N_AGENTS = 8
N_TARGETS = 1
# MODEL
device = 'cpu'
# Qnet = MyModel_4(32,32) #MyModel_1(mlp=fc1)

# Tnet = MyModel_4(32,32)


Qnet = MyModel_4(32,32) #MyModel_1(mlp=fc1)
# Qnet.load_state_dict(torch.load("saved_models/M2_dqn",map_location=torch.device('cpu')))
Tnet = MyModel_4(32,32)
Tnet.load_state_dict(Qnet.state_dict())

# fc1 = Fc1(3,32)
# fc2 = Fc1(32,32)
# fc3 = Fc1(32,32)
# fc4 = FcLin(32,27)
# Qnet = MyModel_3(loc=fc1, dec=fc2, msg=fc3, final = fc4, aggr="add")
# Tnet = MyModel_3(loc=fc1, dec=fc2, msg=fc3, final = fc4, aggr="add")

Tnet.eval()
Qnet.to(device)
Tnet.to(device)
optimizer = torch.optim.Adam(Qnet.parameters())
# optimizer.load_state_dict(torch.load("saved_models/opt_M2_dqn"))

# COLLECT
dataloader = ReplayMemory(200000)
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
batch_size = 256
mask = torch.ones(batch_size*(N_AGENTS+N_TARGETS)).bool()
mask[N_AGENTS::N_AGENTS+1] = False # TODO: this works only for 1 target

# loader = iter(DataLoader(sasr, batch_size=batch_size, shuffle=False))

logger = {"pred_q":[], "rewards":[0], "loss":[]}
plotter = training_plotter(logger)
eps_st, eps_end = 1, 0
gamma_st, gamma_end = 0.99, 0.99
train_steps = round(1e+7 / batch_size)
# time_limit x hz determines the amount of collected data.
# the constant at collect_x is the ratio (collected_sample / trained_sample)
collect_x = round(2 * (TIME_LIMIT*HZ) / batch_size)
update_target_x = int(3e+3 / batch_size)
simulate_x = round(2e+5 / batch_size)
plot_x = round(1e+5 / batch_size)
st = time.time()
for i in range(train_steps):
    gamma = ((gamma_end-gamma_st)*i/train_steps) + gamma_st
    # if i < 1000:
    #     gamma = 0
    Qnet.train()
    states0, actions, states1, rewards = dataloader.sample(batch_size)
    state_action_values = Qnet(states0)[0].gather(1, actions.view(-1,1))
    next_state_values = Tnet(states1)[0].max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values*gamma) + rewards
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # for param in Qnet.parameters():
    #     param.grad.data.clamp_(-1, 1)
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
                                             device = device)
        # TODO: sample from collected data
        sasr = list(zip(obs[:-1], acts, obs[1:], rews))
        dataloader.push(sasr)

    if (i % update_target_x) == 0:
        Tnet.load_state_dict(Qnet.state_dict())

    if (i % simulate_x) == 0:
        print("Step: %d" % i)
        print(time.time()-st)
        print("Simulating at iter %d" % i)
        simulate(n_agents = [N_AGENTS],
                 actors = [bots.flock]+ [GnnActor(Qnet, epsilon = 0.05) for i in range(N_AGENTS-1)] ,
                 time_limit=10, Model = "GNN v3", Iteration = i)


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
#
# h0 = torch.zeros(8,32)
# c0 = torch.ones(8,32)
# h1 = 2*torch.ones(8,32)
# c1 = 3*torch.ones(8,32)
# h2 = 4*torch.ones(8,32)
# c2 = 5*torch.ones(8,32)
# ax = 1
# s0 = torch.stack((h0,c0),axis=ax)
# s1 = torch.stack((h1,c1),axis=ax)
# s2 = torch.stack((h2,c2),axis=ax)
# o0 = obs.clone()
# o1 = obs.clone()
# o2 = obs.clone()
# o0.states = s0
# o1.states = s1
# o2.states = s2
# b = Batch.from_data_list([o0,o1,o2])
