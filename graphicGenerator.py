import numpy as np
import matplotlib.pyplot as plt
import torch

# BATTLE ENVIRONMENT #
colours = {}

# DQN
colours['DQN'] = 'green'

DQNcheckpoints = {}
DQNcheckpoints['1'] = torch.load('./DQNAgents/Battle/RUN01_red93511ep250.tar')
DQNcheckpoints['2'] = torch.load('./DQNAgents/Battle/RUN02_red93483ep250.tar')
DQNcheckpoints['3'] = torch.load('./DQNAgents/Battle/RUN03_red119779ep250.tar')
DQNcheckpoints['4'] = torch.load('./DQNAgents/Battle/RUN04_red102937ep250.tar')
DQNcheckpoints['5'] = torch.load('./DQNAgents/Battle/RUN05_red125000ep250.tar')

DQNeps = {}
DQNcum_rewards = {}
DQNnum_steps = {}
DQNloss_nograd = {}
DQNtime = {}

for key in DQNcheckpoints.keys():
    DQNeps[key] = DQNcheckpoints[key]['episodes']
    DQNcum_rewards[key] = DQNcheckpoints[key]['cum_rewards']
    DQNnum_steps[key] = DQNcheckpoints[key]['num_steps']
    DQNloss_nograd[key] = DQNcheckpoints[key]['loss_nograd']
    DQNtime[key] = DQNcheckpoints[key]['time']['DQNAgents'][-1]
    for i in range(len(DQNcum_rewards[key])):
        if DQNloss_nograd[key][i] == None:
            DQNloss_nograd[key][i] = 0

del DQNloss_nograd['1'][0]

if DQNeps['1'] == DQNeps['2'] == DQNeps['3'] == DQNeps['4'] == DQNeps['5']:
    DQNeps['vector'] = np.arange(1, DQNeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

DQNcum_rewards['mean'] = np.mean((DQNcum_rewards['1'], DQNcum_rewards['2'], DQNcum_rewards['3'],
    DQNcum_rewards['4'], DQNcum_rewards['5']), axis=0)
DQNnum_steps['mean'] = np.mean((DQNnum_steps['1'], DQNnum_steps['2'], DQNnum_steps['3'],
    DQNnum_steps['4'], DQNnum_steps['5']), axis=0)
DQNloss_nograd['mean'] = np.mean((DQNloss_nograd['1'], DQNloss_nograd['2'], DQNloss_nograd['3'],
    DQNloss_nograd['4'], DQNloss_nograd['5']), axis=0)
DQNtime['mean'] = np.mean((DQNtime['1'], DQNtime['2'], DQNtime['3'], DQNtime['4'],
    DQNtime['5']), axis=0)

#DuelingDDQN
colours['DuelingDDQN'] = 'blue'

DuelingDDQNcheckpoints = {}
DuelingDDQNcheckpoints['1'] = torch.load('./DuelingDDQNAgents/Battle/RUN01_red105984ep250.tar')
DuelingDDQNcheckpoints['2'] = torch.load('./DuelingDDQNAgents/Battle/RUN02_red110045ep250.tar')
DuelingDDQNcheckpoints['3'] = torch.load('./DuelingDDQNAgents/Battle/RUN03_red109699ep250.tar')
DuelingDDQNcheckpoints['4'] = torch.load('./DuelingDDQNAgents/Battle/RUN04_red123793ep250.tar')
DuelingDDQNcheckpoints['5'] = torch.load('./DuelingDDQNAgents/Battle/RUN05_red118293ep250.tar')

DuelingDDQNeps = {}
DuelingDDQNcum_rewards = {}
DuelingDDQNnum_steps = {}
DuelingDDQNloss_nograd = {}
DuelingDDQNtime = {}

for key in DuelingDDQNcheckpoints.keys():
    DuelingDDQNeps[key] = DuelingDDQNcheckpoints[key]['episodes']
    DuelingDDQNcum_rewards[key] = DuelingDDQNcheckpoints[key]['cum_rewards']
    DuelingDDQNnum_steps[key] = DuelingDDQNcheckpoints[key]['num_steps']
    DuelingDDQNloss_nograd[key] = DuelingDDQNcheckpoints[key]['loss_nograd']
    DuelingDDQNtime[key] = DuelingDDQNcheckpoints[key]['time']['DuelingDDQNAgents'][-1]
    for i in range(len(DuelingDDQNcum_rewards[key])):
        if DuelingDDQNloss_nograd[key][i] == None:
            DuelingDDQNloss_nograd[key][i] = 0

if DuelingDDQNeps['1'] == DuelingDDQNeps['2'] == DuelingDDQNeps['3'] == DuelingDDQNeps['4'] == DuelingDDQNeps['5']:
    DuelingDDQNeps['vector'] = np.arange(1, DuelingDDQNeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

DuelingDDQNcum_rewards['mean'] = np.mean((DuelingDDQNcum_rewards['1'], DuelingDDQNcum_rewards['2'],
    DuelingDDQNcum_rewards['3'], DuelingDDQNcum_rewards['4'], DuelingDDQNcum_rewards['5']), axis=0)
DuelingDDQNnum_steps['mean'] = np.mean((DuelingDDQNnum_steps['1'], DuelingDDQNnum_steps['2'],
    DuelingDDQNnum_steps['3'], DuelingDDQNnum_steps['4'], DuelingDDQNnum_steps['5']), axis=0)
DuelingDDQNloss_nograd['mean'] = np.mean((DuelingDDQNloss_nograd['1'], DuelingDDQNloss_nograd['2'],
    DuelingDDQNloss_nograd['3'], DuelingDDQNloss_nograd['4'], DuelingDDQNloss_nograd['5']), axis=0)
DuelingDDQNtime['mean'] = np.mean((DuelingDDQNtime['1'], DuelingDDQNtime['2'],
    DuelingDDQNtime['3'], DuelingDDQNtime['4'], DuelingDDQNtime['5']), axis=0)

# A3C
colours['A3C'] = 'gold'

A3Ccheckpoints = {}
A3Ccheckpoints['1'] = torch.load('./A3CAgents/Battle/RUN01_red125000ep250.tar')
A3Ccheckpoints['2'] = torch.load('./A3CAgents/Battle/RUN02_red125000ep250.tar')
A3Ccheckpoints['3'] = torch.load('./A3CAgents/Battle/RUN03_red125000ep250.tar')
A3Ccheckpoints['4'] = torch.load('./A3CAgents/Battle/RUN04_red125000ep250.tar')
A3Ccheckpoints['5'] = torch.load('./A3CAgents/Battle/RUN05_red125000ep250.tar')

A3Ceps = {}
A3Ccum_rewards = {}
A3Cnum_steps = {}
A3Closs_nograd = {}
A3Ctime = {}

for key in A3Ccheckpoints.keys():
    A3Ceps[key] = A3Ccheckpoints[key]['episodes']
    A3Ccum_rewards[key] = A3Ccheckpoints[key]['cum_rewards']
    A3Cnum_steps[key] = A3Ccheckpoints[key]['num_steps']
    A3Closs_nograd[key] = A3Ccheckpoints[key]['loss_nograd']
    A3Ctime[key] = A3Ccheckpoints[key]['time']['A3CAgents'][-1]
    for i in range(len(A3Ccum_rewards[key])):
        if A3Closs_nograd[key][i] == None:
            A3Closs_nograd[key][i] = 0

if A3Ceps['1'] == A3Ceps['2'] == A3Ceps['3'] == A3Ceps['4'] == A3Ceps['5']:
    A3Ceps['vector'] = np.arange(1, A3Ceps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

A3Ccum_rewards['mean'] = np.mean((A3Ccum_rewards['1'], A3Ccum_rewards['2'],
    A3Ccum_rewards['3'], A3Ccum_rewards['4'], A3Ccum_rewards['5']), axis=0)
A3Cnum_steps['mean'] = np.mean((A3Cnum_steps['1'], A3Cnum_steps['2'],
    A3Cnum_steps['3'], A3Cnum_steps['4'], A3Cnum_steps['5']), axis=0)
A3Closs_nograd['mean'] = np.mean((A3Closs_nograd['1'], A3Closs_nograd['2'],
    A3Closs_nograd['3'], A3Closs_nograd['4'], A3Closs_nograd['5']), axis=0)
A3Ctime['mean'] = np.mean((A3Ctime['1'], A3Ctime['2'], A3Ctime['3'], A3Ctime['4'],
    A3Ctime['5']), axis=0)

# PPO
colours['PPO'] = 'red'

PPOcheckpoints = {}
PPOcheckpoints['1'] = torch.load('./PPOAgents/Battle/RUN01_actorred125000ep250.tar')
PPOcheckpoints['2'] = torch.load('./PPOAgents/Battle/RUN02_actorred125000ep250.tar')
PPOcheckpoints['3'] = torch.load('./PPOAgents/Battle/RUN03_actorred125000ep250.tar')
PPOcheckpoints['4'] = torch.load('./PPOAgents/Battle/RUN04_actorred125000ep250.tar')
PPOcheckpoints['5'] = torch.load('./PPOAgents/Battle/RUN05_actorred125000ep250.tar')

PPOeps = {}
PPOcum_rewards = {}
PPOnum_steps = {}
PPOloss_nograd = {}
PPOtime = {}

for key in PPOcheckpoints.keys():
    PPOeps[key] = PPOcheckpoints[key]['episodes']
    PPOcum_rewards[key] = PPOcheckpoints[key]['cum_rewards']
    PPOnum_steps[key] = PPOcheckpoints[key]['num_steps']
    PPOloss_nograd[key] = PPOcheckpoints[key]['loss_nograd']
    PPOtime[key] = PPOcheckpoints[key]['time']['PPOAgents'][-1]
    for i in range(len(PPOcum_rewards[key])):
        if PPOloss_nograd[key][i] == None:
            PPOloss_nograd[key][i] = 0

if PPOeps['1'] == PPOeps['2'] == PPOeps['3'] == PPOeps['4'] == PPOeps['5']:
    PPOeps['vector'] = np.arange(1, PPOeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

PPOcum_rewards['mean'] = np.mean((PPOcum_rewards['1'], PPOcum_rewards['2'],
    PPOcum_rewards['3'], PPOcum_rewards['4'], PPOcum_rewards['5']), axis=0)
PPOnum_steps['mean'] = np.mean((PPOnum_steps['1'], PPOnum_steps['2'],
    PPOnum_steps['3'], PPOnum_steps['4'], PPOnum_steps['5']), axis=0)
PPOloss_nograd['mean'] = np.mean((PPOloss_nograd['1'], PPOloss_nograd['2'],
    PPOloss_nograd['3'], PPOloss_nograd['4'], PPOloss_nograd['5']), axis=0)
PPOtime['mean'] = np.mean((PPOtime['1'], PPOtime['2'],
    PPOtime['3'], PPOtime['4'], PPOtime['5']), axis=0)

# P3O
colours['P3O'] = 'purple'
colours['P3O2'] = 'palevioletred'
colours['P3OM'] = 'magenta'

P3Ocheckpoints = {}
P3Ocheckpoints['1'] = torch.load('./P3OAgents/Battle/1 Net/RUN01_red125000ep250.tar')
P3Ocheckpoints['2'] = torch.load('./P3OAgents/Battle/1 Net/RUN02_red125000ep250.tar')
P3Ocheckpoints['3'] = torch.load('./P3OAgents/Battle/1 Net/RUN03_red125000ep250.tar')
P3Ocheckpoints['4'] = torch.load('./P3OAgents/Battle/1 Net/RUN04_red125000ep250.tar')
P3Ocheckpoints['5'] = torch.load('./P3OAgents/Battle/1 Net/RUN05_red125000ep250.tar')

P3O2checkpoints = {}
P3O2checkpoints['1'] = torch.load('./P3OAgents/Battle/2 Nets/RUN01_actorred125000ep250.tar')
P3O2checkpoints['2'] = torch.load('./P3OAgents/Battle/2 Nets/RUN02_actorred125000ep250.tar')
P3O2checkpoints['3'] = torch.load('./P3OAgents/Battle/2 Nets/RUN03_actorred125000ep250.tar')
P3O2checkpoints['4'] = torch.load('./P3OAgents/Battle/2 Nets/RUN04_actorred125000ep250.tar')
P3O2checkpoints['5'] = torch.load('./P3OAgents/Battle/2 Nets/RUN05_actorred125000ep250.tar')

P3OMcheckpoints = {}
P3OMcheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_0actorred125000ep250.tar')
P3OMcheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_0actorred125000ep250.tar')
P3OMcheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_0actorred125000ep250.tar')
P3OMcheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_0actorred125000ep250.tar')
P3OMcheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_0actorred125000ep250.tar')

P3Oeps = {}
P3Ocum_rewards = {}
P3Onum_steps = {}
P3Oloss_nograd = {}
P3Otime = {}

P3O2cum_rewards = {}
P3O2num_steps = {}
P3O2time = {}

P3OMcum_rewards = {}
P3OMnum_steps = {}
P3OMtime = {}

for key in P3Ocheckpoints.keys():
    P3Oeps[key] = P3Ocheckpoints[key]['episodes']
    P3Ocum_rewards[key] = P3Ocheckpoints[key]['cum_rewards']
    P3Onum_steps[key] = P3Ocheckpoints[key]['num_steps']
    P3Oloss_nograd[key] = P3Ocheckpoints[key]['loss_nograd']
    P3Otime[key] = P3Ocheckpoints[key]['time']['P3OAgents'][-1]

    P3O2cum_rewards[key] = P3O2checkpoints[key]['cum_rewards']
    P3O2num_steps[key] = P3O2checkpoints[key]['num_steps']
    P3O2time[key] = P3O2checkpoints[key]['time']['P3OAgents'][-1]

    P3OMcum_rewards[key] = P3OMcheckpoints[key]['cum_rewards']
    P3OMnum_steps[key] = P3OMcheckpoints[key]['num_steps']
    P3OMtime[key] = P3OMcheckpoints[key]['time']['P3OAgents'][-1]

    for i in range(len(P3Ocum_rewards[key])):
        if P3Oloss_nograd[key][i] == None:
            P3Oloss_nograd[key][i] = 0

if P3Oeps['1'] == P3Oeps['2'] == P3Oeps['3'] == P3Oeps['4'] == P3Oeps['5']:
    P3Oeps['vector'] = np.arange(1, P3Oeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

P3Ocum_rewards['mean'] = np.mean((P3Ocum_rewards['1'], P3Ocum_rewards['2'],
    P3Ocum_rewards['3'], P3Ocum_rewards['4'], P3Ocum_rewards['5']), axis=0)
P3Onum_steps['mean'] = np.mean((P3Onum_steps['1'], P3Onum_steps['2'],
    P3Onum_steps['3'], P3Onum_steps['4'], P3Onum_steps['5']), axis=0)
P3Oloss_nograd['mean'] = np.mean((P3Oloss_nograd['1'], P3Oloss_nograd['2'],
    P3Oloss_nograd['3'], P3Oloss_nograd['4'], P3Oloss_nograd['5']), axis=0)
P3Otime['mean'] = np.mean((P3Otime['1'], P3Otime['2'], P3Otime['3'], P3Otime['4'],
    P3Otime['5']), axis=0)

P3O2cum_rewards['mean'] = np.mean((P3O2cum_rewards['1'], P3O2cum_rewards['2'],
    P3O2cum_rewards['3'], P3O2cum_rewards['4'], P3O2cum_rewards['5']), axis=0)
P3O2num_steps['mean'] = np.mean((P3O2num_steps['1'], P3O2num_steps['2'],
    P3O2num_steps['3'], P3O2num_steps['4'], P3O2num_steps['5']), axis=0)
P3O2time['mean'] = np.mean((P3O2time['1'], P3O2time['2'], P3O2time['3'], P3O2time['4'],
    P3O2time['5']), axis=0)

P3OMcum_rewards['mean'] = np.mean((P3OMcum_rewards['1'], P3OMcum_rewards['2'],
    P3OMcum_rewards['3'], P3OMcum_rewards['4'], P3OMcum_rewards['5']), axis=0)
P3OMnum_steps['mean'] = np.mean((P3OMnum_steps['1'], P3OMnum_steps['2'],
    P3OMnum_steps['3'], P3OMnum_steps['4'], P3OMnum_steps['5']), axis=0)
P3OMtime['mean'] = np.mean((P3OMtime['1'], P3OMtime['2'], P3OMtime['3'], P3OMtime['4'],
    P3OMtime['5']), axis=0)

# plotting figures
figRewards = plt.figure("Cumulative Rewards")
plt.plot(DQNeps['vector'], DQNcum_rewards['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNcum_rewards['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Ccum_rewards['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['5'], color=colours['A3C'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOcum_rewards['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['5'], color=colours['PPO'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Ocum_rewards['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['5'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2cum_rewards['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['mean'], color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMcum_rewards['1'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['2'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['3'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['4'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['5'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Cumulative rewards")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battleCumulativeRewards.png', bbox_inches='tight')

figNumSteps = plt.figure("Number of steps")
plt.plot(DQNeps['vector'], DQNnum_steps['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNnum_steps['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Cnum_steps['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['5'], color=colours['A3C'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOnum_steps['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['5'], color=colours['PPO'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Onum_steps['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['5'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2num_steps['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['mean'], color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMnum_steps['1'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['2'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['3'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['4'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['5'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Number of steps")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battleNumSteps.png', bbox_inches='tight')

figLoss = plt.figure("Loss")
plt.plot(DQNeps['vector'], DQNloss_nograd['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNloss_nograd['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Closs_nograd['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['5'], color=colours['A3C'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Oloss_nograd['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['5'], color=colours['P3O'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battleLoss.png', bbox_inches='tight')

print('Battle Computational Time Table')
print('==================================')
print(f"DQN: {DQNtime['mean']} sec")
print(f"Dueling DDQN: {DuelingDDQNtime['mean']} sec")
print(f"A3C: {A3Ctime['mean']} sec")
print(f"PPO: {PPOtime['mean']} sec")
print(f"P3O: {P3Otime['mean']} sec")
print(f"P3O 2 Nets: {P3O2time['mean']} sec")
print(f"P3O Multi-Nets: {P3OMtime['mean']} sec \n")


PPOcritiCheckpoints = {}
PPOcritiCheckpoints['1'] = torch.load('./PPOAgents/Battle/RUN01_criticred125000ep250.tar')
PPOcritiCheckpoints['2'] = torch.load('./PPOAgents/Battle/RUN02_criticred125000ep250.tar')
PPOcritiCheckpoints['3'] = torch.load('./PPOAgents/Battle/RUN03_criticred125000ep250.tar')
PPOcritiCheckpoints['4'] = torch.load('./PPOAgents/Battle/RUN04_criticred125000ep250.tar')
PPOcritiCheckpoints['5'] = torch.load('./PPOAgents/Battle/RUN05_criticred125000ep250.tar')

PPOcriticLoss_nograd = {}
for key in PPOcritiCheckpoints.keys():
    PPOcriticLoss_nograd[key] = PPOcritiCheckpoints[key]['loss_nograd']
    for i in range(len(PPOcum_rewards[key])):
        if PPOcriticLoss_nograd[key][i] == None:
            PPOcriticLoss_nograd[key][i] = 0

PPOcriticLoss_nograd['mean'] = np.mean((PPOcriticLoss_nograd['1'], PPOcriticLoss_nograd['2'],
    PPOcriticLoss_nograd['3'], PPOcriticLoss_nograd['4'], PPOcriticLoss_nograd['5']), axis=0)

P3O2critiCheckpoints = {}
P3O2critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/2 Nets/RUN01_criticred125000ep250.tar')
P3O2critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/2 Nets/RUN02_criticred125000ep250.tar')
P3O2critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/2 Nets/RUN03_criticred125000ep250.tar')
P3O2critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/2 Nets/RUN04_criticred125000ep250.tar')
P3O2critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/2 Nets/RUN05_criticred125000ep250.tar')

P3O2loss_nograd = {}
P3O2criticLoss_nograd = {}
for key in P3O2critiCheckpoints.keys():
    P3O2loss_nograd[key] = P3O2checkpoints[key]['loss_nograd']
    P3O2criticLoss_nograd[key] = P3O2critiCheckpoints[key]['loss_nograd']
    for i in range(len(P3O2cum_rewards[key])):
        if P3O2loss_nograd[key][i] == None:
            P3O2loss_nograd[key][i] = 0
        if P3O2criticLoss_nograd[key][i] == None:
            P3O2criticLoss_nograd[key][i] = 0

P3O2loss_nograd['mean'] = np.mean((P3O2loss_nograd['1'], P3O2loss_nograd['2'],
    P3O2loss_nograd['3'], P3O2loss_nograd['4'], P3O2loss_nograd['5']), axis=0)
P3O2criticLoss_nograd['mean'] = np.mean((P3O2criticLoss_nograd['1'], P3O2criticLoss_nograd['2'],
    P3O2criticLoss_nograd['3'], P3O2criticLoss_nograd['4'], P3O2criticLoss_nograd['5']), axis=0)

P3OM0critiCheckpoints = {}
P3OM0critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_0criticred125000ep250.tar')

P3OMloss_nograd = {}
P3OM0criticLoss_nograd = {}

for key in P3OM0critiCheckpoints.keys():
    P3OMloss_nograd[key] = P3OMcheckpoints[key]['loss_nograd']
    P3OM0criticLoss_nograd[key] = P3OM0critiCheckpoints[key]['loss_nograd']
    for i in range(len(P3OMcum_rewards[key])):
        if P3OMloss_nograd[key][i] == None:
            P3OMloss_nograd[key][i] = 0
        if P3OM0criticLoss_nograd[key][i] == None:
            P3OM0criticLoss_nograd[key][i] = 0

P3OMloss_nograd['mean'] = np.mean((P3OMloss_nograd['1'], P3OMloss_nograd['2'],
    P3OMloss_nograd['3'], P3OMloss_nograd['4'], P3OMloss_nograd['5']), axis=0)
P3OM0criticLoss_nograd['mean'] = np.mean((P3OM0criticLoss_nograd['1'], P3OM0criticLoss_nograd['2'],
    P3OM0criticLoss_nograd['3'], P3OM0criticLoss_nograd['4'], P3OM0criticLoss_nograd['5']), axis=0)

P3OM1checkpoints = {}
P3OM1checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_1actorred125000ep250.tar')
P3OM1checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_1actorred125000ep250.tar')
P3OM1checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_1actorred125000ep250.tar')
P3OM1checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_1actorred125000ep250.tar')
P3OM1checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_1actorred125000ep250.tar')

P3OM1critiCheckpoints = {}
P3OM1critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_1criticred125000ep250.tar')

P3OM1loss_nograd = {}
P3OM1criticLoss_nograd = {}

for key in P3OM1critiCheckpoints.keys():
    P3OM1loss_nograd[key] = P3OM1checkpoints[key]['loss_nograd']
    P3OM1criticLoss_nograd[key] = P3OM1critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM1loss_nograd[key][i] == None:
            P3OM1loss_nograd[key][i] = 0
        if P3OM1criticLoss_nograd[key][i] == None:
            P3OM1criticLoss_nograd[key][i] = 0

P3OM1loss_nograd['mean'] = np.mean((P3OM1loss_nograd['1'], P3OM1loss_nograd['2'],
    P3OM1loss_nograd['3'], P3OM1loss_nograd['4'], P3OM1loss_nograd['5']), axis=0)
P3OM1criticLoss_nograd['mean'] = np.mean((P3OM1criticLoss_nograd['1'], P3OM1criticLoss_nograd['2'],
    P3OM1criticLoss_nograd['3'], P3OM1criticLoss_nograd['4'], P3OM1criticLoss_nograd['5']), axis=0)

P3OM2checkpoints = {}
P3OM2checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_2actorred125000ep250.tar')
P3OM2checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_2actorred125000ep250.tar')
P3OM2checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_2actorred125000ep250.tar')
P3OM2checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_2actorred125000ep250.tar')
P3OM2checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_2actorred125000ep250.tar')

P3OM2critiCheckpoints = {}
P3OM2critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_2criticred125000ep250.tar')

P3OM2loss_nograd = {}
P3OM2criticLoss_nograd = {}

for key in P3OM2critiCheckpoints.keys():
    P3OM2loss_nograd[key] = P3OM2checkpoints[key]['loss_nograd']
    P3OM2criticLoss_nograd[key] = P3OM2critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM2loss_nograd[key][i] == None:
            P3OM2loss_nograd[key][i] = 0
        if P3OM2criticLoss_nograd[key][i] == None:
            P3OM2criticLoss_nograd[key][i] = 0

P3OM2loss_nograd['mean'] = np.mean((P3OM2loss_nograd['1'], P3OM2loss_nograd['2'],
    P3OM2loss_nograd['3'], P3OM2loss_nograd['4'], P3OM2loss_nograd['5']), axis=0)
P3OM2criticLoss_nograd['mean'] = np.mean((P3OM2criticLoss_nograd['1'], P3OM2criticLoss_nograd['2'],
    P3OM2criticLoss_nograd['3'], P3OM2criticLoss_nograd['4'], P3OM2criticLoss_nograd['5']), axis=0)

P3OM3checkpoints = {}
P3OM3checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_3actorred125000ep250.tar')
P3OM3checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_3actorred125000ep250.tar')
P3OM3checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_3actorred125000ep250.tar')
P3OM3checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_3actorred125000ep250.tar')
P3OM3checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_3actorred125000ep250.tar')

P3OM3critiCheckpoints = {}
P3OM3critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_3criticred125000ep250.tar')

P3OM3loss_nograd = {}
P3OM3criticLoss_nograd = {}

for key in P3OM3critiCheckpoints.keys():
    P3OM3loss_nograd[key] = P3OM3checkpoints[key]['loss_nograd']
    P3OM3criticLoss_nograd[key] = P3OM3critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM3loss_nograd[key][i] == None:
            P3OM3loss_nograd[key][i] = 0
        if P3OM3criticLoss_nograd[key][i] == None:
            P3OM3criticLoss_nograd[key][i] = 0

P3OM3loss_nograd['mean'] = np.mean((P3OM3loss_nograd['1'], P3OM3loss_nograd['2'],
    P3OM3loss_nograd['3'], P3OM3loss_nograd['4'], P3OM3loss_nograd['5']), axis=0)
P3OM3criticLoss_nograd['mean'] = np.mean((P3OM3criticLoss_nograd['1'], P3OM3criticLoss_nograd['2'],
    P3OM3criticLoss_nograd['3'], P3OM3criticLoss_nograd['4'], P3OM3criticLoss_nograd['5']), axis=0)

P3OM4checkpoints = {}
P3OM4checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_4actorred125000ep250.tar')
P3OM4checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_4actorred125000ep250.tar')
P3OM4checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_4actorred125000ep250.tar')
P3OM4checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_4actorred125000ep250.tar')
P3OM4checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_4actorred125000ep250.tar')

P3OM4critiCheckpoints = {}
P3OM4critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_4criticred125000ep250.tar')

P3OM4loss_nograd = {}
P3OM4criticLoss_nograd = {}

for key in P3OM4critiCheckpoints.keys():
    P3OM4loss_nograd[key] = P3OM4checkpoints[key]['loss_nograd']
    P3OM4criticLoss_nograd[key] = P3OM4critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM4loss_nograd[key][i] == None:
            P3OM4loss_nograd[key][i] = 0
        if P3OM4criticLoss_nograd[key][i] == None:
            P3OM4criticLoss_nograd[key][i] = 0

P3OM4loss_nograd['mean'] = np.mean((P3OM4loss_nograd['1'], P3OM4loss_nograd['2'],
    P3OM4loss_nograd['3'], P3OM4loss_nograd['4'], P3OM4loss_nograd['5']), axis=0)
P3OM4criticLoss_nograd['mean'] = np.mean((P3OM4criticLoss_nograd['1'], P3OM4criticLoss_nograd['2'],
    P3OM4criticLoss_nograd['3'], P3OM4criticLoss_nograd['4'], P3OM4criticLoss_nograd['5']), axis=0)

P3OM5checkpoints = {}
P3OM5checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_5actorred125000ep250.tar')
P3OM5checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_5actorred125000ep250.tar')
P3OM5checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_5actorred125000ep250.tar')
P3OM5checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_5actorred125000ep250.tar')
P3OM5checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_5actorred125000ep250.tar')

P3OM5critiCheckpoints = {}
P3OM5critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_5criticred125000ep250.tar')

P3OM5loss_nograd = {}
P3OM5criticLoss_nograd = {}

for key in P3OM5critiCheckpoints.keys():
    P3OM5loss_nograd[key] = P3OM5checkpoints[key]['loss_nograd']
    P3OM5criticLoss_nograd[key] = P3OM5critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM5loss_nograd[key][i] == None:
            P3OM5loss_nograd[key][i] = 0
        if P3OM5criticLoss_nograd[key][i] == None:
            P3OM5criticLoss_nograd[key][i] = 0

P3OM5loss_nograd['mean'] = np.mean((P3OM5loss_nograd['1'], P3OM5loss_nograd['2'],
    P3OM5loss_nograd['3'], P3OM5loss_nograd['4'], P3OM5loss_nograd['5']), axis=0)
P3OM5criticLoss_nograd['mean'] = np.mean((P3OM5criticLoss_nograd['1'], P3OM5criticLoss_nograd['2'],
    P3OM5criticLoss_nograd['3'], P3OM5criticLoss_nograd['4'], P3OM5criticLoss_nograd['5']), axis=0)

P3OM6checkpoints = {}
P3OM6checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_6actorred125000ep250.tar')
P3OM6checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_6actorred125000ep250.tar')
P3OM6checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_6actorred125000ep250.tar')
P3OM6checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_6actorred125000ep250.tar')
P3OM6checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_6actorred125000ep250.tar')

P3OM6critiCheckpoints = {}
P3OM6critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_6criticred125000ep250.tar')

P3OM6loss_nograd = {}
P3OM6criticLoss_nograd = {}

for key in P3OM6critiCheckpoints.keys():
    P3OM6loss_nograd[key] = P3OM6checkpoints[key]['loss_nograd']
    P3OM6criticLoss_nograd[key] = P3OM6critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM6loss_nograd[key][i] == None:
            P3OM6loss_nograd[key][i] = 0
        if P3OM6criticLoss_nograd[key][i] == None:
            P3OM6criticLoss_nograd[key][i] = 0

P3OM6loss_nograd['mean'] = np.mean((P3OM6loss_nograd['1'], P3OM6loss_nograd['2'],
    P3OM6loss_nograd['3'], P3OM6loss_nograd['4'], P3OM6loss_nograd['5']), axis=0)
P3OM6criticLoss_nograd['mean'] = np.mean((P3OM6criticLoss_nograd['1'], P3OM6criticLoss_nograd['2'],
    P3OM6criticLoss_nograd['3'], P3OM6criticLoss_nograd['4'], P3OM6criticLoss_nograd['5']), axis=0)

P3OM7checkpoints = {}
P3OM7checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_7actorred125000ep250.tar')
P3OM7checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_7actorred125000ep250.tar')
P3OM7checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_7actorred125000ep250.tar')
P3OM7checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_7actorred125000ep250.tar')
P3OM7checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_7actorred125000ep250.tar')

P3OM7critiCheckpoints = {}
P3OM7critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_7criticred125000ep250.tar')

P3OM7loss_nograd = {}
P3OM7criticLoss_nograd = {}

for key in P3OM7critiCheckpoints.keys():
    P3OM7loss_nograd[key] = P3OM7checkpoints[key]['loss_nograd']
    P3OM7criticLoss_nograd[key] = P3OM7critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM7loss_nograd[key][i] == None:
            P3OM7loss_nograd[key][i] = 0
        if P3OM7criticLoss_nograd[key][i] == None:
            P3OM7criticLoss_nograd[key][i] = 0

P3OM7loss_nograd['mean'] = np.mean((P3OM7loss_nograd['1'], P3OM7loss_nograd['2'],
    P3OM7loss_nograd['3'], P3OM7loss_nograd['4'], P3OM7loss_nograd['5']), axis=0)
P3OM7criticLoss_nograd['mean'] = np.mean((P3OM7criticLoss_nograd['1'], P3OM7criticLoss_nograd['2'],
    P3OM7criticLoss_nograd['3'], P3OM7criticLoss_nograd['4'], P3OM7criticLoss_nograd['5']), axis=0)

P3OM8checkpoints = {}
P3OM8checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_8actorred125000ep250.tar')
P3OM8checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_8actorred125000ep250.tar')
P3OM8checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_8actorred125000ep250.tar')
P3OM8checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_8actorred125000ep250.tar')
P3OM8checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_8actorred125000ep250.tar')

P3OM8critiCheckpoints = {}
P3OM8critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_8criticred125000ep250.tar')

P3OM8loss_nograd = {}
P3OM8criticLoss_nograd = {}

for key in P3OM8critiCheckpoints.keys():
    P3OM8loss_nograd[key] = P3OM8checkpoints[key]['loss_nograd']
    P3OM8criticLoss_nograd[key] = P3OM8critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM8loss_nograd[key][i] == None:
            P3OM8loss_nograd[key][i] = 0
        if P3OM8criticLoss_nograd[key][i] == None:
            P3OM8criticLoss_nograd[key][i] = 0

P3OM8loss_nograd['mean'] = np.mean((P3OM8loss_nograd['1'], P3OM8loss_nograd['2'],
    P3OM8loss_nograd['3'], P3OM8loss_nograd['4'], P3OM8loss_nograd['5']), axis=0)
P3OM8criticLoss_nograd['mean'] = np.mean((P3OM8criticLoss_nograd['1'], P3OM8criticLoss_nograd['2'],
    P3OM8criticLoss_nograd['3'], P3OM8criticLoss_nograd['4'], P3OM8criticLoss_nograd['5']), axis=0)

P3OM9checkpoints = {}
P3OM9checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_9actorred125000ep250.tar')
P3OM9checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_9actorred125000ep250.tar')
P3OM9checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_9actorred125000ep250.tar')
P3OM9checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_9actorred125000ep250.tar')
P3OM9checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_9actorred125000ep250.tar')

P3OM9critiCheckpoints = {}
P3OM9critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_9criticred125000ep250.tar')

P3OM9loss_nograd = {}
P3OM9criticLoss_nograd = {}

for key in P3OM9critiCheckpoints.keys():
    P3OM9loss_nograd[key] = P3OM9checkpoints[key]['loss_nograd']
    P3OM9criticLoss_nograd[key] = P3OM9critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM9loss_nograd[key][i] == None:
            P3OM9loss_nograd[key][i] = 0
        if P3OM9criticLoss_nograd[key][i] == None:
            P3OM9criticLoss_nograd[key][i] = 0

P3OM9loss_nograd['mean'] = np.mean((P3OM9loss_nograd['1'], P3OM9loss_nograd['2'],
    P3OM9loss_nograd['3'], P3OM9loss_nograd['4'], P3OM9loss_nograd['5']), axis=0)
P3OM9criticLoss_nograd['mean'] = np.mean((P3OM9criticLoss_nograd['1'], P3OM9criticLoss_nograd['2'],
    P3OM9criticLoss_nograd['3'], P3OM9criticLoss_nograd['4'], P3OM9criticLoss_nograd['5']), axis=0)

P3OM10checkpoints = {}
P3OM10checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_10actorred125000ep250.tar')
P3OM10checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_10actorred125000ep250.tar')
P3OM10checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_10actorred125000ep250.tar')
P3OM10checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_10actorred125000ep250.tar')
P3OM10checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_10actorred125000ep250.tar')

P3OM10critiCheckpoints = {}
P3OM10critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_10criticred125000ep250.tar')

P3OM10loss_nograd = {}
P3OM10criticLoss_nograd = {}

for key in P3OM10critiCheckpoints.keys():
    P3OM10loss_nograd[key] = P3OM10checkpoints[key]['loss_nograd']
    P3OM10criticLoss_nograd[key] = P3OM10critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM10loss_nograd[key][i] == None:
            P3OM10loss_nograd[key][i] = 0
        if P3OM10criticLoss_nograd[key][i] == None:
            P3OM10criticLoss_nograd[key][i] = 0

P3OM10loss_nograd['mean'] = np.mean((P3OM10loss_nograd['1'], P3OM10loss_nograd['2'],
    P3OM10loss_nograd['3'], P3OM10loss_nograd['4'], P3OM10loss_nograd['5']), axis=0)
P3OM10criticLoss_nograd['mean'] = np.mean((P3OM10criticLoss_nograd['1'], P3OM10criticLoss_nograd['2'],
    P3OM10criticLoss_nograd['3'], P3OM10criticLoss_nograd['4'], P3OM10criticLoss_nograd['5']), axis=0)

P3OM11checkpoints = {}
P3OM11checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_11actorred125000ep250.tar')
P3OM11checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_11actorred125000ep250.tar')
P3OM11checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_11actorred125000ep250.tar')
P3OM11checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_11actorred125000ep250.tar')
P3OM11checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_11actorred125000ep250.tar')

P3OM11critiCheckpoints = {}
P3OM11critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_11criticred125000ep250.tar')

P3OM11loss_nograd = {}
P3OM11criticLoss_nograd = {}

for key in P3OM11critiCheckpoints.keys():
    P3OM11loss_nograd[key] = P3OM11checkpoints[key]['loss_nograd']
    P3OM11criticLoss_nograd[key] = P3OM11critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM11loss_nograd[key][i] == None:
            P3OM11loss_nograd[key][i] = 0
        if P3OM11criticLoss_nograd[key][i] == None:
            P3OM11criticLoss_nograd[key][i] = 0

P3OM11loss_nograd['mean'] = np.mean((P3OM11loss_nograd['1'], P3OM11loss_nograd['2'],
    P3OM11loss_nograd['3'], P3OM11loss_nograd['4'], P3OM11loss_nograd['5']), axis=0)
P3OM11criticLoss_nograd['mean'] = np.mean((P3OM11criticLoss_nograd['1'], P3OM11criticLoss_nograd['2'],
    P3OM11criticLoss_nograd['3'], P3OM11criticLoss_nograd['4'], P3OM11criticLoss_nograd['5']), axis=0)

P3OM12checkpoints = {}
P3OM12checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_12actorred125000ep250.tar')
P3OM12checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_12actorred125000ep250.tar')
P3OM12checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_12actorred125000ep250.tar')
P3OM12checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_12actorred125000ep250.tar')
P3OM12checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_12actorred125000ep250.tar')

P3OM12critiCheckpoints = {}
P3OM12critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_12criticred125000ep250.tar')
P3OM12critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_12criticred125000ep250.tar')
P3OM12critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_12criticred125000ep250.tar')
P3OM12critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_12criticred125000ep250.tar')
P3OM12critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_12criticred125000ep250.tar')

P3OM12loss_nograd = {}
P3OM12criticLoss_nograd = {}

for key in P3OM12critiCheckpoints.keys():
    P3OM12loss_nograd[key] = P3OM12checkpoints[key]['loss_nograd']
    P3OM12criticLoss_nograd[key] = P3OM12critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM12loss_nograd[key][i] == None:
            P3OM12loss_nograd[key][i] = 0
        if P3OM12criticLoss_nograd[key][i] == None:
            P3OM12criticLoss_nograd[key][i] = 0

P3OM12loss_nograd['mean'] = np.mean((P3OM12loss_nograd['1'], P3OM12loss_nograd['2'],
    P3OM12loss_nograd['3'], P3OM12loss_nograd['4'], P3OM12loss_nograd['5']), axis=0)
P3OM12criticLoss_nograd['mean'] = np.mean((P3OM12criticLoss_nograd['1'], P3OM12criticLoss_nograd['2'],
    P3OM12criticLoss_nograd['3'], P3OM12criticLoss_nograd['4'], P3OM12criticLoss_nograd['5']), axis=0)

P3OM13checkpoints = {}
P3OM13checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_13actorred125000ep250.tar')
P3OM13checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_13actorred125000ep250.tar')
P3OM13checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_13actorred125000ep250.tar')
P3OM13checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_13actorred125000ep250.tar')
P3OM13checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_13actorred125000ep250.tar')

P3OM13critiCheckpoints = {}
P3OM13critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_13criticred125000ep250.tar')
P3OM13critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_13criticred125000ep250.tar')
P3OM13critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_13criticred125000ep250.tar')
P3OM13critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_13criticred125000ep250.tar')
P3OM13critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_13criticred125000ep250.tar')

P3OM13loss_nograd = {}
P3OM13criticLoss_nograd = {}

for key in P3OM13critiCheckpoints.keys():
    P3OM13loss_nograd[key] = P3OM13checkpoints[key]['loss_nograd']
    P3OM13criticLoss_nograd[key] = P3OM13critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM13loss_nograd[key][i] == None:
            P3OM13loss_nograd[key][i] = 0
        if P3OM13criticLoss_nograd[key][i] == None:
            P3OM13criticLoss_nograd[key][i] = 0

P3OM13loss_nograd['mean'] = np.mean((P3OM13loss_nograd['1'], P3OM13loss_nograd['2'],
    P3OM13loss_nograd['3'], P3OM13loss_nograd['4'], P3OM13loss_nograd['5']), axis=0)
P3OM13criticLoss_nograd['mean'] = np.mean((P3OM13criticLoss_nograd['1'], P3OM13criticLoss_nograd['2'],
    P3OM13criticLoss_nograd['3'], P3OM13criticLoss_nograd['4'], P3OM13criticLoss_nograd['5']), axis=0)

P3OM14checkpoints = {}
P3OM14checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_14actorred125000ep250.tar')
P3OM14checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_14actorred125000ep250.tar')
P3OM14checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_14actorred125000ep250.tar')
P3OM14checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_14actorred125000ep250.tar')
P3OM14checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_14actorred125000ep250.tar')

P3OM14critiCheckpoints = {}
P3OM14critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_14criticred125000ep250.tar')
P3OM14critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_14criticred125000ep250.tar')
P3OM14critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_14criticred125000ep250.tar')
P3OM14critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_14criticred125000ep250.tar')
P3OM14critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_14criticred125000ep250.tar')

P3OM14loss_nograd = {}
P3OM14criticLoss_nograd = {}

for key in P3OM14critiCheckpoints.keys():
    P3OM14loss_nograd[key] = P3OM14checkpoints[key]['loss_nograd']
    P3OM14criticLoss_nograd[key] = P3OM14critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM14loss_nograd[key][i] == None:
            P3OM14loss_nograd[key][i] = 0
        if P3OM14criticLoss_nograd[key][i] == None:
            P3OM14criticLoss_nograd[key][i] = 0

P3OM14loss_nograd['mean'] = np.mean((P3OM14loss_nograd['1'], P3OM14loss_nograd['2'],
    P3OM14loss_nograd['3'], P3OM14loss_nograd['4'], P3OM14loss_nograd['5']), axis=0)
P3OM14criticLoss_nograd['mean'] = np.mean((P3OM14criticLoss_nograd['1'], P3OM14criticLoss_nograd['2'],
    P3OM14criticLoss_nograd['3'], P3OM14criticLoss_nograd['4'], P3OM14criticLoss_nograd['5']), axis=0)

P3OM15checkpoints = {}
P3OM15checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_15actorred125000ep250.tar')
P3OM15checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_15actorred125000ep250.tar')
P3OM15checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_15actorred125000ep250.tar')
P3OM15checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_15actorred125000ep250.tar')
P3OM15checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_15actorred125000ep250.tar')

P3OM15critiCheckpoints = {}
P3OM15critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_15criticred125000ep250.tar')
P3OM15critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_15criticred125000ep250.tar')
P3OM15critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_15criticred125000ep250.tar')
P3OM15critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_15criticred125000ep250.tar')
P3OM15critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_15criticred125000ep250.tar')

P3OM15loss_nograd = {}
P3OM15criticLoss_nograd = {}

for key in P3OM15critiCheckpoints.keys():
    P3OM15loss_nograd[key] = P3OM15checkpoints[key]['loss_nograd']
    P3OM15criticLoss_nograd[key] = P3OM15critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM15loss_nograd[key][i] == None:
            P3OM15loss_nograd[key][i] = 0
        if P3OM15criticLoss_nograd[key][i] == None:
            P3OM15criticLoss_nograd[key][i] = 0

P3OM15loss_nograd['mean'] = np.mean((P3OM15loss_nograd['1'], P3OM15loss_nograd['2'],
    P3OM15loss_nograd['3'], P3OM15loss_nograd['4'], P3OM15loss_nograd['5']), axis=0)
P3OM15criticLoss_nograd['mean'] = np.mean((P3OM15criticLoss_nograd['1'], P3OM15criticLoss_nograd['2'],
    P3OM15criticLoss_nograd['3'], P3OM15criticLoss_nograd['4'], P3OM15criticLoss_nograd['5']), axis=0)

P3OM16checkpoints = {}
P3OM16checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_16actorred125000ep250.tar')
P3OM16checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_16actorred125000ep250.tar')
P3OM16checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_16actorred125000ep250.tar')
P3OM16checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_16actorred125000ep250.tar')
P3OM16checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_16actorred125000ep250.tar')

P3OM16critiCheckpoints = {}
P3OM16critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_16criticred125000ep250.tar')
P3OM16critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_16criticred125000ep250.tar')
P3OM16critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_16criticred125000ep250.tar')
P3OM16critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_16criticred125000ep250.tar')
P3OM16critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_16criticred125000ep250.tar')

P3OM16loss_nograd = {}
P3OM16criticLoss_nograd = {}

for key in P3OM16critiCheckpoints.keys():
    P3OM16loss_nograd[key] = P3OM16checkpoints[key]['loss_nograd']
    P3OM16criticLoss_nograd[key] = P3OM16critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM16loss_nograd[key][i] == None:
            P3OM16loss_nograd[key][i] = 0
        if P3OM16criticLoss_nograd[key][i] == None:
            P3OM16criticLoss_nograd[key][i] = 0

P3OM16loss_nograd['mean'] = np.mean((P3OM16loss_nograd['1'], P3OM16loss_nograd['2'],
    P3OM16loss_nograd['3'], P3OM16loss_nograd['4'], P3OM16loss_nograd['5']), axis=0)
P3OM16criticLoss_nograd['mean'] = np.mean((P3OM16criticLoss_nograd['1'], P3OM16criticLoss_nograd['2'],
    P3OM16criticLoss_nograd['3'], P3OM16criticLoss_nograd['4'], P3OM16criticLoss_nograd['5']), axis=0)

P3OM17checkpoints = {}
P3OM17checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_17actorred125000ep250.tar')
P3OM17checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_17actorred125000ep250.tar')
P3OM17checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_17actorred125000ep250.tar')
P3OM17checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_17actorred125000ep250.tar')
P3OM17checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_17actorred125000ep250.tar')

P3OM17critiCheckpoints = {}
P3OM17critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_17criticred125000ep250.tar')
P3OM17critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_17criticred125000ep250.tar')
P3OM17critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_17criticred125000ep250.tar')
P3OM17critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_17criticred125000ep250.tar')
P3OM17critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_17criticred125000ep250.tar')

P3OM17loss_nograd = {}
P3OM17criticLoss_nograd = {}

for key in P3OM17critiCheckpoints.keys():
    P3OM17loss_nograd[key] = P3OM17checkpoints[key]['loss_nograd']
    P3OM17criticLoss_nograd[key] = P3OM17critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM17loss_nograd[key][i] == None:
            P3OM17loss_nograd[key][i] = 0
        if P3OM17criticLoss_nograd[key][i] == None:
            P3OM17criticLoss_nograd[key][i] = 0

P3OM17loss_nograd['mean'] = np.mean((P3OM17loss_nograd['1'], P3OM17loss_nograd['2'],
    P3OM17loss_nograd['3'], P3OM17loss_nograd['4'], P3OM17loss_nograd['5']), axis=0)
P3OM17criticLoss_nograd['mean'] = np.mean((P3OM17criticLoss_nograd['1'], P3OM17criticLoss_nograd['2'],
    P3OM17criticLoss_nograd['3'], P3OM17criticLoss_nograd['4'], P3OM17criticLoss_nograd['5']), axis=0)

P3OM18checkpoints = {}
P3OM18checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_18actorred125000ep250.tar')
P3OM18checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_18actorred125000ep250.tar')
P3OM18checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_18actorred125000ep250.tar')
P3OM18checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_18actorred125000ep250.tar')
P3OM18checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_18actorred125000ep250.tar')

P3OM18critiCheckpoints = {}
P3OM18critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_18criticred125000ep250.tar')
P3OM18critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_18criticred125000ep250.tar')
P3OM18critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_18criticred125000ep250.tar')
P3OM18critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_18criticred125000ep250.tar')
P3OM18critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_18criticred125000ep250.tar')

P3OM18loss_nograd = {}
P3OM18criticLoss_nograd = {}

for key in P3OM18critiCheckpoints.keys():
    P3OM18loss_nograd[key] = P3OM18checkpoints[key]['loss_nograd']
    P3OM18criticLoss_nograd[key] = P3OM18critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM18loss_nograd[key][i] == None:
            P3OM18loss_nograd[key][i] = 0
        if P3OM18criticLoss_nograd[key][i] == None:
            P3OM18criticLoss_nograd[key][i] = 0

P3OM18loss_nograd['mean'] = np.mean((P3OM18loss_nograd['1'], P3OM18loss_nograd['2'],
    P3OM18loss_nograd['3'], P3OM18loss_nograd['4'], P3OM18loss_nograd['5']), axis=0)
P3OM18criticLoss_nograd['mean'] = np.mean((P3OM18criticLoss_nograd['1'], P3OM18criticLoss_nograd['2'],
    P3OM18criticLoss_nograd['3'], P3OM18criticLoss_nograd['4'], P3OM18criticLoss_nograd['5']), axis=0)

P3OM19checkpoints = {}
P3OM19checkpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_19actorred125000ep250.tar')
P3OM19checkpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_19actorred125000ep250.tar')
P3OM19checkpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_19actorred125000ep250.tar')
P3OM19checkpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_19actorred125000ep250.tar')
P3OM19checkpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_19actorred125000ep250.tar')

P3OM19critiCheckpoints = {}
P3OM19critiCheckpoints['1'] = torch.load('./P3OAgents/Battle/MultiNets/RUN01_red_19criticred125000ep250.tar')
P3OM19critiCheckpoints['2'] = torch.load('./P3OAgents/Battle/MultiNets/RUN02_red_19criticred125000ep250.tar')
P3OM19critiCheckpoints['3'] = torch.load('./P3OAgents/Battle/MultiNets/RUN03_red_19criticred125000ep250.tar')
P3OM19critiCheckpoints['4'] = torch.load('./P3OAgents/Battle/MultiNets/RUN04_red_19criticred125000ep250.tar')
P3OM19critiCheckpoints['5'] = torch.load('./P3OAgents/Battle/MultiNets/RUN05_red_19criticred125000ep250.tar')

P3OM19loss_nograd = {}
P3OM19criticLoss_nograd = {}

for key in P3OM19critiCheckpoints.keys():
    P3OM19loss_nograd[key] = P3OM19checkpoints[key]['loss_nograd']
    P3OM19criticLoss_nograd[key] = P3OM19critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM19loss_nograd[key][i] == None:
            P3OM19loss_nograd[key][i] = 0
        if P3OM19criticLoss_nograd[key][i] == None:
            P3OM19criticLoss_nograd[key][i] = 0

P3OM19loss_nograd['mean'] = np.mean((P3OM19loss_nograd['1'], P3OM19loss_nograd['2'],
    P3OM19loss_nograd['3'], P3OM19loss_nograd['4'], P3OM19loss_nograd['5']), axis=0)
P3OM19criticLoss_nograd['mean'] = np.mean((P3OM19criticLoss_nograd['1'], P3OM19criticLoss_nograd['2'],
    P3OM19criticLoss_nograd['3'], P3OM19criticLoss_nograd['4'], P3OM19criticLoss_nograd['5']), axis=0)

P3OMMloss_nograd = np.mean((P3OMloss_nograd['mean'], P3OM1loss_nograd['mean'], P3OM2loss_nograd['mean'],
    P3OM3loss_nograd['mean'], P3OM4loss_nograd['mean'], P3OM4loss_nograd['mean'],
    P3OM5loss_nograd['mean'], P3OM6loss_nograd['mean'], P3OM7loss_nograd['mean'],
    P3OM8loss_nograd['mean'], P3OM9loss_nograd['mean'], P3OM10loss_nograd['mean'],
    P3OM11loss_nograd['mean'], P3OM12loss_nograd['mean'], P3OM13loss_nograd['mean'],
    P3OM14loss_nograd['mean'], P3OM15loss_nograd['mean'], P3OM16loss_nograd['mean'],
    P3OM17loss_nograd['mean'], P3OM18loss_nograd['mean'], P3OM19loss_nograd['mean']), axis=0)
P3OMMcriticLoss_nograd = np.mean((P3OM0criticLoss_nograd['mean'], P3OM1criticLoss_nograd['mean'], P3OM2criticLoss_nograd['mean'],
    P3OM3criticLoss_nograd['mean'], P3OM4criticLoss_nograd['mean'], P3OM4criticLoss_nograd['mean'],
    P3OM5criticLoss_nograd['mean'], P3OM6criticLoss_nograd['mean'], P3OM7criticLoss_nograd['mean'],
    P3OM8criticLoss_nograd['mean'], P3OM9criticLoss_nograd['mean'], P3OM10criticLoss_nograd['mean'],
    P3OM11criticLoss_nograd['mean'], P3OM12criticLoss_nograd['mean'], P3OM13criticLoss_nograd['mean'],
    P3OM14criticLoss_nograd['mean'], P3OM15criticLoss_nograd['mean'], P3OM16criticLoss_nograd['mean'],
    P3OM17criticLoss_nograd['mean'], P3OM18criticLoss_nograd['mean'], P3OM19criticLoss_nograd['mean']), axis=0)

figActorLoss = plt.figure("Actor Loss")
plt.plot(PPOeps['vector'], PPOloss_nograd['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOloss_nograd['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['5'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], P3O2loss_nograd['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2loss_nograd['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(PPOeps['vector'], P3OMMloss_nograd, color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMloss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM1loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM2loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM3loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM4loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM5loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM6loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM7loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM8loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM9loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM10loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM11loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM12loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM13loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM14loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM15loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM16loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM17loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM18loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM19loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Actor loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battleActorLoss.png', bbox_inches='tight')

figCriticLoss = plt.figure("Critic Loss")
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['5'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], P3O2criticLoss_nograd['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(PPOeps['vector'], P3OMMcriticLoss_nograd, color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OM0criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM1criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM2criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM3criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM4criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM5criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM6criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM7criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM8criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM9criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM10criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM11criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM12criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM13criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM14criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM15criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM16criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM17criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM18criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM19criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Critic loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battleCriticLoss.png', bbox_inches='tight')

# BATTLEFIELD ENVIRONMENT #
colours = {}

# DQN
colours['DQN'] = 'green'

DQNcheckpoints = {}
DQNcheckpoints['1'] = torch.load('./DQNAgents/Battlefield/RUN01_red125000ep250.tar')
DQNcheckpoints['2'] = torch.load('./DQNAgents/Battlefield/RUN02_red125000ep250.tar')
DQNcheckpoints['3'] = torch.load('./DQNAgents/Battlefield/RUN03_red125000ep250.tar')
DQNcheckpoints['4'] = torch.load('./DQNAgents/Battlefield/RUN04_red125000ep250.tar')
DQNcheckpoints['5'] = torch.load('./DQNAgents/Battlefield/RUN05_red125000ep250.tar')

DQNeps = {}
DQNcum_rewards = {}
DQNnum_steps = {}
DQNloss_nograd = {}
DQNtime = {}

for key in DQNcheckpoints.keys():
    DQNeps[key] = DQNcheckpoints[key]['episodes']
    DQNcum_rewards[key] = DQNcheckpoints[key]['cum_rewards']
    DQNnum_steps[key] = DQNcheckpoints[key]['num_steps']
    DQNloss_nograd[key] = DQNcheckpoints[key]['loss_nograd']
    DQNtime[key] = DQNcheckpoints[key]['time']['DQNAgents'][-1]
    for i in range(len(DQNcum_rewards[key])):
        if DQNloss_nograd[key][i] == None:
            DQNloss_nograd[key][i] = 0

if DQNeps['1'] == DQNeps['2'] == DQNeps['3'] == DQNeps['4'] == DQNeps['5']:
    DQNeps['vector'] = np.arange(1, DQNeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

DQNcum_rewards['mean'] = np.mean((DQNcum_rewards['1'], DQNcum_rewards['2'], DQNcum_rewards['3'],
    DQNcum_rewards['4'], DQNcum_rewards['5']), axis=0)
DQNnum_steps['mean'] = np.mean((DQNnum_steps['1'], DQNnum_steps['2'], DQNnum_steps['3'],
    DQNnum_steps['4'], DQNnum_steps['5']), axis=0)
DQNloss_nograd['mean'] = np.mean((DQNloss_nograd['1'], DQNloss_nograd['2'], DQNloss_nograd['3'],
    DQNloss_nograd['4'], DQNloss_nograd['5']), axis=0)
DQNtime['mean'] = np.mean((DQNtime['1'], DQNtime['2'], DQNtime['3'], DQNtime['4'],
    DQNtime['5']), axis=0)

#DuelingDDQN
colours['DuelingDDQN'] = 'blue'

DuelingDDQNcheckpoints = {}
DuelingDDQNcheckpoints['1'] = torch.load('./DuelingDDQNAgents/Battlefield/RUN01_red125000ep250.tar')
DuelingDDQNcheckpoints['2'] = torch.load('./DuelingDDQNAgents/Battlefield/RUN02_red125000ep250.tar')
DuelingDDQNcheckpoints['3'] = torch.load('./DuelingDDQNAgents/Battlefield/RUN03_red125000ep250.tar')
DuelingDDQNcheckpoints['4'] = torch.load('./DuelingDDQNAgents/Battlefield/RUN04_red125000ep250.tar')
DuelingDDQNcheckpoints['5'] = torch.load('./DuelingDDQNAgents/Battlefield/RUN05_red125000ep250.tar')

DuelingDDQNeps = {}
DuelingDDQNcum_rewards = {}
DuelingDDQNnum_steps = {}
DuelingDDQNloss_nograd = {}
DuelingDDQNtime = {}

for key in DuelingDDQNcheckpoints.keys():
    DuelingDDQNeps[key] = DuelingDDQNcheckpoints[key]['episodes']
    DuelingDDQNcum_rewards[key] = DuelingDDQNcheckpoints[key]['cum_rewards']
    DuelingDDQNnum_steps[key] = DuelingDDQNcheckpoints[key]['num_steps']
    DuelingDDQNloss_nograd[key] = DuelingDDQNcheckpoints[key]['loss_nograd']
    DuelingDDQNtime[key] = DuelingDDQNcheckpoints[key]['time']['DuelingDDQNAgents'][-1]
    for i in range(len(DuelingDDQNcum_rewards[key])):
        if DuelingDDQNloss_nograd[key][i] == None:
            DuelingDDQNloss_nograd[key][i] = 0

if DuelingDDQNeps['1'] == DuelingDDQNeps['2'] == DuelingDDQNeps['3'] == DuelingDDQNeps['4'] == DuelingDDQNeps['5']:
    DuelingDDQNeps['vector'] = np.arange(1, DuelingDDQNeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

DuelingDDQNcum_rewards['mean'] = np.mean((DuelingDDQNcum_rewards['1'], DuelingDDQNcum_rewards['2'],
    DuelingDDQNcum_rewards['3'], DuelingDDQNcum_rewards['4'], DuelingDDQNcum_rewards['5']), axis=0)
DuelingDDQNnum_steps['mean'] = np.mean((DuelingDDQNnum_steps['1'], DuelingDDQNnum_steps['2'],
    DuelingDDQNnum_steps['3'], DuelingDDQNnum_steps['4'], DuelingDDQNnum_steps['5']), axis=0)
DuelingDDQNloss_nograd['mean'] = np.mean((DuelingDDQNloss_nograd['1'], DuelingDDQNloss_nograd['2'],
    DuelingDDQNloss_nograd['3'], DuelingDDQNloss_nograd['4'], DuelingDDQNloss_nograd['5']), axis=0)
DuelingDDQNtime['mean'] = np.mean((DuelingDDQNtime['1'], DuelingDDQNtime['2'],
    DuelingDDQNtime['3'], DuelingDDQNtime['4'], DuelingDDQNtime['5']), axis=0)

# A3C
colours['A3C'] = 'gold'

A3Ccheckpoints = {}
A3Ccheckpoints['1'] = torch.load('./A3CAgents/Battlefield/RUN01_red125000ep250.tar')
A3Ccheckpoints['2'] = torch.load('./A3CAgents/Battlefield/RUN02_red125000ep250.tar')
A3Ccheckpoints['3'] = torch.load('./A3CAgents/Battlefield/RUN03_red125000ep250.tar')
A3Ccheckpoints['4'] = torch.load('./A3CAgents/Battlefield/RUN04_red125000ep250.tar')
A3Ccheckpoints['5'] = torch.load('./A3CAgents/Battlefield/RUN05_red125000ep250.tar')

A3Ceps = {}
A3Ccum_rewards = {}
A3Cnum_steps = {}
A3Closs_nograd = {}
A3Ctime = {}

for key in A3Ccheckpoints.keys():
    A3Ceps[key] = A3Ccheckpoints[key]['episodes']
    A3Ccum_rewards[key] = A3Ccheckpoints[key]['cum_rewards']
    A3Cnum_steps[key] = A3Ccheckpoints[key]['num_steps']
    A3Closs_nograd[key] = A3Ccheckpoints[key]['loss_nograd']
    A3Ctime[key] = A3Ccheckpoints[key]['time']['A3CAgents'][-1]
    for i in range(len(A3Ccum_rewards[key])):
        if A3Closs_nograd[key][i] == None:
            A3Closs_nograd[key][i] = 0

if A3Ceps['1'] == A3Ceps['2'] == A3Ceps['3'] == A3Ceps['4'] == A3Ceps['5']:
    A3Ceps['vector'] = np.arange(1, A3Ceps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

A3Ccum_rewards['mean'] = np.mean((A3Ccum_rewards['1'], A3Ccum_rewards['2'],
    A3Ccum_rewards['3'], A3Ccum_rewards['4'], A3Ccum_rewards['5']), axis=0)
A3Cnum_steps['mean'] = np.mean((A3Cnum_steps['1'], A3Cnum_steps['2'],
    A3Cnum_steps['3'], A3Cnum_steps['4'], A3Cnum_steps['5']), axis=0)
A3Closs_nograd['mean'] = np.mean((A3Closs_nograd['1'], A3Closs_nograd['2'],
    A3Closs_nograd['3'], A3Closs_nograd['4'], A3Closs_nograd['5']), axis=0)
A3Ctime['mean'] = np.mean((A3Ctime['1'], A3Ctime['2'], A3Ctime['3'], A3Ctime['4'],
    A3Ctime['5']), axis=0)

# PPO
colours['PPO'] = 'red'

PPOcheckpoints = {}
PPOcheckpoints['1'] = torch.load('./PPOAgents/Battlefield/RUN01_actorred125000ep250.tar')
PPOcheckpoints['2'] = torch.load('./PPOAgents/Battlefield/RUN02_actorred125000ep250.tar')
PPOcheckpoints['3'] = torch.load('./PPOAgents/Battlefield/RUN03_actorred125000ep250.tar')
PPOcheckpoints['4'] = torch.load('./PPOAgents/Battlefield/RUN04_actorred125000ep250.tar')
PPOcheckpoints['5'] = torch.load('./PPOAgents/Battlefield/RUN05_actorred125000ep250.tar')

PPOeps = {}
PPOcum_rewards = {}
PPOnum_steps = {}
PPOloss_nograd = {}
PPOtime = {}

for key in PPOcheckpoints.keys():
    PPOeps[key] = PPOcheckpoints[key]['episodes']
    PPOcum_rewards[key] = PPOcheckpoints[key]['cum_rewards']
    PPOnum_steps[key] = PPOcheckpoints[key]['num_steps']
    PPOloss_nograd[key] = PPOcheckpoints[key]['loss_nograd']
    PPOtime[key] = PPOcheckpoints[key]['time']['PPOAgents'][-1]
    for i in range(len(PPOcum_rewards[key])):
        if PPOloss_nograd[key][i] == None:
            PPOloss_nograd[key][i] = 0

if PPOeps['1'] == PPOeps['2'] == PPOeps['3'] == PPOeps['4'] == PPOeps['5']:
    PPOeps['vector'] = np.arange(1, PPOeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

PPOcum_rewards['mean'] = np.mean((PPOcum_rewards['1'], PPOcum_rewards['2'],
    PPOcum_rewards['3'], PPOcum_rewards['4'], PPOcum_rewards['5']), axis=0)
PPOnum_steps['mean'] = np.mean((PPOnum_steps['1'], PPOnum_steps['2'],
    PPOnum_steps['3'], PPOnum_steps['4'], PPOnum_steps['5']), axis=0)
PPOloss_nograd['mean'] = np.mean((PPOloss_nograd['1'], PPOloss_nograd['2'],
    PPOloss_nograd['3'], PPOloss_nograd['4'], PPOloss_nograd['5']), axis=0)
PPOtime['mean'] = np.mean((PPOtime['1'], PPOtime['2'], PPOtime['3'], PPOtime['4'],
    PPOtime['5']), axis=0)

# P3O
colours['P3O'] = 'purple'
colours['P3O2'] = 'palevioletred'
colours['P3OM'] = 'magenta'

P3Ocheckpoints = {}
P3Ocheckpoints['1'] = torch.load('./P3OAgents/Battlefield/1 Net/RUN01_red125000ep250.tar')
P3Ocheckpoints['2'] = torch.load('./P3OAgents/Battlefield/1 Net/RUN02_red125000ep250.tar')
P3Ocheckpoints['3'] = torch.load('./P3OAgents/Battlefield/1 Net/RUN03_red125000ep250.tar')
P3Ocheckpoints['4'] = torch.load('./P3OAgents/Battlefield/1 Net/RUN04_red125000ep250.tar')
P3Ocheckpoints['5'] = torch.load('./P3OAgents/Battlefield/1 Net/RUN05_red125000ep250.tar')

P3O2checkpoints = {}
P3O2checkpoints['1'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN01_actorred125000ep250.tar')
P3O2checkpoints['2'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN02_actorred125000ep250.tar')
P3O2checkpoints['3'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN03_actorred125000ep250.tar')
P3O2checkpoints['4'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN04_actorred125000ep250.tar')
P3O2checkpoints['5'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN05_actorred125000ep250.tar')

P3OMcheckpoints = {}
P3OMcheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_0actorred125000ep250.tar')
P3OMcheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_0actorred125000ep250.tar')
P3OMcheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_0actorred125000ep250.tar')
P3OMcheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_0actorred125000ep250.tar')
P3OMcheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_0actorred125000ep250.tar')

P3Oeps = {}
P3Ocum_rewards = {}
P3Onum_steps = {}
P3Oloss_nograd = {}
P3Otime = {}

P3O2cum_rewards = {}
P3O2num_steps = {}
P3O2time = {}

P3OMcum_rewards = {}
P3OMnum_steps = {}
P3OMtime = {}

for key in P3Ocheckpoints.keys():
    P3Oeps[key] = P3Ocheckpoints[key]['episodes']
    P3Ocum_rewards[key] = P3Ocheckpoints[key]['cum_rewards']
    P3Onum_steps[key] = P3Ocheckpoints[key]['num_steps']
    P3Oloss_nograd[key] = P3Ocheckpoints[key]['loss_nograd']
    P3Otime[key] = P3Ocheckpoints[key]['time']['P3OAgents'][-1]

    P3O2cum_rewards[key] = P3O2checkpoints[key]['cum_rewards']
    P3O2num_steps[key] = P3O2checkpoints[key]['num_steps']
    P3O2time[key] = P3O2checkpoints[key]['time']['P3OAgents'][-1]

    P3OMcum_rewards[key] = P3OMcheckpoints[key]['cum_rewards']
    P3OMnum_steps[key] = P3OMcheckpoints[key]['num_steps']
    P3OMtime[key] = P3OMcheckpoints[key]['time']['P3OAgents'][-1]

    for i in range(len(P3Ocum_rewards[key])):
        if P3Oloss_nograd[key][i] == None:
            P3Oloss_nograd[key][i] = 0

if P3Oeps['1'] == P3Oeps['2'] == P3Oeps['3'] == P3Oeps['4'] == P3Oeps['5']:
    P3Oeps['vector'] = np.arange(1, P3Oeps['1'] + 1, 1)
else:
    raise Exception(f"Warning: analysis data with different length has been detected")

P3Ocum_rewards['mean'] = np.mean((P3Ocum_rewards['1'], P3Ocum_rewards['2'],
    P3Ocum_rewards['3'], P3Ocum_rewards['4'], P3Ocum_rewards['5']), axis=0)
P3Onum_steps['mean'] = np.mean((P3Onum_steps['1'], P3Onum_steps['2'],
    P3Onum_steps['3'], P3Onum_steps['4'], P3Onum_steps['5']), axis=0)
P3Oloss_nograd['mean'] = np.mean((P3Oloss_nograd['1'], P3Oloss_nograd['2'],
    P3Oloss_nograd['3'], P3Oloss_nograd['4'], P3Oloss_nograd['5']), axis=0)
P3Otime['mean'] = np.mean((P3Otime['1'], P3Otime['2'], P3Otime['3'], P3Otime['4'],
    P3Otime['5']), axis=0)

P3O2cum_rewards['mean'] = np.mean((P3O2cum_rewards['1'], P3O2cum_rewards['2'],
    P3O2cum_rewards['3'], P3O2cum_rewards['4'], P3O2cum_rewards['5']), axis=0)
P3O2num_steps['mean'] = np.mean((P3O2num_steps['1'], P3O2num_steps['2'],
    P3O2num_steps['3'], P3O2num_steps['4'], P3O2num_steps['5']), axis=0)
P3O2time['mean'] = np.mean((P3O2time['1'], P3O2time['2'], P3O2time['3'], P3O2time['4'],
    P3O2time['5']), axis=0)

P3OMcum_rewards['mean'] = np.mean((P3OMcum_rewards['1'], P3OMcum_rewards['2'],
    P3OMcum_rewards['3'], P3OMcum_rewards['4'], P3OMcum_rewards['5']), axis=0)
P3OMnum_steps['mean'] = np.mean((P3OMnum_steps['1'], P3OMnum_steps['2'],
    P3OMnum_steps['3'], P3OMnum_steps['4'], P3OMnum_steps['5']), axis=0)
P3OMtime['mean'] = np.mean((P3OMtime['1'], P3OMtime['2'], P3OMtime['3'], P3OMtime['4'],
    P3OMtime['5']), axis=0)

# plotting figures
figRewards = plt.figure("Battlefield Cumulative Rewards")
plt.plot(DQNeps['vector'], DQNcum_rewards['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNcum_rewards['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNcum_rewards['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNcum_rewards['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Ccum_rewards['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Ccum_rewards['5'], color=colours['A3C'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOcum_rewards['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcum_rewards['5'], color=colours['PPO'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Ocum_rewards['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Ocum_rewards['5'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2cum_rewards['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2cum_rewards['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['mean'], color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMcum_rewards['1'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['2'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['3'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['4'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMcum_rewards['5'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Cumulative rewards")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battlefieldCumulativeRewards.png', bbox_inches='tight')

figNumSteps = plt.figure("Battlefield Number of steps")
plt.plot(DQNeps['vector'], DQNnum_steps['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNnum_steps['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNnum_steps['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNnum_steps['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Cnum_steps['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Cnum_steps['5'], color=colours['A3C'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOnum_steps['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOnum_steps['5'], color=colours['PPO'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Onum_steps['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Onum_steps['5'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2num_steps['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2num_steps['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['mean'], color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMnum_steps['1'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['2'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['3'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['4'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OMnum_steps['5'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Number of steps")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battlefieldNumSteps.png', bbox_inches='tight')

figLoss = plt.figure("Battlefield Loss")
plt.plot(DQNeps['vector'], DQNloss_nograd['mean'], color=colours['DQN'], alpha=1, label=f"DQN")
plt.plot(DQNeps['vector'], DQNloss_nograd['1'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['2'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['3'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['4'], color=colours['DQN'], alpha=0.05)
plt.plot(DQNeps['vector'], DQNloss_nograd['5'], color=colours['DQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['mean'], color=colours['DuelingDDQN'], alpha=1, label=f"DuelingDDQN")
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['1'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['2'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['3'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['4'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(DuelingDDQNeps['vector'], DuelingDDQNloss_nograd['5'], color=colours['DuelingDDQN'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['mean'], color=colours['A3C'], alpha=1, label=f"A3C")
plt.plot(A3Ceps['vector'], A3Closs_nograd['1'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['2'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['3'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['4'], color=colours['A3C'], alpha=0.05)
plt.plot(A3Ceps['vector'], A3Closs_nograd['5'], color=colours['A3C'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['mean'], color=colours['P3O'], alpha=1, label=f"P3O 1 Net")
plt.plot(P3Oeps['vector'], P3Oloss_nograd['1'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['2'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['3'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['4'], color=colours['P3O'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3Oloss_nograd['5'], color=colours['P3O'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battlefieldLoss.png', bbox_inches='tight')

print('Battlefield Computational Time Table')
print('==================================')
print(f"DQN: {DQNtime['mean']} sec")
print(f"Dueling DDQN: {DuelingDDQNtime['mean']} sec")
print(f"A3C: {A3Ctime['mean']} sec")
print(f"PPO: {PPOtime['mean']} sec")
print(f"P3O: {P3Otime['mean']} sec")
print(f"P3O 2 Nets: {P3O2time['mean']} sec")
print(f"P3O Multi-Nets: {P3OMtime['mean']} sec \n")

PPOcritiCheckpoints = {}
PPOcritiCheckpoints['1'] = torch.load('./PPOAgents/Battlefield/RUN01_criticred125000ep250.tar')
PPOcritiCheckpoints['2'] = torch.load('./PPOAgents/Battlefield/RUN02_criticred125000ep250.tar')
PPOcritiCheckpoints['3'] = torch.load('./PPOAgents/Battlefield/RUN03_criticred125000ep250.tar')
PPOcritiCheckpoints['4'] = torch.load('./PPOAgents/Battlefield/RUN04_criticred125000ep250.tar')
PPOcritiCheckpoints['5'] = torch.load('./PPOAgents/Battlefield/RUN05_criticred125000ep250.tar')

PPOcriticLoss_nograd = {}
for key in PPOcritiCheckpoints.keys():
    PPOcriticLoss_nograd[key] = PPOcritiCheckpoints[key]['loss_nograd']
    for i in range(len(PPOcum_rewards[key])):
        if PPOcriticLoss_nograd[key][i] == None:
            PPOcriticLoss_nograd[key][i] = 0

PPOcriticLoss_nograd['mean'] = np.mean((PPOcriticLoss_nograd['1'], PPOcriticLoss_nograd['2'],
    PPOcriticLoss_nograd['3'], PPOcriticLoss_nograd['4'], PPOcriticLoss_nograd['5']), axis=0)

P3O2critiCheckpoints = {}
P3O2critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN01_criticred125000ep250.tar')
P3O2critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN02_criticred125000ep250.tar')
P3O2critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN03_criticred125000ep250.tar')
P3O2critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN04_criticred125000ep250.tar')
P3O2critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/2 Nets/RUN05_criticred125000ep250.tar')

P3O2loss_nograd = {}
P3O2criticLoss_nograd = {}
for key in P3O2critiCheckpoints.keys():
    P3O2loss_nograd[key] = P3O2checkpoints[key]['loss_nograd']
    P3O2criticLoss_nograd[key] = P3O2critiCheckpoints[key]['loss_nograd']
    for i in range(len(P3O2cum_rewards[key])):
        if P3O2loss_nograd[key][i] == None:
            P3O2loss_nograd[key][i] = 0
        if P3O2criticLoss_nograd[key][i] == None:
            P3O2criticLoss_nograd[key][i] = 0

P3O2loss_nograd['mean'] = np.mean((P3O2loss_nograd['1'], P3O2loss_nograd['2'],
    P3O2loss_nograd['3'], P3O2loss_nograd['4'], P3O2loss_nograd['5']), axis=0)
P3O2criticLoss_nograd['mean'] = np.mean((P3O2criticLoss_nograd['1'], P3O2criticLoss_nograd['2'],
    P3O2criticLoss_nograd['3'], P3O2criticLoss_nograd['4'], P3O2criticLoss_nograd['5']), axis=0)

P3OM0critiCheckpoints = {}
P3OM0critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_0criticred125000ep250.tar')
P3OM0critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_0criticred125000ep250.tar')

P3OMloss_nograd = {}
P3OM0criticLoss_nograd = {}

for key in P3OM0critiCheckpoints.keys():
    P3OMloss_nograd[key] = P3OMcheckpoints[key]['loss_nograd']
    P3OM0criticLoss_nograd[key] = P3OM0critiCheckpoints[key]['loss_nograd']
    for i in range(len(P3OMcum_rewards[key])):
        if P3OMloss_nograd[key][i] == None:
            P3OMloss_nograd[key][i] = 0
        if P3OM0criticLoss_nograd[key][i] == None:
            P3OM0criticLoss_nograd[key][i] = 0

P3OMloss_nograd['mean'] = np.mean((P3OMloss_nograd['1'], P3OMloss_nograd['2'],
    P3OMloss_nograd['3'], P3OMloss_nograd['4'], P3OMloss_nograd['5']), axis=0)
P3OM0criticLoss_nograd['mean'] = np.mean((P3OM0criticLoss_nograd['1'], P3OM0criticLoss_nograd['2'],
    P3OM0criticLoss_nograd['3'], P3OM0criticLoss_nograd['4'], P3OM0criticLoss_nograd['5']), axis=0)

P3OM1checkpoints = {}
P3OM1checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_1actorred125000ep250.tar')
P3OM1checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_1actorred125000ep250.tar')
P3OM1checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_1actorred125000ep250.tar')
P3OM1checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_1actorred125000ep250.tar')
P3OM1checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_1actorred125000ep250.tar')

P3OM1critiCheckpoints = {}
P3OM1critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_1criticred125000ep250.tar')
P3OM1critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_1criticred125000ep250.tar')

P3OM1loss_nograd = {}
P3OM1criticLoss_nograd = {}

for key in P3OM1critiCheckpoints.keys():
    P3OM1loss_nograd[key] = P3OM1checkpoints[key]['loss_nograd']
    P3OM1criticLoss_nograd[key] = P3OM1critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM1loss_nograd[key][i] == None:
            P3OM1loss_nograd[key][i] = 0
        if P3OM1criticLoss_nograd[key][i] == None:
            P3OM1criticLoss_nograd[key][i] = 0

P3OM1loss_nograd['mean'] = np.mean((P3OM1loss_nograd['1'], P3OM1loss_nograd['2'],
    P3OM1loss_nograd['3'], P3OM1loss_nograd['4'], P3OM1loss_nograd['5']), axis=0)
P3OM1criticLoss_nograd['mean'] = np.mean((P3OM1criticLoss_nograd['1'], P3OM1criticLoss_nograd['2'],
    P3OM1criticLoss_nograd['3'], P3OM1criticLoss_nograd['4'], P3OM1criticLoss_nograd['5']), axis=0)

P3OM2checkpoints = {}
P3OM2checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_2actorred125000ep250.tar')
P3OM2checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_2actorred125000ep250.tar')
P3OM2checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_2actorred125000ep250.tar')
P3OM2checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_2actorred125000ep250.tar')
P3OM2checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_2actorred125000ep250.tar')

P3OM2critiCheckpoints = {}
P3OM2critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_2criticred125000ep250.tar')
P3OM2critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_2criticred125000ep250.tar')

P3OM2loss_nograd = {}
P3OM2criticLoss_nograd = {}

for key in P3OM2critiCheckpoints.keys():
    P3OM2loss_nograd[key] = P3OM2checkpoints[key]['loss_nograd']
    P3OM2criticLoss_nograd[key] = P3OM2critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM2loss_nograd[key][i] == None:
            P3OM2loss_nograd[key][i] = 0
        if P3OM2criticLoss_nograd[key][i] == None:
            P3OM2criticLoss_nograd[key][i] = 0

P3OM2loss_nograd['mean'] = np.mean((P3OM2loss_nograd['1'], P3OM2loss_nograd['2'],
    P3OM2loss_nograd['3'], P3OM2loss_nograd['4'], P3OM2loss_nograd['5']), axis=0)
P3OM2criticLoss_nograd['mean'] = np.mean((P3OM2criticLoss_nograd['1'], P3OM2criticLoss_nograd['2'],
    P3OM2criticLoss_nograd['3'], P3OM2criticLoss_nograd['4'], P3OM2criticLoss_nograd['5']), axis=0)

P3OM3checkpoints = {}
P3OM3checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_3actorred125000ep250.tar')
P3OM3checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_3actorred125000ep250.tar')
P3OM3checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_3actorred125000ep250.tar')
P3OM3checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_3actorred125000ep250.tar')
P3OM3checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_3actorred125000ep250.tar')

P3OM3critiCheckpoints = {}
P3OM3critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_3criticred125000ep250.tar')
P3OM3critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_3criticred125000ep250.tar')

P3OM3loss_nograd = {}
P3OM3criticLoss_nograd = {}

for key in P3OM3critiCheckpoints.keys():
    P3OM3loss_nograd[key] = P3OM3checkpoints[key]['loss_nograd']
    P3OM3criticLoss_nograd[key] = P3OM3critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM3loss_nograd[key][i] == None:
            P3OM3loss_nograd[key][i] = 0
        if P3OM3criticLoss_nograd[key][i] == None:
            P3OM3criticLoss_nograd[key][i] = 0

P3OM3loss_nograd['mean'] = np.mean((P3OM3loss_nograd['1'], P3OM3loss_nograd['2'],
    P3OM3loss_nograd['3'], P3OM3loss_nograd['4'], P3OM3loss_nograd['5']), axis=0)
P3OM3criticLoss_nograd['mean'] = np.mean((P3OM3criticLoss_nograd['1'], P3OM3criticLoss_nograd['2'],
    P3OM3criticLoss_nograd['3'], P3OM3criticLoss_nograd['4'], P3OM3criticLoss_nograd['5']), axis=0)

P3OM4checkpoints = {}
P3OM4checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_4actorred125000ep250.tar')
P3OM4checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_4actorred125000ep250.tar')
P3OM4checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_4actorred125000ep250.tar')
P3OM4checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_4actorred125000ep250.tar')
P3OM4checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_4actorred125000ep250.tar')

P3OM4critiCheckpoints = {}
P3OM4critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_4criticred125000ep250.tar')
P3OM4critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_4criticred125000ep250.tar')

P3OM4loss_nograd = {}
P3OM4criticLoss_nograd = {}

for key in P3OM4critiCheckpoints.keys():
    P3OM4loss_nograd[key] = P3OM4checkpoints[key]['loss_nograd']
    P3OM4criticLoss_nograd[key] = P3OM4critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM4loss_nograd[key][i] == None:
            P3OM4loss_nograd[key][i] = 0
        if P3OM4criticLoss_nograd[key][i] == None:
            P3OM4criticLoss_nograd[key][i] = 0

P3OM4loss_nograd['mean'] = np.mean((P3OM4loss_nograd['1'], P3OM4loss_nograd['2'],
    P3OM4loss_nograd['3'], P3OM4loss_nograd['4'], P3OM4loss_nograd['5']), axis=0)
P3OM4criticLoss_nograd['mean'] = np.mean((P3OM4criticLoss_nograd['1'], P3OM4criticLoss_nograd['2'],
    P3OM4criticLoss_nograd['3'], P3OM4criticLoss_nograd['4'], P3OM4criticLoss_nograd['5']), axis=0)

P3OM5checkpoints = {}
P3OM5checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_5actorred125000ep250.tar')
P3OM5checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_5actorred125000ep250.tar')
P3OM5checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_5actorred125000ep250.tar')
P3OM5checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_5actorred125000ep250.tar')
P3OM5checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_5actorred125000ep250.tar')

P3OM5critiCheckpoints = {}
P3OM5critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_5criticred125000ep250.tar')
P3OM5critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_5criticred125000ep250.tar')

P3OM5loss_nograd = {}
P3OM5criticLoss_nograd = {}

for key in P3OM5critiCheckpoints.keys():
    P3OM5loss_nograd[key] = P3OM5checkpoints[key]['loss_nograd']
    P3OM5criticLoss_nograd[key] = P3OM5critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM5loss_nograd[key][i] == None:
            P3OM5loss_nograd[key][i] = 0
        if P3OM5criticLoss_nograd[key][i] == None:
            P3OM5criticLoss_nograd[key][i] = 0

P3OM5loss_nograd['mean'] = np.mean((P3OM5loss_nograd['1'], P3OM5loss_nograd['2'],
    P3OM5loss_nograd['3'], P3OM5loss_nograd['4'], P3OM5loss_nograd['5']), axis=0)
P3OM5criticLoss_nograd['mean'] = np.mean((P3OM5criticLoss_nograd['1'], P3OM5criticLoss_nograd['2'],
    P3OM5criticLoss_nograd['3'], P3OM5criticLoss_nograd['4'], P3OM5criticLoss_nograd['5']), axis=0)

P3OM6checkpoints = {}
P3OM6checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_6actorred125000ep250.tar')
P3OM6checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_6actorred125000ep250.tar')
P3OM6checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_6actorred125000ep250.tar')
P3OM6checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_6actorred125000ep250.tar')
P3OM6checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_6actorred125000ep250.tar')

P3OM6critiCheckpoints = {}
P3OM6critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_6criticred125000ep250.tar')
P3OM6critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_6criticred125000ep250.tar')

P3OM6loss_nograd = {}
P3OM6criticLoss_nograd = {}

for key in P3OM6critiCheckpoints.keys():
    P3OM6loss_nograd[key] = P3OM6checkpoints[key]['loss_nograd']
    P3OM6criticLoss_nograd[key] = P3OM6critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM6loss_nograd[key][i] == None:
            P3OM6loss_nograd[key][i] = 0
        if P3OM6criticLoss_nograd[key][i] == None:
            P3OM6criticLoss_nograd[key][i] = 0

P3OM6loss_nograd['mean'] = np.mean((P3OM6loss_nograd['1'], P3OM6loss_nograd['2'],
    P3OM6loss_nograd['3'], P3OM6loss_nograd['4'], P3OM6loss_nograd['5']), axis=0)
P3OM6criticLoss_nograd['mean'] = np.mean((P3OM6criticLoss_nograd['1'], P3OM6criticLoss_nograd['2'],
    P3OM6criticLoss_nograd['3'], P3OM6criticLoss_nograd['4'], P3OM6criticLoss_nograd['5']), axis=0)

P3OM7checkpoints = {}
P3OM7checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_7actorred125000ep250.tar')
P3OM7checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_7actorred125000ep250.tar')
P3OM7checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_7actorred125000ep250.tar')
P3OM7checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_7actorred125000ep250.tar')
P3OM7checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_7actorred125000ep250.tar')

P3OM7critiCheckpoints = {}
P3OM7critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_7criticred125000ep250.tar')
P3OM7critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_7criticred125000ep250.tar')

P3OM7loss_nograd = {}
P3OM7criticLoss_nograd = {}

for key in P3OM7critiCheckpoints.keys():
    P3OM7loss_nograd[key] = P3OM7checkpoints[key]['loss_nograd']
    P3OM7criticLoss_nograd[key] = P3OM7critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM7loss_nograd[key][i] == None:
            P3OM7loss_nograd[key][i] = 0
        if P3OM7criticLoss_nograd[key][i] == None:
            P3OM7criticLoss_nograd[key][i] = 0

P3OM7loss_nograd['mean'] = np.mean((P3OM7loss_nograd['1'], P3OM7loss_nograd['2'],
    P3OM7loss_nograd['3'], P3OM7loss_nograd['4'], P3OM7loss_nograd['5']), axis=0)
P3OM7criticLoss_nograd['mean'] = np.mean((P3OM7criticLoss_nograd['1'], P3OM7criticLoss_nograd['2'],
    P3OM7criticLoss_nograd['3'], P3OM7criticLoss_nograd['4'], P3OM7criticLoss_nograd['5']), axis=0)

P3OM8checkpoints = {}
P3OM8checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_8actorred125000ep250.tar')
P3OM8checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_8actorred125000ep250.tar')
P3OM8checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_8actorred125000ep250.tar')
P3OM8checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_8actorred125000ep250.tar')
P3OM8checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_8actorred125000ep250.tar')

P3OM8critiCheckpoints = {}
P3OM8critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_8criticred125000ep250.tar')
P3OM8critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_8criticred125000ep250.tar')

P3OM8loss_nograd = {}
P3OM8criticLoss_nograd = {}

for key in P3OM8critiCheckpoints.keys():
    P3OM8loss_nograd[key] = P3OM8checkpoints[key]['loss_nograd']
    P3OM8criticLoss_nograd[key] = P3OM8critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM8loss_nograd[key][i] == None:
            P3OM8loss_nograd[key][i] = 0
        if P3OM8criticLoss_nograd[key][i] == None:
            P3OM8criticLoss_nograd[key][i] = 0

P3OM8loss_nograd['mean'] = np.mean((P3OM8loss_nograd['1'], P3OM8loss_nograd['2'],
    P3OM8loss_nograd['3'], P3OM8loss_nograd['4'], P3OM8loss_nograd['5']), axis=0)
P3OM8criticLoss_nograd['mean'] = np.mean((P3OM8criticLoss_nograd['1'], P3OM8criticLoss_nograd['2'],
    P3OM8criticLoss_nograd['3'], P3OM8criticLoss_nograd['4'], P3OM8criticLoss_nograd['5']), axis=0)

P3OM9checkpoints = {}
P3OM9checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_9actorred125000ep250.tar')
P3OM9checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_9actorred125000ep250.tar')
P3OM9checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_9actorred125000ep250.tar')
P3OM9checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_9actorred125000ep250.tar')
P3OM9checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_9actorred125000ep250.tar')

P3OM9critiCheckpoints = {}
P3OM9critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_9criticred125000ep250.tar')
P3OM9critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_9criticred125000ep250.tar')

P3OM9loss_nograd = {}
P3OM9criticLoss_nograd = {}

for key in P3OM9critiCheckpoints.keys():
    P3OM9loss_nograd[key] = P3OM9checkpoints[key]['loss_nograd']
    P3OM9criticLoss_nograd[key] = P3OM9critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM9loss_nograd[key][i] == None:
            P3OM9loss_nograd[key][i] = 0
        if P3OM9criticLoss_nograd[key][i] == None:
            P3OM9criticLoss_nograd[key][i] = 0

P3OM9loss_nograd['mean'] = np.mean((P3OM9loss_nograd['1'], P3OM9loss_nograd['2'],
    P3OM9loss_nograd['3'], P3OM9loss_nograd['4'], P3OM9loss_nograd['5']), axis=0)
P3OM9criticLoss_nograd['mean'] = np.mean((P3OM9criticLoss_nograd['1'], P3OM9criticLoss_nograd['2'],
    P3OM9criticLoss_nograd['3'], P3OM9criticLoss_nograd['4'], P3OM9criticLoss_nograd['5']), axis=0)

P3OM10checkpoints = {}
P3OM10checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_10actorred125000ep250.tar')
P3OM10checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_10actorred125000ep250.tar')
P3OM10checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_10actorred125000ep250.tar')
P3OM10checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_10actorred125000ep250.tar')
P3OM10checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_10actorred125000ep250.tar')

P3OM10critiCheckpoints = {}
P3OM10critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_10criticred125000ep250.tar')
P3OM10critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_10criticred125000ep250.tar')

P3OM10loss_nograd = {}
P3OM10criticLoss_nograd = {}

for key in P3OM10critiCheckpoints.keys():
    P3OM10loss_nograd[key] = P3OM10checkpoints[key]['loss_nograd']
    P3OM10criticLoss_nograd[key] = P3OM10critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM10loss_nograd[key][i] == None:
            P3OM10loss_nograd[key][i] = 0
        if P3OM10criticLoss_nograd[key][i] == None:
            P3OM10criticLoss_nograd[key][i] = 0

P3OM10loss_nograd['mean'] = np.mean((P3OM10loss_nograd['1'], P3OM10loss_nograd['2'],
    P3OM10loss_nograd['3'], P3OM10loss_nograd['4'], P3OM10loss_nograd['5']), axis=0)
P3OM10criticLoss_nograd['mean'] = np.mean((P3OM10criticLoss_nograd['1'], P3OM10criticLoss_nograd['2'],
    P3OM10criticLoss_nograd['3'], P3OM10criticLoss_nograd['4'], P3OM10criticLoss_nograd['5']), axis=0)

P3OM11checkpoints = {}
P3OM11checkpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_11actorred125000ep250.tar')
P3OM11checkpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_11actorred125000ep250.tar')
P3OM11checkpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_11actorred125000ep250.tar')
P3OM11checkpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_11actorred125000ep250.tar')
P3OM11checkpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_11actorred125000ep250.tar')

P3OM11critiCheckpoints = {}
P3OM11critiCheckpoints['1'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN01_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['2'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN02_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['3'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN03_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['4'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN04_red_11criticred125000ep250.tar')
P3OM11critiCheckpoints['5'] = torch.load('./P3OAgents/Battlefield/MultiNets/RUN05_red_11criticred125000ep250.tar')

P3OM11loss_nograd = {}
P3OM11criticLoss_nograd = {}

for key in P3OM11critiCheckpoints.keys():
    P3OM11loss_nograd[key] = P3OM11checkpoints[key]['loss_nograd']
    P3OM11criticLoss_nograd[key] = P3OM11critiCheckpoints[key]['loss_nograd']
    for i in range(250):
        if P3OM11loss_nograd[key][i] == None:
            P3OM11loss_nograd[key][i] = 0
        if P3OM11criticLoss_nograd[key][i] == None:
            P3OM11criticLoss_nograd[key][i] = 0

P3OM11loss_nograd['mean'] = np.mean((P3OM11loss_nograd['1'], P3OM11loss_nograd['2'],
    P3OM11loss_nograd['3'], P3OM11loss_nograd['4'], P3OM11loss_nograd['5']), axis=0)
P3OM11criticLoss_nograd['mean'] = np.mean((P3OM11criticLoss_nograd['1'], P3OM11criticLoss_nograd['2'],
    P3OM11criticLoss_nograd['3'], P3OM11criticLoss_nograd['4'], P3OM11criticLoss_nograd['5']), axis=0)

P3OMMloss_nograd = np.mean((P3OMloss_nograd['mean'], P3OM1loss_nograd['mean'], P3OM2loss_nograd['mean'],
    P3OM3loss_nograd['mean'], P3OM4loss_nograd['mean'], P3OM4loss_nograd['mean'],
    P3OM5loss_nograd['mean'], P3OM6loss_nograd['mean'], P3OM7loss_nograd['mean'],
    P3OM8loss_nograd['mean'], P3OM9loss_nograd['mean'], P3OM10loss_nograd['mean'],
    P3OM11loss_nograd['mean']), axis=0)
P3OMMcriticLoss_nograd = np.mean((P3OM0criticLoss_nograd['mean'], P3OM1criticLoss_nograd['mean'], P3OM2criticLoss_nograd['mean'],
    P3OM3criticLoss_nograd['mean'], P3OM4criticLoss_nograd['mean'], P3OM4criticLoss_nograd['mean'],
    P3OM5criticLoss_nograd['mean'], P3OM6criticLoss_nograd['mean'], P3OM7criticLoss_nograd['mean'],
    P3OM8criticLoss_nograd['mean'], P3OM9criticLoss_nograd['mean'], P3OM10criticLoss_nograd['mean'],
    P3OM11criticLoss_nograd['mean']), axis=0)

figActorLoss = plt.figure("Battlefield Actor Loss")
plt.plot(PPOeps['vector'], PPOloss_nograd['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOloss_nograd['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOloss_nograd['5'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], P3O2loss_nograd['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2loss_nograd['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2loss_nograd['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(PPOeps['vector'], P3OMMloss_nograd, color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OMloss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM1loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM2loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM3loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM4loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM5loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM6loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM7loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM8loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM9loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM10loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM11loss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Actor loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battlefieldActorLoss.png', bbox_inches='tight')

figCriticLoss = plt.figure("Battlefield Critic Loss")
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['mean'], color=colours['PPO'], alpha=1, label=f"PPO 2 Nets")
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['1'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['2'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['3'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['4'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], PPOcriticLoss_nograd['5'], color=colours['PPO'], alpha=0.05)
plt.plot(PPOeps['vector'], P3O2criticLoss_nograd['mean'], color=colours['P3O2'], alpha=1, label=f"P3O 2 Nets")
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['1'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['2'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['3'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['4'], color=colours['P3O2'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3O2criticLoss_nograd['5'], color=colours['P3O2'], alpha=0.05)
plt.plot(PPOeps['vector'], P3OMMcriticLoss_nograd, color=colours['P3OM'], alpha=1, label=f"P3O Multi-Nets")
plt.plot(P3Oeps['vector'], P3OM0criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM1criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM2criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM3criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM4criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM5criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM6criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM7criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM8criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM9criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM10criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.plot(P3Oeps['vector'], P3OM11criticLoss_nograd['mean'], color=colours['P3OM'], alpha=0.05)
plt.xlabel("Episodes")
plt.ylabel(f"Critic loss")
plt.xlim(0, 250)
plt.legend()
plt.savefig('battlefieldCriticLoss.png', bbox_inches='tight')
