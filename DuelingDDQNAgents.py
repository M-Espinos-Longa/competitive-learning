import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from DuelingDDQNModel import *
from Dataclasses import SARSD
from operator import itemgetter
from collections import deque
from tqdm import tqdm
import math

class DuelingDDQNAgents(nn.Module):
    """
    Dueling Double Deep Q Network agents with stochastic prioritised sweeping.
    See https://arxiv.org/abs/1511.05952v4 fir further info about prioritised replay
    See https://arxiv.org/abs/1511.06581v3 for further info about Dueling DDQN
    """
    def agents_init(self, agents_info, env, train=True):
        """
        Setup for the agents called when the experiment starts
        Inputs:
            + agents_info (dict) -> key parameters to initialise agents
            {
                num_agents (int): number of learning agents involved in the game
                num_observations (int): number of input features
                num_obs_c (int): observation channels
                num_obs_h (int): height of image observation
                num_obs_w (int): width of image observation
                num_actions (int): number of available actions per agent
                num_hidden_units (list): hidden units per layer
                kernels (list): squared kernels on each conv layer
                strides (list): strides on each convolutional layer
                fmaps (list): feature maps in convolutional layers
                batch_size (int): number of transitions for each optimisation step
                buff_size (int): replay buffer size (batch_size < buff_size)
                alpha (float): scaling probability factor for prioritised replay
                beta_0 (float): importance sampling dynamic initial bounding factor
                offset (float): prevents non-visitations of transitions with null TD error
                lr (float): optimiser learning rate
                policy (string)
                tau (float): temperature parameter from soft-max policy
                epsilon_params (list): [initial epsilon, final epsilon, epsilon decay]
                discount (float): factor applied to the return function
                seed (int): random seed for reproducibility of experiments
                total_training_steps (int)
            }
            + env (object) -> environment
            + train (bool) -> network mode (default: training)
        Outputs:
        """
        # store provided parameters in agents_info
        self.num_agents = agents_info["num_agents"]
        self.num_observations = agents_info["num_observations"]
        self.num_obs_c = agents_info["num_obs_c"]
        self.num_obs_h = agents_info["num_obs_h"]
        self.num_obs_w = agents_info["num_obs_w"]
        self.num_actions = agents_info["num_actions"]
        self.num_hidden_units = agents_info["num_hidden_units"]
        self.kernels = agents_info["kernels"]
        self.strides = agents_info["strides"]
        self.fmaps = agents_info["fmaps"]
        self.batch_size = agents_info["batch_size"]
        self.buff_size = agents_info["buff_size"]
        self.alpha = agents_info["alpha"]
        self.beta_0 = agents_info["beta_0"]
        self.offset = agents_info["offset"]
        self.lr = agents_info["lr"]
        self.policy = agents_info["policy"]
        self.tau = agents_info["tau"]
        self.epsilon_params = agents_info["epsilon_params"]
        self.discount = agents_info["discount"]
        self.seed = agents_info["seed"]
        self.total_training_steps = agents_info["total_training_steps"]
        self.team = agents_info["team"]

        # network mode
        self.train = train

        # set cuda device if train mode and if available
        if self.train:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        # set cpu device (after speed test, found it faster for the given case)
        #self.device = 'cpu'
        print(f"Using {self.device} device")

        # set random seed for each run
        self.rand_generator = np.random.RandomState(self.seed)

        # Compute Huber Loss (mean squared error when error is small, mean
        # absolute error when error is large; predef. param.: beta=1.0.), less sensitive
        # to outliers than MSELoss, sometimes prevents exploding gradients
        # Deactivate mean() operation as we want to include the importance factor
        # in the loss. The mean will be computed manually.
        self.loss_func = nn.SmoothL1Loss(reduction='none')

        # initialise Q networks and target Q networks
        self.q_net = {}
        self.tgt_q_net = {}

        self.q_net[self.team] = DuelingDDQNModel()
        self.tgt_q_net[self.team] = DuelingDDQNModel()
        #self.q_net_blue = DuelingDDQNModel()
        #self.tgt_q_net_blue = DuelingDDQNModel()
        self.q_net[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
            self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        self.tgt_q_net[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
            self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        #self.q_net_blue.model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
        #    self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        #self.tgt_q_net_blue.model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
        #    self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        self.tgt_q_net[self.team].load_state_dict(self.q_net[self.team].state_dict())
        #self.tgt_q_net_blue.load_state_dict(self.q_net_blue.state_dict())

        # initialise model losses
        self.loss = {}
        self.loss[self.team] = None
        #self.loss_red = None
        #self.loss_blue = None

        # define optimisers
        self.optimiser = {}
        self.optimiser[self.team] = torch.optim.Adam(self.q_net[self.team].parameters(), lr=self.lr)
        #self.optimiser_red = torch.optim.Adam(self.q_net_red.parameters(), lr=self.lr)
        #self.optimiser_blue = torch.optim.Adam(self.q_net_blue.parameters(), lr=self.lr)

        # define network modes
        if train:
            self.q_net[self.team].train()
            #self.q_net_red.train()
            #self.q_net_blue.train()
        else:
            self.q_net[self.team].eval()
            #self.q_net_red.eval()
            #self.q_net_blue.eval()
        self.tgt_q_net[self.team].eval()
        #self.tgt_q_net_red.eval()
        #self.tgt_q_net_blue.eval()

        # initialise replay buffer and priorities arrays (prioritised sweeping)
        # programming efficiency: to eliminate old transitions and keep buffer size
        # use deque (fast memory inserting, low speed sampling) if num_insertions >
        # num_samples. Check execution computer speed using tqdm library.
        #self.buffer_red = self.buffer_blue = deque(maxlen=self.buff_size)
        self.team_buffers = {}
        self.team_buffers[self.team] = []
        #self.buffer_red = deque(maxlen=self.buff_size)
        #self.buffer_blue = deque(maxlen=self.buff_size)

        self.prior = {}
        self.prior[self.team] = []
        #self.prior_red = deque(maxlen=self.buff_size)
        #self.prior_blue = deque(maxlen=self.buff_size)

        # initialise importance sampling scaling factor for prioritised sweeping
        self.beta = self.beta_0

        # compute initial epsilon threshold
        if self.policy == 'EpsilonGreedy':
            self.eps_init = self.epsilon_params[0]
            self.eps_fin = self.epsilon_params[1]
            self.eps_dec = self.epsilon_params[2]
            self.eps_thresh = self.eps_fin + (self.eps_init - self.eps_fin) * math.exp(-1. * 0 / self.eps_dec)


    def agents_start(self, observations, env):
        """
        First step of the agents after initialising the experiment. Inputs are
        given by the environment.
        Inputs:
            + observations (dictionary) -> contains observations from all agents.
            Each observation array is constituted as follows:
            {
                'red_0':
                [
                    obstacle [0,1] Channels 1,
                    my_team_presence [0,1] Channels 1,
                    my_team_hp [0,1] Channels 1,
                    my_team_minimap (minimap_mode=True) [0,2] Channels 1,
                    other_team_presence [0,1] Channels 1,
                    other_team_hp [0,1] Channels 1,
                    other_team_minimap (minimap_mode=True) [0,2] Channels 1,
                    binary_agent_id [0,1] Channels 10,
                    one_hot_action (extra_features=True) [0,1] Channels 21,
                    last_reward (extra_features=True) [-0.1,5] Channels 1
                    agent_position (minimap_mode=True) [0,1] Channels 2
                ]
            }
            + env -> environment
        Outputs:
            + actions (dictionary) -> contains actions from all MDP agents. The
            dictionary is organised as presented.
            {
                'red_0': 1, 'red_1': 4, ... , 'blue_10': 14, 'blue_11': 20
            }
        """
        actions = {}
        for agent in env.agents:
            if self.team in agent:
                if self.policy == 'SoftMax':
                    actions[agent] = self.SoftMax(observations[agent], agent)
                elif self.policy == 'EpsilonGreedy':
                    actions[agent] = self.EpsilonGreedy(observations[agent], agent)
                else:
                    raise Exception(f"{self.policy} policy not defined")

        # updating previous observations and actions
        self.prev_observations = observations
        self.prev_actions = actions

        return actions


    def agents_step(self, transitions, num_training_steps, env):
        """
        Step taken by the agents. Inputs are given by env.step().
        Inputs:
            + transitions (tuple) -> MDP transitions
            {
                observations (dictionary) -> contains observations from all agents
                rewards (dictionary) -> contains rewards from all agents
                dones (dictionary) -> boolean indicating termination state of agents
                infos (dictionary) -> agents' information
            }
            + num_training_steps (int) -> assuming same training steps in both teams
            + env (object) -> environment
        Outputs:
            + actions (dictionary) -> contains actions from all MDP agents.
        """
        # copy transitions
        observations = transitions[0]
        rewards = transitions[1]
        dones = transitions[2]

        # initialise actions
        actions = {}

        for agent in env.agents:
            # check agent termination
            if self.team in agent:
                if dones[agent]:
                    print(f"dones: {dones[agent]}")
                    print(f"observation: {observations[agent]}")
                    print(f"rewards: {rewards[agent]}")

                    # insert MDP transition to the replay buffer
                    self.insert_buffer(SARSD(self.prev_observations[agent],
                        self.prev_actions[agent], rewards[agent], observations[agent],
                        dones[agent]), agent)

                    # action required by environment for terminated agents
                    actions[agent] = None
                else:
                    if self.policy == 'SoftMax':
                        actions[agent] = self.SoftMax(observations[agent], agent)
                    elif self.policy == 'EpsilonGreedy':
                        self.eps_thresh = self.eps_fin + (self.eps_init - self.eps_fin) * math.exp(-1. * num_training_steps / self.eps_dec)
                        actions[agent] = self.EpsilonGreedy(observations[agent], agent)
                    else:
                        raise Exception(f"{self.policy} policy not defined")

                    # insert MDP transition to the replay buffer
                    self.insert_buffer(SARSD(self.prev_observations[agent],
                        self.prev_actions[agent], rewards[agent], observations[agent],
                        dones[agent]), agent)

        # update importance sampling beta scaling factor (increasing towards 1
        # by the end of the episode due to inherent biased learning instabilities).
        # beta changes linearly
        self.beta = self.beta_0 + num_training_steps/self.total_training_steps * (1 - self.beta_0)

        # updating previous observations and actions
        self.prev_observations = observations
        self.prev_actions = actions

        return actions


    def agents_end(self, transitions, env):
        """
        Method run when the agents terminate.
        Inputs:
            + transitions (tuple) -> MDP transitions
            {
                observations (dictionary) -> contains observations from all agents
                rewards (dictionary) -> contains rewards from all agents
                dones (dictionary) -> boolean indicating termination state of agents
                infos (dictionary) -> agents' information
            }
            + env (object) -> environment
        Outputs:
            + actions (dict) -> contains actions from all MDP agents
        """
        # copying transitions
        observations = transitions[0]
        rewards = transitions[1]
        dones = transitions[2]

        # initialise actions
        actions = {}

        for agent in env.agents:
            if self.team in agent:
                # insert MDP transition to the replay buffer
                self.insert_buffer(SARSD(self.prev_observations[agent],
                    self.prev_actions[agent], rewards[agent], observations[agent],
                    dones[agent]), agent)

                # void actions
                actions[agent] = None

        return actions


    def train_nets(self):
        """
        Optimisation step to train DuelingDDQN net.
        Inputs:
        Outputs:
            + loss_red (tensor) -> red team model loss
            + loss_blue (tensor) -> blue team model loss
        """
        #for i in range(2):
        train = False
        if len(self.team_buffers[self.team]) >= self.buff_size:
            train = True
                # red team
            #    if i == 0:
            # compute probability vector based on priority array
            scaled_prior = np.array(self.prior[self.team]) ** self.alpha
            prior_prob = scaled_prior / sum(scaled_prior)

            # select random data of batch size from buffers without repetition
            idx = self.rand_generator.choice(len(self.team_buffers[self.team]),
                size=self.batch_size, replace=False, p=prior_prob)
            transitions = list(itemgetter(*idx)(self.team_buffers[self.team]))

            # compute importance sampling from selected transitions to reduce
            # bias from prioritisation (weights are normalised by
            # 1 / max(weights) for stability matters)
            unnorm_imp = (1/len(self.team_buffers[self.team]) * 1/prior_prob[idx]) ** self.beta
            imp = unnorm_imp / max(unnorm_imp)

            # blue team
            #    else:
            #        scaled_prior = np.array(self.prior_blue) ** self.alpha
            #        prior_prob = scaled_prior / sum(scaled_prior)

            #        idx = self.rand_generator.choice(len(self.buffer_blue),
            #            size=self.batch_size, replace=False, p=prior_prob)
            #        transitions = list(itemgetter(*idx)(self.buffer_blue))

            #        unnorm_imp = (1/len(self.buffer_blue) * 1/prior_prob[idx]) ** self.beta
            #        imp = unnorm_imp / max(unnorm_imp)

            # convert importance sampling array to tensor
            imp = torch.from_numpy(imp).to(self.device)

            # initialise actions (no need to be a tensor)
            actions = []

            # arranging data into tensors/lists with proper format (described in nn.Conv2d
            # documentation
            for cc, buff in enumerate(transitions):
                if cc == 0:
                    # conv net
                    states = torch.Tensor([np.transpose(buff.state, (2, 0, 1))]).to(self.device)
                    rewards = torch.Tensor([buff.reward]).to(self.device).unsqueeze(0)
                    next_states = torch.Tensor([np.transpose(buff.next_state, (2, 0, 1))]).to(self.device)
                    masks = torch.Tensor([0 if buff.done else 1]).to(self.device).unsqueeze(0)

                    # linear net
                    #states = torch.Tensor([buff.state]).flatten().unsqueeze(0).to(self.device)
                    #rewards = torch.Tensor([buff.reward]).unsqueeze(0).to(self.device)
                    #next_states = torch.Tensor([buff.next_state]).flatten().unsqueeze(0).to(self.device)
                    #masks = torch.Tensor([0 if buff.done else 1]).unsqueeze(0).to(self.device)
                else:
                    # conv net
                    states = torch.cat((states, torch.Tensor([np.transpose(buff.state, (2, 0, 1))]).to(self.device)), dim=0)
                    rewards = torch.cat((rewards, torch.Tensor([buff.reward]).to(self.device).unsqueeze(0)), dim=0)
                    next_states = torch.cat((next_states, torch.Tensor([np.transpose(buff.next_state, (2, 0, 1))]).to(self.device)), dim=0)
                    masks = torch.cat((masks, torch.Tensor([0 if buff.done else 1]).to(self.device).unsqueeze(0)), dim=0)

                    # linear net
                    #states = torch.cat((states, torch.Tensor([buff.state]).flatten().unsqueeze(0).to(self.device)), dim=0)
                    #rewards = torch.cat((rewards, torch.Tensor([buff.reward]).unsqueeze(0).to(self.device)), dim=0)
                    #next_states = torch.cat((next_states, torch.Tensor([buff.next_state]).flatten().unsqueeze(0).to(self.device)), dim=0)
                    #masks = torch.cat((masks, torch.Tensor([0 if buff.done else 1]).unsqueeze(0).to(self.device)), dim=0)
                actions.append(buff.action)

                # get next state-action values (DDQN: target network state-action value
                # whose action maximises the state-action value from the online learning
                # network)
                with torch.no_grad():
                    #if i == 0:
                    next_q_values_q_net = self.q_net[self.team].forward_pass(next_states)
                    max_ind = torch.argmax(next_q_values_q_net, dim=1, keepdim=True)
                    next_q_values = self.tgt_q_net[self.team].forward_pass(next_states)
                    #else:
                    #    next_q_values_q_net = self.q_net_blue.forward_pass(next_states)
                    #    max_ind = torch.argmax(next_q_values_q_net, dim=1, keepdim=True)
                    #    next_q_values = self.tgt_q_net_blue.forward_pass(next_states)

                # compute update targets of the update
                targets = rewards + self.discount * masks * next_q_values.gather(1, max_ind.long())

                # reset gradients and compute current state action values
                #if i == 0:
                self.optimiser[self.team].zero_grad()
                current_q_values = self.q_net[self.team].forward_pass(states)
                #else:
                #    self.optimiser_blue.zero_grad()
                #    current_q_values = self.q_net_blue.forward_pass(states)

                # programming efficiency: avoid for loop and multiply current
                # q values by a one hot action matrix to obtain corresponding q values
                one_hot_actions = F.one_hot(torch.LongTensor(actions), self.num_actions).to(self.device)
                q_values = torch.sum(current_q_values * one_hot_actions, -1)

                # compute model loss and optimisation step
                #if i == 0:
                # compute the error
                error = self.loss_func(q_values.unsqueeze(1), targets)

                # multiply by importance and compute mean
                self.loss[self.team] = torch.mul(error, imp)
                self.loss[self.team] = torch.mean(self.loss[self.team])

                self.loss[self.team].backward()
                self.optimiser[self.team].step()

                # recompute priorities from buffer replay
                for i, err in zip(idx, error):
                    self.prior[self.team][i] = abs(float(err)) + self.offset

                #else:
                #    self.error_blue = self.loss_func(q_values.unsqueeze(1), targets)

                #    self.loss_blue = torch.mul(self.error_blue, imp)
                #    self.loss_blue = torch.mean(self.loss_blue)

                #    self.loss_blue.backward()
                #    self.optimiser_blue.step()

                #    for i, err in zip(idx, self.error_blue):
                #        self.prior_blue[i] = abs(float(err)) + self.offset
        if train:
            return self.loss[self.team].detach().item(), None#self.loss_blue.detach().item()
        else:
            return None, None


    def update_tgt_nets(self, env):
        """
        Update state-action value target model
        Inputs:
            + env (object) -> environment
        Outputs:
        """
        # red team
        self.tgt_q_net[self.team].load_state_dict(self.q_net[self.team].state_dict())
        self.tgt_q_net[self.team].eval()
        # blue team
        #self.tgt_q_net_blue.load_state_dict(self.q_net_blue.state_dict())
        #self.tgt_q_net_blue.eval()


    def save_net(self, training_steps, eps, cum_rewards, num_steps, loss_nograd, time, team):
        """
        Method called to save all model weights.
        Inputs:
            + training_steps (int) -> number of training steps
            + eps (int) -> number of training episodes
            + cum_rewards (array) -> sum of cumulative rewards per episode
            + num_steps (array) -> number of steps per episode
            + loss_nograd (array) -> array of losses
            + time (float) -> computational time
            + team (string)
        Outputs:
        """
        print(f'Saving {team} team model...')
        #if team == 'red':
        torch.save({
            "model_state_dict": self.q_net[self.team].state_dict(),
            "optimiser_state_dict": self.optimiser[self.team].state_dict(),
            "loss": self.loss[self.team],
            "training_steps": training_steps,
            "episodes": eps,
            "cum_rewards": cum_rewards,
            "num_steps": num_steps,
            "loss_nograd": loss_nograd,
            "time": time
        }, './DuelingDDQNAgents/' + team + f'{training_steps}' + '.tar')

        #elif team == 'blue':
        #    torch.save({
        #        "model_state_dict": self.q_net_blue.state_dict(),
        #        "optimiser_state_dict": self.optimiser_blue.state_dict(),
        #        "loss": self.loss_blue,
        #        "training_steps": training_steps,
        #        "episodes": eps,
        #        "cum_rewards": cum_rewards,
        #        "num_steps": num_steps,
        #        "loss_nograd": loss_nograd,
        #        "time": time
        #    }, './DuelingDDQNAgents/' + team + f'{training_steps}' + '.tar')
        #else:
        #    raise Exception(f'{team} team does not exist')
        print('All data has been saved')

    def load_net(self, training_steps, team):
        """
        Method called to load all model weights.
        Inputs:
            + training_steps (int) -> number of training steps
            + team (string)
        Outputs:
            + eps (int) -> number of training episodes
            + cum_rewards (array) -> sum of cumulative rewards
            + num_steps (array) -> number of steps per episode
            + loss_nograd (array)
        """
        print("Loading model...")
        #if team == 'red':
        checkpoint = torch.load('./DuelingDDQNAgents/' + team + f'{training_steps}' + '.tar')
        self.q_net[team].load_state_dict(checkpoint['model_state_dict'])
        self.tgt_q_net[team].load_state_dict(self.q_net[team].state_dict())
        self.optimiser[team].load_state_dict(checkpoint['optimiser_state_dict'])
        self.loss[team] = checkpoint['loss']
        # network mode
        if self.train:
            self.q_net[team].train()
        else:
            self.q_net[team].eval()

        #elif team == 'blue':
        #    checkpoint = torch.load('./DuelingDDQNAgents/' + team + f'{training_steps}' + '.tar')
        #    self.q_net_blue.load_state_dict(checkpoint['model_state_dict'])
        #    self.tgt_q_net_blue.load_state_dict(self.q_net_blue.state_dict())
        #    self.optimiser_blue.load_state_dict(checkpoint['optimiser_state_dict'])
        #    self.loss_blue = checkpoint['loss']
            # network mode
        #    if self.train:
        #        self.q_net_blue.train()
        #    else:
        #        self.q_net_blue.eval()

        #else:
        #    raise Exception(f'{team} team does not exist')

        # get episodes
        eps = checkpoint['episodes']
        cum_rewards = checkpoint['cum_rewards']
        num_steps = checkpoint['num_steps']
        training_steps = checkpoint['training_steps']
        loss_nograd = checkpoint['loss_nograd']

        print("All data has been loaded")

        return eps, cum_rewards, num_steps, training_steps, loss_nograd


    def insert_buffer(self, MDP_transition, agent):
        """
        Insert MDP transitions into the replay buffer. Information is stored following
        format = [S, A, R, S, done] for each MDP transition.
        Inputs:
            + MDP_transition (tuple) -> MDP transition
            {
                prev_observation (ndarray) -> previous observation
                prev_action (int) -> previous action
                reward (float) -> reinforcement
                observation (ndarray) -> current agent observation
                done (bool) -> indicating if agent is in a terminal state
            }
            + agent (string)
        Outputs:
        """
        if self.team in agent:
        # insert new MDP transitions
        #if 'red' in agent:
            self.team_buffers[self.team].append(MDP_transition)
            self.team_buffers[self.team] = self.team_buffers[self.team][-self.buff_size:]
            #self.buffer_red = self.buffer_red[-self.buff_size:]

            # update priorities (maximum priority for new MDP transitions)
            self.prior[self.team].append(max(self.prior[self.team], default=1))
            self.prior[self.team] = self.prior[self.team][-self.buff_size:]
            #self.prior_red = self.prior_red[-self.buff_size:]

        #elif 'blue' in agent:
        #    self.buffer_blue.append(MDP_transition)
            #self.buffer_blue = self.buffer_blue[-self.buff_size:]

        #    self.prior_blue.append(max(self.prior_blue, default=1))
            #self.prior_blue = self.prior_blue[-self.buff_size:]
        #else:
        #    raise Exception(f'Unidentified {agent} agent')


    def SoftMax(self, observation, agent):
        """
        Soft-max policy.
        Inputs:
            + observation (array) -> information obtained from an agent
            + agent (string) -> corresponding agent
        Outputs:
            + action (int) -> selected action
        """
        # convert observation array into 4D tensor (batch size = 1, Channels, Height, Width)
        obs = torch.from_numpy(np.transpose(observation, (2, 0, 1))).to(self.device)

        # conver observation array into linear net format
        #obs = torch.from_numpy(observation).flatten().to(self.device)

        # current state-action values
        with torch.no_grad():
            #if 'red' in agent:
            qvals = self.q_net[self.team].forward_pass(obs.unsqueeze(0)).squeeze(0)
            #elif 'blue' in agent:
            #    qvals = self.q_net_blue.forward_pass(obs.unsqueeze(0)).squeeze(0)
            #else:
            #    raise Exception(f"Unidentified {agent} agent")

        # compute soft-max policy with normalisation
        num_vec = torch.exp((qvals - min(qvals)) / (self.tau * (max(qvals) - min(qvals)))).cpu()
        prob = np.array(num_vec / torch.sum(num_vec))

        # make sure probabilities sum 1
        prob /= prob.sum()

        # compute argmax with random tie-breaking
        action = self.rand_generator.choice(np.arange(len(prob)), p=prob)

        return action


    def EpsilonGreedy(self, observation, agent):
        """
        Epsilon greedy policy with time step decay.
        Inputs:
            + observation (array) -> information obtained from an agent
            + agent (string)
        Outputs:
            + action (int)
        """
        # explore
        if self.rand_generator.rand() < self.eps_thresh:
            action = self.rand_generator.choice(self.num_actions)
        # greedy action
        else:
            # convert observation array into 4D tensor (batch size = 1, Channels, Height, Width)
            obs = torch.from_numpy(np.transpose(observation, (2, 0, 1))).to(self.device)

            # conver observation array into linear net format
            #obs = torch.from_numpy(observation).flatten().to(self.device)

            # current state-action values
            with torch.no_grad():
            #    if 'red' in agent:
                qvals = self.q_net[self.team].forward_pass(obs.unsqueeze(0)).squeeze(0)
            #    elif 'blue' in agent:
            #        qvals = self.q_net_blue.forward_pass(obs.unsqueeze(0)).squeeze(0)
            #    else:
            #        raise Exception(f"Unidentified {agent} agent")

            action, _ = self.Argmax(qvals)

        return action


    def Argmax(self, qvals):
        """
        Selects greedy action with random tie-breaking.
        Inputs:
            + qvals (tensor) -> state-action values for a given state
        Outputs:
            + action (int) -> index of greedy action
        """
        # setting minimum value to compare with state-action values
        top = float("-inf")
        ties = []

        # comparing state-action values
        for i in range(len(qvals)):
            if qvals[i] > top:
                # updating top value
                top = qvals[i]
                # reseting ties
                ties = []
            # if top is updated initialise ties
            if qvals[i] == top:
                ties.append(i)

        # breaking ties if any
        action = self.rand_generator.choice(ties)
        return action.astype(int), ties
