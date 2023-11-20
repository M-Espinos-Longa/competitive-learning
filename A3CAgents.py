import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from A3CModel import *
from Dataclasses import SARD, VTLLP
from operator import itemgetter
from collections import deque
from tqdm import tqdm
import math

class A3CAgents(nn.Module):
    """
    Asynchronous Advantage Actor-Critic agents.
    Original paper on https://arxiv.org/abs/1602.01783v2.
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
                buff_size (int): common memory size (batch_size < buff_size)
                n_steps (int): nunber of delayed steps for the update
                entropy (float): update term to encourage exploration
                lr (float): optimiser learning rate
                discount (float): factor applied to the return function
                seed (int): random seed for reproducibility of experiments
                max_steps_per_episode (int)
                team (string)
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
        self.n_steps = agents_info["n_steps"]
        self.entropy = agents_info["entropy"]
        self.lr = agents_info["lr"]
        self.discount = agents_info["discount"]
        self.seed = agents_info["seed"]
        self.max_steps_per_episode = agents_info["max_steps_per_episode"]
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
        self.loss_func = nn.SmoothL1Loss()

        # initialise acotr-critic networks
        self.a3c_net = {}
        self.a3c_net[self.team] = A3CModel()
        self.a3c_net[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
            self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        #self.a3c_net_blue.model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
        #    self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)

        # initialise model losses
        self.loss = {}
        self.loss[self.team] = None
        #self.loss_blue = None

        # define optimisers
        self.optimiser = {}
        self.optimiser[self.team] = torch.optim.Adam(self.a3c_net[self.team].parameters(), lr=self.lr)
        #self.optimiser_blue = torch.optim.Adam(self.a3c_net_blue.parameters(), lr=self.lr)

        # define network modes
        if train:
            self.a3c_net[self.team].train()
            #self.a3c_net_blue.train()
        else:
            self.a3c_net[self.team].eval()
            #self.a3c_net_blue.eval()

        self.buffers = {}
        for agent in env.agents:
            if self.team in agent:
                # initialise individual buffers
                self.buffers[agent] = []

        # team buffers
        self.team_buffers = {}
        self.team_buffers[self.team] = []
        #self.buffer_blue = []


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
        # actor policy
        actions = {}
        for agent in env.agents:
            if self.team in agent:
                actions[agent] = self.actor_policy(observations[agent], agent)

        #actions = {agent: self.actor_policy(observations[agent], agent) for agent in env.agents}

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
                    self.insert_buffer(SARD(self.prev_observations[agent],
                        self.prev_actions[agent], rewards[agent], dones[agent]), agent)

                    # action required by environment for terminated agents
                    actions[agent] = None
                else:
                    # actor policy
                    actions[agent] = self.actor_policy(observations[agent], agent)

                    # insert MDP transition to the replay buffer
                    self.insert_buffer(SARD(self.prev_observations[agent],
                        self.prev_actions[agent], rewards[agent], dones[agent]), agent)

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
                self.insert_buffer(SARD(self.prev_observations[agent],
                    self.prev_actions[agent], rewards[agent], dones[agent]), agent)

                # void actions
                actions[agent] = None

        return actions

    def update_n_steps(self, agent_keys):
        """
        Optimisation step to train A3C net.
        Inputs:
            + agent_keys (dict) -> name of all existing agents
        Outputs:
        """
        self.n_step += 1
        torch.autograd.set_detect_anomaly(True)
        if self.n_step >= self.n_steps:
            for agent in agent_keys:
                if self.team in agent:
                    # compute targets backwards
                    self.buffers[agent].reverse()
                    for t, buff in enumerate(self.buffers[agent]):
                        # convert tranistion to tensors
                        state = torch.Tensor([np.transpose(buff.state, (2, 0, 1))]).to(self.device)
                        action = buff.action
                        reward = torch.Tensor([buff.reward]).to(self.device).unsqueeze(0)
                        mask = torch.Tensor([0 if buff.done else 1]).to(self.device).unsqueeze(0)

                    # forward net pass
                    #if 'red' in agent:
                        # forward net pass
                        value, policy = self.a3c_net[self.team].forward_pass(state)

                        if t == 0:
                            # compute first target and pass
                            target = mask * value.clone()

                        else:
                            # compute target and store
                            target = reward + self.discount * target.clone()

                            # compute advantage
                            adv = target.clone() - value.detach()

                            # compute log probability
                            log_prob = F.log_softmax(policy.clone(), dim=1).squeeze(0)
                            log_prob_action = (adv * (log_prob[action].clone())).unsqueeze(0)

                            # compute probability
                            prob = F.softmax(policy, dim=1)

                            self.team_buffers[self.team].append(VTLLP(value, target, log_prob.unsqueeze(0),
                                log_prob_action, prob))

                    #elif 'blue' in agent:
                        # forward pass
                    #    value, policy = self.a3c_net_blue.forward_pass(state)

                    #    if t == 0:
                            # compute first target and pass
                    #        target = mask * value

                    #    else:
                            # compute target and store
                    #        target = reward + self.discount * target

                            # compute advantage
                    #        adv = target - value.detach()

                            # compute log probability
                    #        log_prob = F.log_softmax(policy, dim=1).squeeze(0)
                    #        log_prob_action = (adv * (log_prob[action])).unsqueeze(0)

                            # compute probability
                    #        prob = F.softmax(policy, dim=1)

                    #        self.buffer_blue.append(VTLLP(value, target, log_prob.unsqueeze(0),
                    #            log_prob_action, prob))

                    #else:
                    #    raise Exception(f"Unrecognised {agent} agent")

                        # reset agent buffer
                        if t == len(self.buffers[agent]) - 1:
                            self.buffers[agent] = []
            # reset n_step
            self.n_step = 0


    def train_nets(self, training_steps, agent, ep_steps):
        """
        Optimisation step to train A3C net.
        Inputs:
            + training_steps (int)
            + agent (string)
            + ep_steps (int) -> episode steps
        Outputs:
            + loss_red (tensor) -> red team model loss
            + loss_blue (tensor) -> blue team model loss
        """
        #torch.autograd.set_detect_anomaly(True)
        train = False
        # training loop
        if ep_steps == self.max_steps_per_episode:
            if self.team in agent:
                self.buffers[agent] = [] 

        if len(self.buffers[agent]) >= self.batch_size:
            if self.team in agent:
                train = True
                # compute targets backwards
                self.buffers[agent].reverse()
                for t, buff in enumerate(self.buffers[agent]):
                    # convert tranistion to tensors
                    state = torch.Tensor([np.transpose(buff.state, (2, 0, 1))]).to(self.device)
                    action = buff.action
                    reward = torch.Tensor([buff.reward]).to(self.device).unsqueeze(0)
                    mask = torch.Tensor([0 if buff.done else 1]).to(self.device).unsqueeze(0)

                    # forward net pass
                    value, policy = self.a3c_net[self.team].forward_pass(state)

                    if t == 0:
                        # compute first target and pass
                        target = mask * value
                    else:
                        # compute target and store
                        target = reward + self.discount * target

                        # compute advantage
                        adv = target - value.detach()

                        # compute log probability
                        log_prob = F.log_softmax(policy, dim=1).squeeze(0)
                        log_prob_action = (adv * (log_prob[action])).unsqueeze(0)

                        # compute probability
                        prob = F.softmax(policy, dim=1)

                        self.team_buffers[self.team].insert(0, VTLLP(value, target, log_prob.unsqueeze(0),
                            log_prob_action, prob))

                    # reset agent buffer
                    if t == len(self.buffers[agent]) - 1:
                        self.buffers[agent] = []

                # shuffle data
                #self.rand_generator.shuffle(self.team_buffers[self.team])

                # get transitions
                transitions = self.team_buffers[self.team]

                # recover data from team buffers
                for cc, buff in enumerate(transitions):
                    if cc == 0:
                        values = buff.value
                        targets = buff.target
                        log_probs = buff.log_prob
                        log_prob_actions = buff.log_prob_action
                        probs = buff.prob

                    else:
                        values = torch.cat((values, buff.value), dim=0)
                        targets = torch.cat((targets, buff.target), dim=0)
                        log_probs = torch.cat((log_probs, buff.log_prob), dim=0)
                        log_prob_actions = torch.cat((log_prob_actions, buff.log_prob_action), dim=0)
                        probs = torch.cat((probs, buff.prob), dim=0)

                # compute critic loss
                critic_loss = self.loss_func(values, targets)

                # compute actor loss (the minus sign is because it is a policy gradient
                # ascent, we want to maximise the reward, but pytorch is optimised to minimise
                # a function. Hence, instead of maximising PI policy, we minimise 1 - PI)
                actor_loss = -log_prob_actions.mean()

                # compute entropy loss (to encourage exploration)
                entropy_loss = -self.entropy * (probs * log_probs).sum(dim=1).mean()

                # compute model loss and optimisation step
                #if team == 'red':
                # reset optimiser gradients
                self.optimiser[self.team].zero_grad()

                # compute total loss
                self.loss[self.team] = critic_loss + actor_loss + entropy_loss

                self.loss[self.team].backward()
                self.optimiser[self.team].step()

                # reset team buffers
                self.team_buffers[self.team] = []

        #if len(self.team_buffers[self.team]) >= self.buff_size:
        #    train = True
            #for team in ['red']:
                # red team
                #if team == 'red':
                # randomise and select red team buffer data
        #    transitions = self.rand_generator.choice(self.team_buffers[self.team],
        #        size=self.batch_size, replace=False)

                ## blue team
                #elif team == 'blue':
                #    transitions = self.rand_generator.choice(self.buffer_blue,
                #        size=len(self.buffer_blue), replace=False)

                #else:
                #    raise Exception(f"Unrecognised {team} team")

            # recover data from team buffers
            # for cc, buff in enumerate(transitions):
            #     if cc == 0:
            #         values = buff.value
            #         targets = buff.target
            #         log_probs = buff.log_prob
            #         log_prob_actions = buff.log_prob_action
            #         probs = buff.prob
            #
            #     else:
            #         values = torch.cat((values, buff.value), dim=0)
            #         targets = torch.cat((targets, buff.target), dim=0)
            #         log_probs = torch.cat((log_probs, buff.log_prob), dim=0)
            #         log_prob_actions = torch.cat((log_prob_actions, buff.log_prob_action), dim=0)
            #         probs = torch.cat((probs, buff.prob), dim=0)
            #
            # # compute critic loss
            # critic_loss = self.loss_func(values, targets)
            #
            # # compute actor loss (the minus sign is because it is a policy gradient
            # # ascent, we want to maximise the reward, but pytorch is optimised to minimise
            # # a function. Hence, instead of maximising PI policy, we minimise 1 - PI)
            # actor_loss = -log_prob_actions.mean()
            #
            # # compute entropy loss (to encourage exploration)
            # entropy_loss = -self.entropy * (probs * log_probs).sum(dim=1).mean()
            #
            # # compute model loss and optimisation step
            # #if team == 'red':
            # # reset optimiser gradients
            # self.optimiser[self.team].zero_grad()
            #
            # # compute total loss
            # self.loss[self.team] = critic_loss + actor_loss + entropy_loss
            #
            # self.loss[self.team].backward()
            # self.optimiser[self.team].step()

                #elif team == 'blue':
                #    self.optimiser_blue.zero_grad()

                #    self.loss_blue = critic_loss + actor_loss + entropy_loss

                #    self.loss_blue.backward()
                #    self.optimiser_blue.step()

                #else:
                #    raise Exception(f"Unrecognised {team} team")

        # reset team buffers
        #self.team_buffers[self.team] = []
        #self.buffer_blue = []
        if train:
            return self.loss[self.team].detach().item(), None#self.loss_blue.detach().item()
        else:
            return None, None


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
        torch.save({
            "model_state_dict": self.a3c_net[team].state_dict(),
            "optimiser_state_dict": self.optimiser[team].state_dict(),
            "loss": self.loss[team],
            "training_steps": training_steps,
            "episodes": eps,
            "cum_rewards": cum_rewards,
            "num_steps": num_steps,
            "loss_nograd": loss_nograd,
            "time": time
        }, './A3CAgents/' + team + f'{training_steps}' + '.tar')

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
        if team == 'red':
            checkpoint = torch.load('./A3CAgents/' + team + f'{training_steps}' + '.tar')
            self.a3c_net[team].load_state_dict(checkpoint['model_state_dict'])
            self.optimiser[team].load_state_dict(checkpoint['optimiser_state_dict'])
            self.loss[team] = checkpoint['loss']
            # network mode
            if self.train:
                self.a3c_net[team].train()
            else:
                self.a3c_net[team].eval()

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
            self.buffers[agent].append(MDP_transition)
            self.team_buffers[self.team] = self.team_buffers[self.team][-self.buff_size:]


    def actor_policy(self, observation, agent):
        """
        Actor policy.
        Inputs:
            + observation (array) -> information obtained from an agent
            + agent (string) -> corresponding agent
        Outputs:
            + action (int) -> selected action
        """
        # convert observation array into 4D tensor (batch size = 1, Channels, Height, Width)
        obs = torch.from_numpy(np.transpose(observation, (2, 0, 1))).to(self.device)

        # current state-action values
        with torch.no_grad():
            _, dist = self.a3c_net[self.team].forward_pass(obs.unsqueeze(0))

        # apply softmax layer
        prob = F.softmax(dist.squeeze(0), dim=0).cpu()

        # compute argmax with random tie-breaking
        action = self.rand_generator.choice(np.arange(len(prob)), p=np.array(prob))

        return action
