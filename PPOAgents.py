import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from P3OModel import *
from torch.distributions import Categorical
from Dataclasses import SARD, SATAP
from operator import itemgetter
from collections import deque
from itertools import islice
from tqdm import tqdm
import math

class PPOAgents(nn.Module):
    """
    Proximal Policy Optimisation Asynchronous Actor-Critic agents.
    Original paper on https://arxiv.org/abs/1707.06347v2.
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
                buff_size (int): common buffer size (batch_size < buff_size)
                n_steps (int): nunber of delayed steps for the update
                epochs (int): number of training iterations over the same data
                lambda_gae (float): generalised advantage estimation parameter
                epsilon_clip (float): clipping parameter
                critic_discount (float): applied to critic loss
                entropy (float): update term to encourage exploration
                lr (float): optimiser learning rate
                discount (float): factor applied to the return function
                seed (int): random seed for reproducibility of experiments
                max_steps_per_episode (int)
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
        self.epochs = agents_info["epochs"]
        self.lambda_gae = agents_info["lambda_gae"]
        self.epsilon_clip = agents_info["epsilon_clip"]
        self.critic_discount = agents_info["critic_discount"]
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
        #self.ppo_net = {}
        #self.ppo_net[self.team] = PPOModel()
        self.ppo_net_actor = {}
        self.ppo_net_critic = {}
        self.ppo_net_actor[self.team] = P3OModel()
        self.ppo_net_critic[self.team] = P3OModel()
        #self.ppo_net_red = PPOModel()
        #self.ppo_net_blue = PPOModel()
        #self.ppo_net[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
        #    self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        self.ppo_net_actor[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
            self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)
        self.ppo_net_critic[self.team].model_init(self.num_obs_c, self.num_obs_h, self.num_obs_w,
            self.num_actions, self.num_hidden_units, self.kernels, self.strides, self.fmaps, self.device)

        # initialise model losses
        #self.loss = {}
        #self.loss[self.team] = None
        self.loss_actor = {}
        self.loss_critic = {}
        self.loss_actor[self.team] = None
        self.loss_critic[self.team] = None

        # define optimisers
        #self.optimiser = {}
        #self.optimiser[self.team] = torch.optim.Adam(self.ppo_net[self.team].parameters(), lr=self.lr)
        self.optimiser_actor = {}
        self.optimiser_critic = {}
        self.optimiser_actor[self.team] = torch.optim.Adam(self.ppo_net_actor[self.team].parameters(), lr=self.lr)
        self.optimiser_critic[self.team] = torch.optim.Adam(self.ppo_net_critic[self.team].parameters(), lr=1e-02)

        # define network modes
        if train:
            #self.ppo_net[self.team].train()
            self.ppo_net_actor[self.team].train()
            self.ppo_net_critic[self.team].train()
        else:
            #self.ppo_net[self.team].eval()
            self.ppo_net_actor[self.team].eval()
            self.ppo_net_critic[self.team].eval()

        self.buffers = {}
        for agent in env.agents:
            # initialise individual buffers
            self.buffers[agent] = []

        # team buffers
        self.team_buffers = {}
        self.team_buffers[self.team] = []
        #self.buffer_red = []
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


    def train_nets(self, training_steps, agent, ep_steps):
        """
        Optimisation step to train PPO net.
        Inputs:
            + training_steps (int)
            + agent (string)
            + ep_steps (int) -> number of episode steps
        Outputs:
            + loss_red (tensor) -> red team model loss
            + loss_blue (tensor) -> blue team model loss
        """
        train = False
        if len(self.buffers[agent]) >= self.buff_size or ep_steps == self.max_steps_per_episode:
            if self.team in agent:
                train = True
                # compute targets backwards
                for t, buff in enumerate(reversed(self.buffers[agent])):
                    # convert tranistion to tensors
                    state = torch.Tensor([np.transpose(buff.state, (2, 0, 1))]).to(self.device)
                    action = buff.action
                    reward = torch.Tensor([buff.reward]).to(self.device).unsqueeze(0)
                    mask = torch.Tensor([0 if buff.done else 1]).to(self.device).unsqueeze(0)

                    # forward net pass
                    dist = self.ppo_net_actor[self.team].forward_pass_actor(state)
                    value = self.ppo_net_critic[self.team].forward_pass_critic(state)

                    # in case t = t_max
                    if t == 0:
                        # compute next value, initialise advantage, and pass
                        next_value = mask * value
                        adv = 0
                    else:
                        # compute target and store
                        target = reward + self.discount * next_value

                        # update next value
                        next_value = mask * value

                        # compute TD error
                        delta = target - value

                        # advantage
                        adv = delta + self.discount * self.lambda_gae * adv

                        # compute log policy
                        policy = F.softmax(dist, dim=1)

                        # insert objective data into team buffer
                        self.team_buffers[self.team].insert(0, SATAP(state, action, target.detach(),
                            adv.detach(), policy.detach()))

                #elif 'blue' in agent:
                    # forward pass
                #    value, dist = self.ppo_net_blue.forward_pass(state)

                #    if t == 0:
                        # compute next value, initialise advantage, and pass
                #        next_value = mask * value
                #        adv = 0

                #    else:
                        # compute target and store
                #        target = reward + self.discount * next_value

                        # update next value
                #        next_value = mask * value

                        # compute TD error
                #        delta = target - value

                        # advantage
                #        adv = delta + self.discount * self.lambda_gae * adv

                        # compute log policy
                #        policy = F.softmax(dist, dim=1)

                        # insert objective data into team buffer
                #        self.buffer_blue.append(SATAP(state, action, target.detach(),
                #            adv.detach(), policy.detach()))

                #else:
                #    raise Exception(f"Unrecognised {agent} agent")

                # reset agent buffer
                    if t == len(self.buffers[agent]) - 1:
                        self.buffers[agent] = []

        # compute number and size of mini-batches
        num_mini_batch = {}
        num_mini_batch[self.team] = len(self.team_buffers[self.team]) // self.batch_size
        #num_mini_batch["red"] = len(self.buffer_red) // self.batch_size
        #num_mini_batch["blue"] = len(self.buffer_blue) // self.batch_size

        # training loop
        #for team in ['red', 'blue']:
        for _ in range(self.epochs):
            # shuffle data every epoch
            #self.rand_generator.shuffle(self.team_buffers[self.team])
        #    self.rand_generator.shuffle(self.buffer_blue)

            # mini-batches
            for mb in range(num_mini_batch[self.team]):
                # red team
        #        if team == 'red':
                        # get mini-batch
                transitions = self.team_buffers[self.team][
                    mb*self.batch_size:(mb+1)*self.batch_size]
                    # blue team
        #            elif team == 'blue':
        #                transitions = self.buffer_blue[
        #                    mb*self.batch_size:(mb+1)*self.batch_size]
        #            else:
        #                raise Exception(f"Unrecognised {team} team")

                    # initialise actions
                actions = []

                # recover data from team buffers
                for cc, buff in enumerate(transitions):
                    if cc == 0:
                        states = buff.state
                        targets = buff.target
                        advantages = buff.advantage
                        old_policies = buff.policy
                    else:
                        states = torch.cat((states, buff.state), dim=0)
                        targets = torch.cat((targets, buff.target), dim=0)
                        advantages = torch.cat((advantages, buff.advantage), dim=0)
                        old_policies = torch.cat((old_policies, buff.policy), dim=0)
                    actions.append(buff.action)

        #            if team == 'red':
                dists = self.ppo_net_actor[self.team].forward_pass_actor(states)
                values = self.ppo_net_critic[self.team].forward_pass_critic(states)
                #    elif team == 'blue':
                #        values, dists = self.ppo_net_blue.forward_pass(states)

                # compute ratios (self.eps to prevent NaN if probs drop)
                log_policies = F.log_softmax(dists, dim=1)
                policies = F.softmax(dists, dim=1)
                action_policies = policies[actions]
                old_action_policies = old_policies[actions]
                ratios = action_policies / (old_action_policies + 0.000001)
                #ratios = action_policies - old_action_policies

                # compute surrogates and actor loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # compute critic loss
                critic_loss = self.critic_discount * self.loss_func(values, targets)

                # compute entropy loss (to encourage exploration)
                entropy_loss = -self.entropy * (policies * log_policies).sum(dim=1).mean()

                # compute model loss and optimisation step
                #    if team == 'red':
                # reset optimiser gradients
                #self.optimiser[self.team].zero_grad()
                self.optimiser_actor[self.team].zero_grad()
                self.optimiser_critic[self.team].zero_grad()

                # compute total loss
                #self.loss[self.team] = actor_loss + critic_loss + entropy_loss
                self.optimiser_actor[self.team].zero_grad()
                self.optimiser_critic[self.team].zero_grad()

                self.loss_actor[self.team] = actor_loss + entropy_loss
                self.loss_critic[self.team] = critic_loss

                #self.loss[self.team].backward()
                self.loss_actor[self.team].backward()
                self.loss_critic[self.team].backward()
                #self.optimiser[self.team].step()
                self.optimiser_actor[self.team].step()
                self.optimiser_critic[self.team].step()

                #    elif team == 'blue':
                #        self.optimiser_blue.zero_grad()

                #        self.loss_blue = actor_loss + critic_loss + entropy_loss
#
#                        self.loss_blue.backward()
#                        self.optimiser_blue.step()

        # reset team buffers
        self.team_buffers[self.team] = []
        #self.buffer_red = []
        #self.buffer_blue = []

        if train:
            return self.loss_actor[self.team].detach().item(), self.loss_critic[self.team].detach().item()#, self.loss_blue.detach().item()
        else:
            return None, None

    def save_net(self, training_steps, eps, cum_rewards, num_steps, loss_nograd_actor, loss_nograd_critic, time, team):
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
            "model_state_dict": self.ppo_net_actor[team].state_dict(),
            "optimiser_state_dict": self.optimiser_actor[team].state_dict(),
            "loss": self.loss_actor[team],
            "training_steps": training_steps,
            "episodes": eps,
            "cum_rewards": cum_rewards,
            "num_steps": num_steps,
            "loss_nograd": loss_nograd_actor,
            "time": time
        }, './PPOAgents/' + 'actor' + team + f'{training_steps}' + '.tar')

        torch.save({
            "model_state_dict": self.ppo_net_critic[team].state_dict(),
            "optimiser_state_dict": self.optimiser_critic[team].state_dict(),
            "loss": self.loss_critic[team],
            "training_steps": training_steps,
            "episodes": eps,
            "cum_rewards": cum_rewards,
            "num_steps": num_steps,
            "loss_nograd": loss_nograd_critic,
            "time": time
        }, './PPOAgents/' + 'critic' + team + f'{training_steps}' + '.tar')

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
        checkpoint = torch.load('./PPOAgents/' + 'actor' + team + f'{training_steps}' + '.tar')
        self.ppo_net_actor[self.team].load_state_dict(checkpoint['model_state_dict'])
        self.optimiser_actor[self.team].load_state_dict(checkpoint['optimiser_state_dict'])
        self.loss_actor[self.team] = checkpoint['loss']

        loss_nograd_actor = checkpoint['loss_nograd']

        checkpoint = torch.load('./PPOAgents/' + 'critic' + team + f'{training_steps}' + '.tar')
        self.ppo_net_critic[self.team].load_state_dict(checkpoint['model_state_dict'])
        self.optimiser_critic[self.team].load_state_dict(checkpoint['optimiser_state_dict'])
        self.loss_critic[self.team] = checkpoint['loss']

        # network mode
        if self.train:
            self.ppo_net_actor[self.team].train()
            self.ppo_net_critic[self.team].train()
        else:
            self.ppo_net_actor[self.team].eval()
            self.ppo_net_critic[self.team].eval()

        # get episodes
        eps = checkpoint['episodes']
        cum_rewards = checkpoint['cum_rewards']
        num_steps = checkpoint['num_steps']
        training_steps = checkpoint['training_steps']
        loss_nograd_critic = checkpoint['loss_nograd']

        print("All data has been loaded")

        return eps, cum_rewards, num_steps, training_steps, loss_nograd_actor, loss_nograd_critic


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
        # insert new MDP transitions
        if self.team in agent:
            self.buffers[agent].append(MDP_transition)


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
            dist = self.ppo_net_actor[self.team].forward_pass_actor(obs.unsqueeze(0))


        # apply softmax layer
        prob = F.softmax(dist.squeeze(0), dim=0).cpu()

        # compute argmax with random tie-breaking
        action = self.rand_generator.choice(np.arange(len(prob)), p=np.array(prob))

        return action
