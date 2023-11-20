import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import time

from tqdm import tqdm
from MDPWrapper import *
from DQNAgents import *
from DuelingDDQNAgents import *
from A3CAgents import *
from PPOAgents import *
from P3OAgents import *
from P3OAgents_copy import *
from RandomAgents import *
from pettingzoo.magent import battlefield_v3, battle_v3, adversarial_pursuit_v3, tiger_deer_v3
from collections import deque

class Experiments(object):
    def main_sim(self, num_episodes=250, max_steps=500, steps_per_tgt_update=20000,
        train_freq=1, seed=0, model_load=False, on_policy=True):
        """
        Agent learning gym for swarms.
        Inputs:
            + num_espisodes (int) -> number of iterations per run
            + max_steps (int) -> maximum number of time steps per episode
            + seed (int) -> random seed
            + steps_per_tgt_update (int) -> steps to update target weights
            + model_load (bool) -> indicator to load models
        Outputs:
        """
        # create algorithm objects
        agents = {
            #"DQNAgents": DQNAgents(),
            #"DuelingDDQNAgents": DuelingDDQNAgents(),
            #"A3CAgents": A3CAgents(),
            #"PPOAgents": PPOAgents(),
            "P3OAgents": P3OAgents(),
            #"P3OAgents_copy": P3OAgents_copy(),
            "RandomAgents": RandomAgents()
        }
        algorithm = "P3OAgents"

        # battlefield_v3 environment
        env = battlefield_v3.parallel_env(map_size=46, minimap_mode=True,
           step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.001,
           attack_opponent_reward=0.2, max_cycles=max_steps, extra_features=False)

        # battle_v3 environment
        # env = battle_v3.parallel_env(map_size=25, minimap_mode=True, step_reward=-0.05,
        #    dead_penalty=-0.1, attack_penalty=-0.01, attack_opponent_reward=0.2,
        #    max_cycles=max_steps, extra_features=False)

        # adversarial_pursuit_v3
        # env = adversarial_pursuit_v3.parallel_env(map_size=25, minimap_mode=True,
        #     tag_penalty=-0.2, max_cycles=max_steps, extra_features=False)

        # tiger_deer_v3
        # env = tiger_deer_v3.parallel_env(map_size=25, minimap_mode=True, tiger_step_recover=-0.1,
        #     deer_attacked=-0.1, max_cycles=max_steps)

        # create MDP wrapper
        wrap = MDPWrapper()

        # initialise agents info
        agents_info = {
            "num_agents": env.num_agents,
            "num_observations": 1521,
            "num_obs_c": 9,
            "num_obs_h": 13,
            "num_obs_w": 13,
            "num_actions": 21,
            "num_hidden_units": [256],
            "kernels": [3, 3, 3],
            "strides": [2, 1, 1],
            "fmaps": [32, 32, 32],
            "batch_size": 64,
            "buff_size": 320,
            "n_steps": 25,
            "alpha": 0.99,
            "beta_0": 0.5,
            "epochs": 3,
            "lambda_gae": 0.95,
            "epsilon_clip": 0.2,
            "critic_discount": 1,
            "entropy": 0.01,
            "offset": 0.1,
            "lr": 1e-04,
            "policy": 'SoftMax',
            "tau": 0.22,
            "epsilon_params": [0.2, 0.001, 50000],
            "discount": 0.99,
            "seed": seed,
            "total_training_steps": max_steps * num_episodes,
            "max_steps_per_episode": max_steps,
            "team": 'red'
        }

        # initialise environment info
        env_info = {
            "seed": seed,
            "render": True,
            "max_steps": max_steps,
            "num_episodes": num_episodes,
            "steps_per_tgt_update": steps_per_tgt_update,
            "train_freq": train_freq
        }

        # initialise wandb visualiser
        wandb.init(project="Nature", name="P3OAgents - MultiNets",
            notes=f"battle_v3 environment, hu={agents_info['num_hidden_units']}, \
                kernels={agents_info['kernels']}, strides={agents_info['strides']}, \
                fmaps={agents_info['fmaps']}, lr={agents_info['lr']}, \
                max_steps={max_steps}, epochs={agents_info['epochs']}, \
                batch_size={agents_info['batch_size']}, \
                buff_size={agents_info['buff_size']}, train_freq={train_freq}, \
                epsilon_clip={agents_info['epsilon_clip']}, lambda_gae={agents_info['lambda_gae']}, \
                entropy={agents_info['entropy']}, n_steps={agents_info['n_steps']}, critic_discount={agents_info['critic_discount']}\
                alpha={agents_info['alpha']}, actor-critic: one network, entity=m-espinos-longa")
        wandb.define_metric("Red Training Steps")
        wandb.define_metric("Red Episodes")
        wandb.define_metric("Red Loss", step_metric="Red Training Steps")
        wandb.define_metric("Red Loss Actor", step_metric="Red Training Steps")
        wandb.define_metric("Red Loss Critic", step_metric="Red Training Steps")
        wandb.define_metric("Averaged Red Loss Actor", step_metric="Red Training Steps")
        wandb.define_metric("Averaged Red Loss Critic", step_metric="Red Training Steps")
        wandb.define_metric("Red Cumulative Rewards", step_metric="Red Training Steps")
        wandb.define_metric("Red Episode Rewards", step_metric="Red Training Steps")
        wandb.define_metric("Red Steps", step_metric="Red Episodes")
        #wandb.define_metric("Blue Training Steps")
        #wandb.define_metric("Blue Loss", step_metric="Blue Training Steps")
        #wandb.define_metric("Blue Cumulative Rewards", step_metric="Blue Training Steps")
        #wandb.define_metric("Blue Steps", step_metric="Blue Episodes")

        # initialise cumulative rewards, number of steps, and training per algorithm
        cum_rewards_red = {}
        num_steps_red = {}
        num_episodes_red = {}
        training_steps_red = {}
        if on_policy:
            #loss_red_actor = {}
            #loss_red_critic = {}
            #loss_red = {}
            loss_actor = {}
            loss_critic = {}
            loss_actor[algorithm] = {}
            loss_critic[algorithm] = {}
        else:
            loss_red = {}

        # initialise computational time
        ct = {}

        # define training mode
        train = True

        # initialise global cumulative rewards and number of steps
        cum_rewards_red[algorithm] = deque(maxlen=num_episodes)
        num_steps_red[algorithm] = deque(maxlen=num_episodes)
        num_episodes_red[algorithm] = 0
        training_steps_red[algorithm] = 0
        if on_policy:
            #loss_red_actor[algorithm] = deque(maxlen=num_episodes)
            #loss_red_critic[algorithm] = deque(maxlen=num_episodes)
            #loss_red[algorithm] = deque(maxlen=num_episodes)
            for agent in env.agents:
                if agents_info["team"] in agent:
                    loss_actor[algorithm][agent] = deque(maxlen=num_episodes)
                    loss_critic[algorithm][agent] = deque(maxlen=num_episodes)
        else:
            loss_red[algorithm] = deque(maxlen=num_episodes)

        #cum_rewards_blue[algorithm] = deque(maxlen=num_episodes)
        #num_steps_blue[algorithm] = deque(maxlen=num_episodes)
        #num_episodes_blue[algorithm] = 0
        #training_steps_blue[algorithm] = 0
        #loss_blue[algorithm] = deque(maxlen=num_episodes)

        ct[algorithm] = deque(maxlen=num_episodes)

        # initialise MDP
        wrap.MDP_init(env, env_info, agents, agents_info, train, on_policy, algorithm)

        # initialise loss in case learning starts from scratch
        if len(os.listdir('./' + algorithm)) == 0 or not model_load:
            if on_policy:
                #loss_red_actor[algorithm].append(0)
                #loss_red_critic[algorithm].append(0)
                #loss_red[algorithm].append(0)
                for agent in env.agents:
                    if wrap.agents[algorithm].team in agent:
                        loss_actor[algorithm][agent].append(0)
                        loss_critic[algorithm][agent].append(0)
            else:
                loss_red[algorithm].append(0)

            ct[algorithm].append(0)
            start = time.time()

        # pre-trained model mode active (no computational time tracking)
        else:
            # [USER: select training steps of model to be loaded]
            ts_load_red = '15000'
            #ts_load_blue = '15000'

            # load weights and data
            eps, cr, ns, ts, l_a, l_c = wrap.agents.load_net(ts_load_red, 'red')
            cum_rewards_red[algorithm] += cr
            num_steps_red[algorithm] += ns
            num_episodes_red[algorithm] += eps
            training_steps_red[algorithm] += ts
            if on_policy:
                #loss_red_actor[algorithm] += l_a
                #loss_red_critic[aglorithm] += l_c
                #loss_red[algorithm] += l
                for agent in env.agents:
                    if wrap.agents[algorithm].team in agent:
                        loss_actor[algorithm][agent] += l_a[agent]
                        loss_critic[algorithm][agent] += l_c[agent]
            else:
                loss_red[algorithm] += l

            #eps, cr, ns, ts = wrap.agents.load_net(ts_load_blue, 'blue')
            #cum_rewards_blue[algorithm] += cr
            #num_steps_blue[algorithm] += ns
            #num_episodes_blue[algorithm] += eps
            #training_steps_blue[algorithm] += ts
            #loss_blue[algorithm] += l

        # episode loop
        for episode in tqdm(range(num_episodes)):
            # run episode
            metrics = wrap.MDP_episode(wandb,
                training_steps_red[algorithm], num_episodes_red[algorithm] + 1,#loss_red[algorithm][episode],
                #loss_red[algorithm][episode], loss_blue[algorithm][episode],
                test=False, perf=True)

            # keep track of metrics (off-policy methods)
            wandb.log({"Red Steps": metrics[1],
                "Red Episode Rewards": metrics[0],
                "Red Episodes": num_episodes_red[algorithm] + 1})
                #"Blue Steps": metrics[1],
                #"Blue Episodes": num_episodes_blue[algorithm] + 1})

            # append results
            cum_rewards_red[algorithm].append(metrics[0])
            num_steps_red[algorithm].append(metrics[1])
            num_episodes_red[algorithm] += 1
            training_steps_red[algorithm] += metrics[1]
            if on_policy:
                #loss_red_actor[algorithm].append(metrics[2])
                #loss_red_critic[algorithm].append(metrics[3])
                #loss_red[algorithm].append(metrics[2])
                for agent in wrap.agent_keys:
                    if wrap.agents[algorithm].team in agent:
                        loss_actor[algorithm][agent].append(metrics[2][agent])
                        loss_critic[algorithm][agent].append(metrics[3][agent])
            else:
                loss_red[algorithm].append(metrics[2])

            # save model weights and experimental data every 4 episodes
            if episode == num_episodes - 1:
                print('Saving networks')
                ct[algorithm].append(time.time() - start)
                if on_policy:
                    # wrap.agents[algorithm].save_net(training_steps_red[algorithm],
                    #     num_episodes_red[algorithm], cum_rewards_red[algorithm],
                    #     num_steps_red[algorithm], loss_red_actor[algorithm], loss_red_critic[algorithm], ct, 'red')
                    # wrap.agents[algorithm].save_net(training_steps_red[algorithm],
                    #     num_episodes_red[algorithm], cum_rewards_red[algorithm],
                    #     num_steps_red[algorithm], loss_red[algorithm], ct, 'red')
                    wrap.agents[algorithm].save_net(training_steps_red[algorithm],
                        num_episodes_red[algorithm], cum_rewards_red[algorithm],
                        num_steps_red[algorithm], loss_actor[algorithm], loss_critic[algorithm], ct, 'red')
                else:
                    wrap.agents[algorithm].save_net(training_steps_red[algorithm],
                        num_episodes_red[algorithm], cum_rewards_red[algorithm],
                        num_steps_red[algorithm], loss_red[algorithm], ct, 'red')

        # end of computation
        end = time.time()
        print(f"Computational time: {end - start}")


    def performance_episode(self, max_steps=5000, steps_per_tgt_update=4000, seed=0,
            model_load=True, train=False, on_policy=True):
        """
        Episode performance visualisation test. To call if certain model weights
        want to be loaded and evaluated visually.
        Inputs:
            + max_steps (int) -> maximum number of time steps per episode
            + seed (int) -> random seed
            + steps_per_tgt_update (int) -> steps to update target weights
            + model_load (bool) -> indicator to load models
        Outputs:
        """
        # create algorithm objects
        agents = {
            #"DQNAgents": DQNAgents(),
            #"DuelingDDQNAgents": DuelingDDQNAgents(),
            #"A3CAgents": A3CAgents(),
            #"PPOAgents": PPOAgents(),
            "P3OAgents": P3OAgents(),
            "RandomAgents": RandomAgents()
        }
        algorithm = "P3OAgents"

        # create environment
        env = battlefield_v3.parallel_env(map_size=46, minimap_mode=True,
            step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1,
            attack_opponent_reward=0.2, max_cycles=max_steps, extra_features=False)

        # battle_v3 environment
        # env = battle_v3.parallel_env(map_size=25, minimap_mode=True, step_reward=-0.005,
        #    dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
        #    max_cycles=max_steps, extra_features=False)

        # create MDP wrapper
        wrap = MDPWrapper()

        # initialise agents info
        agents_info = {
            "num_agents": env.num_agents,
            "num_observations": 1521,
            "num_obs_c": 9,
            "num_obs_h": 13,
            "num_obs_w": 13,
            "num_actions": 21,
            "num_hidden_units": [256],
            "kernels": [3, 3, 3],
            "strides": [2, 1, 1],
            "fmaps": [32, 32, 32],
            "batch_size": 64,
            "buff_size": 60000,
            "n_steps": 25,
            "alpha": 0.99,
            "beta_0": 0.5,
            "epochs": 3,
            "lambda_gae": 0.95,
            "epsilon_clip": 0.2,
            "critic_discount": 1,
            "entropy": 0.01,
            "offset": 0.1,
            "lr": 1e-04,
            "policy": 'SoftMax',
            "tau": 0.22,
            "epsilon_params": [0.2, 0.001, 50000],
            "discount": 0.99,
            "seed": seed,
            "total_training_steps": max_steps * 1,
            "max_steps_per_episode": max_steps,
            "team": 'red'
        }

        # initialise environment info
        env_info = {
            "seed": seed,
            "render": True,
            "max_steps": max_steps,
            "num_episodes": 1,
            "steps_per_tgt_update": steps_per_tgt_update,
            "train_freq": 1
        }

        # initialise MDP
        wrap.MDP_init(env, env_info, agents, agents_info, train, on_policy, algorithm)

        # track q networks training steps
        if model_load:
            # [USER: select training steps of model to be loaded]
            ts_load = '125000'

            # load weights and data
            if on_policy:
                eps, cr, ns, ts, ln, ln = wrap.agents[algorithm].load_net(ts_load, 'red')
            else:
                eps, cr, ns, ts, ln = wrap.agents[algorithm].load_net(ts_load, 'red')
            #eps, cr, ns, ts, ln = wrap.agents[algorithm].load_net(ts_load, 'blue')
        else:
            raise Exception("This mode is only supported with model loading")

        # run episode
        metrics = wrap.MDP_episode(wandb, ts, 1, test=False, perf=True)

        # keep track of metrics
        #print(f"Red Loss: {metrics[2]}, Red Cumulative Rewards: {metrics[0]}, Blue Loss: {metrics[4]} Blue Cumulative Rewards: {metrics[3]}, Steps: {metrics[1]}")
        print(f"Red Loss: {metrics[2]}, Red Cumulative Rewards: {metrics[0]}, Steps: {metrics[1]}")
