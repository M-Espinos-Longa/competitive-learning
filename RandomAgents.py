import numpy as np
import random

class RandomAgents():
    """
    Random agents
    """
    def agents_init(self, agents_info, env, train):
        """
        Setup for the agents called when the experiment starts
        Inputs:
            + agents_info (dict) -> key parameters to initialise agents
            {
                num_agents (int): number of learning agents involved in the game
                num_observations (int): number of environment observations per agent
                num_actions (int): number of available actions per agent
                seed (int): random seed for reproducibility of experiments
            }
        Outputs:
        """
        # store provided parameters in agents_info
        self.num_agents = agents_info["num_agents"]
        self.num_observations = agents_info["num_observations"]
        self.num_actions = agents_info["num_actions"]
        self.seed = agents_info["seed"]
        self.team = agents_info["team"]

        # set random seed for each run
        self.rand_generator = np.random.RandomState(self.seed)


    def agents_start(self, observations, env):
        """
        First step of the agents after initialising the experiment Inputs are given
        by the environment.
        Inputs:
            + env -> environment
        Outputs:
            + actions (dictionary) -> contains actions from all MDP agents. The
            dictionary is organised as presented.
            {
                'red_0': 1, 'red_1': 4, ... , 'blue_10': 14, 'blue_11': 20
            }
        """
        # random action selection
        actions = {}
        for agent in env.agents:
            if self.team in agent:
                actions[agent] = self.rand_generator.choice(range(self.num_actions))

        return actions


    def agents_step(self, transitions, num_steps, env):
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
            + num_steps (int)
            + env (object) -> environment
        Outputs:
            + actions (dictionary) -> contains actions from all MDP agents
        """
        # copy transitions
        dones = transitions[2]

        # initialise actions
        actions = {}

        for agent in env.agents:
            if self.team in agent:
                if not dones[agent]:
                    actions[agent] = self.rand_generator.choice(range(self.num_actions))
                else:
                    actions[agent] = None

        return actions


    def agents_end(self, env):
        """
        Method run when the agents terminate.
        Inputs:
            + env (object) -> environment
        Outputs:
            + actions (dict) -> contains actions from all MDP agents
        """
        # initialise actions
        actions = {}

        for agent in env.agents:
            if self.team in agent:
                actions[agent] = None

        return actions
