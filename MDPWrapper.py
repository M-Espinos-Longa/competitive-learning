import numpy as np
import time
from tqdm import tqdm
import wandb

class MDPWrapper(object):
    def MDP_init(self, env, env_info, agents, agents_info, train, on_policy, algorithm):
        """
        Firsth method called when running an experiment. Environment and agent
        objects are created and initialised.
        Inputs:
            + env (object) -> environment
            + env_info (dict) -> environment settings
            {
                seed (int) -> random seed
                render (bool) -> visualisation
                max_steps (int) -> maximum number of steps per episode
                steps_per_tgt_update (int) -> target update frequency
                train_freq (int)
            }
            + agents (object) -> agents under x algorithm
            + agents_info (dict) -> key parameters to initialise agents (specific
              to "TYPEAgents.py")
            + train (bool) -> training mode
            + on_policy (bool) -> as far as on-policy RL methods are concerned
            + algorithm (string)
        Outputs:
        """
        # initialise environment
        seed = env_info["seed"]
        self.render = env_info["render"]
        self.max_steps = env_info["max_steps"]
        self.num_episodes = env_info["num_episodes"]
        self.steps_per_tgt_update = env_info["steps_per_tgt_update"]
        self.train_freq = env_info["train_freq"]
        self.team = agents_info["team"]
        self.env = env
        if self.render:
            self.env.render()

        self.env.seed(seed=seed)
        self.train = train
        self.on_policy = on_policy
        self.algorithm = algorithm

        # competitive play against different agent types (performance episode)
        if isinstance(agents, dict):
            self.agents = {}
            for i, key in enumerate(agents.keys()):
                self.agents[key] = agents[key]
                self.agents[key].agents_init(agents_info, self.env, self.train)
                if i == 0:
                    agents_info["team"] = 'blue'
        else:
            # initialise agents
            self.agents = agents
            self.agents.agents_init(agents_info, self.env, self.train)


    def MDP_episode(self, wandb, training_steps_red, num_episodes_red, test=False, perf=False):
        """
        Method called when running a full episode.
        Inputs:
            + wandb (obj) -> metrics visualisation tool
            + training_steps_red (int) -> number of training steps red team
            + num_episodes_red (int) -> number of training episodes
            + test (bool) -> testing mode
            + perf (bool) -> performance mode
        Outputs:
            + cum_rewards (float) -> sum of swarm's cumulative rewards
            + num_steps (int) -> number of steps required to end the episode
            + total_loss (float) -> total agent model losses
        """
        # initialise outputs
        #cum_rewards = {agent: 0.0 for agent in self.env.agents}
        cum_rewards_red = 0.0
        #cum_rewards_blue = 0.0
        num_steps = 0

        enemy_agents = 20

        # first environment step
        if num_episodes_red > 1:
            init_obs = self.env.reset()
            init_obs = self.env.reset()
        else:
            init_obs = self.env.reset()

        if self.render:
            self.env.render()

        # keep agent keys
        self.agent_keys = self.env.agents

        # if we want two different types of agents (one for each team), merge
        # action dictionaries
        if perf:
            actions = {}
            for key in self.agents.keys():
                actions[key] = self.agents[key].agents_start(init_obs, self.env)
            keys = list(self.agents.keys())
            actions = self.MergeActions(actions[keys[0]], actions[keys[1]])
        else:
            actions = self.agents.agents_start(init_obs, self.env)

        # initialise losses
        if self.on_policy:
            #loss_red_actor = None
            #loss_red_critic = None
            #loss_red = None
            loss_actor = {}
            loss_critic = {}
            for agent in self.agent_keys:
                if self.agents[self.algorithm].team in agent:
                    loss_actor[agent] = None
                    loss_critic[agent] = None
            averaged_loss_actor = None
            averaged_loss_critic = None
        else:
            loss_red = None

        # episode loop
        for ns in tqdm(range(self.max_steps)):
            # environment step
            transitions = self.env.step(actions)

            if self.render:
                self.env.render()

            for key, r in transitions[1].items():
                if self.team in key:
                    cum_rewards_red += r
                #elif 'blue' in key:
                #    cum_rewards_blue += r
                #else:
                #    raise Exception(f'Unidentified {key} agent')
            num_steps += 1
            training_steps_red += 1
            #training_steps_blue += 1

            # check MDP termination
            if all(transitions[2].values()):
                if perf:
                    actions = self.agents[self.algorithm].agents_end(transitions, self.env)
                else:
                    actions = self.agents.agents_end(transitions, self.env)
            # asume training steps from both teams are equal
            else:
                # performance episode mode
                if perf:
                    actions = {}
                    for key in self.agents.keys():
                        actions[key] = self.agents[key].agents_step(transitions, training_steps_red, self.env)
                    keys = list(self.agents.keys())
                    actions = self.MergeActions(actions[keys[0]], actions[keys[1]])
                # training mode
                else:
                    actions = self.agents.agents_step(transitions, training_steps_red, self.env)

            # training step for off-policy methods
            if not self.on_policy:
                if num_steps % self.train_freq == 0 and self.train:
                    loss_red, _ = self.agents[self.algorithm].train_nets()
                    if loss_red != None:
                        # track metrics
                        if self.train:
                            wandb.log({"Red Loss": loss_red,
                                "Red Cumulative Rewards": cum_rewards_red,
                                "Red Training Steps": training_steps_red})
                                #"Blue Loss": loss_blue,
                                #"Blue Cumulative Rewards": cum_rewards_blue,
                                #"Blue Training Steps": training_steps_blue})
                    else:
                        wandb.log({#"Red Loss": loss_red,
                            "Red Cumulative Rewards": cum_rewards_red,
                            "Red Training Steps": training_steps_red})

            # training step for on-policy methods
            if self.on_policy and self.train and num_steps % self.train_freq == 0:
                if perf:
                    #averaged_loss_actor = []
                    #averaged_loss_critic = []
                    #self.agents[self.algorithm].update_n_steps(self.agent_keys)
                    for agent in self.agent_keys:
                        if self.agents[self.algorithm].team in agent:
                            #loss_red, _ = self.agents[self.algorithm].train_nets(training_steps_red, agent, num_steps)
                            loss_red_actor, loss_red_critic = self.agents[self.algorithm].train_nets(training_steps_red, agent, num_steps)
                            # loss_actor[agent], loss_critic[agent] = self.agents[self.algorithm].train_nets(training_steps_red, agent, num_steps)
                            # if loss_actor[agent] != None and loss_critic[agent] != None:
                            #     averaged_loss_actor.append(loss_actor[agent])
                            #     averaged_loss_critic.append(loss_critic[agent])
                else:
                    loss_red, _ = self.agents.train_nets(self.agent_keys, training_steps_red)

                # track metrics
                if loss_red_actor != None and loss_red_critic != None:
                #if loss_red != None:
                #if len(averaged_loss_actor) > 0 and len(averaged_loss_critic) > 0:
                    if self.train:
                        wandb.log({#"Red Loss": loss_red,
                            "Red Loss Actor": loss_red_actor,
                            "Red Loss Critic": loss_red_critic,
                            #"Averaged Red Loss Actor": sum(averaged_loss_actor)/len(averaged_loss_actor),
                            #"Averaged Red Loss Critic": sum(averaged_loss_critic)/len(averaged_loss_critic),
                            "Red Cumulative Rewards": cum_rewards_red,
                            "Red Training Steps": training_steps_red})
                            #"Blue Loss": loss_blue,
                            #"Blue Cumulative Rewards": cum_rewards_blue,
                            #"Blue Training Steps": training_steps_blue})
                else:
                    if self.train:
                        wandb.log({"Red Cumulative Rewards": cum_rewards_red,
                            "Red Training Steps": training_steps_red})
                            #"Red Loss": loss_red,
                            #"Red Loss Actor": loss_red_actor,
                            #"Red Loss Critic": loss_red_critic,


            # target model update (DQN and DuelingDDQN networks)
            if num_steps % self.steps_per_tgt_update == 0 and self.train and not self.on_policy:
                self.agents.update_tgt_nets(self.env)

            # debug step or premature end
            if test or not self.env.agents:
                break

        # close render window
        if self.render:
            self.env.close()
        if self.on_policy:
            return cum_rewards_red, num_steps, loss_red_actor, loss_red_critic
            #return cum_rewards_red, num_steps, loss_red
            #return cum_rewards_red, num_steps, loss_actor, loss_critic
        else:
            return cum_rewards_red, num_steps, loss_red

    def MergeActions(self, actions1, actions2):
        """
        Method called when using two different agent types and want to merge both
        action dictionaries.
        Inputs:
            + actions1 (dict)
            + actions2 (dict)
        Outputs:
            + actions (dict)
        """
        act1 = actions1.copy()
        act2 = actions2.copy()
        act2.update(act1)
        actions = act2

        return actions
