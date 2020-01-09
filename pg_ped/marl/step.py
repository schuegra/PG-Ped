# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:13:20 2019

@author: Philipp
"""

import os
from copy import deepcopy
from random import shuffle
from typing import List

import numpy
import torch
from pg_ped import config
from pg_ped.environment_construction.geometry import update_state_all_positions
from pg_ped.environment_construction.state_representation import generate_kinematics_torch, generate_kinematics_torch2
from pg_ped.marl.agent import Agent

from . import Environment


class SingleAgentStep(object):

    @property
    def number_steps(self):
        return self._steps

    def __init__(self, agent: Agent, environment: Environment, active: bool = True):
        self._agent = agent
        self._environment = environment
        self._steps = 0
        self.active = active
        self._done = False

    def __call__(self, **kwargs):
        """
        """

        failed_action = False
        if not self._done:
            state = self._environment.state

            action, action_index, log_prob, prob, failed_action = self._agent.action(state, [], **kwargs)

            next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            forbidden_actions = []
            i = 0
            while failed is True:

                if (not -1 in forbidden_actions) and (
                        len(forbidden_actions) == len(self._agent._action_space)):  # No action possible
                    self._done = True
                    print('NO ACTION POSSIBLE -> TERMINATE EPISODE')
                    return state, self._done, False

                forbidden_actions += [action_index.detach().cpu().numpy()]  # ; print(forbidden_actions)
                action, action_index, log_prob, prob, _ = self._agent.action(state, forbidden_actions,
                                                                             **kwargs)
                next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
                i += 1

            try:
                action_to_pass = action.clone()  # .cpu()
            except Exception:
                action_to_pass = deepcopy(action)
            self._agent.observe(
                state.detach().cpu().numpy(),
                action_to_pass,
                action_index.detach().cpu().numpy(),
                log_prob.clone(),
                prob.clone(),
                next_state.detach().cpu().numpy(),
                reward,
                None,  # state value
                None  # next state value
            )
            self._steps += 1

        if self._done is False:
            self._done = self._environment.done(agent_identity=self._agent.identity, **kwargs)

        return self._environment.state, self._done, False

    def optimize(self, **kwargs) -> None:
        if self.active:
            self._agent.optimize(self._steps, **kwargs)

    def save_model(self, path: str, **kwargs) -> None:
        '''
            Saves the policy parameters to pytorch model.
        '''
        self._agent.save_model(path, **kwargs)


class SingleAgentStepVadereEventDrivenUpdate(SingleAgentStep):

    def _mem(self, state, action, action_index, log_prob, prob):
        self._state = state
        self._action = action
        self._action_index = action_index
        self._log_prob = log_prob
        self._prob = prob

    def mem_next_state(self, next_state):
        self._next_state = next_state

    def mem_reward(self, reward):
        self._reward = reward

    def observe(self):
        self._agent.observe(
            self._state,
            self._action,
            self._action_index,
            self._log_prob,
            self._prob,
            self._next_state,
            self._reward,
            None,
            None
        )

    def __call__(self, **kwargs):
        failed_action = False
        if not self._done:
            state = self._environment.state

            action, action_index, log_prob, prob, failed_action = self._agent.action(state, [], **kwargs)

            next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            forbidden_actions = []
            i = 0
            while failed is True:

                if (not -1 in forbidden_actions) and (
                        len(forbidden_actions) == len(self._agent._action_space)):  # No action possible
                    self._done = True
                    print('NO ACTION POSSIBLE -> TERMINATE EPISODE')
                    return state, self._done, False

                forbidden_actions += [action_index.detach().cpu().numpy()]  # ; print(forbidden_actions)
                action, action_index, log_prob, prob, _ = self._agent.action(state, forbidden_actions,
                                                                             **kwargs)
                next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
                i += 1

            try:
                action_to_pass = action.clone()  # .cpu()
            except Exception:
                action_to_pass = deepcopy(action)
            self._mem(
                state.detach().cpu().numpy(),
                action_to_pass,
                action_index.detach().cpu().numpy(),
                log_prob.clone(),
                prob.clone(),
            )
            self._steps += 1

        if self._done is False:
            self._done = self._environment.done(agent_identity=self._agent.identity, **kwargs)

        return self._environment.state, self._done, False

class SingleAgentStep1Step(SingleAgentStep):

    def __call__(self, **kwargs):
        """
        """

        failed_action = False
        if not self._done:
            state = self._environment.state

            action, action_index, log_prob, prob, failed_action = self._agent.action(state, [], **kwargs)

            next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            forbidden_actions = []
            i = 0
            while failed is True:

                if (not -1 in forbidden_actions) and (
                        len(forbidden_actions) == len(self._agent._action_space)):  # No action possible
                    self._done = True
                    print('NO ACTION POSSIBLE -> TERMINATE EPISODE')
                    return state, self._done, True

                forbidden_actions += [action_index.detach().cpu().numpy()]  # ; print(forbidden_actions)
                action, action_index, log_prob, prob, _ = self._agent.action(state, forbidden_actions,
                                                                             **kwargs)
                next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
                i += 1

            old_mode = deepcopy(kwargs['mode'])
            kwargs['mode'] = 'simulate'
            _, _, _, next_state_value, _ = self._agent.action(next_state, forbidden_actions, **kwargs)
            kwargs['mode'] = old_mode

            # print(action)
            try:
                action_to_pass = action.detach().cpu()
            except Exception:
                action_to_pass = deepcopy(action)
            self._agent.observe(state.detach().cpu(),
                                action_to_pass,
                                action_index.detach().cpu(),
                                log_prob.detach().cpu(),
                                prob.detach().cpu(),
                                next_state.detach().cpu(),
                                reward,
                                None,
                                next_state_value.detach().cpu())
            kwargs['reward_sums'][self._agent.identity][kwargs['current_episode']] += [reward]
            self._steps += 1

        if self._done is False:
            self._done = self._environment.done(agent_identity=self._agent.identity, **kwargs)

        return self._environment.state, self._done, False

    def optimize(self, **kwargs) -> None:
        if self.active:
            self._agent.optimize(1, **kwargs)


class SingleAgentStep1Step2(SingleAgentStep):

    def __call__(self, **kwargs):
        """
        """

        failed_action = False
        if not self._done:
            state = self._environment.state
            state = generate_kinematics_torch(state, self._agent.identity, self._environment._initial_state, **kwargs)

            action, action_index, log_prob, value, failed_action = self._agent.action(state, [], **kwargs)

            next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            forbidden_actions = []
            i = 0
            while failed is True:

                if (not -1 in forbidden_actions) and (
                        len(forbidden_actions) == len(self._agent._action_space)):  # No action possible
                    self._done = True
                    print('NO ACTION POSSIBLE -> TERMINATE EPISODE')
                    return state, self._done, True

                forbidden_actions += [action_index.detach().cpu().numpy()]  # ; print(forbidden_actions)
                action, action_index, log_prob, prob, _ = self._agent.action(state, forbidden_actions,
                                                                             **kwargs)
                next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
                i += 1

            next_state = generate_kinematics_torch(next_state, self._agent.identity, self._environment._initial_state,
                                                   **kwargs)
            old_mode = deepcopy(kwargs['mode'])
            kwargs['mode'] = 'simulate'
            _, _, _, next_state_value, _ = self._agent.action(next_state, forbidden_actions, **kwargs)
            kwargs['mode'] = old_mode

            # print(action)
            try:
                action_to_pass = action.detach().cpu()
            except Exception:
                action_to_pass = deepcopy(action)
            self._agent.observe(state.detach().cpu(),
                                action_to_pass,
                                action_index.detach().cpu(),
                                log_prob.detach().cpu(),
                                value.detach().cpu(),
                                next_state.detach().cpu(),
                                reward,
                                None,
                                next_state_value.detach().cpu())
            kwargs['reward_sums'][self._agent.identity][kwargs['current_episode']] += [reward]
            self._steps += 1

        if self._done is False:
            self._done = self._environment.done(agent_identity=self._agent.identity, **kwargs)

        return self._environment.state, self._done, False

    def optimize(self, **kwargs) -> None:
        if self.active:
            self._agent.optimize(1, **kwargs)


class SingleAgentStepExperienceReplay(SingleAgentStep1Step):

    def optimize(self, number_samples: int, **kwargs) -> None:
        if self.active:
            kwargs['number_samples'] = number_samples
            self._agent.optimize(self.number_steps, **kwargs)


class SingleAgentStepExperienceReplay2(SingleAgentStep1Step2):

    def optimize(self, number_samples: int, **kwargs) -> None:
        if self.active:
            kwargs['number_samples'] = number_samples
            self._agent.optimize(self.number_steps, **kwargs)


class SingleAgentStepActorCritic(SingleAgentStep):

    def __call__(self, **kwargs):
        """
        """

        if not self._done:
            state = self._environment.state

            action, action_index, log_prob, prob, state_value = self._agent.action(state, [], **kwargs)
            next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            forbidden_actions = []
            while failed is True and len(forbidden_actions) < len(self._agent._action_space):
                forbidden_actions += [action_index.detach().cpu().numpy()]
                if len(forbidden_actions) == len(self._agent._action_space):
                    return state, True
                action, action_index, log_prob, prob, state_value = self._agent.action(state, forbidden_actions,
                                                                                       **kwargs)
                next_state, reward, failed = self._environment.step(action, self._agent.identity, **kwargs)
            if self._agent.identity == 0:
                print(str(int(action_index)))
            _, _, _, _, next_state_value = self._agent.action(next_state, forbidden_actions, **kwargs)

            try:
                action_to_pass = action.clone()
            except Exception:
                action_to_pass = deepcopy(action)
            self._agent.observe(state.detach().cpu().numpy(),
                                action_to_pass,
                                action_index.detach().cpu().numpy(),
                                log_prob.clone(),
                                prob,
                                next_state.detach().cpu().numpy(),
                                reward,
                                state_value.clone(),
                                next_state_value)
            self._steps += 1

        self._done = self._environment.done(agent_identity=self._agent.identity, **kwargs)
        return self._environment.state, self._done

    def optimize(self, **kwargs) -> None:
        if self.active:
            self._agent.optimize(self.number_steps, **kwargs)


class SingleAgentStepActorCritic1Step(SingleAgentStepActorCritic):

    def optimize(self, **kwargs) -> None:
        if self.active:
            self._agent.optimize(1, **kwargs)


class MultiAgentStepBase(object):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        self._single_agent_steps = [SingleAgentStep(agent, environment) for agent in agents]
        self._training = training

    def __call__(self):
        raise NotImplementedError


class MultiAgentStepSequentialEpisodic(MultiAgentStepBase):

    def __call__(self, **kwargs):
        '''
           Sequentially lets the agents interact with the environment.
           The order of interaction is randomly chosen.
           
           returns:
               State of the environment after each agent has interacted with
               it.
        '''
        shuffle(self._single_agent_steps)
        one_is_done = False
        failed = False
        for step in self._single_agent_steps:
            state, done, failed_step = step(**kwargs)
            failed = failed or failed_step
            if done is True:
                one_is_done = True
                # break

        if (self._training is True) and (one_is_done is True) and (failed is False):
            self.optimize(**kwargs)

        if (one_is_done is True) and (self._training is False):
            self.track_rewards(**kwargs)

        return state, one_is_done, failed

    def optimize(self, **kwargs) -> None:
        '''
            Makes the single agent loops call the optimizers of the action
            selectors.
        '''
        for sas in self._single_agent_steps:
            sas.optimize(**kwargs)

    def save_models(self, path: str, **kwargs) -> None:
        '''
            Saves the policy paramters to pytorch models.
        '''

        for i, sas in enumerate(self._single_agent_steps):
            sas.save_model(path, **kwargs)

    def track_rewards(self, reward_sums: List[List[float]], **kwargs):
        '''
            Track reward of episode.
        '''
        for i, sas in enumerate(self._single_agent_steps):
            transitions = sas._agent._replay_memory._memory[-sas.number_steps:]
            reward_sum = sum([x.reward for x in transitions])
            reward_sums[i] += [reward_sum]


class MultiAgentStepVadereSync(MultiAgentStepSequentialEpisodic):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        self._single_agent_steps = [SingleAgentStepVadereEventDrivenUpdate(agent, environment) for agent in agents]
        self._training = training

    def __call__(self, **kwargs):
        one_is_done = False
        failed = False
        for step in self._single_agent_steps:
            state, done, failed_step = step(**kwargs)
            failed = failed or failed_step
            if done is True:
                one_is_done = True

        # Make step
        config.cli.ctr.nextStep(config.cli.sim.getSimTime() + kwargs['time_per_step'])

        # Evaluate Reward function
        self._eval_reward_functions(**kwargs)

        # Update positions
        positions_dict = config.cli.pers.getPosition2DList()
        positions_in_order_of_vadere = list(positions_dict.values())
        position_runner = [positions_in_order_of_vadere[-1]]
        positions_other = [pos for pos in positions_in_order_of_vadere[:-1]]
        positions = position_runner + positions_other
        state = update_state_all_positions(state, positions, **kwargs)
        for step in self._single_agent_steps:
            step._environment._state = state
            step.mem_next_state(state.detach().cpu().numpy())
            step.observe()


        if (self._training is True) and (one_is_done is True) and (failed is False):
            self.optimize(**kwargs)

        if (one_is_done is True) and (self._training is False):
            self.track_rewards(**kwargs)

        return state, one_is_done, failed


    def _eval_reward_functions(self, **kwargs):
        for step in self._single_agent_steps:
            kwargs['initial_state'] = step._environment._initial_state
            r = step._environment._reward_function(step._state, step._agent.identity, **kwargs)
            step.mem_reward(r)

class MultiAgentStepSequentialEpisodicNStepTD(MultiAgentStepSequentialEpisodic):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True, n: int = 1) -> None:
        self._single_agent_steps = [SingleAgentStep1Step(agent, environment) for agent in agents]
        self._training = training
        self._n = n

    def __call__(self, **kwargs):
        '''
           Sequentially lets the agents interact with the environment.
           The order of interaction is randomly chosen.

           returns:
               State of the environment after each agent has interacted with
               it.
        '''

        shuffle(self._single_agent_steps)
        one_is_done = False
        failed = False
        for step in self._single_agent_steps:
            state, done, failed_step = step(**kwargs)
            failed = failed or failed_step
            if done is True:
                one_is_done = True
                # break

        number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + kwargs['current_step']
        if (number_of_time_steps_since_training_start % self._n == 0) and self._training and (
                not failed) and number_of_time_steps_since_training_start >= kwargs['start_learning']:
            self.optimize(**kwargs)

        if (self._training is False):
            self.track_rewards(**kwargs)

        return state, one_is_done, failed


class MultiAgentStepSequentialEpisodicExperienceReplay(MultiAgentStepSequentialEpisodicNStepTD):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True, n: int = 1) -> None:
        super().__init__(agents, environment, training, n)
        self._single_agent_steps = [SingleAgentStepExperienceReplay(agent, environment) for agent in agents]


class MultiAgentStepSequentialEpisodicExperienceReplayPerTypeTraining(MultiAgentStepSequentialEpisodicNStepTD):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True, n: int = 1) -> None:
        # As in MultiAgentStepSequentialEpisodicExperienceReplay but with active=False
        super().__init__(agents, environment, training, n)
        self._single_agent_steps = [SingleAgentStepExperienceReplay(agent, environment, active=False) for agent in
                                    agents]

        # To only train representer of type
        all_types = [sas._agent.type for sas in self._single_agent_steps]
        types = numpy.unique(all_types)
        for type in types:
            self._single_agent_steps[all_types.index(type)].active = True


class MultiAgentStepSequentialEpisodicExperienceReplayPerTypeTraining3(MultiAgentStepSequentialEpisodicNStepTD):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True, n: int = 1) -> None:
        # As in MultiAgentStepSequentialEpisodicExperienceReplay but with active=False
        super().__init__(agents, environment, training, n)
        self._single_agent_steps = [SingleAgentStepExperienceReplay2(agent, environment, active=False) for agent in
                                    agents]

        # To only train representer of type
        all_types = [sas._agent.type for sas in self._single_agent_steps]
        types = numpy.unique(all_types)
        for type in types:
            self._single_agent_steps[all_types.index(type)].active = True


class MultiAgentStepSequentialEpisodicExperienceReplayPerTypeTraining2(MultiAgentStepSequentialEpisodic):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        # As in MultiAgentStepSequentialEpisodic but with active=False
        super().__init__(agents, environment, training)

        # To only train representer of type
        all_types = [sas._agent.type for sas in self._single_agent_steps]
        types = numpy.unique(all_ids)
        for type in types:
            self._single_agent_steps[all_ids.index(type)].active = True


class MultiAgentStepOneVsAll(MultiAgentStepBase):

    def __init__(self, active_agent: int, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        super().__init__(agents, environment)
        self._episode = MultiAgentStepSequentialEpisodic(agents, environment, training)
        self._fix_all_but_active_agent(active_agent)

    def __call__(self, **kwargs):
        '''
           Holds the parameters of all agents but the current agent fixed while
           all agents interact with the environment.

           returns:
               State of the environment after each agent has interacted with
               it.
        '''

        return self._episode(**kwargs)

    def _fix_all_but_active_agent(self, active_agent: int) -> None:
        for sas in self._episode._single_agent_steps:
            sas.active = False
        self._episode._single_agent_steps[active_agent].active = True

    def save_models(self, path: str, **kwargs) -> None:
        '''
            Saves the policy paramters to pytorch models.
        '''

        self._episode.save_models(path, **kwargs)


class MultiAgentStepSequentialEpisodic1StepTDAC(MultiAgentStepSequentialEpisodic):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        self._single_agent_steps = [SingleAgentStepActorCritic1Step(agent, environment) for agent in agents]
        self._training = training

    def __call__(self, **kwargs):
        '''
           Sequentially lets the agents interact with the environment.
           The order of interaction is randomly chosen.

           returns:
               State of the environment after each agent has interacted with
               it.
        '''

        shuffle(self._single_agent_steps)
        one_is_done = False
        for step in self._single_agent_steps:
            state, done = step(**kwargs)
            if done is True:
                one_is_done = True

        if self._training is True:
            self.optimize(**kwargs)

        return state, one_is_done


class MultiAgentStepSequentialEpisodicA2C(MultiAgentStepSequentialEpisodic):

    def __init__(self, agents: List[Agent], environment: Environment, training: bool = True) -> None:
        self._single_agent_steps = [SingleAgentStepActorCritic(agent, environment) for agent in agents]
        self._training = training

    def __call__(self, **kwargs):
        '''
           Sequentially lets the agents interact with the environment.
           The order of interaction is randomly chosen.

           returns:
               State of the environment after each agent has interacted with
               it.
        '''

        shuffle(self._single_agent_steps)
        one_is_done = False
        for step in self._single_agent_steps:
            state, done = step(**kwargs)
            if done is True:
                one_is_done = True

        if self._training is True and one_is_done is True:
            self.optimize(**kwargs)

        return state, one_is_done
