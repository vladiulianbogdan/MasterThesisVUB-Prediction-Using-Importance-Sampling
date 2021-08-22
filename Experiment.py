import re
import numpy as np
import sys
import time
from gmpy2 import mpfr
from Utility import peek_line
from Tokenizer import Tokenizer

def increment_state_counter(mapping, robot, state, number_of_states):
    try:
        mapping[robot][state] += 1
    except KeyError:
        mapping[robot] = np.zeros(number_of_states)
        mapping[robot][state] += 1
 

def add(mappings, key, element):
  try:
    mappings[key].append(element)
  except KeyError:
    mappings[key] = [element]

class Experiment:
    def __init__(self, file, fsm):
        self.steps_for_robot = {}

        # TODO: add the number of states here
        self.fsm = fsm
        self.state_counter_for_robot = {}
        self.rewards = []
        self.number_of_robots = 0

        # Skip the first lines
        while True:
            line = peek_line(file)
            if line.startswith("[INFO]") or line.startswith("Light is at"):
                file.readline()
                continue
            else:
                break
        

        robot_index = -1
        state_counter_for_robot = {}

        while not peek_line(file).startswith("[INFO]"):
            line = file.readline()

            # if not re.match(r'--t (\d+) --o (\d+)', line) is None:
            if not re.match(r'Obj --t', line) is None:
                parts = line.split(" ")
                current_value = round(float(parts[4]), 2)
                current_time = int(parts[2])
                # match = re.match(r'--t (\d+) --o (\d+)', line)
                self.rewards.append(current_value)
            elif line.startswith("--t"):
                builder = ExperimentStepBuilder()
                tokens = Tokenizer(line)
                while tokens.has_more_tokens():
                    token = tokens.next_token()
                    
                    if token == "--t":
                        index = int(tokens.next_token())
                        builder.set_step_index(index)

                        if (index == 0):
                            state_counter_for_robot = {}
                            robot_index += 1

                        builder.set_robot_index(robot_index)
                        # set also the reward
                        builder.set_reward(self.rewards[index])
                    elif token.startswith("--s"):
                        match = re.match(r'--s(\d+)', token)
                        state_index = int(match.group(1))
                        builder.set_state_index(state_index)

                        increment_state_counter(self.state_counter_for_robot, robot_index, state_index, len(self.fsm.states))

                        # Skip the identifier
                        tokens.next_token()
                    elif token == "--n":
                        number_of_neighbours = int(tokens.next_token())
                        builder.set_number_of_neighbours(number_of_neighbours)
                    elif token == "--f":
                        ground_sensor_reading = float(tokens.next_token())
                        builder.set_ground_sensor_reading(ground_sensor_reading)
                    elif token.startswith("--c"):
                        match = re.match(r'--c(\d+)', token)
                        condition = int(match.group(1))
                        builder.set_condition(condition)

                        transition_type = int(tokens.next_token()) # type of the transition
                        builder.set_transition_type(transition_type)

                        value = int(tokens.next_token()) # value (with the new code is always 1)
                        builder.set_value(value)

                        probability = float(tokens.next_token()) # probability that the transition was active
                        builder.set_probability(probability)
                    elif token == "--a":
                        active_transitions = int(tokens.next_token())
                        builder.set_active_transition(active_transitions)
                
                step = builder.build()
                # step.debug()
                add(self.steps_for_robot, robot_index, step)
                self.number_of_robots = robot_index + 1
                
            elif line.startswith("Score"):
                parts = line.split(" ")
                self.result = float(parts[1])
            elif len(line) == 0:
                break
            else:
                line = file.readline()
                print("Unknown line", line, line.startswith("[INFO]"), peek_line(file), peek_line(file).startswith("[INFO]"))
    
    def get_active_states_statistics(self):
        state_counter = {}

        for robot_index in range(0, self.number_of_robots):
            steps = self.steps_for_robot[robot_index]
            for step in steps:
                try:
                    state_counter[step.state_index] += 1
                except KeyError:
                    state_counter[step.state_index] = 1
        
        return state_counter


    def calculate_state_values_intermediate_rewards(self, first_visit=True, discounted_factor=0.98):
        timesteps = len(self.steps_for_robot[0])
        overall_state_values = np.zeros(len(self.fsm.states))
        overall_state_proportional_values = np.zeros(len(self.fsm.states))
        num_of_robots = float(len(self.steps_for_robot.keys()))
        per_robot_reward = self.result / float(num_of_robots)
        state_proportional_rewards = np.zeros(len(self.fsm.states))
        state_number_of_visits = np.zeros(len(self.fsm.states))

        state_values = []
        state_values_proportional = []

        debug = True

        for s in range(0, len(self.fsm.states)):	
            state_values.append(mpfr(0))
            state_values_proportional.append(mpfr(0))

        for robot_index, episode in self.steps_for_robot.items(): 
            g = mpfr(0)
            g_prop = [] #np.zeros(self.number_of_states())
            for s in range(0, len(self.fsm.states)):
                g_prop.append(mpfr(0))

            per_robot_reward = self.result/float(num_of_robots)

            for index, step in reversed(list(enumerate(episode[:-1]))):
                state = step.state_index
                per_robot_reward = self.rewards[index] / float(num_of_robots) # reward for each robot/episode
                if (per_robot_reward < 0):
                    per_robot_reward = 0
                for s in range(0, len(self.fsm.states)):
                    state_proportional_rewards[s] = ((self.state_counter_for_robot[step.robot_index][s] / float(timesteps)) * per_robot_reward)	

                g = mpfr(g * discounted_factor + per_robot_reward)
                g_prop[state] = mpfr(g_prop[state] * discounted_factor + state_proportional_rewards[state])

                # if index < 2:
                #     print(g, g_prop, state_proportional_rewards)

                if per_robot_reward > 0 and (not state in map(lambda step: step.state_index, episode[0:index]) or not first_visit):
                    state_number_of_visits[state] += 1
                    state_values[state] += g
                    state_values_proportional[state] += g_prop[state]
                    # print(index, g,  per_robot_reward, state_values / state_number_of_visits)
            # print("debuggging", state_values, state_values_proportional, state_number_of_visits


        for s in range(0, len(self.fsm.states)):
            if state_number_of_visits[s] != 0:
                overall_state_values[s] = state_values[s] / state_number_of_visits[s]
                overall_state_proportional_values[s] = state_values_proportional[s] / state_number_of_visits[s]

        return overall_state_values, overall_state_proportional_values

    def estimate_state_values_intermediate_rewards_state_pruning(self, old_fsm, new_fsm, discount_factor=0.98):
        nominator = []
        overall_state_values = []
        number_of_state_visits = []
        num_of_robots = float(len(self.steps_for_robot.keys()))

        for s in range(0, len(new_fsm.states)):		
            nominator.append(mpfr(0))
            overall_state_values.append(mpfr(0))
            number_of_state_visits.append(mpfr(0))

        for robot_index, episode in self.steps_for_robot.items():
            run_possible = True
            # A run is a sequence of state between two intermediate rewards.
            states_in_this_run = []

            for index, step in list(enumerate(episode)):
                state_index = step.state_index
                per_robot_reward = self.rewards[index] / float(num_of_robots) # reward for each robot/episode
                states_in_this_run.append(state_index)
                number_of_state_visits[state_index] += 1

                if (new_fsm.states[state_index].is_active == False):
                    # We only considers run that contain active states
                    run_possible = False
                
                if per_robot_reward > 0:
                    if run_possible == True:
                        states_in_this_run = reversed(states_in_this_run)
                        for state_index_in_run in states_in_this_run:
                            nominator[state_index_in_run] += per_robot_reward
                            per_robot_reward *= discount_factor

                    run_possible = True
                    states_in_this_run = []

        for s in range(0, len(new_fsm.states)):
            if (number_of_state_visits[s] != 0):	
                overall_state_values[s] = nominator[s] / number_of_state_visits[s]

        return overall_state_values, number_of_state_visits



    def estimate_value_states_importance_sampling_intermediate(self, old_fsm, new_fsm, first_visit=True, discount_factor=0.98):
        timesteps = len(self.steps_for_robot[0])
        num_of_robots = float(len(self.steps_for_robot.keys())) # number of robots or episodes per experiment
        weighted_importance_sampling_state_estimation = []
        ordinary_importance_sampling_state_estimation = []
        proportional_weighted_importance_sampling_state_estimation = []
        proportional_ordinary_importance_sampling_state_estimation = []

        for i in range(len(new_fsm.states)):
            weighted_importance_sampling_state_estimation.append(mpfr(0.0))
            ordinary_importance_sampling_state_estimation.append(mpfr(0.0))
            proportional_weighted_importance_sampling_state_estimation.append(mpfr(0.0))
            proportional_ordinary_importance_sampling_state_estimation.append(mpfr(0.0))

        nominator = []
        weight_denominator = []
        ordinary_denominator = []
        proportional_nominator = []

        rewards = []
        prop_rewards = []
        importance_samplings = []
		
        for s in range(0, len(new_fsm.states)):		
            nominator.append(mpfr(0))
            weight_denominator.append(mpfr(0))
            ordinary_denominator.append(mpfr(0))
            proportional_nominator.append(mpfr(0))

            rewards.append([])
            importance_samplings.append([])
            prop_rewards.append([])

        for robot_index, episode in self.steps_for_robot.items(): 
            importance_sampling_coefficient = mpfr(1.0)
            behavior_prob_transition = mpfr(1.0)
            target_prob_transition = mpfr(1.0)

            state_proportional_rewards = np.zeros(len(new_fsm.states))

            g = mpfr(0)
            g_prop = []
            for s in range(0, len(new_fsm.states)):		
                g_prop.append(mpfr(0))

            ordinary_denominator = np.zeros(len(new_fsm.states))
            visited = np.zeros(len(new_fsm.states))
            last_active_state = -1

            for index, step in reversed(list(enumerate(episode[:-1]))):
                state = step.state_index
                index = step.step_index
                per_robot_reward = self.rewards[index] / float(num_of_robots) # reward for each robot/episode
                if (per_robot_reward < 0):
                    per_robot_reward = 0
                for s in range(0, len(new_fsm.states)):
                    state_proportional_rewards[s] = ((self.state_counter_for_robot[step.robot_index][s] / float(timesteps)) * per_robot_reward)

                next_state = episode[index + 1]
                previous_state = episode[index - 1]
                tr_neighbors = episode[index + 1].number_of_neighbours
                tr_ground = episode[index + 1].ground_sensor_reading

                res = old_fsm.probability_of_state_transition(state, next_state.state_index, tr_neighbors, tr_ground) # old_fsm[state].prob_of_reaching_state(next_state, old_fsm, tr_neighbors[index + 1], tr_ground[index + 1])

                behavior_prob_transition *= res

                if (new_fsm.states[state].is_active == True):
                    if (new_fsm.states[next_state.state_index].is_active == False):
                        continue

                    last_active_state = state
                    res = new_fsm.probability_of_state_transition(state, next_state.state_index, tr_neighbors, tr_ground) # probability that the target policy transitions from the previous state to the current state

                    target_prob_transition *= res
                else:
                    # There is no previous state. We can stop
                    if (index == 0):
                        break

                    # In a sequence like 5, 4, 3, 3, 2, 1, if state 3 was removed the transition 2 - 4 will be considerate.
                    # So if we are in an inactive state and the previous state is also inactive, we will continue to the next step.
                    if (new_fsm.states[previous_state.state_index].is_active == False):
                        continue
                    else:
                        # If there was no previously active state just continue because there is nothing to compute.
                        if (last_active_state == -1):
                            continue
                        res = new_fsm.probability_of_state_transition(previous_state.state_index, last_active_state, tr_neighbors, tr_ground) # probability that the target policy transitions from the previous state to the current state

                        target_prob_transition *= res

                if target_prob_transition == 0:
                    target_prob_transition = sys.float_info.epsilon

                if behavior_prob_transition == 0:
                    behavior_prob_transition = sys.float_info.epsilon

                importance_sampling_coefficient = mpfr(target_prob_transition / behavior_prob_transition)

                g = mpfr(g * discount_factor + per_robot_reward)
                g_prop[state] = mpfr(g_prop[state] * discount_factor + state_proportional_rewards[state])

                if per_robot_reward > 0 and (not state in map(lambda step: step.state_index, episode[0:index]) or not first_visit):
                    rewards[state].append(g)
                    prop_rewards[state].append(g_prop[state])
                    importance_samplings[state].append(importance_sampling_coefficient)

                    nominator[state] += g

                    weight_denominator[state] += importance_sampling_coefficient
                    proportional_nominator[state] += g_prop[state] 
                    ordinary_denominator[state] += 1

                visited[state] = 1

        for s in range(0, len(new_fsm.states)):
            sum_importance_samplings = mpfr(0)

            for r in range(0, len(rewards[s])):
                sum_importance_samplings += importance_samplings[s][r]

                weighted_importance_sampling_state_estimation[s] += mpfr(rewards[s][r]) * mpfr(importance_samplings[s][r])
                ordinary_importance_sampling_state_estimation[s] += rewards[s][r] * importance_samplings[s][r]
                proportional_weighted_importance_sampling_state_estimation[s] += prop_rewards[s][r] * importance_samplings[s][r]
                proportional_ordinary_importance_sampling_state_estimation[s] += prop_rewards[s][r] * importance_samplings[s][r]

            if sum_importance_samplings != 0:
                weighted_importance_sampling_state_estimation[s] /= sum_importance_samplings
                proportional_weighted_importance_sampling_state_estimation[s] /= sum_importance_samplings

            if len(rewards[s]) != 0:
                ordinary_importance_sampling_state_estimation[s] /= mpfr(len(rewards[s]))
                proportional_ordinary_importance_sampling_state_estimation[s] /= mpfr(len(rewards[s]))

        return weighted_importance_sampling_state_estimation, ordinary_importance_sampling_state_estimation, proportional_weighted_importance_sampling_state_estimation, proportional_ordinary_importance_sampling_state_estimation


class ExperimentStep:
    def __init__(self, step_index, state_index, number_of_neighbours,
        ground_sensor_reading, condition, transition_type, value, probability, active_transitions, reward, robot_index):
        self.step_index = step_index
        self.state_index = state_index
        self.number_of_neighbours = number_of_neighbours
        self.ground_sensor_reading = ground_sensor_reading
        self.condition = condition
        self.transition_type = transition_type
        self.value = value
        self.probability = probability
        self.active_transitions = active_transitions
        self.reward = reward
        self.robot_index = robot_index
    
    def debug(self):
        print(
            self.step_index,
            self.state_index,
            self.number_of_neighbours,
            self.ground_sensor_reading,
            self.condition,
            self.transition_type,
            self.value,
            self.probability,
            self.active_transitions,
            self.reward,
            self.robot_index
        )

class ExperimentStepBuilder:
    def __init__(self):
        self.step_index = None
        self.state_index = None
        self.number_of_neighbours = None
        self.ground_sensor_reading = None
        self.condition = None
        self.transition_type = None
        self.value = None
        self.probability = None
        self.active_transitions = None
        self.reward = None
        self.robot_index = None

    def set_step_index(self, step_index):
        self.step_index = step_index
        return self
    
    def set_state_index(self, state_index):
        self.state_index = state_index
        return self
    
    def set_number_of_neighbours(self, number_of_neighbours):
        self.number_of_neighbours = number_of_neighbours
        return self

    def set_ground_sensor_reading(self, ground_sensor_reading):
        self.ground_sensor_reading = ground_sensor_reading
        return self
    
    def set_condition(self, condition):
        self.condition = condition
        return self
    
    def set_transition_type(self, transition_type):
        self.transition_type = transition_type
        return self

    def set_value(self, value):
        self.value = value
        return self

    def set_probability(self, probability):
        self.probability = probability
        return self

    def set_active_transition(self, active_transitions):
        self.active_transitions = active_transitions
        return self
    
    def set_reward(self, reward):
        self.reward = reward
        return self

    def set_robot_index(self, index):
        self.robot_index = index
        return self

    def build(self):
        return ExperimentStep(self.step_index,
            self.state_index,
            self.number_of_neighbours,
            self.ground_sensor_reading,
            self.condition,
            self.transition_type,
            self.value,
            self.probability,
            self.active_transitions,
            self.reward,
            self.robot_index)