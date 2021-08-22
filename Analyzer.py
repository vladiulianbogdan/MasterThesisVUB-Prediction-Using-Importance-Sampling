import Tokenizer
import re
import sys
import os
import random
import copy
from StateMachine import StateMachine


# First of all, a very useful tip,
# If you run automode adding -r before —fsm-config  you get a graphical representation of the FSM.

# The FSM description language is a bit complicated so I’ll try to do my best.
#  (so complicated that at the top of AutoMoDeLogAnalyzer.py there is a description of the grammar in BNF)

# Let’s start with the state, the definition is like this 

# —s<state_index> <state_type> <state_parameters> —n<state_index> <number of transitions> <transitions>

# So the state has an index ( from 0 to the number of states -1) that is used in all the tokens linked to that state.
# Following the state index there is the  type that specifies which behavior is executed while in that state with, if necessary, the parameters needed by the behavior (such as —att0 4.47 ).
# Then there is the number of transitions (so —n0 3 indicates that state 0 has 3 transitions) and then the specification of the transitions.
# Each transitions is defined like this 

# —n<state index>x<transition index> <transition target> —c<state index>x<transition index> <transition type> <parameters>

# So if you are considering States and transitions in order, you can ignore the various indexes 
# (and if not so the order in which you consider the transitions of a state does not change anything).
# In the end, you need from a transition definition is the target and all the information to calculate the probability that are the type (that tells you how to calculate the probability) 
# and the parameters (that are in the formula to calculate the probability).
# The parameters are defined like this

# —p<state index>x<transition index> <value of parameter p> [—w<state index>x<transition index> <value of parameter w>] 
# (where parameter w is not always required).

# Another thing that needs clarification is that the transition target is not a state index.
# Let’s take, for instance, —n0x0 2, this is the first transition of state 0 and points to the second state of the possible targets of State 0 that are [1, 2, 3, 4] so this transition
# points to State 3. Similarly, the possible targets for State 1 are [0,2,3,4] and State 2 are [0,1,3,4] and so on.
# Considering all the transitions in State 0 of the FSM in your email they point to, respectively, state 3, state 1 and  state 1.
# This system has been conceived this way so that it would be easier to interface automode with irace (or at least this is what one of the guys that made it told me)

from FsmHistory import FsmHistory
import numpy as np

commandline_separator = "-------------------------------------------------------------------------------------"	

def perform_analysis(fsmHistory, newFSM):
    count = 0
    weighted_state_estimation = np.zeros(len(newFSM.states))
    original_state_estimation = np.zeros(len(newFSM.states))
    proportional_weighted_state_estimation = np.zeros(len(newFSM.states))
    proportional_original_state_estimation = np.zeros(len(newFSM.states))
    overall_discounted_state_value = np.zeros(len(fsmHistory.fsm.states))
    overall_discounted_proportional_state_value = np.zeros(len(fsmHistory.fsm.states))

    first_visit = False
    discount_factor = 0.99

    for experiment in fsmHistory.experiments:
        count += 1
        weighted_is_state_estimation, ordinary_is_estimation, proportional_weighted_is_state_estimation, proportional_ordinary_is_estimation = experiment.estimate_value_states_importance_sampling_intermediate(fsmHistory.fsm, newFSM, first_visit, discount_factor)
        discounted_state_value, discounted_propotional_state_value = experiment.calculate_state_values_intermediate_rewards(first_visit, discount_factor)
        # discounted_state_value = experiment.calculate_vpi_for_experiment()
        # discounted_propotional_state_value = experiment.calculate_proportional_vpi_for_experiment()

        overall_discounted_state_value += discounted_state_value
        overall_discounted_proportional_state_value += discounted_propotional_state_value

        for state in range(0, len(newFSM.states)):
            weighted_state_estimation[state] += weighted_is_state_estimation[state]
            original_state_estimation[state] += ordinary_is_estimation[state]
            proportional_weighted_state_estimation[state] += proportional_weighted_is_state_estimation[state]
            proportional_original_state_estimation[state] += proportional_ordinary_is_estimation[state]

    overall_discounted_state_value /= len(fsmHistory.experiments)
    weighted_state_estimation /= len(fsmHistory.experiments)
    original_state_estimation /= len(fsmHistory.experiments)
    overall_discounted_proportional_state_value /= len(fsmHistory.experiments)
    proportional_weighted_state_estimation /= len(fsmHistory.experiments)
    proportional_original_state_estimation /= len(fsmHistory.experiments)

    print("\n Off-policy analysis using configurable parameters of the new FSM")
    print("Parameters: First visit only:", first_visit, "; discount factor:", discount_factor)
    print("State values of the original FSM                 : {0}".format([i for i in overall_discounted_state_value]))		
    print("Weighted state estimation                        : {0}".format([i for i in weighted_state_estimation]))
    print("Ordinary state estimation                        : {0}".format([i for i in original_state_estimation]))
    print("Proportional state values of the original FSM    : {0}".format([i for i in overall_discounted_proportional_state_value]))
    print("Proportional weighted state estimation           : {0}".format([i for i in proportional_weighted_state_estimation]))
    print("Proportional ordinary state estimation           : {0}".format([i for i in proportional_original_state_estimation]))

    print_performance_estimation(fsmHistory, overall_discounted_state_value, weighted_state_estimation, proportional_weighted_state_estimation, proportional_original_state_estimation, original_state_estimation, overall_discounted_proportional_state_value, "Discounted ")

def print_performance_estimation(fsm_history, vpi_all, wei_is, wei_is_proportional, ord_is_proportional, ord_is, discounted_propotional_state_value, prefix=""):
	average_original_reward = 0
	number_of_robots = fsm_history.experiments[0].number_of_robots

	for experiment in fsm_history.experiments:
		average_original_reward += experiment.result
	average_original_reward /= len(fsm_history.experiments)

	average_wei_reward = 0.0
	for s in range(0,len(wei_is)):
		state_contribution = 1.0
		if vpi_all[s] != 0.0 :			
			state_contribution =  wei_is[s]/vpi_all[s] # old test vpi_all[s]/wei_is[s]
			# print(s, state_contribution)
		average_wei_reward += average_original_reward * state_contribution
		# print("sum", average_original_reward * state_contribution)
			#break

	average_wei_reward = average_wei_reward/(len(wei_is))	

	average_prop_reward = 0.0
	if discounted_propotional_state_value is None:
		for s in wei_is_proportional:
			average_prop_reward += s
		average_prop_reward *= number_of_robots
	else:
		for s in range(0,len(wei_is_proportional)):
			state_contribution = 1.0
			if discounted_propotional_state_value[s] != 0.0 :			
				state_contribution =  wei_is_proportional[s] / discounted_propotional_state_value[s] # old test vpi_all[s]/wei_is[s]
				# print("aici", s, state_contribution, wei_is_proportional[s], discounted_propotional_state_value[s])
			average_prop_reward += average_original_reward * state_contribution
		# print("sum", average_original_reward * state_contribution)
			#break

	average_prop_reward = average_prop_reward / len(wei_is_proportional)
	
	average_ord_prop_reward = 0.0
	for s in ord_is_proportional:
		average_ord_prop_reward += s
	
	print(average_ord_prop_reward, "nunmber of robots", number_of_robots, len(fsm_history.experiments))
	average_ord_prop_reward *= number_of_robots
		
	average_ord_reward = 0.0
	for s in range(0,len(ord_is)):
		state_contribution = 1.0
		if vpi_all[s] != 0.0 :
			state_contribution =  ord_is[s]/vpi_all[s]

		average_ord_reward +=  average_original_reward * state_contribution
	
	average_ord_reward = average_ord_reward/(len(ord_is))
	
	print("\n Performance estimation")
	print(commandline_separator)
	print( prefix + "Average performance of the original FSM                              : {0}".format(round(average_original_reward,3)))
	print( prefix + "WIS Expected average performance of the pruned FSM                   : {0}".format(round(average_wei_reward,3)))
	print( prefix + "OIS Expected average performance of the pruned FSM                   : {0}".format(round(average_ord_reward,3)))
	print( prefix + "WIS Expected average performance with the proportional reward        : {0}".format(round(average_prop_reward,3)))
	print( prefix + "OIS Expected average performance with the proportional reward        : {0}".format(round(average_ord_prop_reward,3)))

def state_pruning_analysis(fsm_history, new_FSM):
	average_original_reward = 0
	number_of_robots = fsm_history.experiments[0].number_of_robots
	overall_state_values_reference = np.zeros(len(new_FSM.states))
	overall_state_values = np.zeros(len(new_FSM.states))
	overall_number_of_state_visits = np.zeros(len(new_FSM.states))

	for experiment in fsm_history.experiments:
		state_values_reference, number_of_state_visits = experiment.estimate_state_values_intermediate_rewards_state_pruning(fsm_history.fsm, fsm_history.fsm)
		state_values_estimation, number_of_state_visits = experiment.estimate_state_values_intermediate_rewards_state_pruning(fsm_history.fsm, new_FSM)

		for i in range(0, len(overall_state_values_reference)):
			overall_state_values_reference[i] += state_values_reference[i]
			overall_state_values[i] += state_values_estimation[i]

			overall_number_of_state_visits[i] += number_of_state_visits[i]

	for s in range(0, len(overall_state_values_reference)):
		overall_state_values_reference[s] /= len(fsm_history.experiments)
		overall_state_values[s] /= len(fsm_history.experiments)
		overall_number_of_state_visits[s] /= len(fsm_history.experiments)

	proportional_values = []
	total_number_of_state_visits = 0

	for state_visit in overall_number_of_state_visits:
		total_number_of_state_visits += state_visit

	for state_visit in overall_number_of_state_visits:
		proportional_values.append(state_visit / total_number_of_state_visits)

	for experiment in fsm_history.experiments:
		average_original_reward += experiment.result
	average_original_reward /= len(fsm_history.experiments)

	average_wei_reward = 0.0
	for s in range(0, len(overall_state_values)):
		state_contribution = 0.0

		if overall_state_values_reference[s] != 0.0 :			
			state_contribution =  overall_state_values[s] / overall_state_values_reference[s]
			# print(s, state_contribution)
		average_wei_reward += proportional_values[s] * average_original_reward * state_contribution
		# print("sum", average_original_reward * state_contribution)
			#break

	print("Average performace", average_original_reward)
	print("Performance estimation with pruning", average_wei_reward)


def find_and_alterate_state(state, fsm_description_parts):
	fsm_description_parts_copy = fsm_description_parts.copy();
	for i in range(0, len(fsm_description_parts) - 1):
		state_description = '--p%dx' % state

		if (fsm_description_parts[i].startswith(state_description)):
			value = float(fsm_description_parts[i + 1])
			alterated_value = round(random.uniform(0, value + 5), 2)
			fsm_description_parts_copy[i + 1] = str(alterated_value)
			break
	
	return fsm_description_parts_copy

second_parameter = sys.argv[2]


if (second_parameter == '-generate-fsm-exp'):
	threshold = float(sys.argv[3])

	folder = sys.argv[1]
	fsms_stats = []

	for root, dirs, files in os.walk(folder):
		for file in files:
			fsm_history_file = os.path.join(root,file)
			print("Analyze file", fsm_history_file)

			file = open(fsm_history_file, "r")
			history = FsmHistory(file)
			active_states, states_stats = history.get_active_states_statistics(threshold)

			# # # We discard all the FSMs that have less than 3 active states
			# if (active_states < 3):
			# 	continue
			
			fsms_stats.append((fsm_history_file, history, states_stats, active_states))

	def takeThird(elem):
		return elem[3]
	
	fsms_stats.sort(key=takeThird, reverse=True)
	print(fsms_stats)

	# for i in range(20):
	for i in range(len(fsms_stats)):
		fsm_stat = fsms_stats[i]
		fsm_description_parts = fsm_stat[1].fsm.fsm_description.split(" ")
		state_stats = fsm_stat[2]

		# produce 4 FSMs by alterating the most 2 active transitions
		# for z in range(0, 8):
		# 	first_state = states_stats[0][0]
		# 	second_state = states_stats[1][0]
		# 	new_fsm = find_and_alterate_state(first_state, fsm_description_parts)
		# 	new_fsm = find_and_alterate_state(second_state, new_fsm)
		# 	print(' '.join(new_fsm))

		for z in range(0, 30):
			random_number_of_alteration = random.randint(1, len(states_stats))

			current_state_machine = fsm_description_parts
			for j in range(random_number_of_alteration):
				state = states_stats[j][0]
				current_state_machine = find_and_alterate_state(state, current_state_machine)
			print(' '.join(current_state_machine).replace("\n", ""))
elif (second_parameter == '-generate-fsm-exp-drastic'):
	threshold = float(sys.argv[3])

	folder = sys.argv[1]
	fsms_stats = []

	for root, dirs, files in os.walk(folder):
		for file in files:
			fsm_history_file = os.path.join(root,file)
			print("Analyze file", fsm_history_file)

			file = open(fsm_history_file, "r")
			history = FsmHistory(file)
			active_states, states_stats = history.get_active_states_statistics(threshold)

			# # # We discard all the FSMs that have less than 3 active states
			# if (active_states < 3):
			# 	continue
			
			fsms_stats.append((fsm_history_file, history, states_stats, active_states))

	# for i in range(20):
	for i in range(len(fsms_stats)):
		fsm_stat = fsms_stats[i]
		fsm_description_parts = fsm_stat[1].fsm.fsm_description.split(" ")
		state_stats = fsm_stat[2]

		for z in range(0, 30):
			fsm_description_parts_copy = fsm_description_parts.copy();
			for i in range(0, len(fsm_description_parts) - 1):
				if "." in fsm_description_parts[i]:
					value = float(fsm_description_parts[i])
					decider = random.random()
					if (decider < 0.4):
						alterated_value = round(random.uniform(0, value + 5), 2)
						fsm_description_parts_copy[i] = str(alterated_value)					

			print(' '.join(fsm_description_parts_copy).replace("\n", ""))
elif (second_parameter == '-as'):
	fsm_history_file = sys.argv[1]

	state_to_be_removed = 0
	file = open(fsm_history_file, "r")
	history = FsmHistory(file)

	while state_to_be_removed < len(history.fsm.states):
		print("Analysis removing state", state_to_be_removed)
		new_fsm = copy.deepcopy(history.fsm)
		new_fsm.set_state_active(state_to_be_removed, False)
		state_to_be_removed += 1

		state_pruning_analysis(history, new_fsm)
else:
	fsm_history_file = sys.argv[1]

	input_parameters_as_string = ' '.join(sys.argv[3:])

	new_fsm = StateMachine.parse_state_machine(input_parameters_as_string)
	file = open(fsm_history_file, "r")
	history = FsmHistory(file)
	print("Done!")

	perform_analysis(history, new_fsm)
