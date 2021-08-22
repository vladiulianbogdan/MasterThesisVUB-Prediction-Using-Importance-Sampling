import re
from Tokenizer import Tokenizer
from Transition import Transition
from State import State
import time

class StateMachine:
    def __init__(self):
        self.states = []
        self.identifier = ""
        self.fsm_description = ""
        self.transitions = []

    def add_state(self, state):
        self.states.append(state)

    def add_transition(self, transition):
        self.transitions.append(transition)

    def set_identifier(self, identifier):
        self.identifier = identifier

    def set_state_active(self, state_index, is_active):
        self.states[state_index].set_active(is_active)

        for transition in self.transitions:
            if transition.source == state_index or transition.target == state_index:
                transition.is_active = False

    def probability_of_state_transition(self, source, destination, num_neighbors, ground_sensor, path=[]):
        if (source in path):
            return 0.0
			
        active_transitions = []
		
        for transition in self.transitions:
            if(transition.is_active and transition.source == source):
                active_transitions.append(transition)

        denominator =  1 if len(active_transitions) == 0 else len(active_transitions)

        if (source == destination):	
            return 1.0 / denominator
		
        prob = 0 #Start with 0 since there can be more than one transition to target

        for transition in active_transitions:
            if (transition.target == destination and transition.source == source):
                tprob = transition.get_transition_probability(num_neighbors, ground_sensor)
                prob += tprob / float(denominator) #Add the probability of taking that transition

		#if(self.id == 0):
		#	print("State {0} probability of reaching directly state {1} : {2}".format(self.id, target, prob))

        if prob == 0:
            new_path = path
            for transition in active_transitions:
                new_path.append(source)
                nprob = self.probability_of_state_transition(transition.target, destination, num_neighbors, ground_sensor, new_path)
                nprob = transition.get_transition_probability(num_neighbors, ground_sensor)
                if (nprob > prob):
                    prob = nprob
        
        if prob == 0:
            prob = 0.0

        return prob

    def debug(self):
        for state in self.states:
            print("State")
            print("     State index", state.state_index)
            print("     State type", state.state_type)
            print("     Attributes", state.attributes)
            print("     Number of transitions", state.number_of_transitions)
            print("")
        print("")

        for transition in self.transitions:
            print("Transition")
            print("     From:", transition.source)
            print("     To:", transition.target)
            print("     Type:", transition.type)
            print("     P:", transition.value_of_parameter_p)
            if hasattr(transition, 'value_of_parameter_w'):
                print("     W:", transition.value_of_parameter_w)

    # TODO: make static
    @classmethod
    def parse_state_machine(cls, inputString):
        tokens = Tokenizer(inputString)
        number_of_states = 0
        state_machine_id = 0
        state_machine = StateMachine()
        state_machine.fsm_description = inputString

        print(inputString)

        while (tokens.has_more_tokens()):
            current_token = tokens.peek()
            print("current_token", current_token)

            if (current_token == '--fsm-config'):
                tokens.next_token()
                state_machine_id = tokens.next_token()
                state_machine.set_identifier(state_machine_id)
            elif (current_token == '--nstates'):
                tokens.next_token()
                number_of_states = tokens.next_token()
            else:
                new_state = State(tokens)
                state_machine.add_state(new_state)

                # Create transitions
                token = tokens.peek()
                while (re.match(r'--n(\d+)x(\d+)', token) != None and re.match(r'--s(\d+)', token) == None):
                    new_transition = Transition(tokens)
                    state_machine.add_transition(new_transition)
                    token = tokens.peek()

        return state_machine