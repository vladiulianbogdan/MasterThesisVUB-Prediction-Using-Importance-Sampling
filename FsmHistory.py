from StateMachine import StateMachine
from Experiment import Experiment
from Utility import peek_line

class FsmHistory:
    def __init__(self, file):
        fsm_description = file.readline()

        self.experiments = []
        self.fsm = StateMachine.parse_state_machine(fsm_description)

        while True:
            if (len(peek_line(file)) == 0):
                break

            experiment = Experiment(file, self.fsm)
            self.experiments.append(experiment)
    
    def get_active_states_statistics(self, threshold):
        active_states_statistics = {}
        total = 0
        number_of_active_states = 0

        for state in self.fsm.states:
            active_states_statistics[state.state_index] = 0

        for experiment in self.experiments:
            state_counters = experiment.get_active_states_statistics();

            for state_index in state_counters.keys():
                active_states_statistics[state_index] += state_counters[state_index]
                total += state_counters[state_index]
        
        active_states_as_array = []
        for state_index in active_states_statistics.keys():
            active_percentage = active_states_statistics[state_index] / total
            active_states_as_array.append((state_index, active_percentage))

            if (active_percentage > threshold):
                number_of_active_states += 1

        # take second element for sort
        def takeSecond(elem):
            return elem[1]

        active_states_as_array.sort(key=takeSecond, reverse=True)

        return number_of_active_states, active_states_as_array


