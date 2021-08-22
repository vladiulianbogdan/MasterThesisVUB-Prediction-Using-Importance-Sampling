import re
import math

class Transition:
    def __init__(self, tokens):
        token = tokens.next_token()
        match = re.match(r'--n(\d+)x(\d+)', token)
        self.is_active = True
        self.source = int(match.group(1))
        self.transition_index = int(match.group(2))

        self.target = Transition.compute_real_target(self.source, int(tokens.next_token()))

        # skip —c<state index>x<transition index> since it is not needed
        tokens.next_token()
        self.type = int(tokens.next_token())

        # skip —p<state index>x<transition index> since it is not needed
        token = tokens.peek()
        if (re.match(r'--p(\d+)x(\d+)', token) != None):
            tokens.next_token()
            self.value_of_parameter_p = float(tokens.next_token())

        token = tokens.peek()
        if (re.match(r'--w(\d+)x(\d+)', token) != None):
            # skip —w<state index>x<transition index> since it is not needed
            tokens.next_token()
            self.value_of_parameter_w = float(tokens.next_token())

        token = tokens.peek()
        if (re.match(r'--p(\d+)x(\d+)', token) != None):
            tokens.next_token()
            self.value_of_parameter_p = float(tokens.next_token())

    @classmethod
    def compute_real_target(cls, source, target_index):
        if (target_index < source):
            return target_index
        else:
            return target_index + 1
    
    def set_active(is_active):
        self.is_active = is_active

    def get_transition_probability(self, num_neighbors=0, ground_sensor=-1):
        type = self.type
        prob = self.value_of_parameter_p
        blackGroundThreshold = 0.1
        whiteGroundThreshold = 0.95		
        type_name = ["BlackFloor", "GrayFloor", "WhiteFloor", "NeighborsCount", "InvertedNeighborsCount", "FixedProbability"]	
        if type == 0 and ( ground_sensor >= blackGroundThreshold or ground_sensor < 0) :
            prob = 0.0
            #print("State {0} Transition {1} type 0 -> prob {2}".format(self.id, transition,prob))
        elif type == 1 and ( ground_sensor >= whiteGroundThreshold or ground_sensor < blackGroundThreshold or ground_sensor < 0):
            prob = 0.0
            #print("State {0} Transition {1} type 1 -> prob {2}".format(self.id, transition,prob))
        elif type == 2 and ( ground_sensor < whiteGroundThreshold or ground_sensor < 0) :	
            prob = 0.0
            #print("State {0} Transition {1} type 2 -> prob {2}".format(self.id, transition,prob))q
        elif type == 3 :
            prob = 1.0 / (1.0 + math.exp(self.value_of_parameter_w * (prob - num_neighbors)))
            #print("State {0} Transition {1} type 3 -> prob {2}".format(self.id, transition,prob))
        elif type == 4:
            prob = 1.0 - 1.0 / (1.0 + math.exp(self.value_of_parameter_w * (prob - num_neighbors)))
            #if(self.id == 0):
            #print("State {0} Transition {1} type 4 -> prob {2} [ p {3} | w {4} | n {5}]".format(self.id, transition,prob,self.transition_p[transition],self.transition_w[transition],num_neighbors))		
        #if prob == 0:# and transition == 0:
            #print("State {0} Transition {1} type {2} ground {3} neighbors {4} P {5} -> prob {6}".format(self.id, transition,type_name[type],ground_sensor,num_neighbors,self.transition_p[transition],prob))

        return prob 