import re

class State:
    def __init__(self, tokens):
        # Parse state information
        state_declaration = tokens.next_token()

        match = re.match(r'--s(\d+)', state_declaration)
        self.state_index = int(match.group(1))
        
        self.state_type = int(tokens.next_token())

        token = tokens.peek()
        self.attributes = []
        while (re.match(r'--n(\d+)', token) == None):
            self.attributes.append(tokens.next_token())
            self.attributes.append(tokens.next_token())
            token = tokens.peek()

        token = tokens.next_token()
        n_state_index = re.match(r'--n(\d+)', token)
        self.number_of_transitions = tokens.next_token()
        self.is_active = True
    
    def set_active(self, is_active):
        self.is_active = is_active