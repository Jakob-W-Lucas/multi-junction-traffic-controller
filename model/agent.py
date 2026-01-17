import torch
import random
import numpy as np
import torch.optim as optim
from collections import deque
from model.graphnn import TrafficGNN

class TrafficAgent:
    """
    Reinforcement Learning Agent (Double DQN with potentially PER)

    Input is a graph representing the current simulation state

    Output is a weighting representing the current priority of roads to maximise traffic flow

    Output is then converted (through value decomposition) into a phase choice - which set of lights will maximise reward
    """
    def __init__(self, num_features, action_space_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0):
        self.gnn = TrafficGNN(num_features=num_features)

        self.optimizer = optim.Adam(self.gnn.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss() #MSELoss is industry standard for QLearning

        #Create Experience Replay (TODO Prioritised?)
        self.memory = deque(maxlen=2000)

        self.gamma = gamma #Discount factor
        self.epsilon = epsilon_start # Exploration vs Exploitation
        self.epsilon_min = 0.05 #Dynamic - changes according to how mature the model is
        self.epsilon_decay = 0.995 #How fast the agent matures

        self.steps_since_last_switch = 0
        self.current_phase_index = 0

        #Linear variables that weight get_action into prioritising the currently active traffic lights, to prevent constant switching. Uncomment only if neccesary
        #self.inertia_start_value = 10.0
        #self.inertia_decay_time = 15.0

    def get_action(self, state, valid_phases, training=True):
        """
        Based upon the state results, instead of just choosing the highest Q value (greedy Q learning) we run each lane's score through a combination of metrics
        to determine the best traffic light phase to turn on and off, considering factors such as current time existing lights have been on for etc.
        """

        self.steps_since_last_switch += 1 #Increment time since last switch, basically makes it more likely for a switch to be approved

        #Start with Exploration - if is chosen choose a random action from the list of valid phases
        if training and random.random() < self.epsilon:
            action = random.randint(0, len(valid_phases) - 1)
            
        #Move on to Exploitation
        else:
            with torch.no_grad(): #Dont want to update weights thsi time... this will be done seperately
                raise NotImplementedError("exploitation of phases not yet implemented... need to finalise formatting first")

        self._update_switch_timer(action)
        return action
    
    def _update_switch_timer(self, new_action):
        if new_action != self.current_phase_index: #if the action chosen is different
            self.steps_since_last_switch = 0 #Reset timer
            self.current_phase_index = new_action #Change current on light
    
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=32):
        raise NotImplementedError("train_step Not Yet Implemented")

    