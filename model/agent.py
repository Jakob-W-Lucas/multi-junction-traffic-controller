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
        self.memory = deque(maxlen=10000)

        self.gamma = gamma #Discount factor
        self.epsilon = epsilon_start # Exploration vs Exploitation
        self.epsilon_min = 0.05 #Dynamic - changes according to how mature the model is
        self.epsilon_decay = 0.995 #How fast the agent matures

        self.steps_since_last_switch = 0
        self.current_phase_index = 0

        self.target_gnn = TrafficGNN(num_features=num_features) #Create a target network - representing a "lagged" version of the original net that updates slowly
        self.target_gnn.load_state_dict(self.gnn.state_dict()) #Load the current weights (since gnn is randomly initialised)
        self.target_gnn.eval() #Set to evaluation mode

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
            with torch.no_grad(): #Dont want to update weights this time... this will be done seperately
                lane_scores = self.gnn(state.x, state.edge_index) #Get Score for EVERY lane from the Policy Network [Num_Lanes, 1]
                
                phase_scores = []
                for phase_indices in valid_phases: #Value Decomposition (Sum lanes -> Phase Score)
                    score = lane_scores[phase_indices].sum().item() #Sum Q-values of lanes in this phase
                    
                    #Custom weightings to score here
                    
                    phase_scores.append(score)
                
                action = np.argmax(phase_scores)

        self._update_switch_timer(action) #Update Timer if neccessary
        return action
    
    def _update_switch_timer(self, new_action):
        if new_action != self.current_phase_index: #if the action chosen is different
            self.steps_since_last_switch = 0 #Reset timer
            self.current_phase_index = new_action #Change current on phase
    
    def update_memory(self, state, action, reward, next_state, done, phases):
        self.memory.append((state, action, reward, next_state, done, phases))
    
    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size: #Do not train if memory is less than batch
            return
        
        minibatch = random.sample(self.memory, batch_size) #Randomly sample a collection of memories. This line will change if using PER

        total_loss = 0
        self.optimizer.zero_grad()

        for state, action, reward, next_state, done, phases in minibatch:
        
            target_q = reward
            if not done:
                with torch.no_grad():
                    next_scores_policy = self.gnn(next_state.x, next_state.edge_index) #Grab the next lanes scores according to the current policy
                    next_scores_target = self.target_gnn(next_state.x, next_state.edge_index) #Grab the next lanes scores according to the lagging, target policy

                    next_phase_scores = []
                    for phase in phases: #For each of the avaliable phases
                        next_phase_scores.append(next_scores_policy[phase].sum().item()) #Figure out each's priority
                    best_next_action = np.argmax(next_phase_scores) #Figure out the best action for this
                    
                    target_value = next_scores_target[phases[best_next_action]].sum() #For the chosen action, figure out the lagged score
                    
                    target_q = reward + self.gamma * target_value.item() #Change the target q value according to this chosen action lagged score

            curr_scores_policy = self.gnn(state.x, state.edge_index) #Calculate the current lane policy scores
            current_q = curr_scores_policy[phases[int(action)]].sum() #Calculate the current q value

            loss = self.criterion(current_q, torch.tensor(target_q)) #Calculate the loss - defined as the difference between the target_q value for the next state adn the current q value.
            total_loss += loss #Increment the loss to backpropagate later
            

        (total_loss / batch_size).backward()
        self.optimizer.step()

        tau = 0.005 # Represents the proportion of change between the existing target network and the new, modified learner model
        
        for target_param, local_param in zip(self.target_gnn.parameters(), self.gnn.parameters()): #Update the target network a tiny bit based upon this new training network
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    