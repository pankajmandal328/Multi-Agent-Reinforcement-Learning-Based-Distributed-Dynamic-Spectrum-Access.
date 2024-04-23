"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class QLearning:
    def __init__(self, environment, lr=0.01, gamma=0.9, e_greedy=0.95):
        self.power_properties = environment.power_properties
        self.power_levels = environment.power_levels
        self.state_space = environment.power_levels
        self.action_space = environment.action
        self.table, self.tablePd = generate_table(self.power_properties, np.max(self.power_levels), self.state_space, self.action_space)
        self.epsilon = e_greedy
        self.gamma = gamma
        self.lr = lr
        self.action_size = np.size(self.action_space)
        self.state = 0
        
    def choose_action(self, state):
        # This is how to choose an action
        state_actions = self.table[state, :]
        if (np.random.uniform() < self.epsilon):  # act non-greedy or state-action have no value
            while True:
                action = np.random.randint(self.action_size)
                if state_actions[action] > np.NINF:
                    break
        else:  # act greedy
            action = np.argmax(state_actions)  # replace argmax to idxmax as argmax means a different function in newer version of pandas
        return action

    def learn(self, s, a, r, s_):
        q_predict = self.table[s, a]
        if s_ != 'terminal': 
            q_target = r + self.gamma * self.table[s_, :].max()
        else: #For last reward in finite environment after reach the end (only for episodic task)
            q_target = r
        if q_predict!=np.NINF:
            self.table[s, a] += self.lr * (q_target - q_predict)


class plotter:
    def __init__(self, episodes):
        self.episodes = episodes
        self.data_memory = np.zeros(np.int(episodes))
        self.axis_memory = np.zeros(np.int(episodes))
        self.index = 0
        
    def record(self, rate, episode_th):
        self.data_memory[episode_th] = rate
        self.axis_memory[episode_th] = episode_th
            
    def plot(self, color = 'b', title="DSA", ax="steps", ay="data rate (bits/s/Hz)", label="Q-Learning", grid=100, smoother=0, fig=2):
        plot_interval = np.int(self.episodes/grid)
        rate_set = np.zeros(grid)
        axis_set = np.zeros(grid)
        if smoother < 0.1:
            smoother = 0.1
        smoother = np.log10(10*smoother) #To make logaritmic scale for ease to see as linear scale
        while plot_interval-(1-smoother)*plot_interval<1:
            smoother+=0.01
        for i in range(grid):
            rate_set[i]=np.sum(self.data_memory[np.int((i+(1-smoother))*plot_interval):(i+1)*plot_interval])/plot_interval/smoother
            axis_set[i]=self.axis_memory[i*plot_interval]
        # plt.rcParams["figure.figsize"] = (16,9)
        plt.figure(fig)
        # plt.ticklabel_format(style='sci', axis='x',useOffset=False, scilimits=(0,0))
        # plt.plot(self.axis_memory, self.data_memory, color, linewidth=1, alpha=0.10)
        plt.plot(axis_set, rate_set, color, linewidth=1, label=label)
        plt.legend(loc='lower right', fontsize='x-large')
        plt.axis([0, self.episodes, np.min(rate_set)-(np.max(rate_set)-np.min(rate_set))*0.1, np.max(rate_set)+(np.max(rate_set)-np.min(rate_set))*0.1])
        # plt.axis([0, self.episodes, 20, 27]) #For fix axis
        plt.xlabel(ax)
        plt.ylabel(ay)
        plt.minorticks_on()
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor',alpha=0.4)
        plt.suptitle(title, fontsize='x-large', fontweight='bold')
        plt.show()  
        
def generate_table(power_properties,TP_max,state_space,action_space):
    table = np.zeros((np.size(state_space,0), action_space.__len__()),dtype=np.float32)
    table[0,1]=np.NINF; table[len(state_space)-1,2]=np.NINF
    tablePd = pd.DataFrame(table, index=state_space, columns=action_space)
    return table, tablePd  