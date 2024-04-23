"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import fungsi_q as fn
import numpy as np


class Lingkungan():
    def __init__(self,N, power_properties, antena_gain, path_loss_PU, path_loss_users, interference_limit=fn.db2lin(-105), Bandwidth=2e6, No=4e-18, percentages_occupy=90, kappa=[0.01,0.05]):
        self.action = np.array([0,-1,1]) # [unchange, decrease, increase] by power_step mW
        self.power_properties = power_properties
        self.n_Lic_net = N
        self.power_levels = np.arange(power_properties[0],power_properties[2],power_properties[1])
        self.state_space = self.power_levels
        self.action_space = self.action
        self.antena_gain = antena_gain
        self.path_loss_BS_Inc = path_loss_PU
        self.path_loss_users = path_loss_users
        self.interference_limit = interference_limit #mW
        self.Bandwidth = Bandwidth
        self.No =No
        self.kappa = kappa
        self.percent = percentages_occupy/100
        self.P_inter=self.interference_limit*np.array(self.path_loss_BS_Inc)
        self.data_rate_val=0
        
    def step(self, action, channel_user,state, P_agent, P_ratio, l, n):
        if P_agent[n]!=self.state_space[state]:
            sss=0
        temp_state = self.state_space[state]
        temp_state = temp_state + self.power_properties[1]*self.action_space[action]
        new_state = np.argmin(abs(self.state_space-temp_state))
        if sum(P_agent)-P_agent[n]+self.state_space[new_state]>np.max(self.state_space): 
            new_state=state #If BS Power more than limmit, cancel the moves
        P_agent[n]= self.state_space[new_state]
        precoder = np.matmul(np.linalg.pinv(channel_user),np.diag(np.sqrt(P_agent))) #ZF Precoder
        reward, power_percent = self.reward_step(channel_user, precoder, P_ratio, l, n)
        return reward, new_state, power_percent
        
    def reward_step(self, channel_user, precoder, P_ratio, l, n):
        Pactual=np.matmul(fn.herm(precoder[:,n]),precoder[:,n]).real #Exactly at the transmit antenna for user n
        penalties = self.kappa[0]*np.heaviside(self.P_inter[l]*P_ratio*(self.percent-0.025)-Pactual, 0)*np.abs(self.P_inter[l]*P_ratio*self.percent-Pactual) + self.kappa[1]*np.heaviside(Pactual-self.P_inter[l]*P_ratio*(self.percent+0.025),0)*np.abs(Pactual-self.P_inter[l]*P_ratio*self.percent)
        power_percent = 100 - (self.P_inter[l]*P_ratio-Pactual)/(self.P_inter[l]*P_ratio)*100
        HF=np.matmul(channel_user, precoder)
        self.data_rate(HF,n) 
        reward_final=- penalties
        return reward_final, power_percent
    
    def data_rate(self, HF, n): #ZF
        Power=np.abs(HF[n,n])
        denominator = self.Bandwidth*self.No
        self.data_rate_val= np.log2(1+self.antena_gain*Power/np.sqrt(self.path_loss_users[n])/denominator)
        return self.data_rate_val
        
