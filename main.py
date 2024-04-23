from environment_q import Lingkungan
from RL_Brain import QLearning, plotter
import fungsi_q as fn
import numpy as np
import progressbar
import time

repeat = 1
steps = 10000
explore_steps = 5000
plotter1 = plotter(steps) #reward
plotter2 = plotter(steps) #datarate
plotter3 = plotter(steps) #power_Margin

L = 1 #Number of Licensee network
N_l=[2] #Number of users each Licensee network
distance_radar_bs = [15e3, 15e3, 15e3, 15e3, 15e3] #meters
N=sum(N_l)
distane_users_bs_max = 51 #meters
users_distance = np.random.randint(50,distane_users_bs_max,[N]) #meters
M = 10 # Transmit antennas each BS

Pm = fn.db2lin(43) #mW in BS
power_properties = [0,10,Pm] # power min, power step, power max;
Tx_gain= 13 #dB
Rx_interference_limit=fn.db2lin(-102) #dBm to mW
No = 4e-18 #mW/Hz or equal to-174 dBm/Hz
Bandwidth = 30e6 #Hz
f_c = 2.8e9 # frequency carier Hz
percentages_occupy = 80 #percentages power to occupy
zeros_penalties_range = 2.5 # percentages_occupy +/- this value, are zeros penalties 
kappa = [0.1,0.1] #Penalties factor below percentages_occupy and above percentages_occupy

antena_gain = fn.db2lin(Tx_gain) # dB to Lin
pl_incumbent = fn.db2lin(23.8*np.log10(distance_radar_bs)+41.2+20*np.log10(f_c/5e9)) # dB to Lin # Winner FSPL
pl_users = fn.db2lin(23.8*np.log10(users_distance)+41.2+20*np.log10(f_c/5e9)) # dB to Lin # Winner Urban
env = Lingkungan(N, power_properties, antena_gain, pl_incumbent, pl_users, Rx_interference_limit, Bandwidth, No, percentages_occupy, kappa)
P_agent = np.zeros((N)) #Actual power before precoder 

start = time.time()
for it in range(repeat):
    agent=[]; aa=[];
    for k in range (N):
        agent.append(QLearning(env, lr=1e-3, gamma=0.9, e_greedy=1))
        aa.append(agent[k].tablePd)
    H=[]; denominator_P_ratio=0
    for l in range (L):
        H.append(fn.H(N_l[l],M))
        denominator_P_ratio += np.trace(np.matmul(H[l],fn.herm(H[l]))).real
    P_ratio = np.zeros((N))
    for l in range (L):
        P_ratio[sum(N_l[0:l]):sum(N_l[0:(l+1)])]=np.diagonal(np.matmul(H[l],fn.herm(H[l])).real)/denominator_P_ratio
    ##main
    bar = progressbar.ProgressBar(maxval=steps/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range (steps):
        accumulator_r=0; accumulator_dr=0; accumulator_P=0; agent_counter=0;
        for l in range (L):
            for n in range (N_l[l]):
                action = agent[agent_counter+n].choose_action(agent[agent_counter+n].state)
                r, s_, perc_power = env.step(action, H[l], agent[agent_counter+n].state, P_agent[agent_counter:sum(N_l[0:(l+1)])], P_ratio[agent_counter+n], l, n) ##action, channel, state power at tx antennas of l-th BS, Bs index, index user n
                agent[agent_counter+n].learn(agent[agent_counter+n].state, action, r, s_)
                agent[agent_counter+n].state = s_ #New state
                P_agent[agent_counter+n]=env.state_space[agent[agent_counter+n].state] #Power allocation before precoder
                agent[agent_counter+n].epsilon -= 1/explore_steps
                accumulator_r+=r
                # print(r)
                accumulator_dr+=env.data_rate_val
                accumulator_P+=perc_power
            agent_counter+=N_l[l]
        plotter1.record(plotter1.data_memory[i]*it/(it+1)+accumulator_r/N/(it+1),i)
        plotter2.record(plotter2.data_memory[i]*it/(it+1)+accumulator_dr/N/(it+1),i)
        plotter3.record(plotter3.data_memory[i]*it/(it+1)+accumulator_P/N/(it+1),i)
        # print('percent %.1f' % (accumulator_P/N),
        #       'reward %.2f' % (accumulator_r/N))
        if i%10 == 0:
            bar.update(i/10 + 1)
    # print(np.average(plotter2.data_memory[explore_steps+1:steps]))
    # for k in range (N):
    #     print(agent[k].state)
    # print("")
    # print("")
    bar.finish()
    print("Nl=",N_l," it=", it)
    print("Datarate average=", np.average(plotter2.data_memory[explore_steps:steps-1]))  
    print("Penalties reward=", np.average(plotter1.data_memory[explore_steps:steps-1]))
    print("% Power load average=", np.average(plotter3.data_memory[explore_steps:steps-1]))
    print ("Time= ", time.time()-start)
    print("")
    time.sleep(0.2)

reward = plotter1.data_memory; np.save("Result/reward_n."+str(N)+"_Q-Learning_ZF.npy", reward)
datareate = plotter2.data_memory; np.save("Result/datareate_n."+str(N)+"_Q-Learning_ZF.npy", datareate)     
power_tot = plotter3.data_memory; np.save("Result/powerusage_n."+str(N)+"_Q-Learning_ZF.npy", power_tot)     
akumulasi_dr=np.average(datareate[explore_steps:steps-1])
akumulasi_penalti=np.average(reward[explore_steps:steps-1])
akumulasi_power=np.average(power_tot[explore_steps:steps-1])
plotter1.plot(grid=100, color = 'r', title='Reward Multi-agent DSA', label="L="+str(L)+" N="+str(N), ay="average reward / user", smoother=1, fig=1)
plotter2.plot(grid=100, color = 'g', title='Data Rate Multi-agent DSA', label="L="+str(L)+" N="+str(N), ay="average data rate (bits/s/Hz/user)", smoother=1, fig=2)
plotter3.plot(grid=100, color = 'b', title='% Power Utilization Multi-agent DSA', label="L="+str(L)+" N="+str(N), ay="average % power", smoother=1, fig=3)
print("Datarate average=",akumulasi_dr)
print("Penalties reward=", akumulasi_penalti)
print("% Power load average=",akumulasi_power)
print("Nl=",N_l," Final!")
print("Datarate average=",akumulasi_dr)
print("Penalties reward=", akumulasi_penalti)
print("% Power load average=",akumulasi_power)