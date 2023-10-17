    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:45:42 2021

@author: shamano
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 16:14:59 2021

@author: shamano
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:08:48 2021

@author: shamano
"""

# Produces the figure in the paper for the sensitivity to iterations


import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import configparser




PARTIAL = False

FAILURE_RATE_CONSTRAINTS = 0.100001

EPSMIN = 0
EPSN = 0
EPSINC = 0
ITERMIN = 0
ITERINC = 0
ITERN = 0
NTRIALS = 0
SAMPLESMIN = 0
SAMPLESN = 0
SAMPLESINC = 0
FAILURE_PROB = 0


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = '16'



def read_configuration(fname):
    config = configparser.ConfigParser()
    print("Reading configuration file ",fname)
    config.read(fname)
    global NVERTICES,EPSMIN,EPSINC,EPSN,ITERMIN,ITERINC,ITERN,NTRIALS,SAMPLESINC,SAMPLESMIN,SAMPLESN,FAILURE_PROB
    
         
    if config['MAIN']['NVERTICES'] is None:
        print('Missing configuration parameter ',NVERTICES)
    else:
         NVERTICES = int(config['MAIN']['NVERTICES'])
                  
    if config['MAIN']['SAMPLESMIN'] is None:
        print('Missing configuration parameter ',SAMPLESMIN)
    else:
         SAMPLESMIN = float(config['MAIN']['SAMPLESMIN']) 
         
    if config['MAIN']['SAMPLESN'] is None:
        print('Missing configuration parameter ',SAMPLESN)
    else:
         SAMPLESN = int(config['MAIN']['SAMPLESN']) 
        
    if config['MAIN']['SAMPLESINC'] is None:
        print('Missing configuration parameter ',SAMPLESINC)
    else:
         SAMPLESINC = float(config['MAIN']['SAMPLESINC']) 
         
    if config['MAIN']['ITERMIN'] is None:
        print('Missing configuration parameter ',ITERMIN)
    else:
         ITERMIN = int(config['MAIN']['ITERMIN']) 
         
    if config['MAIN']['ITERINC'] is None:
        print('Missing configuration parameter ',ITERINC)
    else:
         ITERINC = int(config['MAIN']['ITERINC']) 
        
    if config['MAIN']['ITERN'] is None:
        print('Missing configuration parameter ',ITERN)
    else:
         ITERN = int(config['MAIN']['ITERN']) 
         
    if config['MAIN']['NTRIALS'] is None:
        print('Missing configuration parameter ',NTRIALS)
    else:
         NTRIALS = int(config['MAIN']['NTRIALS'])
         
    if config['MAIN']['FAILURE_PROB'] is None:
        print('Missing configuration parameter ',FAILURE_PROB)
    else:
         FAILURE_PROB = float(config['MAIN']['FAILURE_PROB']) 
         
    print('Done reading configuration')



if __name__ == "__main__":
    
    # THIS PRODUCES THE CHART IN THE INITIAL SUBMISSION
    # NVERTICES = 20
    # PREFIX= 9
    # read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
    # samples_list =  [(SAMPLESMIN + i*SAMPLESINC) for i in range(SAMPLESN)]
    # failures_list = [0.21, 0.18, 0.15, 0.16, 0.1, 0.13, 0.13, 0.17, 0.13, 0.12, 0.1, 0.08, 0.13, 0.12, 0.16, 0.17, 0.13, 0.09, 0.14, 0.12, 0.14, 0.11, 0.12, 0.12, 0.07, 0.08, 0.11, 0.1, 0.08, 0.15, 0.05, 0.09, 0.07, 0.08, 0.08, 0.11, 0.1, 0.07, 0.1, 0.12, 0.08, 0.05, 0.1, 0.12, 0.09, 0.07, 0.08, 0.06, 0.07, 0.12, 0.12, 0.12, 0.08, 0.1, 0.09]
    # figsamples, axsamples = plt.subplots()
    # color1 = 'tab:red'
    # color2 = 'tab:blue'
    # axsamples.set_ylabel('failure probability',fontsize=15)
    # axsamples.set_xlabel('$S$',fontsize=15)
    # samples = np.array(samples_list)
        # failures =np.array(failures_list)
    # axsamples.plot(samples,failures,color=color1,linewidth=2) 
    # figsamples.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.grid(b=True,which='both',axis='both')
    # plt.ylim(0,1.05*np.max(failures))
    # plt.show()
    # figsamples.savefig('chart.pdf')

    # THIS IS THE NEW PLOTTING METHOD
    N_START = 1
    N_END = 50
    read_configuration("NEWCASE/Samples/1/config.txt")
    samples_list =  [(SAMPLESMIN + i*SAMPLESINC) for i in range(SAMPLESN)]
    success_series = np.zeros((N_END,len(samples_list)))

    for INDEX in range(1,N_END+1):

        read_configuration("NEWCASE/Samples/{}/config.txt".format(INDEX))
        complete_budget = pickle.load(open("NEWCASE/Samples/{}/complete_budget_series.dat".format(INDEX),"rb"))

        start = 0
        for i in range(len(samples_list)):
            aslice = complete_budget[start:start+NTRIALS]
            aslice = np.array(aslice)
            start += NTRIALS
            success_series[INDEX-1,i] = (aslice<0).sum()/NTRIALS
           # time_deviation[i] = np.std(aslice)
           # time_min[i] = np.min(aslice)
           # time_max[i] = np.max(aslice)
           
    mean_v = np.zeros((len(samples_list),))
    std_v = np.zeros((len(samples_list),))
    for i in range(len(samples_list)):
        v = success_series[:,i]
        mean_v[i] = np.mean(v)
        std_v[i] = np.std(v)
        
    figtime, axtime = plt.subplots()
    color = 'tab:red'
    colormin = 'tab:blue'
    colormax = 'tab:green'
    colorconst = 'tab:blue'
    axtime.set_ylabel('$\Pr[C(\mathcal{P})>B]$')
    axtime.set_xlabel('Number of samples ($S$)')
    samples_a = np.array(samples_list)
    #axtime.errorbar(iterations_a,time_series,yerr=time_deviation,color=color,linewidth=4)
    axtime.plot(samples_a,mean_v ,color=color,linewidth=2)
    axtime.fill_between(samples_a, mean_v-std_v, mean_v+std_v , alpha=0.3,facecolor='tab:green')
    axtime.plot([0,samples_list[-1]],[FAILURE_PROB,FAILURE_PROB],color=colorconst,linewidth=2)
    axtime.set_xlim([0, samples_list[-1]])
    axtime.annotate('$P_f$',xy=(10,0.1),xytext=(1,0.115))
    
    #axtime.fill_between(iterations_a, time_series-time_deviation, time_series+time_deviation , alpha=0.3,facecolor=colormax)
    
    #axtime.plot(iterations_a,time_series - time_deviation,color=colormin,linewidth=2)
    #axtime.plot(iterations_a,time_series + time_deviation,color=colormin,linewidth=2)
    axtime.tick_params(axis='y')
    figtime.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(b=True,which='major',axis='both')
    plt.ylim(0,1.05*np.max(success_series))
    plt.show()
    figtime.savefig('samples_dependency.pdf')