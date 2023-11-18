import random
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

sys.path.insert(0, currentdir)

def act(stochastic, ob, vm_rule, removeVM):

    '''ob= [task_executeTime[0], remainNumDAGTask[1], remain_DDL[2], task_NumChildren[3],
            task_executeTime_real[4], vm_QueueTime[5], task_relativeFinishTime[6],
            vm_cost[7], vm_speed[8], vm_workload[9], vm_capacity[10], vm_remainTime[11], 
            Num_of_task_InQueue[12]]'''
    # VM selection rule: TS, ET, CU, PRICE,\\ TIQ, VMR, LFT, NIQ, \\ NOC, NOR, RDL

    prob = []                       # priority values of all candidate machines
    for i in range(len(ob)):
        prob.append([])
        prob[-1] = vm_rule(TS=ob[i][0], ET=ob[i][4], CU=ob[i][8], PRICE=ob[i][7],
                           TIQ=ob[i][5], VMR=ob[i][11], LFT=ob[i][6], NIQ=ob[i][12],
                           NOC=ob[i][3], NOR=ob[i][1], RDL=ob[i][2])

    if removeVM is not None:
        prob[removeVM] = 100000000

    if stochastic:
        VMindex = removeVM
        while VMindex == removeVM: 
            VMindex = random.choice(len(prob), p=prob)
    else:
        VMindex = np.argmin(prob)   # Select the machine with the smallest priority value

    return VMindex
