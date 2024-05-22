import operator
import math
import random
import numpy as np
# import pandas as pd
import multiprocessing

from deap import base
from deap import creator
from deap import tools
from deap import gp
from deaps import MTalgorithms as algorithms
# from deaps import MTgp as gp

import inspect
# import csv
import os
import sys
import time
import pickle

from workflow_scheduling.baselines import logger
from workflow_scheduling.env.cloud_env_maxPktNum import cloud_simulator
from workflow_scheduling.policy import policy_learn
from workflow_scheduling.policy.policy_learn import ensure_dir_exist
# from HEFT import heft50test
from workflow_scheduling.baselines import draw_figure


## sys.inputs
# 1: jod_id --> training data seed
# 2: the generation way of training data
# 3: pool size
# 4: batch size / evals on each generation in training
# 5: test data seed
# 6: test scenarios = 0~7 or all
# 7: evals on each test scenario
# 8: mutation rate
# 9: # of elites in each generation will be tested
# 10: \x, gamma
# 11: algorithm name
tstart = time.time()

# GP-related Parameters
popSize = 1024              # Popliation size
genNum = 100                 # Number of generations + 1 
c_rate = 0.8                # Crossover rate
m_rate = 0.15               # Mutation rate
workflow_types = 12         # Number of workflow types in the sampling set
train_workflowNum = 30     # Number of workflows contained in a training problem instances
                            ## i.e., Randomly sampling 30 times from the sampling set.
elitistNum = 10             # Number of elites saved in each generation
elite_eval_test = 1         # The top $n$ individuals of each generation will be evaluated in the test set
algoName = 'GP'             # Used to name the algorithm, used as the name of the saved file


# Generate Test Set
test_seed = 2023            
random.seed(test_seed)
np.random.seed(test_seed)

## sampling pool of different application scenarios
Test_Sample_Set = {'0':[0,1,2,3, 4,5,6,7, 8,9,10,11], '1': [0,1,2,3], '2': [4,5,6,7], '3': [8,9,10,11]} 
test_evalNum = 50           # Number of test problem instances in the test set used to average the objectives
test_workflowNum = 30       # Number of workflows contained in a test problem instances

test_tample_set_inx = 'all'       
if test_tample_set_inx=='all':    # Tested on all application scenarios
    test_DATA = []
    for inx in list(Test_Sample_Set.keys()):
        test_DATA.append(Test_Sample_Set[inx])
else:                       # Tested on a specific application scenario
    test_DATA = [Test_Sample_Set[test_tample_set_inx]]
sceNum = len(test_DATA)

test_dataset = np.zeros([sceNum, test_evalNum, test_workflowNum])
for i,data in enumerate(test_DATA):
    data = np.array(data)
    x = data[np.random.choice(len(data), size=(test_evalNum,test_workflowNum), replace=True)]
    x = x.astype(np.int64)
    test_dataset[i] = x
test_dataset =  test_dataset.astype(np.int64) 


# Generate Training Set (with rotation)
train_seed = 1 
random.seed(int(train_seed))
np.random.seed(int(train_seed))
GP_seed = 1000

train_evalNum = 3           # Number of training problem instances are used to evaluate each individuals per generation
train_dataset = np.random.randint(0,workflow_types,(genNum+1, train_evalNum, train_workflowNum ))
train_dataset = train_dataset.astype(np.int64)


# Environment 
gamma = 12      # Parameters used to determine the workflow deadline: [1, 12, 24, 36] respectively means that the workflow deadline
                ## is [gamma/dataset.vmVCPU.max()]=[1/48, 1/4, 1/2, 3/4] times that of its makespan if all its tasks are executed on the VMs with unit speed
args = {"algo": algoName, "traffic pattern": "CONSTANT", "seed": 0,"dataset": train_dataset, "envid": 6, "arrival rate": [0.01],
        "gamma": gamma, "WorkflowNum": train_workflowNum, "compare": None, "record": False}

## Save function
parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, parentdir)
def save_log(datas, fname):
    filename1 = f'{fname}_{args["algo"]}_{args["traffic pattern"]}_envid{args["envid"]}_gamma{args["gamma"]}.pickle'
    complete_filename1 = os.path.join(parentdir, "Saved_Results_", filename1)
    ensure_dir_exist(complete_filename1)
    f1 = open(complete_filename1, 'wb')
    pickle.dump(datas, f1)
    f1.close()
    print("==>> %s stored: %s " % (fname, complete_filename1))

## Fitness Function in Training Phase
def trainDWS(individual):

    func = toolbox.compile(expr = individual)

    base_env = cloud_simulator(args)  ## cloud environment setting
    eval_seg = policy_learn.learn(individual.genID, base_env, func,
                                                      individual.eval_index, print_individual=False)

    return eval_seg['objectives'], eval_seg['VM_cost'], eval_seg['SLA_penalty'], eval_seg['VM_execHour'], eval_seg['VM_totHour'], eval_seg['missDeadlineNum']
       
## Fitness Function in Training Phase
def testDWS(individual):

    func = toolbox.compile(expr = individual)

    args = {"algo": algoName, "traffic pattern": "CONSTANT", "seed": 0,"dataset": test_dataset, "envid": 6, "arrival rate": [0.01],
            "gamma": gamma, "WorkflowNum": test_workflowNum, "compare": None, "record": False}

    base_env = cloud_simulator(args)  ## cloud environment setting
    eval_seg = policy_learn.learn(individual.genID, base_env, func,
                                                      individual.eval_index, print_individual=False)

    return eval_seg['objectives'], eval_seg['VM_cost'], eval_seg['SLA_penalty'], eval_seg['VM_execHour'], eval_seg['VM_totHour'], eval_seg['missDeadlineNum']
       

## Test performance of rules in the archive
def unique(list1):      ### only test on unique indiviudals
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)

    unique_indexs = []
    for y in unique_list:
        count = list1.count(y)
        index_list = []
        index = -1
        for i in range(0, count):
            index = list1.index(y, index + 1)
            index_list.append(index)
        unique_indexs.append(index_list)

    return unique_list, unique_indexs

def testPerf(rules, rule_gen): 

    runNum = len(rules)
    allRules = []
    Log_ans=np.zeros((sceNum, runNum, rule_gen, test_evalNum))
    Log_cost=np.zeros((sceNum, runNum, rule_gen, test_evalNum))
    Log_penalty=np.zeros((sceNum, runNum, rule_gen, test_evalNum))
    Log_execHour=np.zeros((sceNum, runNum, rule_gen, test_evalNum))
    Log_totHour=np.zeros((sceNum, runNum, rule_gen, test_evalNum))
    Log_missDDL = np.zeros((sceNum, runNum, rule_gen, test_evalNum)) 
    
    Output= {'objective': Log_ans, 'cost':Log_cost, 'penalty': Log_penalty, \
    'VM_execHour': Log_execHour, 'VM_totHour': Log_totHour, 'missDeadlineNum':Log_missDDL }

    for geni in rules:
        for rulej in geni[:rule_gen]:   # The top $elite_eval_test$ in each generation will be test
            allRules.append(rulej)

    unique_rules, unique_indexs = unique(allRules)
    uniRules = [toolbox.clone(ind) for ind in unique_rules]

    for i in range(sceNum):
        for ind in uniRules:
            ind.genID = i 
            ind.output1 = [] # cost
            ind.output2 = [] # penalty
            ind.output3 = [] # VM_execHour
            ind.output4 = [] # VM_totHour
            ind.output5 = [] # missDeadlineNum
        ### separately calculate fitness on each training instance
        for j in range(test_evalNum): 
            for ind in uniRules:
                ind.eval_index = j
            fitnesses = toolbox.map(toolbox.evaluateTest, uniRules)
            fitnesses=list(fitnesses)
            for ind, fit in zip(uniRules, fitnesses):
                ind.output1.append(fit[1])
                ind.output2.append(fit[2])
                ind.output3.append(fit[3])
                ind.output4.append(fit[4])
                ind.output5.append(fit[5])  
            for k, item in enumerate(list(Output.keys())):
                fit = [temp[k] for temp in fitnesses]
                output_statistic = Output[item]
                for t,indexs in enumerate(unique_indexs):#range(len(
                    for ins in indexs:
                        output_statistic[i, ins//rule_gen,ins%rule_gen,j] = fit[t]
                Output[item] = output_statistic

    return Output 


# Primitive Set
## Define new functions
def protectedDiv(left, right):
    if round(right,4) == 0:
        return 1
    else:
        return left / right

pset = gp.PrimitiveSet("main", 11, prefix='ARG')    # The second arity is size of the terminal set
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(math.cos, 1)    # a random constant

## Rename terminals
pset.renameArguments(ARG0='TS')  # task_size of a task
pset.renameArguments(ARG1='ET')  # execute_time_period of a task
pset.renameArguments(ARG2='CU')  # compute_unit of a VM
pset.renameArguments(ARG3='PRICE')  # price of a VM in one hour
pset.renameArguments(ARG4='TIQ')  # the total execution time in the VM queue
pset.renameArguments(ARG5='VMR')  # remaining available Time of a VM
pset.renameArguments(ARG6='LFT')  # the relative latest_finish_time = ET + TIQ
pset.renameArguments(ARG7='NIQ')  # the number of tasks in the VM queue
pset.renameArguments(ARG8='NOC')  # the number of successor tasks (children) of this task
pset.renameArguments(ARG9='NOR')  # the number of remaining tasks in this workflow
pset.renameArguments(ARG10='RDL')  # the remaining deadline time of the workflow == deadline - current_time


# GP Initialization
## create container
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

## create methods
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)  #
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluateTest", testDWS)
toolbox.register("evaluate", trainDWS)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest, k=elitistNum)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) 

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))


# Main 
def main():

    random.seed(GP_seed)
    np.random.seed(GP_seed)

    pop = toolbox.population(n=popSize)
    hof = tools.HallOfFame(elitistNum)      # the top $elitistNum$ individuals in history

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_ddl = tools.Statistics(lambda ind: ind.output5)   # 'missDeadlineNum'

    mstats = tools.MultiStatistics(fitness=stats_fit, ddlNum=stats_ddl) 
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)

    ## Training Phase
    bestOne, log, archive = algorithms.SGP(pop, toolbox, c_rate, m_rate, genNum, train_evalNum,
                                   stats=mstats, halloffame=hof, verbose=True)

    runtime = [(time.time() - tstart) / 3600]
    print("Training Time Elapsed --->", runtime)

    ## Test Phase
    outputs = testPerf(archive, elite_eval_test) 

    return bestOne, log, hof, archive, outputs, runtime


if __name__ == "__main__":

    ## If multiple processes are required, use the three commented out statements below
    # pool = multiprocessing.Pool()         # Use all processor
    # toolbox.register("map", pool.map)

    bestOneGen, logbook, best, archive, testResult, runtime = main()

    # pool.close()    
   
    nn = {'1':"Logbook","2":"Best","3":"BestEachGen", "4": "Archive", "5": "Runtime", '6': "testResult"}
    save_log(logbook,nn["1"])
    save_log(best,nn["2"])  # best 10 in history at last generation (global)
    save_log(bestOneGen,nn["3"]) # best one in each generation
    save_log(archive,nn["4"])  # elites in each generation
    save_log(runtime,nn["5"])
    save_log(testResult,nn["6"]) 
    

    print("**************** Completed ****************")
    logger.record_tabular("Total Time Elapsed", (time.time() - tstart) / 3600)  ##save as hours
    logger.dump_tabular()

    # Draws
    draw_figure.trainCurve(logbook)
    draw_figure.testCurve(testResult['objective'])     

    print(best[0].fitness.values, "\nVMSR is: ", best[0])    
