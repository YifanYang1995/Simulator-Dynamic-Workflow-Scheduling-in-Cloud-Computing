import operator
import time,math
import random
import numpy as np
import pandas as pd

from deap import base
from deap import creator
from deap import tools
from deap import gp

from workflow_scheduling.baselines import logger
from workflow_scheduling.env.cloud_env_maxPktNum import cloud_simulator
from workflow_scheduling.policy import policy_learn
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

# Parameter Settings
genNum = 0              # Number of generations + 1 
workflow_types = 12     # Number of workflow types in the sampling set
total_workflow_num = 30 # Number of workflows contained in a training problem instances
                        ## i.e., Randomly sampling 30 times from the sampling set.
elitistNum = 10         # Number of elites saved in each generation
gamma = 12              # Parameters used to determine the workflow deadline: [1, 12, 24, 36] respectively means that the workflow deadline
                        ## is [gamma/dataset.vmVCPU.max()]=[1/48, 1/4, 1/2, 3/4] times that of its makespan if all its tasks are executed on the VMs with unit speed
algoName = 'GP'         # Used to name the algorithm, used as the name of the saved file
eval_eachIND = 3        # Number of training problem instances are used to evaluate each individuals per generation


# Generate Training Set
train_seed = 2023 
random.seed(train_seed)
np.random.seed(train_seed)

train_dataset = np.random.randint(0,workflow_types,(genNum+1, eval_eachIND, total_workflow_num ))
train_dataset = train_dataset.astype(np.int64)


# Fitness Function 
args = {"algo": algoName, "traffic pattern": "CONSTANT", "seed": 0,"dataset": train_dataset, "envid": 6, "arrival rate": [0.01],
        "gamma": gamma, "WorkflowNum": total_workflow_num, "compare": None, "record": True} # record: to record the choices under each decision point

def evalDWS(individual):

    func = toolbox.compile(expr = individual)   # Convert GPtree to expression

    base_env = cloud_simulator(args)            # Initial setting of the cloud environment 
    eval_seg = policy_learn.learn(individual.genID, base_env, func,
                                                      individual.eval_index, print_individual=False)

    return eval_seg['objectives'], eval_seg['VM_cost'], eval_seg['SLA_penalty'], eval_seg['VM_execHour'], eval_seg['VM_totHour'], eval_seg['missDeadlineNum']
            

# PrimitiveSet
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
# pset.addEphemeralConstant('rand', lambda:random.randint(-1, 1))
pset.addPrimitive(math.cos, 1)


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
pset.renameArguments(ARG10='RDL') # the remaining deadline time of the workflow == deadline - current_time


# GP Initialization
## create container
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

## create methods
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)  #
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)


if __name__ == "__main__":

    # Randomly generate a GP tree
    t1 = time.time()
    random.seed(10)
    np.random.seed(10)    
    ind = toolbox.individual()
    print('tree depth ----> ', ind.height)
    ind.genID = 0       # the index of the selected problem instance in the training set
    ind.eval_index = 0  # the index of the selected problem instance in the training set

    ans, a1, a2, a3, a4, a5 =  evalDWS(ind)

    print("**************** Completed ****************")
    logger.record_tabular("Total Time Elapsed", (time.time() - tstart) )  
    logger.dump_tabular()

    print("\nVMSR is: ", ind)  
    print("**************** Details ****************")
    print('Total cost is ', ans)
    print('rental cost --> ', a1)
    print('ddl penalty --> ', a2)
    print('VM_execHour --> ', a3)
    print('VM_totHour --> ', a4)
    print('missDeadlineNum --> ', a5)


    
   




      