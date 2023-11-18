import os, sys, inspect, csv
from workflow_scheduling.env import dataset
import numpy as np
from workflow_scheduling.policy import scheduling_policy as pi

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(data)


# Process of using a specific policy ('func') to process a specific problem instance 
def learn(data_index1,          # the first index of the used problem instance 
          base_env,             # initial environment
          func,                 # the specific policy (i.e., VM selection rule used for selecting the best VM to execute a task)
          data_index2,          # the scond index of the used problem instance            
          print_individual): 
    
    # Results of this problem instance (a.k.a, an episode) processed by the policy 'func' 
    eval_seg = traj_segment_generator(data_index1, base_env, stochastic=False, # 'stochastic' represents whether replace 'func' with random selection
                                      func=func,data_index2=data_index2) 

    best_fitness = eval_seg['VM_cost']+eval_seg['SLA_penalty']

    if print_individual:
        print(f"Individual: Total Costs: {best_fitness}, VM_cost: {eval_seg['VM_cost']}, SLA_penalty: {eval_seg['SLA_penalty']}\n")
        print("============================= Evaluate an Individual ==============================")

    return eval_seg


# State Matrix: information about all available machines
def state_info_construct(env):

    ob = []     # output: observation

    # Information about the service object (i.e., a specific task)
    task_executeTime = env.nextWrf.get_taskProcessTime(env.nextTask)        ## task_size
    totNumDAGTask = env.nextWrf.get_totNumofTask()                          ## the total number of tasks in this workflow
    remainNumDAGTask = totNumDAGTask - env.nextWrf.get_completeTaskNum()    ## remain pending tasks in this workflow
    Deadline = env.nextWrf.get_Deadline()                                   ## is arrival_time + deadline_length
    remain_DDL = Deadline - env.nextTimeStep                                ## remain due time of this workflow
    task_NumChildren = env.nextWrf.get_NumofSuccessors(env.nextTask)        ## number of successors of the task
    task_ob = [task_executeTime, remainNumDAGTask, remain_DDL, task_NumChildren]

    # Information about rented machines: If the task is processed on each machine 
    for vm in env.vm_queues:   
        vm_QueueTime = vm.vmQueueTime()                                     ## TIQ: total processing time of all tasks queuing in this VM
        vm_cost = 0
        vm_speed = vm.get_cpu()
        task_executeTime_real = task_executeTime / vm_speed                 ## ET
        task_LatestFinishTime = vm.vmLatestTime()+ task_executeTime_real    ## LFT = ET + TIQ
        vm_workload = vm.get_utilization(env.nextWrf, env.nextTask)
        vm_capacity = vm.get_capacity(env.nextWrf, env.nextTask)            ## how many tasks can processed in one hour
        vm_remainTime = env.VMRemainAvaiTime[vm.vmid]                       ## unit: seconds
        vm_NumInQueue = vm.vmQueue.qlen()                                   ## Number of pending tasks on the machine
        ob.append([])
        ob[-1] = task_ob + [task_executeTime_real, vm_QueueTime, task_LatestFinishTime, vm_cost, 
                vm_speed, vm_workload, vm_capacity, vm_remainTime, vm_NumInQueue] 

    # Information about new machines 
    for dcind in range(env.dcNum): 
        for cpuNum in dataset.vmVCPU:      
            dc = dataset.datacenter[dcind] 
            vm_QueueTime = 0                 
            vm_cost = dc[-1]/2 * cpuNum                                     ## price
            vm_speed = cpuNum                                                   
            task_executeTime_real = task_executeTime / vm_speed             ## ET
            task_LatestFinishTime = vm_QueueTime + task_executeTime_real    ## LFT = TIQ +ET
            vm_workload = 0
            vm_capacity = 60*60 / (env.nextWrf.get_taskProcessTime(env.nextTask)/cpuNum)  
            vm_remainTime = 60*60
            vm_NumInQueue = 0
            ob.append([])   
            ob[-1] = task_ob + [task_executeTime_real, vm_QueueTime, task_LatestFinishTime, vm_cost, 
                    vm_speed, vm_workload, vm_capacity, vm_remainTime, vm_NumInQueue] 


    return ob


# An episode
def traj_segment_generator(data_index1, env, stochastic, func, data_index2): 

    ep_num = 0      
    ep_num_Max = 1  # only has one episode
    eventNum = 0    # number of time steps in an episode
    new = True      # marks if we're on first timestep of an episode    

    env.input_generation_index(data_index1)         # input the index of problem instance (episode index)
    env.input_evaluate_index(data_index2)           # input the index of problem instance (episode index)
    env.reset()                                     # initial the environment with seed=0
    ob = state_info_construct(env)                  # obtain the initial observation after activating the first task

    totalCost = []
    vmexecHour = []
    vmtotHour = []
    vmcost = []
    SLApenalty = []
    missDDLNum = []

    while True:

        if env.VMtobeRemove is None:                # Are there any unselectable machines in the queue
            removeVMindex = None                    # in most cases
        else: 
            removeVMindex = env.vm_queues_id.index(env.VMtobeRemove)

        action = pi.act(stochastic, ob, func, removeVMindex)  # output the index of the selected machine

        # If terminate
        if ep_num >= ep_num_Max: 

            return {"record": totalCost,
                    "objectives": np.mean(totalCost),
                    "VM_execHour": np.mean(vmexecHour), "VM_totHour": np.mean(vmtotHour),
                    "VM_cost": np.mean(vmcost), "SLA_penalty": np.mean(SLApenalty), 
                    "missDeadlineNum": np.mean(missDDLNum)}
        
        # move to next time step
        _, _, _, _, new = env.step(action) # 可以在这里output出VM-related信息
        ob = state_info_construct(env)
  
        if new:     # Whether to start a new episode

            totalCost.append(env.episode_info["VM_cost"] + env.episode_info["SLA_penalty"])
            vmexecHour.append(env.episode_info["VM_execHour"])
            vmtotHour.append(env.episode_info["VM_totHour"]) #  VM_totHour is the total rent hours of all VMs
            vmcost.append(env.episode_info["VM_cost"])
            SLApenalty.append(env.episode_info["SLA_penalty"])
            missDDLNum.append(env.episode_info["missDeadlineNum"])

            ep_num += 1

            if ep_num < data_index2:
                env.input_evaluate_index(ep_num)    # use next problem instance

            env.reset() 
            ob = state_info_construct(env)

        eventNum += 1

def fitness_normalization(x):
    x = np.asarray(x).flatten()
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, x

