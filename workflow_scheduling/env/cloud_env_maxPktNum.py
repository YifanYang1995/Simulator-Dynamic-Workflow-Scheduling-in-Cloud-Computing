import numpy as np
# import pandas as pd
from workflow_scheduling.baselines import logger
import csv
import math
import os, sys, inspect, random
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from workflow_scheduling.env.stats import Stats
from workflow_scheduling.env.poissonSampling import one_sample_poisson
from workflow_scheduling.env.vm import VM
from workflow_scheduling.env import dataset
from workflow_scheduling.env.workflow import Workflow
from workflow_scheduling.env.simqueue import SimQueue
from workflow_scheduling.env.simsetting import Setting


vmidRange = 10000

def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)

def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):

    def __init__(self, args):

        self.set = Setting(args)
        self.baseHEFT = args["compare"]     # None
        self.dataset = args["dataset"]
        self.GENindex = None
        self.indEVALindex = None       
        self.TaskRule = None                # input th task selection rule here, if has

        if self.set.is_allocate_trace_record:
            self.df = {}
            __location__ = os.getcwd() + '\Saved_Results_'
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s_gamma%s.csv' % (args["algo"],  args["seed"], args["arrival rate"], args["gamma"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'VM ID', 'VM speed', 'Price', 'VM Rent Start Time', 'VM Rent End Time', 'VM Pending Index' ]) # 6 + 6 + 6 columns     
    
    def close(self):
        print("Environment id %s is closed" % (self.set.envid))

    def _init(self):

        self.usr_queues = []            # [usr1:[workflows, ...], usr2:[workflows, ...]], e.g., user1 stores 30 workflows
        self.vm_queues = []             # [VM1, VM2, ...] each VM is a class
        self.vm_queues_id = []          # the vmid of each VM in self.vm_queues
        self.vm_queues_cpu = []
        self.vm_queues_rentEndTime = []
        self.usrNum = self.set.usrNum   ## useless for one cloud
        self.dcNum = self.set.dcNum     ## useless for one cloud
        self.wrfNum = self.set.wrfNum
        self.totWrfNum = self.set.totWrfNum
        self.VMtypeNum = len(dataset.vmVCPU) ## number of VM types
        self.numTimestep = 0            # indicate how many timesteps have been processed
        self.completedWF = 0
        self.VMRemainingTime = {}       # {vmid1:time, vmid2:time}
        self.VMRemainAvaiTime = {}      # reamin available time  = leased time period - vm_total_execute_time
        self.VMrentInfos = {}           # {VMid: [rent start time, rent end time]}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wrfNum, self.set.history_len)) 
        self.VMcost = 0
        self.SLApenalty = 0
        self.wrfIndex = 0
        self.usrcurrentTime = np.zeros(self.usrNum)  # Used to record the current moment of the user
        self.remainWrfNum = 0           # Record the number of packets remained in VMs
        self.missDeadlineNum = 0
        self.VMrentHours = 0  
        self.VMexecHours = 0  

        # IMPORTANT: used to get the ready task for the next time step
        self.firstvmWrfLeaveTime = []   # Record the current timestamp on each VM
        self.firstusrWrfGenTime = np.zeros(self.usrNum)  # Arrival time of the first inactive workflow in each user's workflow set

        self.uselessAllocation = 0
        self.VMtobeRemove = None

        self.usr_respTime = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wrfNum)) 


        # upload all workflows with their arrival time to the 'self.firstusrWrfGenTime'
        for i in range(self.usrNum):
            self.usr_queues.append(SimQueue())
            workflowsIDs = self.dataset[self.GENindex][self.indEVALindex]   
            for appID in workflowsIDs:
                self.workflow_generator(i, appID)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime() 

        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr() 
        self.PrenextTimeStep = self.nextTimeStep
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt() # obtain the root task of the first workflow in the self.nextUsr
        temp = self.nextWrf.get_allnextTask(self.finishTask)   # Get all real successor tasks of the virtual workflow root task
                
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]
        if len(temp) > 1:  # the next task has parallel successor tasks
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.stat = Stats(self.set)

    # Generate one workflow at one time
    def workflow_generator(self, usr, appID):

        wrf = dataset.wset[appID]
        nextArrivalTime = one_sample_poisson(
                                self.set.get_individual_arrival_rate(self.usrcurrentTime[usr], usr, appID), 
                                self.usrcurrentTime[usr])

        self.remainWrfNum += 1
        # add workflow deadline to the workflow
        pkt = Workflow(self.usrcurrentTime[usr], wrf, appID, usr, dataset.wsetSlowestT[appID], self.set.dueTimeCoefs[usr, appID], self.wrfIndex) #self.set.gamma / max(dataset.vmVCPU))

        self.usr_queues[usr].enqueue(pkt, self.usrcurrentTime[usr], None, usr, 0) # None means that workflow has not started yet
        self.usrcurrentTime[usr] = nextArrivalTime
        self.totWrfNum -= 1
        self.wrfIndex +=1


    def reset(self):
        random.seed(self.set.seed)
        np.random.seed(self.set.seed)
        self._init()

    def input_task_rule(self, rule):
        self.TaskRule = rule

    def input_generation_index(self, genID): ## the first index in the dataset
        self.GENindex = genID

    def input_evaluate_index(self, evalNum): ## the second index in the dataset
        self.indEVALindex = evalNum    

    def generate_vmid(self):
        vmid = np.random.randint(vmidRange, size=1)[0]
        while vmid in self.VMRemainingTime:
            vmid = np.random.randint(vmidRange, size=1)[0]
        return vmid

    def get_nextWrfFromUsr(self):       # Select the User with the smallest timestamp
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstPktTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstPktTime     # Returns the user and arrival time of the minimum arrival time of the workflow in the current User queue.

    def get_nextWrfFromVM(self):        # Select the machine with the smallest timestamp
        if len(self.firstvmWrfLeaveTime) > 0:
            vmInd = np.argmin(self.firstvmWrfLeaveTime)
            firstPktTime = self.firstvmWrfLeaveTime[vmInd]
            return vmInd, firstPktTime  # Returns vm-id and the minimum end time of the current VM
        else:
            return None, math.inf

    def get_nextTimeStep(self):

        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromVM() 
        if tempnextTimeStep > tempnextTimeStep1:  # task ready time > VM minimum time
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
                                        # The next step is to process the VM and update it to the timestep of the VM.
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.vm_queues[self.nextUsr].get_firstDequeueTask() # Only returns time, does not process task
        else:  # tempnextTimeStep <= tempnextTimeStep1
            if tempnextTimeStep == math.inf:   ## tempnextTimeStep：when self.usr_queues.queue is []
                self.nextTimeStep = None       ## tempnextTimeStep1：when self.firstvmWrfLeaveTime is []
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep # Next step is to process user & Update to user's timeStep
                self.nextisUsr = True    # Activate new Workflow from Usr_queue
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt() # The current first task in the selected user

    def update_VMRemain_infos(self):
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues_rentEndTime[ind] - self.nextTimeStep
            maxTimeStep = max(self.vm_queues[ind].currentTimeStep, self.nextTimeStep) # consider idle gap in VM
            self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[ind] - maxTimeStep - self.vm_queues[ind].vmQueueTime() 

    def remove_expired_VMs(self):
        removed_keys = []        
        Indexes = np.where(np.array(self.vm_queues_rentEndTime) < self.nextTimeStep + 0.00001)[0]

        if not self.nextisUsr:
            nextvmid = self.vm_queues[self.nextUsr].vmid
            if self.nextUsr in Indexes:
                self.VMtobeRemove = nextvmid 
            else:
                self.VMtobeRemove = None

        for ind in Indexes:  
            a = self.vm_queues[ind].get_pendingTaskNum()
            if a ==0:       
                removed_keys.append(self.vm_queues_id[ind])

        for key in removed_keys:

            del self.VMRemainingTime[key]
            del self.VMRemainAvaiTime[key]
            ind = self.vm_queues_id.index(key)
            del self.vm_queues_id[ind]
            del self.vm_queues_cpu[ind]
            del self.vm_queues_rentEndTime[ind]
            vm = self.vm_queues.pop(ind)
            del self.firstvmWrfLeaveTime[ind]
            del vm              

        if not self.nextisUsr:
            if nextvmid in self.vm_queues_id:
                self.nextUsr = self.vm_queues_id.index(nextvmid)   
            else:
                logger.log('nextvmid is not in self.vm_queues_id')
                self.uselessAllocation +=1
                logger.log('-----> wrong index:', self.uselessAllocation)


    def extend_specific_VM(self, VMindex):
        key = self.vm_queues_id[VMindex]
        maxTimeStep = max(self.vm_queues[VMindex].currentTimeStep, self.nextTimeStep)
        self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[VMindex] - maxTimeStep - self.vm_queues[VMindex].vmQueueTime() # has idle gap
        while self.VMRemainAvaiTime[key] < -0.00001 : # ignore system error
            self.VMRemainAvaiTime[key] += self.set.VMpayInterval
            self.vm_queues[VMindex].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime[VMindex] = self.vm_queues[VMindex].rentEndTime
            self.update_VMcost(self.vm_queues[VMindex].loc, self.vm_queues[VMindex].cpu, True) 
            self.VMrentInfos[key] = self.VMrentInfos[key][:4] + [self.vm_queues[VMindex].rentEndTime] #self.VMrentInfos[key][-1]+dataset.vmPrice[self.vm_queues[VMindex].cpu]]           


    def record_a_completed_workflow(self, ddl_penalty):

        if self.set.is_allocate_trace_record:        
            Workflow_Infos = [self.nextWrf.appArivalIndex, self.nextWrf.appID,
                            self.nextWrf.generateTime, self.nextTimeStep, self.nextWrf.deadlineTime, ddl_penalty]

            for task in range(len(self.nextWrf.executeTime)):

                Task_Infos = [task, self.nextWrf.app.nodes[task]['processTime'], self.nextWrf.executeTime[task], 
                            self.nextWrf.readyTime[task], self.nextWrf.enqueueTime[task], self.nextWrf.dequeueTime[task]]

                VM_Infos = self.VMrentInfos[self.nextWrf.processDC[task]] + [self.nextWrf.pendingIndexOnDC[task]]

                write_csv_data(self.pkt_trace_file, Workflow_Infos + Task_Infos + VM_Infos)


    # Check whether the machine's lease period needs to be extended
    def extend_remove_VMs(self): 
        expiredVMid = []
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues[ind].rentEndTime-self.nextTimeStep
            self.VMRemainAvaiTime[key] = self.VMRemainingTime[key] - self.vm_queues[ind].pendingTaskTime

            if self.VMRemainAvaiTime[key] <= 0:
                if self.vm_queues[ind].currentQlen == 0: # to be removed
                    expiredVMid.append(key) 
                else:
                    while self.VMRemainAvaiTime[key] <= 0:
                        self.VMRemainingTime[key] += self.set.VMpayInterval
                        self.VMRemainAvaiTime[key] += self.set.VMpayInterval
                        self.vm_queues[ind].update_vmRentEndTime(self.set.VMpayInterval)
                        self.update_VMcost(self.vm_queues[ind].loc, self.vm_queues[ind].cpu, True)

        if len(expiredVMid) > 0:  # Really remove here
            if not self.nextisUsr:
                nextvmid = self.vm_queues[self.nextUsr].vmid

            for key in expiredVMid:
                del self.VMRemainingTime[key]
                del self.VMRemainAvaiTime[key]
                ind = self.vm_queues_id.index(key)
                del self.vm_queues_id[ind]
                del self.vm_queues_cpu[ind]
                del self.vm_queues_rentEndTime[ind]
                vm = self.vm_queues.pop(ind)
                del self.firstvmWrfLeaveTime[ind]
                del vm        

            # If there is deletion, you need to adjust the index corresponding to self.nextUsr
            if not self.nextisUsr:
                if nextvmid in self.vm_queues_id:
                    self.nextUsr = self.vm_queues_id.index(nextvmid)   
                else:
                    print('wrong')
            

    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        # decode the action: the index of the vm which ranges from 0 to len(self.vm_queues)+self.vmtypeNum*self.dcNum

        # 1) Map & Dispatch
        # maping the action to the vm_id in current VM queue
        diff = action - len(self.vm_queues)
        if diff > -1:  # a new VM is deployed
            vmid = self.generate_vmid()  # Randomly generate a set of numbers to name
            dcid = 0    # This is used to distinguish the number of different resource types for each DC and can be omitted
            vmtype = diff % self.VMtypeNum 
    
            selectedVM = VM(vmid, dataset.vmVCPU[vmtype], dcid, dataset.datacenter[dcid][0], self.nextTimeStep, self.TaskRule)
            self.vm_queues.append(selectedVM)
            self.firstvmWrfLeaveTime.append(selectedVM.get_firstTaskDequeueTime()) #new VM is math.inf
            self.vm_queues_id.append(vmid)
            self.vm_queues_cpu.append(dataset.vmVCPU[vmtype]) 
            self.update_VMcost(dcid, dataset.vmVCPU[vmtype], True)
            selectedVMind = -1
            self.VMRemainingTime[vmid] = self.set.VMpayInterval  # initialize the remaining time for the new VM
            self.VMRemainAvaiTime[vmid] = self.set.VMpayInterval            
            self.vm_queues[selectedVMind].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime.append(self.vm_queues[selectedVMind].rentEndTime) 
            self.VMrentInfos[vmid] = [vmid, dataset.vmVCPU[vmtype],  dataset.vmPrice[dataset.vmVCPU[vmtype]], 
                                      self.nextTimeStep, self.vm_queues[selectedVMind].rentEndTime]     
        else:
            selectedVMind = action


        reward = 0
        self.PrenextUsr, self.PrenextTimeStep, self.PrenextTask = self.nextUsr, self.nextTimeStep, self.nextTask 


        # dispatch nextWrf to selectedVM and update the wrfLeaveTime on selectedVM 
        parentTasks = self.nextWrf.get_allpreviousTask(self.PrenextTask)
        if len(parentTasks) == len(self.nextWrf.completeTaskSet(parentTasks)): # all its predecessor tasks have been done, just double-check
            processTime =  self.vm_queues[selectedVMind].task_enqueue(self.PrenextTask, self.PrenextTimeStep, self.nextWrf)
 
            self.VMexecHours += processTime/3600                                                                                                              
            self.firstvmWrfLeaveTime[selectedVMind] = self.vm_queues[selectedVMind].get_firstTaskDequeueTime() # return currunt timestap on this machine

            self.extend_specific_VM(selectedVMind) 


        # 2) Dequeue nextTask
        if self.isDequeue:      # True: the nextTask should be popped out 
            if self.nextisUsr:  # True: the nextTask to be deployed comes from the user queue
                self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                _, _ = self.usr_queues[self.PrenextUsr].dequeue() # Here is the actual pop-up of the root task 
                self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstPktEnqueueTime() 
                                                            # Updated with the arrival time of the next workflow
                self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_appID(), self.nextWrf.get_generateTime()) # record
            else:               # the nextTask to be deployed comes from the vm queues
                _, _ = self.vm_queues[self.PrenextUsr].task_dequeue() # Here nextTask actually starts to run
                self.firstvmWrfLeaveTime[self.PrenextUsr] = self.vm_queues[self.PrenextUsr].get_firstTaskDequeueTime()
                                                            # Update the current TimeStamp in this machine


        # 3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)   # all successor tasks of the current self.finishTask
                                    # and one successor task has already enqueued

        if len(temp_Children_finishTask) > 0:
            self.dispatchParallelTaskNum += 1

        while True: 

            # self.nextWrf is completed
            while len(temp_Children_finishTask) == 0:  # self.finishTask is the final task of self.nextWrf
                
                if self.nextisUsr:  # for double-check: Default is False
                    # Because it corresponds to self.finishTask, if temp==0, it means it cannot be entry tasks
                    print('self.nextisUsr maybe wrong')
                _, app = self.vm_queues[self.nextUsr].task_dequeue()  
                self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime() 
                        # If there is no task on the VM, math.inf will be returned
                if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):     # self.nextWrf has been completed
                    respTime = self.nextTimeStep - self.nextWrf.get_generateTime()
                    self.usr_respTime[app.get_originDC()][app.get_appID()] += respTime
                    self.usr_received_wrfNum[app.get_originDC()][app.get_appID()] += 1                    
                    self.completedWF += 1
                    self.remainWrfNum -= 1
                    ddl_penalty = self.calculate_penalty(app, respTime)
                    self.SLApenalty += ddl_penalty
                    self.record_a_completed_workflow(ddl_penalty)
                    del app, self.nextWrf

                self.get_nextTimeStep()
                if self.nextTimeStep is None:
                    break
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            if self.nextTimeStep is None:
                break

            # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
            if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                to_be_next = None
                while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                    temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                    temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                    if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) >0:
                        self.dispatchParallelTaskNum += 1
                    else: 
                        to_be_next = temp_nextTask
                        break

                if to_be_next is not None: 
                    self.nextTask = to_be_next
                    if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                        self.isDequeue = False
                    else:
                        self.isDequeue = True
                    break

                else: # Mainly to loop this part
                    _, _ = self.vm_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                    self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                    self.get_nextTimeStep() 
                    self.update_VMRemain_infos()
                    self.remove_expired_VMs()                        
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                    self.dispatchParallelTaskNum = 0                     
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

            else: # i.e., len(temp_Children_finishTask)<=self.dispatchParallelTaskNum
                # self.nextTask is the last imcompleted successor task of the self.finishTask
                if not self.isDequeue:      # Defaults to True
                    print('self.isDequeue maybe wrong')      
                self.get_nextTimeStep()
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                    
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.dispatchParallelTaskNum = 0 # Restart recording the number of successor tasks of self.finishTask
                if self.nextTimeStep is not None:
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        self.numTimestep = self.numTimestep + 1  ## useless for GP
        self.notNormalized_arr_hist = self.stat.update_arrival_rate_history() ## useless for GP

        done = False
        if self.remainWrfNum == 0:
            if len(self.firstvmWrfLeaveTime) == 0:
                done = True
            elif self.firstvmWrfLeaveTime[0] == math.inf and self.firstvmWrfLeaveTime.count(self.firstvmWrfLeaveTime[0]) == len(self.firstvmWrfLeaveTime):
                done = True

        if done:
            reward = -self.VMcost-self.SLApenalty   
            self.episode_info = {"VM_execHour": self.VMexecHours, "VM_totHour": self.VMrentHours,  # VM_totHour is the total rent hours of all VMs
                    "VM_cost": self.VMcost, "SLA_penalty": self.SLApenalty, "missDeadlineNum": self.missDeadlineNum}
            # print('Useless Allocation has ----> ',self.uselessAllocation)
            self._init()  ## cannot delete  

        return reward, self.usr_respTime, self.usr_received_wrfNum, 0, done


    # calculate the total VM cost during an episode
    def update_VMcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.VMcost += temp * dataset.vmPrice[cpu]      # (dataset.datacenter[dc][-1])/2 * cpu
        self.VMrentHours += temp


    def calculate_penalty(self, app, respTime):
        appID = app.get_appID()
        threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold,5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            return 1+dataset.wsetBeta[appID]*(respTime-threshold)/3600
