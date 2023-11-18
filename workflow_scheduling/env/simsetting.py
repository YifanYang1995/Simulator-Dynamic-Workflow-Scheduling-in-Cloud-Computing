import numpy as np
import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
import workflow_scheduling.env.dataset as dataset

traffic_density = {"CONSTANT": 1,
                   "LINEAR_INCREASE": [0.1, 0.00025], # [base_density, increased rate]
                   "LINEAR_DECREASE": [1, -0.00025],
                   "PERIODIC":{0: 0.65, 1: 0.55, 2: 0.35, 3: 0.25, 4: 0.2, 5: 0.16, 6: 0.16, 7: 0.2, 8: 0.4, 9: 0.55, 10: 0.65, 11: 0.75, 12: 0.79, 13: 0.79, 14: 0.85, 15: 0.86, 16: 0.85, 17: 0.83, 18: 0.8, 19: 0.79, 20: 0.76, 21: 0.76, 22: 0.69, 23: 0.6}}

traffic_type = ["CONSTANT", "LINEAR_INCREASE", "LINEAR_DECREASE", "PERIODIC"]
traffic_dist = ["EVEN", "UNEVEN"]



class Setting(object):

    # history_len = 2  # the number of history data (response time & utilization) for each variable will be used
    state_info_sample_period = 50  # state will be recoreded every 50 seconds
    dataformat = "csv"
    ac_ob_info_required = False
    epsilon = 0 # 0.1  # used in RL for exploration, epsilon greedy
    is_pkt_trace_record = False
    save_nn_iteration_frequency = 1  # every 20 timesteps

    def __init__(self, args):
        # algorithm = ['RANDOM', 'WEIGHTED_RANDOM', 'RL', 'OPTIMAL']
        self.algo = args["algo"]
        self.traf_type = args["traffic pattern"]
        self.traf_dist = "EVEN"
        self.seed = args["seed"]
        self.arrival_rate_list = args["arrival rate"]
        self.is_allocate_trace_record = args["record"]
        if "REINFORCE learning rate" in args.keys():
            self.REINFORCE_learn_rate = args["REINFORCE learning rate"]
        # self.input_arrival_rate = int(args["arrival rate"])
        # self.history_len = 3  # the number of history data (response time & utilization) for each variable will be used
        if "hist_len" in args.keys():
            self.history_len = args["hist_len"]
        else:
            self.history_len = 2  # the number of history data (response time & utilization) for each variable will be used
        self.timeStep = 1800
        self.respTime_update_interval = 0.5  ## seems useless# (sec) the time interval used in averaging the response time
        self.util_update_interval = 0.5 ## seems useless
        self.arrival_rate_update_interval = self.timeStep
        self.warmupPeriod = 30  # unit: second ## seems useless
        self.envid = args["envid"] ## is num of self._init(self,num)
        self.gamma = args["gamma"]
        self.pkt_trace_sample_freq = 10  ##
        self.WorkflowNum = args["WorkflowNum"]
        self._init(self.envid)
        self.VMpayInterval = 60 * 60  ## 

        # used for gd scheduling inherit from cpp
        self.dlt = 1.0  # self.dlt * avg_resp_time / util
        self.mu = [100.0] * 5
        self.beta = 0.1  # self.beta * synchronization_cost

        # ================================================================================
        # setting 0: used in RL scheduling test

    
    def _init(self, num):
        if num == 0:  # Latin America network
            # cities = [b'Miami, FL', b'Pueblo Viejo, Puerto Rico', b'Caracas, Venezuela', b'Medellin, Colombia', b'Rio de Janeiro, Brazil', b'Sao Paulo, Brazil', b'Buenos Aires, Argentina', b'Santiago, Chile']
            self.maxSimTime = 1800 + self.warmupPeriod  # unit: second
            latency_matrix = np.array([[0.0,25.0,44.0,32.0,112.0,112.0,106.0,140.0],
                                       [25.0,0.0,69.0,57.0,137.0,137.0,131.0,165.0],
                                       [44.0,69.0,0.0,76.0,156.0,156.0,150.0,184.0],
                                       [32.0,57.0,76.0,0.0,144.0,144.0,138.0,172.0],
                                       [112.0,137.0,156.0,144.0,0.0,55.0,61.0,35.0],
                                       [112.0,137.0,156.0,144.0,55.0,0.0,6.0,30.0],
                                       [106.0,131.0,150.0,138.0,61.0,6.0,0.0,34.0],
                                       [140.0,165.0,184.0,172.0,35.0,30.0,34.0,0.0]])
            latency_matrix = np.multiply(latency_matrix, 1e-3)
            latency = np.multiply(latency_matrix, 0.5 * 10)

            candidate = [0, 2, 7]  # 3 ctls 2400 capacities 2x900+1x600
            self.ctlNum = len(candidate)
            self.schNum = latency.shape[0]

            candidate.sort()
            self.sch2ctlLink = latency[:,candidate]  # size: [schNum, ctlNum]

            larger_ctl = [0, 5, 7]
            capacity = np.array([600] * (self.schNum))
            capacity[larger_ctl] = 900
            self.ctlRate = capacity[candidate]

            # self.pktRate = [610*4] * self.schNum
            self.noPktbyCtl = [0] * self.ctlNum
            self.avgCtlRespTime = [0.] * self.ctlNum
            # TODO GET THE POPULATION of each city
            self.population = {"EVEN": {"CONSTANT": [self.input_arrival_rate]*self.schNum, "LINEAR_INCREASE": [390]*self.schNum, "LINEAR_DECREASE": [390]*self.schNum, "PERIODIC": [450]*self.schNum}, "UNEVEN": []}

        elif num == 1:  # Asia Sprint Network k-center 14 nodes
            self.maxSimTime = 1800 + self.warmupPeriod  # unit: second
            latency_matrix = [
                [0.0, 0.035, 0.021, 0.075, 0.048, 0.025, 0.106, 0.074, 0.087, 0.081, 0.04, 0.069, 0.055, 0.052],
                [0.035, 0.0, 0.056, 0.11, 0.078, 0.06, 0.072, 0.041, 0.051, 0.045, 0.005, 0.034, 0.089, 0.025],
                [0.021, 0.056, 0.0, 0.06, 0.03, 0.047, 0.131, 0.096, 0.115, 0.102, 0.061, 0.09, 0.036, 0.074],
                [0.075, 0.11, 0.06, 0.0, 0.03, 0.101, 0.182, 0.151, 0.166, 0.155, 0.115, 0.145, 0.037, 0.127],
                [0.048, 0.078, 0.03, 0.03, 0.0, 0.052, 0.155, 0.121, 0.139, 0.129, 0.088, 0.116, 0.007, 0.1],
                [0.025, 0.06, 0.047, 0.101, 0.052, 0.0, 0.133, 0.045, 0.121, 0.109, 0.066, 0.093, 0.057, 0.078],
                [0.106, 0.072, 0.131, 0.182, 0.155, 0.133, 0.0, 0.044, 0.022, 0.05, 0.076, 0.04, 0.16, 0.095],
                [0.074, 0.041, 0.096, 0.151, 0.121, 0.045, 0.044, 0.0, 0.012, 0.014, 0.045, 0.006, 0.131, 0.064],
                [0.087, 0.051, 0.115, 0.166, 0.139, 0.121, 0.022, 0.012, 0.0, 0.026, 0.063, 0.022, 0.145, 0.081],
                [0.081, 0.045, 0.102, 0.155, 0.129, 0.109, 0.05, 0.014, 0.026, 0.0, 0.049, 0.012, 0.136, 0.07],
                [0.04, 0.005, 0.061, 0.115, 0.088, 0.066, 0.076, 0.045, 0.063, 0.049, 0.0, 0.04, 0.096, 0.031],
                [0.069, 0.034, 0.09, 0.145, 0.116, 0.093, 0.04, 0.006, 0.022, 0.012, 0.04, 0.0, 0.123, 0.06],
                [0.055, 0.089, 0.036, 0.037, 0.007, 0.057, 0.16, 0.131, 0.145, 0.136, 0.096, 0.123, 0.0, 0.107],
                [0.052, 0.025, 0.074, 0.127, 0.1, 0.078, 0.095, 0.064, 0.081, 0.07, 0.031, 0.06, 0.107, 0.0]]
            latency = np.multiply(latency_matrix, (0.5*10))

            candidate = [0, 1, 7, 10]  # 4 controllers ga from TNSM paper (total capacity 3000)
            # candidate = [0, 1, 2, 7, 10, 13]  # 6 controllers: add 2 and 13 (total capacity 4800)
            # candidate = [0, 1, 2, 4, 5, 7, 10, 13]  # 8 controllers: add 4 and 5  (total capacity 6300)
            # candidate = [0, 1, 4, 6, 5, 9, 10, 12]  #ga
            # candidate = [3, 6, 5, 13, 9, 2, 12, 10]  # k-center
            self.ctlNum = len(candidate)
            self.schNum = latency.shape[0]

            candidate.sort()
            self.sch2ctlLink = latency[:,candidate]   # size: [schNum, ctlNum]

            # larger_ctl = [3, 6, 5, 13, 9, 2, 12, 10, 8]
            larger_ctl = [0, 1, 2, 3, 4, 6, 13]
            capacity = np.array([600] * (self.schNum))
            capacity[larger_ctl] = 900
            self.ctlRate = capacity[candidate]

            # self.pktRate = [610*4] * self.schNum
            self.noPktbyCtl = [0] * self.ctlNum
            self.avgCtlRespTime = [0.] * self.ctlNum
            # population: maximum arrival rate for each scheduler
            self.population = {"EVEN": {"CONSTANT": [self.input_arrival_rate] * self.schNum, "LINEAR_INCREASE": [390] * self.schNum,
                                        "LINEAR_DECREASE": [390] * self.schNum, "PERIODIC": [450] * self.schNum},
                               "UNEVEN": []}

        elif num == 2: # ======== Europe network 15 ============
            self.maxSimTime = 1800 + self.warmupPeriod  # unit: second
            latency_matrix = [[0.0, 11.0, 16.0, 18.0, 16.0, 28.0, 22.0, 31.0, 37.0, 37.0, 34.0, 34.0, 39.0, 38.0, 59.0],
                              [11.0, 0.0, 6.0, 7.0, 5.0, 17.0, 11.0, 18.0, 27.0, 23.0, 23.0, 24.0, 28.0, 28.0, 49.0],
                              [16.0, 6.0, 0.0, 15.0, 4.0, 14.0, 6.0, 13.0, 20.0, 19.0, 20.0, 24.0, 26.0, 21.0, 45.0],
                              [18.0, 7.0, 15.0, 0.0, 13.0, 10.0, 20.0, 25.0, 35.0, 32.0, 16.0, 16.0, 21.0, 35.0, 55.0],
                              [16.0, 5.0, 4.0, 13.0, 0.0, 18.0, 10.0, 16.0, 24.0, 22.0, 24.0, 28.0, 29.0, 25.0, 45.0],
                              [28.0, 17.0, 14.0, 10.0, 18.0, 0.0, 8.0, 15.0, 22.0, 21.0, 6.0, 10.0, 11.0, 24.0, 47.0],
                              [22.0, 11.0, 6.0, 20.0, 10.0, 8.0, 0.0, 7.0, 14.0, 13.0, 14.0, 18.0, 20.0, 15.0, 35.0],
                              [31.0, 18.0, 13.0, 25.0, 16.0, 15.0, 7.0, 0.0, 7.0, 7.0, 21.0, 25.0, 26.0, 22.0, 30.0],
                              [37.0, 27.0, 20.0, 35.0, 24.0, 22.0, 14.0, 7.0, 0.0, 7.0, 28.0, 32.0, 33.0, 29.0, 41.0],
                              [37.0, 23.0, 19.0, 32.0, 22.0, 21.0, 13.0, 7.0, 7.0, 0.0, 27.0, 31.0, 32.0, 28.0, 38.0],
                              [34.0, 23.0, 20.0, 16.0, 24.0, 6.0, 14.0, 21.0, 28.0, 27.0, 0.0, 4.0, 17.0, 30.0, 53.0],
                              [34.0, 24.0, 24.0, 16.0, 28.0, 10.0, 18.0, 25.0, 32.0, 31.0, 4.0, 0.0, 21.0, 33.0, 57.0],
                              [39.0, 28.0, 26.0, 21.0, 29.0, 11.0, 20.0, 26.0, 33.0, 32.0, 17.0, 21.0, 0.0, 22.0, 55.0],
                              [38.0, 28.0, 21.0, 35.0, 25.0, 24.0, 15.0, 22.0, 29.0, 28.0, 30.0, 33.0, 22.0, 0.0, 51.0],
                              [59.0, 49.0, 45.0, 55.0, 45.0, 47.0, 35.0, 30.0, 41.0, 38.0, 53.0, 57.0, 55.0, 51.0, 0.0]]
            latency_matrix = np.multiply(latency_matrix, 1e-3)
            latency = np.multiply(latency_matrix, 0.5 * 10)

            self.candidate = [1,3,5]  #[0, 1, 3, 5, 9]
            self.dcNum = len(self.candidate)
            self.usrNum = latency.shape[0]
            self.candidate.sort()
            self.usr2dc = latency[:, self.candidate]

        elif num == 3:  # North America
            self.maxSimTime = 7200 + self.warmupPeriod  # unit: second
            
            latency_matrix = np.array(dataset.latencymap)
            latency_matrix = np.multiply(latency_matrix, 1e-3)
            latency = np.multiply(latency_matrix, 0.5 * 10)

            self.candidate = [26, 30, 31, 33, 35, 39, 43, 44, 46, 47, 48, 53, 62, 69, 76]
            # self.dcInd = candidate
            self.dcNum = len(self.candidate)
            self.usrNum = latency.shape[0]

            self.candidate.sort()
            self.usr2dc = latency[:, self.candidate]  # size: [schNum, ctlNum]
            self.wrfNum = len(dataset.wset)
            self.arrival_rate = {}   # {usr1: {app1: arrivalRate, app2: arrivalRate}}
            for i in range(self.usrNum):
                self.arrival_rate[i] = {}
                for a in range(len(dataset.wset)):
                    self.arrival_rate[i][a] = dataset.request[i]
            # self.population = {"EVEN": {"CONSTANT": dataset.request, "LINEAR_INCREASE": [390]*self.schNum, "LINEAR_DECREASE": [390]*self.schNum, "PERIODIC": [450]*self.schNum}, "UNEVEN": []}

        elif num == 4:  # =========== India network 9 ==============
            self.maxSimTime = 1800 + self.warmupPeriod  # unit: second
            latency_matrix = [[0.0, 35.0, 106.0, 74.0, 87.0, 81.0, 40.0, 69.0, 52.0],
                              [35.0, 0.0, 72.0, 41.0, 51.0, 45.0, 5.0, 34.0, 25.0],
                              [106.0, 72.0, 0.0, 44.0, 22.0, 50.0, 76.0, 40.0, 95.0],
                              [74.0, 41.0, 44.0, 0.0, 12.0, 14.0, 45.0, 6.0, 64.0],
                              [87.0, 51.0, 22.0, 12.0, 0.0, 26.0, 63.0, 22.0, 81.0],
                              [81.0, 45.0, 50.0, 14.0, 26.0, 0.0, 49.0, 12.0, 70.0],
                              [40.0, 5.0, 76.0, 45.0, 63.0, 49.0, 0.0, 40.0, 31.0],
                              [69.0, 34.0, 40.0, 6.0, 22.0, 12.0, 40.0, 0.0, 60.0],
                              [52.0, 25.0, 95.0, 64.0, 81.0, 70.0, 31.0, 60.0, 0.0]]
            latency_matrix = np.multiply(latency_matrix, 1e-3)
            latency = np.multiply(latency_matrix, 0.5 * 10)

            candidate = [0, 2, 8]  # 3 ctls 2400 capacities 2x900+1x600
            # candidate = [0, 2, 8, 5]   # 4 ctls 3000 capacities 2x900+2x600
            self.ctlNum = len(candidate)
            self.schNum = latency.shape[0]

            candidate.sort()
            self.sch2ctlLink = latency[:, candidate]  # size: [schNum, ctlNum]

            larger_ctl = [0, 7, 8, 1]
            capacity = np.array([600] * (self.schNum))
            capacity[larger_ctl] = 900
            self.ctlRate = capacity[candidate]

            # self.pktRate = [610*4] * self.schNum
            self.noPktbyCtl = [0] * self.ctlNum
            self.avgCtlRespTime = [0.] * self.ctlNum
            # TODO GET THE POPULATION of each city
            self.population = {
                "EVEN": {"CONSTANT": [self.input_arrival_rate] * self.schNum, "LINEAR_INCREASE": [390] * self.schNum,
                         "LINEAR_DECREASE": [390] * self.schNum, "PERIODIC": [450] * self.schNum}, "UNEVEN": []}

        elif num == 5:  # small network used for debugging
            self.maxSimTime = 1000 + self.warmupPeriod
            latency_matrix = np.array([[0.005, 0.08, 0.04], [0.08, 0.001, 0.1]])#, [0.04, 0.1, 0.04]])
            latency = np.multiply(latency_matrix, 0.5)
            self.candidate = [0, 1]
            self.dcNum = len(self.candidate)
            self.usrNum = latency.shape[0]
            self.candidate.sort()
            self.usr2dc = latency[:, self.candidate]
            # self.dcInd = candidate

        else:
            latency_matrix = np.array([[0]])
            latency = np.multiply(latency_matrix, 0.5)
            self.candidate = [0] ## useless for Workflow Scheduling
            self.dcNum = len(self.candidate)
            self.usrNum = latency.shape[0]  # = 1
            self.candidate.sort()  ## useless for Workflow Scheduling
            self.usr2dc = latency[:, self.candidate] ## useless for Workflow Scheduling

        self.wrfNum = len(dataset.wset)
        self.arrival_rate = {}  # {usr1: {app1: arrivalRate, app2: arrivalRate}}
        for i in range(self.usrNum):
            self.arrival_rate[i] = {}
            for a in range(len(dataset.wset)):
                self.arrival_rate[i][a] = self.arrival_rate_list[i] # dataset.request[i]  ## 对应于第几个user的arrival rate
        self.dueTimeCoefs = np.ones((self.usrNum, self.wrfNum))/max(dataset.vmVCPU)*self.gamma   # coefficient used to get different due time for each app from each user
        self.totWrfNum = self.WorkflowNum ## total workflow number!!!!!!    ###48 is the max_number of vmVCPU

    def get_individual_arrival_rate(self, time, usrcenter, app):
        if self.traf_type == "CONSTANT":
            den = traffic_density[self.traf_type]
        else:
            if re.match(r"^LINEAR.", self.traf_type):
                den = traffic_density[self.traf_type][0] + traffic_density[self.traf_type][-1]*time
            elif self.traf_type == "PERIODIC":
                hr = int(time/75)%24  # we consider two periods in one hour
                den = traffic_density[self.traf_type][hr]
            else:
                print("cannot get the arrival rate!!!!!!!!")
                den = None
        return den*self.arrival_rate[usrcenter][app]


