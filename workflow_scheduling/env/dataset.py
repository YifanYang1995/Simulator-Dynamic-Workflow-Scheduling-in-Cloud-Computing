import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
import networkx as nx


vmVCPU = [2, 4, 8, 16, 32, 48]  # EC2 m5; compute_units
request = np.array([1])*0.01 ## arrival rate
datacenter = [ (0, 'East, USA', 0.096)]
vmPrice = {2: 0.096, 4:0.192, 8: 0.384, 16: 0.768, 32: 1.536, 48: 2.304}

from workflow_scheduling.env.buildDAGfromXML import buildGraph
wset = []
wsetTotProcessTime = []
Easy = False
if Easy:
    for i,j in zip(['Montage']*3, ['Montage_3', 'Montage_5', 'Montage_6']):
                    # 'CyberShake_1000', 'Inspiral_1000', 'Montage_1000', 'Sipht_1000'
        dag, wsetProcessTime = buildGraph(f'{i}', parentdir+f'/workflow_scheduling/env/dax/{j}.xml')
        wset.append(dag)
        wsetTotProcessTime.append(wsetProcessTime)
else:
    for i,j in zip(['CyberShake', 'Inspiral', 'Montage', 'Sipht']*3, 
                   ['CyberShake_30', 'Inspiral_30', 'Montage_25', 'Sipht_30', 
                    'CyberShake_50', 'Inspiral_50', 'Montage_50', 'Sipht_60',
                    'CyberShake_100', 'Inspiral_100', 'Montage_100', 'Sipht_100']):
                    # 'CyberShake_1000', 'Inspiral_1000', 'Montage_1000', 'Sipht_1000'
        dag, wsetProcessTime = buildGraph(f'{i}', parentdir+f'/workflow_scheduling/env/dax/{j}.xml')
        wset.append(dag)
        wsetTotProcessTime.append(wsetProcessTime)

from workflow_scheduling.env.get_DAGlongestPath import get_longestPath_nodeWeighted
wsetSlowestT = []
for app in wset:
    wsetSlowestT.append(get_longestPath_nodeWeighted(app))

wsetBeta = []
for app in wset:
    wsetBeta.append(0.24)
