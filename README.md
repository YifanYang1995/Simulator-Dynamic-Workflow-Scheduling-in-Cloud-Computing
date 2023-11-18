# Simulator - Dynamic Workflow Scheduling in Cloud Computing 

Dynamic workflow scheduling (DWS) plays a crucial role in cloud computing. This simulator aims to make intelligent decisions in terms of workflow allocation and VM provisioning by **VM selection rules (VMSRs)**. The objective is to minimize the total costs associated with VM rental fees and deadline violation penalties. Certainly, the objective function can also be customized according to needs.  
## Getting started


| **File Name**             | **Description**                                              |
| ------------------------- | ------------------------------------------------------------ |
| MAIN-OneTimeSimulation.py | Evaluate a problem instance (with 30 workflows) using a randomly generated VM selection rule (in form of a GP tree) |
| MAIN-SGP.py               | Using single-tree GPHH (SGP) to solve dynamic workflow scheduling problems, and individuals in each generation are evaluated on multiple problem instances.  <br />Training and test phases integrated into this file |
