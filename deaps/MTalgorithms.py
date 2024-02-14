#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random, operator
from tkinter import E

from deap import tools
import numpy as np

# self-define for single-tree GPHH
def SGP(population, toolbox, cxpb, mutpb, ngen, evalNum, stats=None,
             halloffame=None, verbose=__debug__):  
    """This algorithm reproduce the simplest evolutionary algorithm as SGP"""

    # Revised Genetic Operators：first allocate the index and then execute the operators 
    def varFW(population, toolbox, cxpb, mutpb): 
        offspring = [toolbox.clone(ind) for ind in population]  # input population already be processed by the ternament selection

        # allocate indexs for each genetic operator
        crossover_index = []
        mutation_index = []
        reprod_index=[]

        for i in range(len(offspring)):
            r = random.random()
            if r < mutpb:
                mutation_index.append(i)
            elif r < cxpb + mutpb :
                crossover_index.append(i)
            else:
                reprod_index.append(i)     

        # Reproduction
        for i in range(len(reprod_index)):                   
            offspring[reprod_index[i]], = [toolbox.clone(offspring[reprod_index[i]])]
            del offspring[reprod_index[i]].fitness.values            

        # Crossover
        for i in range(1, len(crossover_index), 2):
            offspring[crossover_index[i-1]], offspring[crossover_index[i]] = toolbox.mate(offspring[crossover_index[i-1]], offspring[crossover_index[i]])
            del offspring[crossover_index[i-1]].fitness.values, offspring[crossover_index[i]].fitness.values

        # Mutation
        for i in range(len(mutation_index)): 
                  
            offspring[mutation_index[i]], = toolbox.mutate(offspring[mutation_index[i]]) 
            del offspring[mutation_index[i]].fitness.values    

        return offspring

    rule_Log = []               # record the elites found in each generation
    bestOne_Log=[]              # record the best one individual in each generation

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = population
    for ind in invalid_ind:
        ind.genID = 0           # the first index of problem instances in the training set
        ind.output1 = []        # cost
        ind.output2 = []        # penalty
        ind.output3 = []        # VM_execHour
        ind.output4 = []        # VM_totHour
        ind.output5 = []        # missDeadlineNum
    # separately calculate fitness on each training instance
    for i in range(evalNum): 
        for ind in invalid_ind:
            ind.eval_index = i  # the second index of problem instances in the training set
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)             
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.output1.append(fit[1])
            ind.output2.append(fit[2])
            ind.output3.append(fit[3])
            ind.output4.append(fit[4])
            ind.output5.append(fit[5])  
    # add the real fitness.values to each individual
    for ind in invalid_ind:
        ind.fitness.values = np.mean(ind.output1)+np.mean(ind.output2),

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, evals=operator.attrgetter("height")(halloffame[0]), **record) 
    if verbose:
        print(logbook.stream)


    for gen in range(1, ngen + 1):

        # Find Elites   
        elites_in_population = toolbox.selectElitism(population)
        elites = [toolbox.clone(ind) for ind in elites_in_population]
        rule_Log.append([toolbox.clone(ind) for ind in elites]) # 
        bestOne_Log.append(toolbox.clone(elites[0]))

        # Parent Selection, Genetic Operators
        offspring = toolbox.select(population, len(population)-len(elites))   
        offspring = varFW(offspring, toolbox, cxpb, mutpb)
        offspring[0:0] = elites     # add elites 

        # Evaluate the offspring
        invalid_ind = offspring
        for ind in invalid_ind:
            ind.genID = gen         # the first index of problem instances in the training set
            ind.output1 = []        # cost
            ind.output2 = []        # penalty
            ind.output3 = []        # VM_execHour
            ind.output4 = []        # VM_totHour
            ind.output5 = []        # missDeadlineNum
        # separately calculate fitness on each training instance
        for i in range(evalNum): 
            for ind in invalid_ind:
                ind.eval_index = i  # the second index of problem instances in the training set
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)             
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.output1.append(fit[1])
                ind.output2.append(fit[2])
                ind.output3.append(fit[3])
                ind.output4.append(fit[4])
                ind.output5.append(fit[5])  
        # add the real fitness.values to each individual
        for ind in invalid_ind:
            ind.fitness.values = np.mean(ind.output1)+np.mean(ind.output2),

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring 

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, evals=operator.attrgetter("height")(halloffame[0]), **record) #, nevals=len(invalid_ind)
        if verbose:
            print(logbook.stream)

    # Elite in the last generation
    elites_in_population = toolbox.selectElitism(population)
    elites = [toolbox.clone(ind) for ind in elites_in_population]
    rule_Log.append(elites)    
    bestOne_Log.append(elites[0])

    return bestOne_Log, logbook, rule_Log

