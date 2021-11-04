from anytree import *
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import pickle
import numpy as np
import time
import math

from tree_functions import *

import warnings
warnings.filterwarnings("ignore")

#################################################################################################################################
#                                                                                                                               #
#                                                       GENETIC FUNCTIONS                                                       #
#                                                                                                                               #
#################################################################################################################################
def create_population(n, operandes, operations, max_deepness=5):
    """ Create initial population """
    population = []
    for i in range(n):
        individu = createRandomTree(operandes, operations=operations, deepness=random.randint(1,max_deepness),
                                    density=max(0.4,random.random()), max_shifting=20)
        population.append(individu)
    return population


def evaluate(individu, df, operandes, operations, ope_dic, ind=0, height_matter=True):
    """ Evaluate a genome """
    df_ = df.copy()
    a = executeTree(individu, operandes, operations, df_, ope_dic)
    if a is None:
        return float('inf')
    else:
        df['eval'] = a
    
    mse = (np.square(df['p'].to_numpy() - df['eval'].to_numpy())).mean(axis=0)       

    if np.isnan(mse):
        return float('inf')

    if height_matter:
        score = mse+individu.height
    else:
        score = mse
    
    return score


def mutate_pop(pop, operandes, operations, max_depth=None):
    """Modify individual"""
    for i in range(len(pop)):
        if random.random() < 0.05:
            pop[i] = mutate(pop[i], operandes, operations, subTree_ratio=1, max_depth=max_depth)
        if random.random() < 0.2:
            pop[i] = mutate(pop[i], operandes, operations, product_ratio=0.2, max_depth=max_depth)
        if random.random() < 0.07:
            pop[i] = mutate(pop[i], operandes, operations, removeTree_ratio=1, max_depth=max_depth)
        if random.random() < 0.1:
            pop[i] = mutate(pop[i], operandes, operations, operande_ratio=0.3, max_depth=max_depth)
        if random.random() < 0.1:
            pop[i] = mutate(pop[i], operandes, operations, operation_ratio=0.3, max_depth=max_depth)
    return pop


def evaluate_population(population, df_dic, operandes, operations, ope_dic, height_matter=True):
    """ Grade the population. Return a list of tuple (individual, fitness) sorted from most graded to less graded. """
    graded_individual = []
    for k,individual in enumerate(population):
        if not is_valid(individual):
            print('Tree invalide dans evaluate_population')
            s = 0
        else:
            s = evaluate(individual, df_dic, operandes, operations, ope_dic, ind=k+1, height_matter=height_matter)
        graded_individual.append((individual, s))

    return sorted(graded_individual, key=lambda x: x[1])


def evolve_population(population, df_dic, operandes, operations, ope_dic,
                      graded_prop=0.2, non_graded_prop=0.05, max_depth=None, height_matter=True):
    """ Make the given population evolving to his next generation. """

    # Evaluate population
    sorted_pop_fit = evaluate_population(population, df_dic, operandes, operations, ope_dic, height_matter=height_matter)
    n = len(sorted_pop_fit)
    best_ind = sorted_pop_fit[0][0]
    bst_fitness = sorted_pop_fit[0][1]
    avg_fitness = sum([i[1] for i in sorted_pop_fit]) / n

    # Select parents
    sorted_pop = [i[0] for i in sorted_pop_fit]
    parents = sorted_pop[:int(graded_prop * n)]

    # Randomly add other individuals to promote genetic diversity
    for individual in sorted_pop[int(graded_prop * n):]:
        if random.random() < non_graded_prop:
            parents.append(individual)

    # Mutate some individuals
    parents = mutate_pop(parents, operandes, operations, max_depth=max_depth)

    # Crossover parents to create children
    parents_len = len(parents)
    desired_len = n - parents_len
    children = []
    while len(children) < desired_len:
        father = random.choice(parents)
        mother = random.choice(parents)

        child = mix(father, mother, max_depth=5)

        if is_valid(child):
            simplify(child, ope_dic)
            children.append(child)

    # The next generation is ready
    parents.extend(children)

    return parents, avg_fitness, bst_fitness, best_ind


def geneticAlgo(df_dic, operandes, operations, ope_dic,
                    population_size=100, n_generations=10,
                    graded_prop=0.2, non_graded_prop=0.05,
                    verbose=True, checkpoint=True, max_depth=None):
    """ Optimizes hyperparameters with a genetic algorithm """

    population = create_population(population_size, operandes, operations)

    for generation in range(n_generations):
        t0 = time.time()
        
        height_matter=False
        if generation > 0:
            height_matter=True
            
        population, avg_fitness, bst_fitness, best_ind = evolve_population(population, df_dic, operandes, operations, ope_dic,
                                                              graded_prop=graded_prop,
                                                              non_graded_prop=non_graded_prop,
                                                              max_depth=max_depth,
                                                              height_matter=height_matter)

        if verbose:
            show(best_ind)
            print(f'Population {generation + 1} : Avg fitness = {avg_fitness} Best fitness = {bst_fitness}')
            print('Chrono : ',time.time()-t0)
            print(f'Best ind : ')
            print("(height : ",best_ind.height,")",sep="")
        if checkpoint:
            f = open('best_tree.p', 'wb')
            pickle.dump(best_ind, f)
            f.close()
            pickle.dump(population, open('last_population.p', 'wb'))

    if verbose:
        print('\nEvaluation finale')
    score = evaluate_population(population, df_dic, operandes, operations, ope_dic)
    if verbose:
        print('Best score : ', score[0][1])
        print('Best genome : ', score[0][0])

    return score[0][0]



#################################################################################################################################
#                                                                                                                               #
#                                                           FUNCTION                                                            #
#                                                                                                                               #
#################################################################################################################################
def get_primes(n):
    primes = [2]
    for i in range(3,n):
        a = True
        for p in primes:
            if i%p == 0:
                a=False
                break
        if a:
            primes.append(i)
    return primes




#################################################################################################################################
#                                                                                                                               #
#                                                               MAIN                                                            #
#                                                                                                                               #
#################################################################################################################################
    
if __name__ == '__main__':

    ope_dic = {
    "+" : lambda a, b : a + b,
    "-" : lambda a, b : a - b,
    "*" : lambda a, b : a * b,
    "/" : lambda a, b : a / b,
    "**" : lambda a,b : a.astype(float)**(np.abs(b.astype(float))),
    "l":lambda a,b : np.log(abs(a.astype(float))),
    }

    print('calculating primes')
    primes = np.array(get_primes(10000))
    print('plotting primes')
    plt.plot(primes)
    plt.show()
    
    

    df = pd.DataFrame({'i' : [i+1 for i in range(len(primes))], 'p' : primes, '1' : [1 for i in range(len(primes))]})
    operandes = ['i','1']
    operations=['+','-','*','/','l']

    max_depth = 9
    best_tree = geneticAlgo(df, operandes, operations, ope_dic,
                    population_size=1000, n_generations=200,
                    graded_prop=0.2, non_graded_prop=0.05,
                    verbose=True, checkpoint=True, max_depth=max_depth)

    show(best_tree)
    pickle.dump(best_tree, open('best_tree.p','wb'))

    df['eval'] = executeTree(best_tree, operandes, operations, df, ope_dic)
    df['p'].plot(label='primes')
    df['eval'].plot(label='tree')
    plt.legend()
    plt.show()
        

