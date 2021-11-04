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
def get_genome(tree):
    return [float(node.name[1]) for node in PreOrderIter(tree) if node.is_leaf]

def set_genome(tree, genome):
    genome_tree = get_genome(tree)
    nodes = [node for node in PreOrderIter(tree) if node.is_leaf]
    for i in range(len(genome_tree)):
        nodes[i].name[1] = genome[i]
    return tree


def create_population(n, length):
    """ Create initial population """
    population = []
    for i in range(n):
        individu = [random.random() for _ in range(length)]
        population.append(individu)
    return population


def evaluate(tree, individu, df, operandes, operations, ope_dic):
    """ Evaluate a genome """
    df_ = df.copy()
    set_genome(tree, individu)
    a = executeTree(tree, operandes, operations, df_, ope_dic)
    
    if a is None:
        return float('inf')
    else:
        df['eval'] = a
    
    mse = (np.square(df['p'].to_numpy() - df['eval'].to_numpy())).mean(axis=0)       

    if np.isnan(mse):
        return float('inf')

    return mse


def mutate_pop(pop):
    """Modify individual"""
    for p in range(len(pop)):
        prop = 0
        if random.random() < 0.2:
            prop = 0.2
        elif random.random() < 0.3:
            prop = 0.1
        mask = [random.random()<prop for _ in pop[p]]
        for i,m in enumerate(mask):
            if m:
                coeff = random.random()*(1.5-0.5)+0.5
                pop[p][i] = pop[p][i]*coeff

        if random.random() < 0.05:
            ind = random.randint(0, len(pop[p])-1)
            pop[p][ind] = random.random()*3*random.choice([-1,1])
            
    return pop


def evaluate_population(tree, population, df_dic, operandes, operations, ope_dic, height_matter=True):
    """ Grade the population. Return a list of tuple (individual, fitness) sorted from most graded to less graded. """
    graded_individual = []
    for k,individual in enumerate(population):
        s = evaluate(tree, individual, df_dic, operandes, operations, ope_dic)
        graded_individual.append((individual, s))

    return sorted(graded_individual, key=lambda x: x[1])


def get_child(p1, p2):
    mask = [random.random()>0.5 for _ in p1]
    child = [p1[i] if mask[i] else p2[i] for i in range(len(mask))]
    return child


def evolve_population(tree, population, df_dic, operandes, operations, ope_dic,
                      graded_prop=0.2, non_graded_prop=0.05, max_depth=None):
    """ Make the given population evolving to his next generation. """

    # Evaluate population
    sorted_pop_fit = evaluate_population(tree, population, df_dic, operandes, operations, ope_dic)
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
    parents = mutate_pop(parents)

    # Crossover parents to create children
    parents_len = len(parents)
    desired_len = n - parents_len
    children = []
    while len(children) < desired_len:
        p1 = random.choice(parents)
        p2 = random.choice(parents)

        child = get_child(p1, p2)

        children.append(child)

    # The next generation is ready
    parents.extend(children)

    return parents, avg_fitness, bst_fitness, best_ind



def genetic_finetuning(tree, df_func, ope_dic, operandes, operations,
                    population_size=100, n_generations=10,
                    graded_prop=0.2, non_graded_prop=0.05,
                    verbose=True, checkpoint=True):
    """ Optimizes hyperparameters with a genetic algorithm """

    genome = get_genome(tree)
    

    population = create_population(population_size, len(genome))

    for generation in range(n_generations):
        t0 = time.time()
        
        height_matter=False
        if generation > 0:
            height_matter=True
            
        population, avg_fitness, bst_fitness, best_genome = evolve_population(tree, population, df_func, operandes, operations, ope_dic,
                                                              graded_prop=graded_prop,
                                                              non_graded_prop=non_graded_prop)
        set_genome(tree, best_genome)
        if verbose:
            #show(tree)
            print(f'Population {generation + 1} : Avg fitness = {avg_fitness} Best fitness = {bst_fitness}')
            #print('Chrono : ',time.time()-t0)
        if checkpoint:
            pickle.dump(tree, open('best_tree_finetuned.p', 'wb'))
            pickle.dump(population, open('last_population.p', 'wb'))

    if verbose:
        print('\nEvaluation finale')
    score = evaluate_population(tree, population, df_func, operandes, operations, ope_dic)
    if verbose:
        print('Best score : ', score[0][1])

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
    "s":lambda a,b : a.astype(float)**0.5,
    }

    primes = np.array(get_primes(10000))

    
    tree = pickle.load(open('best_tree_1.p','rb'))

    df_func = pd.DataFrame({'i' : [i+1 for i in range(len(primes))], 'p' : primes, '1' : [1 for i in range(len(primes))]})
    operandes = ['i','1']
    operations=['+','-','*','/','**','l','s']

    print('mse : ',evaluate(tree, get_genome(tree), df_func, operandes, operations, ope_dic))

    best_tree = genetic_finetuning(tree, df_func, ope_dic, operandes, operations,
                                   population_size=10000, n_generations=100,
                                   graded_prop=0.2, non_graded_prop=0.05,
                                   verbose=True, checkpoint=True)

    show(best_tree)
    pickle.dump(best_tree, open('best_tree_finetuned.p','wb'))

    df['eval'] = executeTree(best_tree, operandes, operations, df, ope_dic)
    df['p'].plot(label='primes')
    df['eval'].plot(label='tree')
    plt.legend()
    plt.show()
        

