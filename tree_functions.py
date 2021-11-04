from anytree import *
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy


def createRandomTree(operandes, operations=['+','-','*','/'], deepness=3, density=0.7, max_shifting=50):
    """Create first individual"""
    tree = Node(random.choice(operations))
    tree = addTree(tree, tree, deepness, operandes, operations=operations, treshold=1-density, max_shifting=max_shifting)
    return tree
    
def addTree(tree, parent, n, operandes, operations=['+','-','*','/'], treshold=0, max_shifting=10):
    if n==0:
        return tree
    elif n==1 or random.random()<treshold:
        a = Node([random.choice(operandes), random.random()], parent=parent)
        b = Node([random.choice(operandes), random.random()], parent=parent)
        addTree(tree, a, 0, operandes, operations=operations, treshold=treshold, max_shifting=max_shifting)
        return addTree(tree, b, 0, operandes, operations=operations, treshold=treshold, max_shifting=max_shifting)
    else:
        a = Node(random.choice(operations),parent=parent)
        b = Node(random.choice(operations),parent=parent)
        addTree(tree, a, n-1, operandes, operations=operations, treshold=treshold, max_shifting=max_shifting)
        return addTree(tree, b, n-1, operandes, operations=operations, treshold=treshold, max_shifting=max_shifting)


def is_valid(tree):
    leaves = [node.name for node in PostOrderIter(tree) if node.is_leaf]
    for l in leaves:
        if type(l) is str:
            return False
    return True

def executeTree(tree, operandes, operations, df, dic_operations):
    """Execute les opérations de l'arbre"""
    # Check validity        
    tree_nodes = [node.name for node in PostOrderIter(tree)]
    tree_nodes_df = [df[i[0]] if i not in operations else i for i in tree_nodes]
    tree_nodes_flatten = [i[0] if i not in operations else i for i in tree_nodes]

    counter = 0
    while len(tree_nodes) > 1:
        counter += 1
        if counter >= 100:
            return None
        new_tree_nodes = []
        new_tree_nodes_df = []
        new_tree_nodes_flatten = []
        i = 0
        while i < len(tree_nodes):
            if tree_nodes_flatten[i] not in operations and tree_nodes_flatten[i+1] not in operations and tree_nodes_flatten[i+2] in operations:
                # Name of columns
                a = tree_nodes_flatten[i]
                b = tree_nodes_flatten[i+1]
                # Product values
                s_a = tree_nodes[i][1]
                s_b = tree_nodes[i+1][1]
                # Pandas series of two operandes
                adf = tree_nodes_df[i]
                bdf = tree_nodes_df[i+1]
                # Operation
                o = tree_nodes[i+2]

                new_tree_nodes_df.append(dic_operations[o](adf*s_a,bdf*s_b))
                new_tree_nodes.append([a+o+b, 1])
                new_tree_nodes_flatten.append(a+o+b)
                i += 3
            else:
                new_tree_nodes.append(tree_nodes[i])
                new_tree_nodes_df.append(tree_nodes_df[i])
                new_tree_nodes_flatten.append(tree_nodes_flatten[i])
                i += 1
        tree_nodes = new_tree_nodes
        tree_nodes_df = new_tree_nodes_df
        tree_nodes_flatten = new_tree_nodes_flatten
    return new_tree_nodes_df[0]



    

def mix(tree1, tree2, max_depth=None):
    """Create a child from parents"""
    father = copy.deepcopy(tree1)
    mother = copy.deepcopy(tree2)
    child = copy.deepcopy(mother)

    tree_nodes_f = [node for node in PreOrderIter(father) if not node.is_leaf]
    tree_nodes_m = [node for node in PreOrderIter(mother, maxlevel=mother.height) if not node.is_leaf]
    if len(tree_nodes_m) == 0:
        return tree1 
    
    if max_depth == None:
        node_f = random.choice(tree_nodes_f)
        node_m = random.choice(tree_nodes_m)
        node_m.children = [node_m.children[0], node_f]
    else:
        node_f = random.choice(tree_nodes_f)
        node_m = random.choice(tree_nodes_m)
        node_m.children = [node_m.children[0], node_f]
        i = 0
        while node_m.root.height > max_depth:
            node_f = random.choice(tree_nodes_f)
            node_m = random.choice(tree_nodes_m)
            node_m.children = [node_m.children[0], node_f]
            i += 1
            if i%10 == 0:
                print(i)
                

##    node_f = random.choice(tree_nodes_f)
##    node_m = random.choice(tree_nodes_m)
##    if max_depth != None:
##        while node_m.depth + node_f.height > max_depth:
##            node_m = random.choice(tree_nodes_m)
##
##    node_m.children = [node_m.children[0], node_f]
        
    return mother




def mutate(tree, operandes, operations, product_ratio=0., operande_ratio=0., operation_ratio=0., subTree_ratio=0., removeTree_ratio=0., max_depth=None):
    """mutate tree"""
    if not is_valid(tree):
        return createRandomTree(operandes, operations=operations, deepness=random.randint(1,4), density=0.4+random.random(), max_shifting=30)

    if product_ratio > 0:
        leaves = tree.leaves
        mask = [random.random()<product_ratio for _ in leaves]
        l = [l for i,l in enumerate(leaves) if mask[i]]
        for l in l:
            try:
                coeff = random.random()*(2-0.5)+0.5 # random coeff in range [0.5, 2]
                l.name = [l.name[0], l.name[1]*coeff]
            except:
                pass    
    
    if operande_ratio > 0:
        leaves = tree.leaves
        mask = [random.random()<operande_ratio for _ in leaves]
        l = [l for i,l in enumerate(leaves) if mask[i]]
        for l in l:
            l.name[0] = random.choice(operandes)
    
    if operation_ratio > 0:
        nodes = [node for node in PreOrderIter(tree, maxlevel=tree.height)]
        nodes = [node for node in PreOrderIter(tree)]
        mask = [random.random()<operation_ratio for _ in nodes]
        l = [l for i,l in enumerate(nodes) if mask[i]]
        for l in l:
            l.name = random.choice(operations)
    
    if subTree_ratio > 0:
        nodes = [node for node in PreOrderIter(tree)]
        nodes = [i for i in nodes if not i.is_leaf]
        if len(nodes) > 0:
            node_1 = random.choice(nodes)
            if max_depth != None:
                m = max_depth-node_1.depth
            else:
                m = 3
            if m > 1:
                new_tree = createRandomTree(operandes, operations=operations, deepness=random.randint(1,m), density=random.random())
                node_1.children = [node_1.children[0], new_tree]
    
    if removeTree_ratio > 0:
        nodes = [node for node in PreOrderIter(tree, maxlevel=tree.height-1)]
        nodes = [i for i in nodes if not i.is_leaf]
        new_leaf = Node([random.choice(operandes), random.random()*4 - 2])
        
        if len(nodes) > 0:
            node_1 = random.choice(nodes)
            node_1.children = [node_1.children[0], new_leaf]


    if not is_valid(tree):
        return createRandomTree(operandes, operations=operations, deepness=random.randint(1,4), density=0.4+random.random())

    return tree



def show(tree):
    """print tree in a beautiful way"""
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))
    return
    




