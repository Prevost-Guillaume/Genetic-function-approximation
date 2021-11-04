# Genetic-function-approximation

The goal of this project is to find a way to approximate any function.  
The main difficulty is to find a general way to represent any formula, which can be optimized by genetic algorithm.  
We have chosen to use binary expression trees for this purpose.   

### Binary expression trees
The advantages of binary trees are that it is easy to manage the operational priority, as well as the implementation in genetic algorithm.  
A binary expression tree is a tree where the nodes are operations, and the leaves are values (operands).  
*Here is an example of a binary expression tree:*  
<img src=https://github.com/Prevost-Guillaume/Genetic-function-approximation/blob/main/images/expression_tree.png width="600" height="400">  
This tree represents the formula __((5+z)/(-8))\*4^2__  
  
The goal of our project is therefore to find the tree that will best fit the desired function.  
For example, we tried to find the formula that was closest to the function that associates the nth prime number to n. Such a formula obviously does not exist, but we will see if it is possible to get close to it.  
It is however good to note that in our case, the leaves of the tree will not be numbers but columns of a pandas dataframe. We have two columns: 
* A column named "i" which contains the numbers of our search interval, this is the equivalent of the variable. In our example, it goes from 1 to 1230 by steps of 1.
* A column named "1" which contains the value 1, and allows to add constants in the formula.  

Moreover, the sheets also contain a coefficient by which the operands ("i" and "1") will be multiplied immediately. A second step of finetuning these coefficients will intervene, once the tree is found.  

### Genetic algorithm
We use a genetic algorithm to optimize the tree. The main challenge here was crossover. We finally chose to mix two trees by replacing the node of one tree by the node of another tree.  
For the mutation, we implemented different types of mutations which occur at different frequencies:  
* The change of the operation of a node
* The change of the operand of a leaf
* The change of the multiplier coefficient of a sheet
* The deletion of a node of the tree
* Replacing a node of the tree by a new tree

The fitness function is the mean squared error between the function found by the tree and the true function.  
As regularization, we decided to limit the depth of trees (to 9 here).  

  
### Results
We get the following formula :  
  __ln(1.14\*i)\*(((0.16\*i+0.78\*i)+(0.12/0.07))+0.19\*i)__
  __= ln(1.14\*x) * (1.12\*x + 0.57)__

<img src=https://github.com/Prevost-Guillaume/Genetic-function-approximation/blob/main/images/f(x).png width="600" height="400">  
<img src=https://github.com/Prevost-Guillaume/Genetic-function-approximation/blob/main/images/approx.png width="600" height="400">  

mse : 207.98  
mae : 10.62  



We also explored the possibility of using such a tree for feature processing in a data science problem for example. 
The numerical columns of the dataframe become the operands ("i" and "1" here).
We use the score of the ML model (f1, mse, etc.) trained on the column created by the tree (or on the df + this column) as the fitness function of an individual, and the genetic algorithm will look for the combination of columns leading to the best possible score.  

### Further work
* Optimize the tree architecture and the multiplicative coefficients in the same algorithm.
* Try a larger max depth to approximate more complex functions.
* Explore other fields of application of this algorithm.
