## Table of Contents

- [Table of Contents](#table-of-contents)
- [Lab 1: Set Cover Problem](#lab-1-set-cover-problem)
  - [Algorithms Implemented](#algorithms-implemented)
  - [Experimental Setup](#experimental-setup)
    - [Instances](#instances)
  - [Results and Discussion](#results-and-discussion)
  - [Results](#results)
  - [Conclusion](#conclusion)
- [Lab 2: Traveling Salesman Problem](#lab-2-traveling-salesman-problem)
  - [Algorithms implemented](#algorithms-implemented-1)
    - [Greedy Algorithm](#greedy-algorithm)
    - [Genetic Algorithm](#genetic-algorithm)
      - [Fitness Function](#fitness-function)
      - [Parent Selection](#parent-selection)
      - [Crossover](#crossover)
      - [Mutation](#mutation)
      - [Other tweaks](#other-tweaks)
      - [Parameters](#parameters)
  - [Results](#results-1)
  - [Discussion](#discussion)
- [Lab 3F: Short Path between two cities](#lab-3f-short-path-between-two-cities)
- [Lab 3: n-Puzzle description](#lab-3-n-puzzle-description)
  - [Algorithms implemented](#algorithms-implemented-2)
    - [Informed Search](#informed-search)
      - [Best-First Search](#best-first-search)
      - [A\* Search](#a-search)
    - [Heuristic Function](#heuristic-function)
      - [Manhattan Distance](#manhattan-distance)
  - [Results](#results-2)
  - [Discussion](#discussion-1)
- [Project: Symbolic Regression using Genetic Programming](#project-symbolic-regression-using-genetic-programming)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
  - [Methodology](#methodology)
    - [Technologies Used](#technologies-used)
    - [Approach](#approach)
    - [Implementation Details](#implementation-details)
      - [**Evolutionary Strategy and Selection Mechanisms**](#evolutionary-strategy-and-selection-mechanisms)
        - [**Elitism Mechanism**](#elitism-mechanism)
        - [**Stopping Criteria**](#stopping-criteria)
        - [**Parameter Selection and Justification**](#parameter-selection-and-justification)
        - [**Evolutionary Process Recap**](#evolutionary-process-recap)
  - [Results](#results-3)


## Lab 1: Set Cover Problem 

The **Set Cover Problem** involves a universe of elements and a collection of sets, each containing some of these elements. The goal is to select the fewest sets (or minimize the total cost of selected sets) such that every element in the universe is covered by at least one selected set. This problem is NP-hard, meaning it's computationally difficult to find an optimal solution for large instances.

### Algorithms Implemented
I tried three different algorithms to solve the Set Cover problem:

1. **Hill Climbing** with multiple mutations: In this version of hill climbing, the algorithm is initialized with a random solution and then multiple mutations are applied to the current solution during each iteration. The strength of these mutations (the number of mutations) decreases as iterations progress. This helps balance exploration and exploitation.
2. **Simulated Annealing**: This algorithm allows the search to escape local optima by accepting worse solutions with a probability that decreases as the algorithm progresses. The probability of accepting suboptimal solutions is controlled by a temperature parameter that gradually cools down.
3. **Tabu Search**: This approach avoids revisiting recently explored solutions by maintaining a tabu list, which helps to escape local optima and diversify the search space.

### Experimental Setup
The algorithms were tested on a set of six instances of the Set Cover problem, with the following characteristics:

#### Instances
|# INSTANCE| Universe Size | Number of Sets | Density|
|----------|--------------|---------------|--------|
| 1 | 100 | 10 | 0.2 |
| 2 | 1,000 | 100 | 0.2 |
| 3 | 10,000 | 1,000 | 0.2 |
| 4 | 100,000 | 10,000 | 0.1 |
| 5 | 100,000 | 10,000 | 0.2 |
| 6 | 100,000 | 10,000 | 0.3 |

### Results and Discussion
Each algorithm was run on all problem instances, and the performance was evaluated in terms of solution cost, runtime, and the number of function calls required to reach a solution. The results indicate that:

- **Hill Climbing** (HC) produced competitive results with relatively low computational cost, making it a good trade-off between efficiency and accuracy.
- **Simulated Annealing** (SA) is the fastest algorithm but does not perform as well as Tabu search or Hill Climbing. 
- **Tabu Search** (TS)  performs better than the other algorithms but requires more time to converge (~140x more time than HC, ~350x more calls to the fitness function than SA and 16x more calls than HC).

So HC seems to be the best trade-off between performance and time.

### Results
Hill Climbing used for all the instances in the following experiments.

| # INSTANCE | Best fitness | # OF CALLS |
|----------|----------|----------|
| 1 | -292.92 | 0 |
| 2 | -7689.43 | 19   |
| 3 | -741479.28| 292  |
| 4 | -112751635.59 |  435 |
| 5 |  -238795309.18 |  489 |
| 6 | -370743077.11 | 474  |


### Conclusion
After extensive testing, **Hill Climbing** was selected as the best trade-off in terms of solution cost, runtime performance, and the number of function calls. 

----

## Lab 2: Traveling Salesman Problem
The Traveling Salesman Problem (TSP) is an NP-hard problem in combinatorial optimization. Given a list of cities and the distances between each pair of cities, the task is to find the shortest possible tour that visits each city exactly once and returns to the original city.

### Algorithms implemented

The laboratory task is to implement two algorithm: one fast but more approximated and one slower but more precise. The algorithms implemented are the Greedy Algorithm and the Genetic Algorithm.

#### Greedy Algorithm
The Greedy Algorithm is a simple algorithm that starts from a random city and at each step selects the nearest city that has not been visited yet. The algorithm stops when all the cities have been visited.

#### Genetic Algorithm
The Genetic Algorithm is a metaheuristic inspired by the process of natural selection. 
The Genetic Algorithm initializes a population of random TSP tour solutions. Each solution is evaluated based on its fitness, which is inversely related to the total tour distance. Parents for the next generation are selected using methods like Roulette Wheel, Tournament, or Rank Selection, which balance fitness and randomness. Selected parents then undergo crossover and mutation to generate new offspring, repeating the process until termination. 

##### Fitness Function
The fitness function is used to evaluate the quality of the individuals. In this laboratory task, the fitness function is based on the total distance of the path. The fitness of an individual is calculated as the inverse of the total distance of the path, so that the higher the fitness, the better the individual.

##### Parent Selection
The parent selection process is used to select the individuals that will be used to generate the next generation. There are several methods that can be used for parent selection, such as roulette wheel selection, tournament selection, and rank selection. All these methods have been tried in this laboratory task to find the best one.

1. The roulette wheel selection method is based on the fitness of the individuals. The probability of an individual being selected is proportional to its fitness.

2. The tournament selection method is based on selecting a random subset of individuals and then selecting the best individual from that subset.

3. The rank selection method is based on ranking the individuals based on their fitness and then selecting the best individuals.


After some tests, the best parent selection method has been found to be the *tournament* method.

##### Crossover
The crossover process is used to generate new individuals by combining the genetic information of the parents. There are several methods that can be used for tsp, such as Edge Recombination and Inver Over. All these methods have been tried in this laboratory task to find the best one.

1. The edge recombination crossover method is based on the edges of the parents. The method starts by selecting a random edge from one of the parents and then selects the next edge based on the edges that are adjacent to the current edge.

2. The inver over crossover method is based on taking a random element from one parent and then selecting an edge of this element from the other parent and then preserve this edge in the offspring, swapping the other elements between them as appear in the first parent, in the offspring.

After some tests, the best crossover method has been found to be the *edge recombination* method.

##### Mutation
Mutation is used to introduce diversity in the population and prevent premature convergence. There are several methods that can be used for tsp, such as Inverse, and Scramble. All these methods have been tried in this laboratory task to find the best one.

1. The inversion mutation method is based on selecting a random subset of the path and then inverting the order of the cities in that subset.

2. The scramble mutation method is based on selecting a random subset of the path and then shuffling the order of the cities in that subset.

After some tests, the best mutation method has been found to be the *inversion* method.
The mutation rate is reduced over time to allow the algorithm to converge to a better solution.

##### Other tweaks
In order to improve the performance of the genetic algorithm, some tweaks have been made to the algorithm. These tweaks include:

1. Replacement Rate: replacing part of population where offsprings and parents compete.
2. Diversity Threshold: Maintain population diversity.


##### Parameters

| Parameter | Value | Note |
| --- | --- | --- |
| Population Size | 200 | 500 for China*|
| Number of Generations | 50000 | |
| Number of Parents | 40 | |
| Initial Mutation Rate | 0.8 | |
| Replacement Rate | 0.5 |  |
| Diversity Threshold | 0.7 | Periodic check |

*China has a large number of cities (726) and requires a larger population size to explore the search space effectively.

### Results

| State | Number of Cities | Greedy Path Lenght | Genetic Path Lenght |
| --- | --- | --- | --- |
| China | 726 | 63962.92 km | 54559.40 km |
| Italy | 46 | 4436.03 km | 4245.04 km |
| Russia | 167 | 42334.16 km | 36044.49 km |
| USA | 326 | 48050.03 km | 40481.11 km |
| Vanuatu | 8 | 1475.53 km | 1345.54 km |


### Discussion

* Comparison of Algorithms: The Genetic Algorithm generally outperforms the Greedy Algorithm, especially in larger instances, due to its ability to explore the solution space more effectively. However, the Greedy Algorithm is faster and can provide a good approximation for smaller instances.
* Impact of Parameters: Adjusting parameters such as population size (notably for China), mutation rate, and diversity threshold can significantly influence the Genetic Algorithm's performance.
* Scalability: While the Genetic Algorithm performs well, scalability remains a challenge for very large TSP instances, suggesting the need for further optimization techniques or more powerful computational resources.
* Future Enhancements: Consider integrating Greedy Algorithm to initialize the Genetic Algorithm and exploring other crossover and mutation methods.


## Lab 3F: Short Path between two cities
This script implements a **Greedy Best-First Search** algorithm to find a short path between two randomly chosen cities in Italy. It reads city coordinates from a CSV file, computes pairwise geodesic distances, and constructs a graph where cities are nodes and edges are added based on distance constraints. The algorithm iteratively selects the nearest unvisited city until reaching the destination. 

## Lab 3: n-Puzzle description
The n-Puzzle is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile missing. The puzzle also exists in other sizes, particularly the smaller 8-puzzle. If the size is 3×3 tiles, the puzzle is called the 8-puzzle or 9-puzzle, respectively, for the number of tiles and the number of spaces. The goal of the puzzle is to place the tiles in order by making sliding moves that use the empty space.
It's a (n^2-1)-puzzle, where n is the number of rows and columns.

### Algorithms implemented
#### Informed Search
##### Best-First Search
Best-First Search is a search algorithm that explores a graph by expanding the most promising node chosen according to a specified rule. The algorithm uses a priority queue to keep track of the nodes that need to be explored.

##### A* Search
A* Search is a search algorithm that finds the shortest path between the initial and the final node. It uses a heuristic function to estimate the cost of the cheapest path through a node. The algorithm uses a priority queue to keep track of the nodes that need to be explored.

The main difference between Best-First Search and A* Search is that A* Search uses both the cost to reach a node and the heuristic function to estimate the cost of the cheapest path through a node, while Best-First Search only uses the heuristic function.

#### Heuristic Function
The heuristic function is used to estimate the cost of the cheapest path from the current node to the goal node. In this laboratory task, the heuristic function used is the Manhattan distance, which is the sum of the horizontal and vertical distances between the current node and the goal node.
##### Manhattan Distance
The Manhattan distance is the sum of the horizontal and vertical distances between the current node and the goal node. It is calculated as the absolute difference between the x-coordinates and the y-coordinates of the current node and the goal node.

### Results
The initial state is generated randomly performing 1000 steps from the goal state.
The following results correspond to a 3x3 puzzle:

| Algorithm | Time (s) | Path Length (Quality) | N. actions evaluated (Cost) | Efficiency (Quality/Cost) |
|-----------|----------|-----------------------|-----------------------------| ------------------------- |
| Best-First| 0.0151  | 42                    | 502                          |  0.0837                 |
| A*        | 0.0732    | 24                   | 3010                        |  0.0080                  |


### Discussion

From the result we can see that the Best-First Search algorithm is more efficient than the A* Search algorithm. The Best-First Search algorithm evaluates fewer actions and has a higher efficiency than the A* Search algorithm. The Best-First Search algorithm is faster than the A* Search algorithm because it evaluates fewer actions and has a higher efficiency. However, A* find the optimal path, while Best-First Search does not guarantee the optimal path. The A* is also infeasible for large problems (e.g. 5x5) because it evaluates a large number of actions and become very slow.

## Project: Symbolic Regression using Genetic Programming

### Abstract
This project explores the application of Genetic Programming (GP) for Symbolic Regression, aiming to evolve mathematical expressions that approximate given datasets. We employ evolutionary techniques such as crossover, mutation, and selection to iteratively refine expressions. The project implements a robust evaluation framework ensuring numerical stability and meaningful symbolic representations. Results demonstrate the effectiveness of GP in discovering compact and accurate mathematical models.

### Introduction
Symbolic Regression (SR) is a powerful approach that aims to discover mathematical expressions capable of accurately modeling a given dataset, without relying on predefined functional forms. Unlike traditional regression techniques, which assume a fixed model structure, SR explores a vast space of possible mathematical operations, seeking expressions that not only fit the data but also remain interpretable. One of the most effective methodologies for performing SR is **Genetic Programming (GP)**, an evolutionary algorithm that iteratively refines candidate solutions through mechanisms inspired by natural selection. By employing evolutionary operators such as **crossover** and **mutation**, GP is able to generate, modify, and optimize mathematical expressions over successive generations, gradually improving their accuracy and generalization.

The goal of this project is to develop a robust **GP framework** for symbolic regression, ensuring that the generated expressions remain both **mathematically valid** and **numerically stable**. A critical aspect of this implementation is the use of **safe mathematical operations**, which help prevent numerical issues such as division by zero or logarithms of negative numbers. The evolutionary process is guided by a **fitness function** based on **Mean Squared Error (MSE)**, ensuring that the discovered expressions accurately approximate the target dataset while avoiding overfitting.

To maintain the integrity of the generated expressions, the project places specific constraints on the evolutionary process. The implementation strictly enforces the use of **safe mathematical functions**, preventing undefined or unstable operations that could compromise the reliability of the expressions. Additionally, the design of the genetic programming framework is tailored to handle **symbolic tree representations**, ensuring that operators and operands are combined in a structurally valid manner. The evaluation process is conducted on predefined datasets, allowing for a controlled assessment of the framework’s effectiveness without introducing domain-specific assumptions.

This project aims to develop an efficient and interpretable symbolic regression system, capable of autonomously discovering meaningful mathematical relationships from data.

### Methodology
#### Technologies Used
- **Programming Language:** Python
- **Libraries:** NumPy (for numerical computations)
- **Tools:** Jupyter Notebook (for interactive development)

#### Approach
The Genetic Programming (GP) framework follows a structured evolutionary process designed to iteratively improve symbolic expressions:

1. **Initialization:** The process begins by randomly generating a population of mathematical expressions, represented as syntax trees. Each tree consists of nodes that define mathematical operations and operands. This random initialization ensures diversity in the population, allowing exploration of a wide range of potential solutions. 

2. **Evaluation:** Each individual tree is evaluated by computing its fitness using the Mean Squared Error (MSE) between its predicted outputs and the actual values from the training dataset. The lower the MSE, the better the expression approximates the dataset.

3. **Selection:** The tournament selection method is applied to choose individuals for reproduction. A subset of individuals is randomly selected from the population, and the one with the best fitness (lowest MSE) is chosen as a parent. This selection strategy ensures that better solutions have a higher chance of propagating to the next generation while maintaining diversity.

4. **Crossover:** Two selected parent trees undergo crossover, where random subtrees are exchanged between them. This process promotes genetic recombination, allowing new expressions to inherit useful features from both parents, which may lead to improved fitness.
    The crossover operation follows a **subtree swapping** approach:
    1. A **random subtree** is selected from **Parent 1** (the recipient).
    2. A **random subtree** is selected from **Parent 2** (the donor).
    3. The selected subtree from Parent 2 replaces the selected subtree in Parent 1.
    4. The newly created individual is validated to ensure that it remains within predefined structural constraints.


5. **Mutation** Mutation is a key operation in symbolic regression using genetic programming. It introduces variations into the population, preventing premature convergence and promoting diversity. Several mutation operators are applied with different probabilities to modify the structure of individuals in the syntax tree. **Mutation Methods**:
- **Subtree Mutation** Replaces a randomly selected subtree within an individual with a newly generated random subtree. This allows for significant structural changes while preserving the overall tree representation.
- **Point Mutation (Operator and Value Mutation)**
  - **Operator mutation:** Replaces a randomly selected operator node with another operator of the same type.
  - **Value mutation:** Applies a small perturbation to a numerical constant or modifies a variable reference.
- **Hoist Mutation**: Promotes a randomly selected subtree to the root, simplifying the expression while preserving potentially useful substructures.
- **Expansion Mutation**: Replaces a randomly selected leaf node with a new randomly generated subtree, increasing the complexity of the expression.
- **Collapse Mutation**: Simplifies an expression by replacing an internal node (operator) with a constant value.

1. **Pruning & Refinement:** To prevent excessive growth of expression trees (known as bloat), pruning techniques are applied. These include replacing deep or complex subtrees with simpler alternatives while preserving functional integrity. This step ensures that evolved expressions remain interpretable and computationally efficient.

2. **Termination:** The evolutionary process stops when a predefined stopping criterion is met, such as no further improvement in MSE for a set number of generations. At this point, the best-performing expression is selected as the final solution.

#### Implementation Details
The implementation is based on a hierarchical tree structure where each node represents an operation or operand. The following key aspects define the GP framework:

- **Mathematical Operations:** The framework supports a set of unary and binary mathematical operations to construct expressions:
  - Unary operators include `sin`, `cos`, `log`, `sqrt`, `tan`, `tanh`, `sinh`, `cosh`, `abs`, `log10`, and `log2`. These functions operate on a single input and provide essential transformations required for symbolic regression.
  - Binary operators include `+`, `-`, `*`, `/`, `mod`, and `**` (exponentiation). These allow complex relationships to be modeled within the expressions.
  To ensure numerical stability and avoid undefined behaviors, the project implements **safe versions** of mathematical operations. These functions handle edge cases such as division by zero, logarithm of non-positive numbers, and overflow errors:

    ```python
    import numpy as np

    def safe_div(x, y):
        return x / (y + 1e-8)  # Avoid division by zero

    def safe_log(x):
        return np.log(np.abs(x) + 1e-8)  # Avoid log of zero or negative values

    def safe_sqrt(x):
        return np.sqrt(np.abs(x))  # Ensure non-negative input for sqrt
    ```
    
- **Tree Representation:** Expressions are structured as binary trees, where:
  - Leaf nodes represent variables (e.g., `x[0]`, `x[1]`) or constants.
  - Internal nodes represent operators that define mathematical relationships between subtrees.
    **Node Class**
    The `Node` class defines the structure of each node in the tree. It consists of:
    - `value`: The numerical value (for constants) or variable name (e.g., `"x[0]"`) for leaf nodes.
    - `op`: The operator used in internal nodes.
    - `left` and `right`: Pointers to child nodes (only relevant for operator nodes).
    The class provides some functionalities such as a deep copy method and string representation (a human-readable representation).
- **Tree Compilation**
To efficiently evaluate the evolved expressions, the **compile_tree** function converts the tree into a Python lambda function.
    - **Steps of Compilation**
      1. **Tree Traversal (`node_to_str`)**:
         - Recursively converts the tree into a string representation of a valid Python expression.
         - Special handling is applied to **safe operations** (e.g., `safe_divide`, `safe_pow`, `safe_mod`) to ensure numerical robustness.
      2. **Lambda Function Construction**:
         - The generated expression string is embedded into a lambda function:  
           ```python
           lambda x: <generated_expression>
           ```
         - The function is evaluated in a restricted execution environment to prevent security risks.
      3. **Safe Execution**:
         - If any exception occurs during compilation, a fallback function is used that returns a zero array of the appropriate shape.

      By compiling trees into lambda functions, the evaluation of expressions becomes highly efficient, allowing for rapid fitness computations during evolutionary training.

- **Mutation Operators:** The framework implements multiple mutation strategies to introduce variability:
  1. **Subtree Mutation**
     - A target node is randomly chosen from the tree.
     - A new subtree is generated with a random depth within the range **[1,4]**.
     - The selected node is replaced with the new subtree.
     - If the replaced node was the root, the new subtree becomes the new root.
     - Ensures the resulting tree does not exceed the maximum depth (**10**) or node count (**40**).

  2. **Point Mutation (Operator and Value Mutation)**
     - **Operator mutation:**
       - Selects a random operator node.
       - Replaces it with another operator of the same type (binary or unary).
       - Ensures valid children nodes are assigned to maintain a proper syntax tree.
     - **Value mutation:**
       - If the node contains a numerical constant, a small perturbation is applied.
       - The perturbation is scaled by a **mutation factor (0.1)**.
       - If the node represents a variable (`x[i]`), it may be randomly reassigned to another variable.

  3. **Hoist Mutation**
     - Selects a random subtree from the tree.
     - Promotes the subtree to replace the entire tree.
     - Helps in reducing complexity by eliminating unnecessary branches.

  4. **Expansion Mutation**
     - A leaf node (a constant or variable) is randomly selected.
     - It is replaced with a newly generated subtree.
     - The depth of the new subtree is randomly chosen within the range **[1,4]**.
     - Ensures the resulting tree does not exceed depth and node limits.

  5. **Collapse Mutation**
     - Selects a random internal node (operator).
     - Replaces it with a randomly generated constant.
     - Helps in pruning unnecessary computations and simplifying expressions.

    **Mutation Selection and Constraints**
    - A mutation type is chosen based on the following probability distribution:


    | Mutation Type       | Probability |
    |---------------------|------------|
    | Subtree Mutation   | 30%        |
    | Operator Mutation  | 20%        |
    | Value Mutation     | 20%        |
    | Hoist Mutation     | 10%        |
    | Expansion Mutation | 10%        |
    | Collapse Mutation  | 10%        |


    - The selected mutation is applied to the individual.
    - After mutation, the tree is validated:
      - Ensures the tree does not exceed the maximum depth (**10**) or node count (**40**).
    - If limits are exceeded, pruning is applied.

    This structured mutation approach ensures a balance between **exploration** (introducing new structures) and **exploitation** (refining existing solutions), ultimately improving the symbolic regression model’s ability to find optimal expressions.



- **Crossover Strategy:** A random subtree from one parent is swapped with a subtree from another parent, enabling genetic recombination. The crossover process ensures that structural integrity is maintained, allowing evolved expressions to retain valid mathematical meaning.
  - **1. Offspring Initialization**
    - The offspring starts as a **copy of Parent 1**, ensuring that modifications do not alter the original individual.
    - A **random node** (subtree) is selected from Parent 1 as the **crossover target**.
    - The **donor subtree** is randomly chosen from Parent 2.
  - **2. Structural Validation**
    Once the subtree is transferred, its structure is validated:
    - If the **target node** in Parent 1 is a **binary operator**:
        - Ensures that the donor subtree has both left and right children.
        - If necessary, missing children are filled with newly generated random subtrees.
    - If the **target node** is a **unary operator**:
        - Ensures the donor subtree has at least one child.
        - If missing, a random subtree is generated.
    - If the **target node** is a **leaf node**:
        - If it represents a **variable**, it may be reassigned to another variable index.
        - If it is a **constant**, it may be replaced with a newly generated constant.
  - **3. Integration into Offspring**
     - If the target node was the **root** of Parent 1, the offspring is entirely replaced by the donor subtree.
     - Otherwise, the **parent node** of the selected subtree in Parent 1 is updated to reference the donor subtree.

  - **4. Constraint Enforcement**
    - After the crossover, the offspring is passed through `sanitize_tree()` to ensure:
      - The tree does not exceed the **maximum depth (10)**.
      - The tree does not exceed the **maximum node count (40)**.
      - The resulting structure remains a valid mathematical expression.

- **Fitness Calculation:** The Mean Squared Error (MSE) is computed between the predicted values from the evolved function and the actual dataset values (y). This provides a quantitative measure of accuracy, guiding the evolution towards optimal solutions.


##### **Evolutionary Strategy and Selection Mechanisms**

The core evolutionary process in this symbolic regression framework follows a **steady-state genetic programming (GP) approach**, where the population undergoes iterative updates through **selection, crossover, mutation, and replacement**. A crucial aspect of this implementation is the use of **elitism**, which ensures that the best solutions persist across generations, thereby maintaining genetic quality and preventing performance degradation.

###### **Elitism Mechanism**
Elitism is a fundamental technique in evolutionary algorithms that helps preserve high-quality solutions by directly carrying over the best individuals from one generation to the next. In this implementation, the elitism mechanism ensures that a fixed proportion of the top-performing individuals (based on fitness) are directly transferred to the next generation without modification.

The **elitism rate** is a critical hyperparameter. A high elitism rate ensures strong preservation of top solutions, but it may reduce population diversity, leading to premature convergence. Conversely, a very low elitism rate may cause good solutions to be lost, slowing down convergence. The chosen elitism rate in this implementation is set as 5% of the population, striking a balance between exploration and exploitation.

###### **Stopping Criteria**
The algorithm implements two primary stopping conditions to optimize computational efficiency and prevent unnecessary iterations:
1. **Fixed Generation Limit**: The evolutionary process runs for a maximum of 150 generations (GENERATIONS = 150). If no early stopping condition is met, the algorithm halts after this predefined number of iterations.
2. **Early Stopping Based on No Improvement**: The algorithm tracks the best fitness value found so far. If no improvement is observed for 20 consecutive generations, the process is terminated early.

These criteria together strike a balance between exploration (allowing the algorithm to evolve sufficiently complex solutions) and computational efficiency (stopping when further evolution is unlikely to yield improvements).

###### **Parameter Selection and Justification**
The choice of hyperparameters is critical for ensuring that the algorithm effectively explores the solution space while maintaining convergence speed. These parameters were carefully chosen after an extensive tuning process, involving a try-and-error approach to identify the configurations that performed best among all tested solutions. The following parameters were chosen:


| **Parameter**       | **Value** | **Justification** |
|---------------------|----------|-------------------|
| **Population Size** | 1500     | Provides a diverse pool of solutions, reducing the risk of premature convergence. |
| **Max Depth**       | 7        | Limits the complexity of the evolved expressions, preventing excessive tree growth. |
| **Generations**     | 150      | Ensures a sufficient number of iterations for optimization while maintaining efficiency. |
| **Tournament Size** | 15       | Balances selection pressure, allowing strong solutions to be favored without losing diversity. Adapted to the large population. |
| **Elitism Rate**    | 5% | Ensures top solutions are preserved while still allowing innovation. |
| **Mutation Probability** | 50%| Maintains genetic diversity, ensuring that new structures continue to emerge. |
| **Crossover probabilty** | 100% | Ensures that crossover is always applied (exception for the elite), promoting genetic recombination. |

Each parameter was chosen based on typical settings in **genetic programming** and adjusted to maintain a balance between exploration and exploitation.

###### **Evolutionary Process Recap**
The algorithm follows a structured **evolutionary cycle**:

1. **Initialization:**  
   - A population of **1500 randomly generated symbolic expression trees** is created, each constrained by a maximum depth of **7**.
   - The dataset is loaded, determining the number of input variables.

2. **Evaluation:**  
   - Each individual's fitness is computed using **Mean Squared Error (MSE)**, assessing how well the generated function approximates the given data.

3. **Selection and Reproduction:**  
   - **Elitism:** The best **5%** of individuals are automatically carried over.
   - **Tournament Selection:**  
     - Two parents are chosen via a **tournament selection** mechanism, where **15** individuals are randomly sampled.
     - The fittest individual from the sampled set is selected as a parent.
   - **Crossover:** The selected parents undergo subtree crossover, combining portions of their structures.
   - **Mutation:**  Offspring undergo **mutation with a 50% probability**, introducing local modifications to maintain diversity.
   - **Population Replacement:** The new generation replaces the previous one.

4. **Termination:**  
   - If no improvement occurs for **20 consecutive generations**, the algorithm stops early.
   - Otherwise, the process continues until **150 generations** are completed.



### Results
The efficiency of the implemented genetic programming framework allows the algorithm to process all 8 problem instances within a reasonable timeframe. Thanks to the use of **Python's `eval()` function**, which dynamically compiles evolved expressions into executable functions, the evaluation of symbolic models is significantly accelerated. This optimization enables the entire evolutionary process to complete in approximately 15-20 minutes all problems.

The obtained results showcase a diverse range of mathematical expressions evolved through the genetic programming framework, each evaluated based on its Mean Squared Error (MSE) performance. Among the generated functions, **f1** stands out as the most accurate model, achieving an exceptionally low MSE of **7.1259e-34**, indicating an almost perfect fit to the data. This function is a simple sine transformation of the first variable, suggesting that the underlying relationship in the dataset might be well captured by trigonometric behavior. Similarly, **f5** also demonstrates near-perfect accuracy, with an MSE of **5.5728e-18**, though its trivial structure `(x[1] - x[1])` implies it may not carry meaningful predictive power.

In contrast, functions such as **f2** and **f8** exhibit significantly higher MSE values (**1.5524e+13** and **5.0586e+05**, respectively), indicating poor generalization and over-complexity. These functions involve deep compositions of hyperbolic, logarithmic, and trigonometric operations, making them susceptible to numerical instability and overfitting. Their complexity suggests that the evolutionary process may have led to unnecessary intricate expressions that fail to capture the true underlying pattern effectively.

Other functions, such as **f3** and **f4**, achieve moderate performance, with MSE values of **0.35665** and **0.068457**, respectively. While **f3** incorporates multiple nested operations, including absolute values, hyperbolic tangents, and sinusoidal components, **f4** maintains a simpler yet still effective structure, leveraging sine and cosine transformations. Their comparatively lower MSE values indicate that these expressions may hold some predictive relevance, though their interpretability and robustness should be further analyzed.

Furthermore, **f6** and **f7** also display varying degrees of effectiveness, with MSE values of **7.9451e-08** and **3.6774e+02**, respectively. While **f6** appears to provide a relatively strong approximation with minor numerical adjustments, **f7** involves hyperbolic cosine transformations that introduce significant distortions, as reflected in its larger error.

| Function | MSE Value       | 
|----------|----------------|
| f1       | 7.1259e-34     | 
| f2       | 1.5524e+13     | 
| f3       | 0.35665        | 
| f4       | 0.068457       | 
| f5       | 5.5728e-18     | 
| f6       | 7.9451e-08     | 
| f7       | 3.6774e+02     |
| f8       | 5.0586e+05     | 



Overall, the results demonstrate the strengths and limitations of the genetic programming approach in symbolic regression. While some evolved functions exhibit remarkable accuracy, others suffer from excessive complexity and poor generalization. The findings highlight the importance of balancing expressiveness and simplicity in the evolutionary process, ensuring that generated expressions remain both interpretable and predictive.

