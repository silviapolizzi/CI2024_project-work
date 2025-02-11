
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


##### Evolutionary Strategy and Selection Mechanisms

The core evolutionary process in this symbolic regression framework follows a **steady-state genetic programming (GP) approach**, where the population undergoes iterative updates through **selection, crossover, mutation, and replacement**. A crucial aspect of this implementation is the use of **elitism**, which ensures that the best solutions persist across generations, thereby maintaining genetic quality and preventing performance degradation.

###### Elitism Mechanism
Elitism is a fundamental technique in evolutionary algorithms that helps preserve high-quality solutions by directly carrying over the best individuals from one generation to the next. In this implementation, the elitism mechanism ensures that a fixed proportion of the top-performing individuals (based on fitness) are directly transferred to the next generation without modification.

The **elitism rate** is a critical hyperparameter. A high elitism rate ensures strong preservation of top solutions, but it may reduce population diversity, leading to premature convergence. Conversely, a very low elitism rate may cause good solutions to be lost, slowing down convergence. The chosen elitism rate in this implementation is set as 5% of the population, striking a balance between exploration and exploitation.

###### Stopping Criteria
The algorithm implements two primary stopping conditions to optimize computational efficiency and prevent unnecessary iterations:
1. **Fixed Generation Limit**: The evolutionary process runs for a maximum of 150 generations (GENERATIONS = 150). If no early stopping condition is met, the algorithm halts after this predefined number of iterations.
2. **Early Stopping Based on No Improvement**: The algorithm tracks the best fitness value found so far. If no improvement is observed for 20 consecutive generations, the process is terminated early.

These criteria together strike a balance between exploration (allowing the algorithm to evolve sufficiently complex solutions) and computational efficiency (stopping when further evolution is unlikely to yield improvements).

###### Parameter Selection and Justification
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

###### Evolutionary Process Recap
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

