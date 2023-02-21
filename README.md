# Optimization-for-Machine-Learning
Summer semester '23, University of Leipzig, 10-INF-XXXXX

Draft list of topics:
1. Introduction: Examples of optimization in machine learning
   - Empirical risk minimization
   - Matrix completion/recommender systems
   - Learning in dynamical systems

1.  Basic concepts of convex optimization

    - Analysis of gradient descent algorithm 
    - Optimality conditions (KKT)
    - Projection onto convex sets (constrained optimization)
    - Polyak step size


2.  Non-smooth convex optimization (regularized risk minimization)

    -   Mirror descent algorithm: minimizing over 
    -   Mirror descent stochastic approximation
  
3.  Stochstic Gradiend Descent

5.  Online Convex Optimization
    - Follow-The-Regularised-Leader (FTRL) algorithm

6.  Introduction to Bandits:
    - Exploration-Explotation aglorithm

7.  Large scale learning:
    - Approximation risk optimization
    - Asymptotic analysis, uniform convergance bounds
    - Trade-off of the performances
  

8.  Deep learning optimization (ADAM, momentum..., implementation from
    scratch)

    -  Accelerated methods
    -  Natural gradient descent

9.  Projects: 
    - Matrix completion (low rank approximation)
    - Optimization in deep linear learning models 
    - Learning in dynamical sytems

 Goals:
  - Understand the basic mathematical concepts of convex optimization
  - Understand the difference between online and offline optimization methods
  - Analyse the performace of optimization algorithms from a statistical-learning perspective
  - Apply the optimization methods to machine learning problems

<!--We first cover two introductory topics-->
<!--1. Linear algebra-->
  <!--- Subspaces-->
  <!--- Orthogonality-->
  <!--- The pseudo-inverse-->
  <!--- the singular value decomposition-->
<!--2. Probability Theory-->

<!--We then proceed with the following four themes commonly seen in data science-->

<!--3. Network analysis-->
  <!--- Graphs and the Laplace matrix-->
  <!--- The spectrum of a graph-->
  <!--- Markov processes in networks-->
  <!--- Centrality measures-->
<!--4. Machine learning-->
  <!--- Data, models, and learning-->
  <!--- Regeression in statistical models-->
  <!--- Principal component analysis (method for dimension reduction)-->
  <!--- Support vector machines (binary classification method)-->
<!--5. Topological data analysis-->
  <!--- Simplicial complexes and homology-->
<!--6. Matrices and tensors-->
  <!--- Low rank matrices and tensors-->
---

## Course Information 
- From April 4th through July 5th 2023
- Tuesdays 11:15-12:45 (Lecture)
- Wednesdays 15:15 - 16:45 (Seminar)
<!--- SG 2-14-->

- Contact: katerina.papagiannouli(at)mis.mpg.de
- Office hours: Tuesdays and Wednesdays after class, and by email.

Grading scheme:
- 10% Homework: assigned every other week, hand in 1 problem for grading.
- 40% Project: Due 30.06 in class: Pick one project from the list
- 50% Exam: written theory exam covering entire course, mainly computations and examples.

---
## Course Schedule
| Date      | Topics                                                           |
|-----------|------------------------------------------------------------------|
| Tue 04.04 | Inroduction to Optimization for machine learning
| Wed 05.04 | Basic concepts of mathemactical convex optimization                                                                 |
| Tue 11.04 | Analysis of gradient descent algorithm                                                            |    
| Wed 12.04 | Exercise: Implementation of gradient descent algorithm                                                               |
| Tue 18.04 | Non-smooth convex optimization/ regularised risk minimizatiom                                                                |
| Wed 19.04 | Implementation of mirror descent algorithm                                                              |
| Tue 25.04 | Stochastic gradient descent                                                              |
| Wed 26.04 |  Implementation of SGD                                                              |
| Tue 02.05 |  Online Convex Optimization                                                                |
| Wed 03.05 |  Implementation of FTRL algorithm                                                                |
| Tue 09.05 |  Intro to Bandits                                                              |
| Wed 10.05 |  Implementation of Exploration-Explotation algorithm                                                                |
| Tue 16.05 |  Large scale learning: empirical risk minimization   ERM                                                            |
| Wed 17.05 |  Exercises                                                                |
| Tue 23.05 |  Asymptotic analysis of ERM                                                               |
| Wed 24.05 |  Exercises/ projects assigned                                                              |
| Tue 30.05 |  Trade-off of the performances                                                              |
| Wed 31.05 |  Analysis of SVM and its performance through different algorithms                                                                |
| Tue 06.06 |  Deep learning optimization part 1                                                              |
| Wed 07.06 |  Implemention Adam, momentum..                                                                |
| Tue 13.06 |  Deep learning part 2: Natural Gradient descent                                                                |
| Wed 14.06 |  Implementation of natural gradient descent                                                                |
| Tue 20.06 |  Discussion of the projects                      |
| Wed 21.06 |                                                                 |
| Tue 27.06 |  Review topics                                                               |
| Wed 28.06 |  Submission of the projects                                                              |
| Tue 04.07 |  Review topics                                                             |
| Wed 05.07 |  Practice final exam                                                                |

---

## Python und Jupyter Notebooks

<!-- This repository contains the [Jupyter Notebooks](https://github.com/skfairchild/MathData-Winter22-23) from the class.

In order to use the notebooks:

* Download the notebooks (Click on the green `Code` Button or download as Zip File or use a Git Client such as [Github Desktop](https://desktop.github.com) oder [Sublime](https://www.sublimemerge.com)).
* Download the newest version of Juila [here](https://julialang.org/downloads/).
* Start Juila.
* Enter the package manager by putting in `]` in the package manager.
* `add IJulia`
* Leave the package manager with a backspace.
* `using IJulia` 
* `notebook()` 

Then a browser window should open, in which the local saved notebooks can be opened.D

Other material from the [Julia Academy](https://github.com/JuliaAcademy):

* [Introduction to Julia](https://github.com/JuliaAcademy/Introduction-to-Julia)

* [Data Science](https://github.com/JuliaAcademy/DataScience)

* [Foundations of Machine Learning](https://github.com/JuliaAcademy/Foundations-of-Machine-Learning)

* [Data Frames](https://github.com/JuliaAcademy/DataFrames)

--- -->

## Literature
The following materials are chosen to complement the [course lecture
notes](https://raw.githubusercontent.com/KarinaPapayia/Optimization-for-Machine-Learning/main/OML.pdf)

[Optimization in Machine Learning]() Sra, Nowozin & Wright

### 1. Convex Optimization

[Convex optimization](https://web.stanford.edu/~boyd/cvxbook), Boyd & Vandenberghe

[Introductory lectures on convex optimization](), Nesterov

### 2. Optimization for machine learning
[Optimization for machine learning](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Optimization%20for%20Machine%20Learning%20%5BSra%2C%20Nowozin%20%26%20Wright%202011-09-30%5D.pdf)

### 3. Online convex optimization

[Intro to Online Convex Optimization](https://arxiv.org/pdf/1909.05207.pdf), Elad Hazan


