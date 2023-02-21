# Optimization-for-Machine-Learning
Summer semester '23, University of Leipzig, 10-INF-XXXXX

Draft list of topics:
1. Introduction: Examples of optimization in machine learning
   - Empirical risk minimization
   - Matrix completion/recommender systems
   - Learning in dynamical systems

1.  Basic concepts of convex optimization

    - Analysis of gradient descent algorithm 
    - Optimality condition (KKT)
    - Projection onto convex sets (constrained optimization)
    - Polyak step size


2.  Non-smooth convex optimization (regularised risk minimization)

    -   Mirror descent algorithm 
    -   Stochastic approximation mirror descent algorithm
  
3.  Stochstic Gradiend Descent

5.  Online Convex Optimization
    - Follow-The-Regularised-Leader (FTRL) algorithm

6.  Introduction to Bandits:
    - Exploration-Explotation aglorithm

7.  Large scale learning:
    - Approximation risk optimization
    - Asymptotic analysis, uniform convergance bounds
    - tradeoff of the performances
  

8.  Deep learning optimization (ADAM, momentum..., implementation from
    scratch)

    -  Accelerated methods
    -  Natural gradient descent

9.  Projects: 
    - Matrix completion (low rank approximation)
    - Optimization in deep linear learning models 
    - Learning in dynamical sytems
<!--Goals:-->
<!--- Understand the definitions of standard data science terms, and the associated mathematical terms-->
<!--- Understand the proofs of how commonly used techniques in data science work-->
<!--- Implement the algorithms and examples with a computer program-->
<!--- Investigate the math behind your favorite topic in data science-->

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
- 40% Project: Due 18.01 in class: Pick a data science topic and learn about the math behind it. Must include 1 proof and 1 example (~2 pages)
- 50% Exam: written theory exam covering entire course, mainly computations and examples.

---
## Course Schedule
| Date      | Topics                                                           |
|-----------|------------------------------------------------------------------|
| Tue 04.04 | Convex optimization I. (Definitions, examples, gradient descent) |
| Wed 05.04 | Gradient descent implementation                                  |
| Tue 11.04 | Convex optimization II.  (Duality, constrained optimization)     |
| Wed 12.04 |                                                                  |
| Tue 18.04 | Non-smooth convex optimization                                   |
| Wed 19.04 |                                                                  |
| Tue 25.04 |                                                                  |
| Wed 26.04 |                                                                  |
| Tue 02.05 |                                                                  |
| Wed 03.05 |                                                                  |
| Tue 09.05 |                                                                  |
| Wed 10.05 |                                                                  |
| Tue 16.05 |                                                                  |
| Wed 17.05 |                                                                  |
| Tue 23.05 |                                                                  |
| Wed 24.05 |                                                                  |
| Tue 30.05 |                                                                  |
| Wed 31.05 |                                                                  |
| Tue 06.06 |                                                                  |
| Wed 07.06 |                                                                  |
| Tue 13.06 |                                                                  |
| Wed 14.06 |                                                                  |
| Tue 20.06 |                                                                  |
| Wed 21.06 |                                                                  |
| Tue 27.06 |                                                                  |
| Wed 28.06 |                                                                  |
| Tue 04.07 |                                                                  |
| Wed 05.07 |                                                                  |
| Tue 11.07 |                                                                  |
| Wed 12.07 |                                                                  |

---

## Julia und Jupyter Notebooks

This repository contains the [Jupyter Notebooks](https://github.com/skfairchild/MathData-Winter22-23) from the class.

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

---

## Literature
The following materials are chosen to complement the [course lecture
notes](https://raw.githubusercontent.com/KarinaPapayia/Optimization-for-Machine-Learning/main/OML.pdf)

[Optimization in Machine Learning]() Sra, Nowozin & Wright

### 1. Convex Optimization

[Convex optimization](https://web.stanford.edu/~boyd/cvxbook), Boyd & Vandenberghe

[Introductory lectures on convex optimization](), Nesterov

### 
