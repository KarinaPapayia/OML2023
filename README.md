## UPDATE 20.04
The lecture is available on [Moodle](https://moodle2.uni-leipzig.de/course/view.php?id=44197)

# Optimization for Machine Learning
Summer semester '23, University of Leipzig, 10-INF-DS301


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
  
3.  Stochastic Gradient Descent

5.  Online Convex Optimization
    - Follow-The-Regularised-Leader (FTRL) algorithm

6.  Introduction to Bandits:
    - Exploration-Exploitation algorithm

7.  Large scale learning:
    - Approximation risk optimization
    - Asymptotic analysis, uniform convergence bounds
    - Trade-off of the performances
  

8.  Deep learning optimization (ADAM, momentum..., implementation from
    scratch)

    -  Accelerated methods
    -  Natural gradient descent

9.  Projects: 
    - Matrix completion (low rank approximation)
    - Optimization in deep linear learning models 
    - Learning in dynamical systems

 Goals:
  - Understand the basic mathematical concepts of convex optimization
  - Understand the difference between online and offline optimization methods
  - Analyse the performance of optimization algorithms from a statistical-learning perspective
  - Apply the optimization methods to machine learning problems

---

## Course Information 
- From April 5th through July 5th 2023
- Mondays (from April 17th) 17:15 &ndash; 18:45 (Seminar) Pierre Bréchet 
- Wednesdays 15:15 &ndash; 16:45 (Lectures) Katerina Papagiannouli

- Contact: katerina.papagiannouli(at)mis.mpg.de, pierre.brechet(at)mis.mpg.de
- Office hours: Mondays and Wednesdays after class, and by email.

Grading scheme:
- 10% Homework: assigned every other week, hand in 1 problem for grading.
- 40% Project: Due 28.06 in class
- 50% Exam: written theory exam covering entire course, mainly computations and examples.

---
## Course Schedule

| Date        | Topics                                                                                 | HM         |
|-------------|----------------------------------------------------------------------------------------|------------|
| Wed 05.04   | Introduction to Optimization for machine learning                                      |
| *Mon 10.04* | *No class (Easter Monday)*                                                             |
| Wed 12.04   | Basic concepts of convex optimization / Analysis of gradient descent algorithm         |
| Mon 17.04   | Ex01 &mdash; Implementation of gradient descent algorithm. Gradient Manipulations.     |
| Wed 19.04   | Non-smooth convex optimization/ regularised risk minimization                          |
| Mon 24.04   | Ex02 &mdash; Linear models, SVM with gradient descent.  **Hand-out HM1**               |            |
| Wed 26.04   | Stochastic gradient descent                                                            |
| *Mon 01.05* | *No class (Tag der Arbeit)*                                                            |
| Wed 03.05   | Online Convex Optimization                                                             |
| Mon 08.05   | Discuss HM 01 &mdash; Ex03 &mdash; SGD implementation                                  | **01 due** |
| Wed 10.05   | Intro to Bandits                                                                       |
| Mon 15.05   | Ex04 &mdash; Online convex optimization. **Hand-out HM2**                              |
| Wed 17.05   | Large scale learning: empirical risk minimization   ERM                                |
| Mon 22.05   | Discuss HM02 &mdash; Ex05                                                              | **02 due** |
| Wed 24.05   | Asymptotic analysis of ERM. **Hand-out HM3**                                           |
| *Mon 29.05* | *No class (Pflingstmontag)*                                                            |
| Wed 31.05   | Trade-off of the performances                                                          |
| Mon 05.06   | Discuss HM 03 &mdash; Analysis of SVM and its performance through different algorithms | **03 due** |
| Wed 07.06   | Optimization in neural networks part 1                                                 |
| Mon 12.06   | Ex07 &mdash; Implementation Adam, SGD with momentum. **Hand-out HM4**                  |
| Wed 14.06   | Optimization in neural networks part 2: Natural Gradient descent                       |
| Mon 19.06   | Discuss HM04 &mdash; Ex08 &mdash;  Implementation of natural gradient descent          | **04 due** |
| Wed 21.06   | Discussion of the projects                                                             |
| Mon 26.06   | Projects                                                                               |
| Wed 28.06   | Review topics                                                                          |
| Mon 03.07   | **Submission of the projects**                                                         |
| Wed 05.07   | Review topics                                                                          |
| Mon 10.07   | Practice final exam                                                                    |

## Python, Julia and Jupyter Notebooks

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
The following materials are chosen to complement the course lecture
notes that will be made available. 


### 1. Convex Optimization

[Convex optimization](https://web.stanford.edu/~boyd/cvxbook), Boyd & Vandenberghe

Introductory lectures on convex optimization, Nesterov

### 2. Optimization for machine learning
Optimization in Machine Learning, Sra, Nowozin & Wright

### 3. Online convex optimization

[Intro to Online Convex Optimization](https://arxiv.org/pdf/1909.05207.pdf), Elad Hazan

### Miscellaneous 

[Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

