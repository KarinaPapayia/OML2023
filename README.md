# Optimization-for-Machine-Learning
Summer semester '23, University of Leipzig, 10-INF-XXXXX

Draft list of topics:

1.  convex optimization (2 lectures)

    -   Gradient descent

2.  Nonsmooth convex optimization (1 lecture)

    -   Mirror descent

3.  Regularized risk minimization

    -   

4.  SGD

5.  Online convex problem / Follow the regularized leader

6.  Bandits

7.  large scale learning, trade-off approximation risk optimization
    risk, asymptotic analysis tradeoff of the performances, uniform
    convergence bounds

8.  deep learning optimization (ADAM, momentum\..., implementation from
    scratch)

    -   Accelerated methods

9.  Project

Goals:
- Understand the definitions of standard data science terms, and the associated mathematical terms
- Understand the proofs of how commonly used techniques in data science work
- Implement the algorithms and examples with a computer program
- Investigate the math behind your favorite topic in data science

We first cover two introductory topics
1. Linear algebra
  - Subspaces
  - Orthogonality
  - The pseudo-inverse
  - the singular value decomposition
2. Probability Theory

We then proceed with the following four themes commonly seen in data science

3. Network analysis
  - Graphs and the Laplace matrix
  - The spectrum of a graph
  - Markov processes in networks
  - Centrality measures
4. Machine learning
  - Data, models, and learning
  - Regeression in statistical models
  - Principal component analysis (method for dimension reduction)
  - Support vector machines (binary classification method)
5. Topological data analysis
  - Simplicial complexes and homology
6. Matrices and tensors
  - Low rank matrices and tensors
---

## Course Information 
- From April 3rd through July 5th 2023
- Tuesdays 11:15-12:45 (Lecture)
- Wednesdays 15:15 - 16:45 (Seminar)
- SG 2-14

- Contact: katerina.papagiannouli(at)mis.mpg.de
- Office hours: Tuesdays and Wednesdays after class, and by email.

Grading scheme:
- 10% Homework: assigned every other week, hand in 1 problem for grading.
- 40% Project: Due 18.01 in class: Pick a data science topic and learn about the math behind it. Must include 1 proof and 1 example (~2 pages)
- 50% Exam: written theory exam covering entire course, mainly computations and examples.

---
## Course Schedule

| Date | Topics |
|------|--------|
|04.04|   |
|05.04|   |
|11.04|   |
|12.04|   |
|18.04|   |
|19.04|   |
|25.04|   |
|26.04|   |
|02.05|   |
|03.05|   |
|09.05|   |
|10.05|   |
|16.05|   |
|17.05|   |
|23.05|   |
|24.05|   |
|30.05|   |
|31.05|   |
|06.06|   |
|07.06|   |
|13.06|   |
|14.06|   |
|20.06|   |
|21.06|   |
|27.06|   |
|28.06|   |
|04.07|   |
|05.07|   |
|11.07|   |
|12.07|   |

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
The following materials are chosen to complement the [course lecture notes](https://raw.githubusercontent.com/skfairchild/MathData-Winter22-23/main/MathData.pdf)

### 1. Linear Algebra

[The Fundamental Theorem of Linear Algebra](https://www.engineering.iastate.edu/~julied/classes/CE570/Notes/strangpaper.pdf), Gilbert Strang.

### 2. Probability Theory

[Wikipedia Artikel](https://en.wikipedia.org/wiki/Probability_theory)

[Basic Probability Theory](https://faculty.math.illinois.edu/~r-ash/BPT/BPT.pdf), Robert B. Ash.

### 3. Network Analysis

[Spectral Graph Theory](https://mathweb.ucsd.edu/~fan/research/revised.html)
(insbesondere Kapitel 1), Fan Chung.

[Spectral Graph Theory](https://resources.mpi-inf.mpg.de/departments/d1/teaching/ws11/SGT/) (insbesondere Vorlesung 5), Thomas Sauerwald and He Sun

[Graph Theory in the Information Age](https://mathweb.ucsd.edu/~fan/wp/graph.pdf), Fan Chung.

[Computer Science Theory for the Information Age](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/) (insbesondere Notes 5), Venkatesan Guruswami and Ravi Kannan.

### 4. Machine Learning

[Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) (insbesondere Kapitel 8-12), Marc Peter Deisenroth, A. Aldo Faisal und Cheng Soon Ong.

[Neural Network Theory](http://www.pc-petersen.eu/Neural_Network_Theory.pdf), Philipp Christian Petersen

### 5. Topological Data Analysis

[Topological Data Analysis Spring 2020](https://www.few.vu.nl/~botnan/lecture_notes.pdf), Magnus Bakke Botnan.

[Topological Data Analysis](https://fugacci.github.io/home/notes.html), Ulderico Fugacci.

### 6. Matrices and Tensors

[Geometric Methods on Low-Rank Matrix and Tensor Manifolds](https://link.springer.com/content/pdf/10.1007%2F978-3-030-31351-7_9.pdf), André Uschmajew and Bart Vandereycken.

[Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf),
Tamara G. Kolda und Brett W. Bader.

