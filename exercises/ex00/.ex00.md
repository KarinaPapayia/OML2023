# Exercise 00

Python refresher: [Lean Python in Y minutes](https://learnxinyminutes.com/docs/python/) 
<!--## Configuration of the environments-->

This "exercise" is to set up the different coding environments for the lecture.


## Python 

Many package managers for Python are available, including

- Conda / Miniconda
- venv module / virtualenv

We use the `venv` module in order to create our development environment. If you are comfortable with conda you probably do not need what follows. 

#### Creating the virtual environment (venv module)

Assuming you have [Python 3.9 or above
installed](https://www.python.org/downloads/) **and** in your `PATH` variable
(check it with: `python3 --version`, should be >= 3.9), clone the repository
and create the virtual environment in `/path/to/OML2023/exercises`:

```bash

$ git clone https://github.com/katerinapapya/OML2023.git . 

$ cd OML2023/exercises

$ python3 -m venv .venv  # creates a virtual environment inside OML2023 using the module venv with name '.venv'
```

The environment will host all the different pip packages, and has the same python version as `python3`. It can be activated with

```bash
$ source .venv/bin/activate
```

and deactivate with 

```bash
(.venv)$ deactivate
```

More informations can be found [here](https://python.land/virtual-environments/virtualenv#How_to_create_a_Python_venv).

#### Installing the required packages

Once the virtual environment is created and activated (`source .venv/bin/activate` inside the `OML2023` directory), install the required packages with 

```bash
(.venv)$ pip install -r requirements.txt
```

This will install all the dependencies required to the environment.
(**WARNING**: if you had `jupyter` or `ipython` previously in your `PATH`, you should
`deactivate && source .venv/bin/activate` again to update it. Check you are
using the  environment binaries with `which jupyter` etc.).

#### Add the kernel to Jupyter

The kernel for the newly created local environment has to be added to the ones
Jupyter can access to, like so:

```bash
(.venv)$ ipython kernel install --name "OML23" --user
```
The name is simply to identify it later, you can choose any of your liking. The `--user` option make the install only for the current user (no `sudo` necessary).

Then, the kernel should be accessible through Jupyter.

### Jupyter 

The `jupyter` package (inside the requirements) can be called with

```bash
(.venv)$ jupyter notebook
```

to open a browser and select a notebook. Select the notebook `ex00/ex00.ipynb`. 


<!--## Numpy Tour-->

<!--Once Python and Numpy installed in the virtual environment, carry on with `ex00.pdf` to get used to the Numpy library (if you're not already).-->



<!--### Julia-->
