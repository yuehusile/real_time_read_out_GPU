# FKLab Data Analysis in Python #

### What is this repository for? ###

* This repository contains a collection of utilities and analysis tools that are used in the FKLab at NERF. 
* Tools that are commonly used by several people would end up in this repository, whereas experiment-specific scripts or functions generally should go in a separate user repository.

## How do I get set up? ##

### Installing python ###
* Before using any python tools for data analysis, you will have to install a python distribution on your computer. I strongly recommend that you install the [Anaconda distribution](http://continuum.io/downloads). Just follow the installation instruction and make sure to select the option to add the Anaconda directory to your bash shell PATH environment variable.
* You should occasionally update the Anaconda distribution or individual python modules, which you can do with the [conda command](http://conda.pydata.org/docs/faq.html#managing-packages).

### Prerequisites ###
 * For multi-taper spectral analysis the *spectrum* module is required. Install using `pip install spectrum`.

### Using python ###
* It is preferable to use the ipython interface to python, rather than the default python shell (check out the additional [features](http://ipython.org/ipython-doc/stable/interactive/tutorial.html)).
* Tip: add a python script to the $HOME/.ipython/profile_default/startup folder with commands that you want to be executed at the beginning of every session.
* In the process of data analysis or writing new data analysis tools, one generally works side-by-side with ipython (to execute commands) and a text editor (to write new scripts or functions). Whereas I prefer to use ipython in the terminal in combination with a simple text editor (e.g. geany), others prefer an integrated development environment (IDE) similar to Matlab (e.g. [spyder](https://pythonhosted.org/spyder/), which is included in the Anaconda distribution). A third option is to use the [ipython notebook](http://ipython.org/ipython-doc/stable/notebook/notebook.html) in the browser.

### Getting the FKLab python tools for users (not developers) ###
* If you do not intend to contribute to the development of the tools in this repository, then you can simply download a [ZIP](https://bitbucket.org/kloostermannerflab/dataanalysispython/get/master.zip) file and extract the file somewhere on your system.

### Getting the FKLab python tools for users + developers ###
* If you will both use the existing tools and develop new tools, then you should make a local clone of the repository.
* First, make sure you have git installed (run `sudo apt-get install git` in a terminal). Perform a one-time configuration of git by setting your name and email address: `git config --global user.email your.email@address` and `git config --global user.name "Your Name"`.
* Next, create a dedicated development folder in your home folder and clone the repository by following this [link](https://bitbucket.org/kloostermannerflab/dataanalysispython/downloads#clone) and executing the provided command. Make sure to select the HTTPS option and not the SSH option.

### Make code accessible to python ###
* To be able to import modules from this repository into python, you will have to add the folder to the python search path. For this, add the following two lines to $HOME/.profile:

```
#!bash

PYTHONPATH="$HOME/Development/git/dataanalysispython"
export PYTHONPATH

```

### Documentation ###

* [Documentation](http://kloostermannerflab.bitbucket.io) consists of an automatically created reference based on the comments in the code and a set of ipython notebooks that demonstrate the use of several tools in the repository.

## Contribution guidelines ##

### Git workflow ###

* Familiarize yourself with [git concepts and the basic git commands](https://www.atlassian.com/git/tutorials/setting-up-a-repository) before continuing.
* In particular, familiarize yourself with the [branching workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).
* We use a simple [workflow](https://guides.github.com/introduction/flow/). The idea is that there is a single [central repository](https://bitbucket.org/kloostermannerflab/dataanalysispython) to which all developers have read/write access.
* The **master** branch is a clean branch that you **should never** commit changes to directly. The central repository has been configured such that you cannot push changes to the master branch by accident.
* For every branch that you will push to the central repository, you have to open an [issue](https://bitbucket.org/kloostermannerflab/dataanalysispython/issues?status=new&status=open) on Bitbucket. The name of the branch should follow the following format: issue##-<initials>-short-description, where ## should be replaced by the issue number and <initials> are a unique identifier (in capitals) of the person who created the branch. For example: issue34-FK-add-decoding-tools.
* As soon as you would like help with or feedback on new code in a branch, or you think the new code is ready to be merged into the master branch, then you open a [pull request](https://bitbucket.org/kloostermannerflab/dataanalysispython/pull-requests/) (PR) on Bitbucket. Other lab members can then review your code, give comments and suggestions, before it the PR is accepted.
* TODO: describe workflow in more detail

### Writing good code ###

* We mainly follow the [PEP8](http://www.python.org/dev/peps/pep-0008/) style guide.
* Also, try to adhere to the [Zen of Python](https://www.python.org/dev/peps/pep-0020/).
* TODO: more guidelines

### Documenting your code ###

* All function that are exposed to the end-user should be documented properly. For the most part we follow the documentation style of the numpy module as described [here](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
* Please take a look at existing documented functions in the repository for examples.

## Who do I talk to? ##

* Fabian Kloosterman