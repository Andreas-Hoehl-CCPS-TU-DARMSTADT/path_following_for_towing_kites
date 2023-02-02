# Trajectory Tracking and Path Following MPC for Towing Kites

This project contains the implementation of the simulations presented in my paper: **Paper Title**. available at: _not
yet published_.
Please have a look at the paper and cite it if you use this or some parts of this project in your own work.

## Overview

It consists of the following main parts:

* Calculation of optimal trajectories for a towing kite system.
* Implementation of Trajectory Tracking MPC
* Implementation of Path Following MPC
* Implementation of an offline learned Hybrid Model
* Simulation and Control Framework
* Quick Start Setup
* Examples

## Installation and Setup

To run this project you need nothing but a python 3.8 environment and install all packages specified in _requirements.txt_

Once you have that the first thing you should do is run _setup.py_.
This will create a folder named _data_ in your project root with the following structure:

    .
    ├── optimal_trajectories        # Containts the optimal trajectories
    ├── residual_models             # Containts the trained residual model with training history
    ├── simulation                  # All closed-loop simulations are stored here
    ├── pgf_data                    # Contains the data needed to create pgf plots in latex
    ├── test_data                   # Contains the data to test the residual model
    └── training_data               # Contains the data to train the residual model

It also creates all the components, such as the optimal trajectories, that are needed for further experiments.
Since it also generates the training data and then trains the residual model, this can take a few minutes.
After the script is finished, you can go through all the created folders and examine the files created in them.

## Usage

A good place to start (after the initial setup!) is to run some scripts in the _experiments_ folder.
Each script has its own brief documentation and should give an idea of how to create your own experiments.

However, the three most important use-cases are probably:

* Running a closed-loop simulation
    * have a look at _simulate.py_ and look at it documentation (a brief look at _interfaces.py_ to understand the type
      of the parameter settings also cannot hurt)
* inspecting a simulation result
    * have a look at _view_simulation.py_
* Calculating an optimal trajectory
    * have a look at _optimal_trajectory/calculate_optimal_trajectory.py_

After that feel free to experiment. You could for example change the plant parameter in _model_zoo.py_ and see what
happens.
