# Regulation Games for Trustworthy Machine Learning
## Repo Structure
`main.py` file contains the starting point. The parameters for the program
are in `parameters.py` file.
The rest of the content is divided into four parts:
-  Code to visualize Pareto frontier using pre-computed results and generate agents' loss function arrays
-  Code to run ParetoPlay game simulations
-  Code to train a model during calibration using an algorithm
-  Result visulization tools

## Part 1: Pre-calculated Pareto Frontier
Pre-calculated results are saved in `previous_results\loss_functions\{dataset}\{method}\` where dataset is the dataset used and method is the algorithm used. 
There is a sample csv in `previous_results\loss_functions\utkface\fairPATE\` that shows the format of the result csv file. Note that the results provided there are example only and should not be used to run game simulations. 
`previous_results\loss_functions\pf_surface_plot.ipynb` generates a visualization of the Pareto frontier using the results csv file.
`previous_results\loss_functions\loss_functions_helper.ipynb` takes in the results and generates loss function arrays that the game agents use during the game simulation.

## Part 2: ParetoPlay Game Simulations
`simulation.py` stores functions that runs a game simulation. `agents.py` store game agents classes. ParetoPlay parameters are defined in `parameters.py`. 

Sample bash code for running the scripts can be found in the `experiments\`.

```
EROOT= [path that points to the directory] \
RROOT= [path that points to where intermediate models should be stored] \

poetry run python $EROOT/main.py \
    --experiment_name [Name of the experiment for creating an experiment folder] \
    --prev_results_dir $EROOT/previous_results/ \
    --save_path $EROOT/results/ \
    --path $RROOT/ \
    --data_dir [path that points to the dataset] \
    --num_rounds [Number of game rounds] \
    --init_priv [Initial privacy specification] \
    --init_fair [Initial fairness specification] \
    --dataset [Dataset name] \
    --priority [Agent that moves first] \
    --step_size_priv [Privacy parameter step size] \
    --step_size_fair [Fairness parameter step size] \
    --step_size_decay [Step size decay factor] \
    --builder_lambda [Model builder lambda when weighing its own loss. Used in fairPATE to weight accuracy vs coverage] \
    --lambda_priv [Lambda priavcy] \
    --lambda_fair [Lambda fairness] \
    --C_priv [Privacy penalty factor] \
    --C_fair [Fairness penalty factor] \
    --goal_priv [Privacy regulator's constraint] \
    --goal_fair [Fairness regulator's constraint] \
    --calibration [Whether to include calibration round]
```

## Part 3: Calibration
`calibration.py` contains functions that train one model instance using the specified algorithm during the calibration step. 
The code belonging to specified algorithms is stored under `baselines/`. 
Current version of the code supports FairPATE. 

## Part 4: Result Visualization
`visualizations/` contains tools for graphing the game simulation results. 
`visualization/visualization.py` has functions that are automatically called (unless specified otherwise) at the end of each game simulation to graph losses and parameters trajectories of that particular game. The figures are stored with the game results.
`visualization/graphing_notebook.py` contains more functions to graph results in different styles and aggregate results from multiple simulations with different starting points. 


## Environment 
This project uses [Poetry](https://python-poetry.org) for dependency management. To start developing, first [install `poetry`](https://python-poetry.org/docs/master/#installing-with-the-official-installer) then:

```
>> cd .
>> poetry install
```

Poetry will install all dependencies and create an appropriate virtual environment for the project. To get started with a notebook server: 

```
>> poetry run jupyter lab
```

Otherwise, to run python scripts in the environment:
```
>> poetry run python <script>.py
```

For example:
```
>> poetry run bash experiments/utkface_experiment.sh
```

To get the location of the environment:
```
>> poetry run which python
```

### Running Tests
```
poetry run python -m  pytest tests/test_jax_nonjax.py --rootdir="."
```
