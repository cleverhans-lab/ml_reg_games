import os
import pandas as pd
import numpy as np

from agents import GameRunner, ModelBuilder, Regulator, Interaction
from parameters import set_dataset


def init_game(args):
    '''
        Initialize the game including the agents and relevant variables
    '''
    # load the table format loss functions
    print("Loading the losses", flush=True)
    loss_dir = args.prev_results_dir+"loss_functions/"+args.dataset+"/"+args.algorithm
    loss_builder_acc = np.load(loss_dir+'/builder_loss_acc.npy')  
    loss_privacy = np.load(loss_dir+'/privacy_loss.npy')
    loss_fairness = np.load(loss_dir+'/fairness_loss.npy')
    priv_fair_values = np.load(loss_dir+'/priv_fair_values.npy')
    
    # also load the corresponding input epsilon and fairness parameter
    priv_values = priv_fair_values[:,0]
    fair_values = priv_fair_values[:,1]
    
    
    # other methods may not have coverage
    if args.algorithm == 'fairPATE':
        loss_builder_cov = np.load(loss_dir+'/builder_loss_cov.npy')
        losses = np.squeeze(np.stack((-1 * loss_builder_acc, loss_privacy, loss_fairness, -1 * loss_builder_cov), axis=-1))
    else:
        losses = np.squeeze(np.stack((-1 * loss_builder_acc, loss_privacy, loss_fairness), axis=-1))

    # initialize the agents without the loss functions. those are distrubuted later
    model_builder = ModelBuilder(args)
    regulators = Regulator(args)
    # game runner takes in the cost function (outputs) and the input parameters
    game_runner = GameRunner(args, 
                             losses,
                             priv_values, 
                             fair_values,
                             [model_builder, regulators],
                             args.calibration)
    
    # if we are using different datasets for regulators and builder, load those as well
    if args.regulator_dataset:
        loss_dir = args.prev_results_dir+"loss_functions/"+args.regulator_dataset+"/"+args.algorithm
        loss_builder_acc_2 = np.load(loss_dir+'/builder_loss_acc.npy')  
        loss_privacy_2 = np.load(loss_dir+'/privacy_loss.npy')
        loss_fairness_2 = np.load(loss_dir+'/fairness_loss.npy')
        priv_fair_values_2 = np.load(loss_dir+'/priv_fair_values.npy')
        
        # also load the corresponding input epsilon and gamma
        priv_values_2 = priv_fair_values_2[:,0]
        fair_values_2 = priv_fair_values_2[:,1]
        # so far this is only implemented for fairPATE
        loss_builder_cov_2 = np.load(loss_dir+'/builder_loss_cov.npy')
        losses_2 = np.squeeze(np.stack((-1 * loss_builder_acc_2, loss_privacy_2, loss_fairness_2, -1 * loss_builder_cov_2), axis=-1))

        # store these in game runner
        game_runner.set_to_two_datasets(losses_2, priv_values_2, fair_values_2)
    
    return game_runner, model_builder, regulators
    

def run_game(args, game_runner, model_builder, regulators):
    '''
        Run an instance of ParetoPlay game
    '''
    def log_init(game_runner, model_builder, regulators, curr_param, C_priv, C_fair):
        # initialize an interaction for the starting point
        inter_init = Interaction(round=0)
        inter_init.round_params = curr_param
        # get the losses at the current parameters
        loss_combined, loss_b, loss_p, loss_f, acc, cov = model_builder.get_losses(curr_param, C_priv, C_fair)
        if args.algorithm == 'dpsgd-g-a':
            # dpsgd-g-a's tau does not correspond directly to fairness gap
            args.goal_fair = loss_f
            regulators.update_goal_fair(loss_f)
    
        inter_init.losses = [[loss_combined, loss_b, loss_p, loss_f, acc, cov]]
        # if using two datasets, then also get the losses of the regulator's dataset
        if args.regulator_dataset:
            loss_combined, loss_b, loss_p, loss_f, acc, cov = regulators.get_losses(curr_param, C_priv, C_fair)
            # append these losses to inter as well
            inter_init.losses.append([loss_combined, loss_b, loss_p, loss_f, acc, cov])
        # log the round
        game_runner.register_interaction(inter_init)
        game_runner.results_to_df()
        
        return loss_combined, loss_b, loss_p, loss_f, acc, cov
    
    
    # initiliaze current parameters if they are provided initially
    if args.init_priv and args.init_fair: 
        curr_param = [args.init_priv, args.init_fair]
    else:
        curr_param = None
    C_priv = args.C_priv
    C_fair = args.C_fair

    # initialize results file path
    file_path = args.save_path+args.experiment_name+'/df.parquet.gzip'
    # check if the file already exist
    if not os.path.exists(file_path):
        # initialize new game if not
        curr_round = 1
    else:
        # read the latest result if yes
        df_inter_results = pd.read_parquet(file_path)
        df_last = df_inter_results.tail(1)
        curr_round = int(df_last['round']) + 1
        # game runner updates parameters
        game_runner.sync(int(df_last['round']) + 1, df_inter_results)
        if args.algorithm == 'fairPATE':
            curr_param = [float(df_last['epsilon']), float(df_last['gamma'])]
        elif args.algorithm == 'dpsgd-g-a':
            curr_param = [float(df_last['epsilon']), float(df_last['tau'])]
        else:
            raise Exception("Algorithm not implemented!")

        model_builder.update_step_size(curr_round)

    # start the game (or continue)
    for i in range(curr_round, args.num_rounds+1):
        print("Game round "+str(i)+"---------------------------------------------")
        # game runner takes the PF and distributes the loss functions
        if args.calibration or i == 1:
            # if there is no calibration then game runner only needs to distribute the loss functions once
            game_runner.distribute_losses()
        # initialize a variable to track the current interaction
        inter = Interaction(round=i)
        # if regulators move first
        if args.priority == 'regulators':
            # check if this is the first round
            if i == 1:
                # pick the initial parameters in the first round if it was not provided
                if not curr_param:
                    curr_param = regulators.regulators_starting_point()
                    # update the goal parameters as well
                    args.goal_priv, args.goal_fair = curr_param
                print("Initial parameters chosen by regulators: "+str(curr_param), flush=True)
                
                # log the initial parameters and losses
                log_init(game_runner, model_builder, regulators, curr_param, C_priv, C_fair)

            # if regulators take a step
            priv_step, fair_step = regulators.best_response(curr_param)
            # model builder takes a step
            curr_param, loss_combined, loss_b, loss_p, loss_f, acc, cov = model_builder.best_response(curr_param, C_priv, C_fair, priv_step, fair_step)
           
        else:
            # check if this is the first round
            if i == 1:
                if not curr_param:
                    # if chosen by model builder then these are not the goal parameters
                    curr_param = model_builder.choose_starting_point()
                print("Initial parameters chosen by model builder: "+str(curr_param), flush=True)
                # log the initial params and losses
                loss_combined, loss_b, loss_p, loss_f, acc, cov = log_init(game_runner, model_builder, regulators, curr_param, C_priv, C_fair)
            else:
                # if regulators take a step
                priv_step, fair_step = regulators.best_response(curr_param)
                # model builder takes a step
                curr_param, loss_combined, loss_b, loss_p, loss_f, acc, cov = model_builder.best_response(curr_param, C_priv, C_fair, priv_step, fair_step)
           
        # update interaction
        inter.round_params = curr_param
        inter.losses = [[loss_combined, loss_b, loss_p, loss_f, acc, cov]]
        # if using two datasets, then also get the losses of the regulator's dataset
        if args.regulator_dataset:
            loss_combined, loss_b, loss_p, loss_f, acc, cov = regulators.get_losses(curr_param, C_priv, C_fair)
            # append these losses to inter as well
            inter.losses.append([loss_combined, loss_b, loss_p, loss_f, acc, cov])
        
        # log the round
        game_runner.register_interaction(inter)
        # write to df
        game_runner.results_to_df()
        
        # calibration round
        if args.calibration:
            calibration_results = game_runner.train_calibration_model(curr_param)
            game_runner.update_losses(calibration_results)
            game_runner.calibration_to_df(calibration_results)
        
        print(inter, flush=True)
        
        # save now in case of preemption, writing one round at a time
        if args.save:
            results_df = game_runner.return_results_df()
            results_df.to_parquet(args.save_path+args.experiment_name+'/df.parquet.gzip', compression='gzip')     

    return game_runner.return_results_df()


def run_simulation(args):
    # initialize the game
    print("Initializing the game", flush=True)
    # initialize all agents and game runner
    game_runner, model_builder, regulators = init_game(args)
    print("Running the game", flush=True)
    return run_game(args, game_runner, model_builder, regulators)
    
