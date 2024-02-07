import os

from simulation import run_simulation
from parameters import parse_params
from visualizations.visualization import plot_figures

from calibration import train_student_governance_game

def setup_to_file(args):
    '''
        Write the setup of the game to a txt file
    '''
    # game parameters
    f = open(args.save_path+args.experiment_name+"/setup.txt", "w")
    f.write("Dataset: "+str(args.dataset_list)+"\n")
    f.write("Number of Rounds: "+ str(args.num_rounds)+"\n")
    f.write("Priority: "+str(args.priority)+"\n")
    f.write("Initial Privacy Budget: "+ str(args.init_priv)+ "\n")
    f.write("Initial Fairness Constraint: "+ str(args.init_fair)+ "\n")
    f.write("Step Size Privacy: "+ str(args.step_size_priv)+"\n")
    f.write("Step Size Fairness: "+ str(args.step_size_fair)+"\n")
    f.write("Model Builder Lambda: "+str(args.builder_lambda)+"\n")
    f.write("Step Decay Factor: "+str(args.step_size_decay)+"\n")
    f.write("Model Builder lambda_priv: "+str(args.lambda_priv)+"\n")
    f.write("Model Builder lambda_fair: "+str(args.lambda_fair)+"\n")
    f.write("Privacy Regulator Penalty Scalar: "+str(args.C_priv)+"\n")
    f.write("Fairness Regulator Penalty Scalar: "+str(args.C_fair)+"\n")

    f.close()
    

def save_results(args, results_df):
    '''
        Create a new folder for this experiment and save all the relevant information and results in there
    '''
    save_folder_path = args.save_path+args.experiment_name
    
    # write the setup of the experiment
    setup_to_file(args)
    
    # write the game results
    results_df.to_parquet(args.save_path+args.experiment_name+'/df.parquet.gzip', compression='gzip') 
    
    # generate and save the visualizations
    two_datasets = False
    if args.regulator_dataset:
        two_datasets = True
    plot_figures(results_df, save_folder_path, args.algorithm, two_datasets=two_datasets)
        
        
def main(args):
    # create a folder for this experiment
    save_folder_path = args.save_path+args.experiment_name
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        
    # run the game simulation
    results_df = run_simulation(args)
    
    # save the results
    if args.save:
        save_results(args, results_df)


def test(args, params):
    train_student_governance_game(args, params)
    
    
if __name__ == "__main__":
    args = parse_params()

    main(args)
    #test(args, [3, 0.01])