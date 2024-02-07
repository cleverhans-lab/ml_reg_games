import seaborn as sns
import matplotlib.pyplot as plt


def plot_figures(results_df, save_folder_path, algorithm='fairPATE', two_datasets=False):
    '''
        Plot all the relevant figures using the experiment results
    '''
    results_df["loss_build"] = 100 + results_df["loss_build"]
    plot_losses(results_df, save_folder_path, two_datasets)
    plot_parameters(results_df, save_folder_path, algorithm, two_datasets)
    
    
def plot_losses(results_df, save_folder_path, two_datasets):
    '''
        Plot and save the losses vs time
    '''
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    results_df = results_df[results_df['agent'] != 'calibration']
    if two_datasets:
        sns.lineplot(data=results_df, x="round", y="accuracy", hue='dataset', ax=axes[0])
        sns.lineplot(data=results_df, x="round", y="coverage", hue='dataset', ax=axes[1])
        sns.lineplot(data=results_df, x="round", y="privacy cost", hue='dataset', ax=axes[2])
        sns.lineplot(data=results_df, x="round", y="max fairness gap", hue='dataset', ax=axes[3])
    else:
        sns.lineplot(data=results_df, x="round", y="accuracy", ax=axes[0])
        sns.lineplot(data=results_df, x="round", y="coverage", ax=axes[1])
        sns.lineplot(data=results_df, x="round", y="privacy cost", ax=axes[2])
        sns.lineplot(data=results_df, x="round", y="max fairness gap", ax=axes[3])
    plt.savefig(save_folder_path+"/losses.pdf")
    

def plot_parameters(results_df, save_folder_path, algorithm='fairPATE', two_datasets=False):
    '''
        Plot and save the parameters vs time
    '''
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    results_df = results_df[results_df['agent'] != 'calibration']
    if two_datasets:
        sns.lineplot(data=results_df, x="round", y="epsilon", hue='dataset', ax=axes[0])
    else:
        sns.lineplot(data=results_df, x="round", y="epsilon", ax=axes[0])
    if algorithm == 'fairPATE':
        if two_datasets:
            sns.lineplot(data=results_df, x="round", y="gamma", hue='dataset', ax=axes[1])
        else:
            sns.lineplot(data=results_df, x="round", y="gamma", ax=axes[1])
    elif algorithm == 'dpsgd-g-a':
        sns.lineplot(data=results_df, x="round", y="tau", ax=axes[1])
    plt.savefig(save_folder_path+"/parameters.pdf")
    
      