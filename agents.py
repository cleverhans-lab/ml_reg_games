from dataclasses import dataclass

import numpy as np
import pandas as pd
import math
from scipy.interpolate import griddata
from calibration import train_student_governance_game
from parameters import set_dataset



@dataclass
class Interaction():
    round: int
    round_params: float=0, 
    losses: float=0


@dataclass
class Metric():
    epsilon: float
    gamma: float
    achieved_epsilon: float
    achieved_gamma: float
    accuracy: float
    coverage: float
        

class Agent():
    def __init__(self, args, name: str) -> None:
        self.args = args
        self.name = name
        self.algorithm = args.algorithm
        self.loss_b, self.loss_priv, self.loss_fair, self.priv_values, self.fair_values = None, None, None, None, None
        self.loss_b_inter, self.loss_priv_inter, self.loss_fair_inter, self.priv_values_inter, self.fair_values_inter = None, None, None, None, None
        self.achieved_priv, self.achieved_fair = None, None
        self.acc, self.cov = None, None
        self.acc_inter, self.cov_inter = None, None
        # builder lambdas for the regulators
        self.lambda_priv = args.lambda_priv
        self.lambda_fair = args.lambda_fair
        # in certain scenarios multiply the step size by a factor beforehand to ensure all agents have the same step size in the same round
        self.step_size = np.array([args.step_size_priv, args.step_size_fair])
        if args.priority == "model_builder" and name == "model_builder":  
            self.step_size = self.step_size * 1/self.args.step_size_decay
    
    
    def update_achieved(self, achieved_priv, achieved_fair, priv_values, fair_values):
        '''
            Update the achieved values
            Doing this because loss_priv and loss_fair does not preserve the actual values
        '''
        achieved, _, _ = self.interpolate_losses([achieved_priv, achieved_fair], priv_values, fair_values)
        self.achieved_priv = achieved[0]
        self.achieved_fair = achieved[1]

        
    def update_loss(self, loss_b, loss_priv, loss_fair, priv_values, fair_values, acc, cov=None):
        '''
            Update the loss functions
        '''
        # these are in table format
        self.loss_b =loss_b
        self.loss_priv =loss_priv
        self.loss_fair =loss_fair
        self.priv_values = priv_values
        self.fair_values = fair_values
        self.acc, self.cov = acc, cov
        
        
        if self.algorithm == 'fairPATE':
            losses_inter, self.priv_values_inter, self.fair_values_inter = self.interpolate_losses([loss_b, loss_priv, loss_fair, acc, cov], priv_values, fair_values)
            self.cov_inter = losses_inter[4]
        else:
            losses_inter, self.priv_values_inter, self.fair_values_inter = self.interpolate_losses([loss_b, loss_priv, loss_fair, acc], priv_values, fair_values)
        
        
        # these are the losses
        self.loss_b_inter = losses_inter[0]
        self.loss_priv_inter = losses_inter[1]
        self.loss_fair_inter = losses_inter[2]
        self.acc_inter = losses_inter[3]

    
    def interpolate_losses(self, losses, priv_values, fair_values):
        '''
            Interpolate the losses into a grid format
            :param priv_values: epsilon of points
            :param fair_values: gamma of points
            :param loss: loss value specific to each agent
        '''
        x = priv_values
        y = fair_values
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X,Y = np.meshgrid(xi,yi)
        losses_inter = []
        for z in losses:
            losses_inter.append(griddata((x,y),z,(X,Y), method='linear'))
        
        return losses_inter, xi, yi
        
        
    def best_response(self, curr_param, C_priv, C_fair, priv_step, fair_step):
        '''
            Calculates the gradient at a particular point using the loss array and
            takes a step according to the step size
            :param curr_param: current parameter. Calculates the gradient at this point
            :param C_priv: privacy regulator's scalar multiplier
            :param C_fair: fairness regulator's scalar multiplier
            :param priv_step: privacy regulator's gradient multiplied by C_priv
            :param fair_step: fairness regulator's gradient multiplied by C_fair
            :return: the updated params, builder loss, priv loss, fair loss, acc, cov
        '''
        
        def closest_point(target, points):
            array = np.asarray(points)
            return  (np.abs(array - target)).argmin()
        
        # interpolate the parameters
        interpolated_priv, interpolated_fair = self.priv_values_inter, self.fair_values_inter
        # interpolate the losses
        interpolated_loss_b, interpolated_loss_priv, interpolated_loss_fair = self.loss_b_inter, self.loss_priv_inter, self.loss_fair_inter
        
        # calculate builder's gradient
        d_priv = self.priv_values_inter[1]-self.priv_values_inter[0]
        d_fair = self.fair_values_inter[1]-self.fair_values_inter[0]
        grad_builder = np.gradient(interpolated_loss_b, d_fair, d_priv)
        
        # find the index of the closest value
        # Note: this we are doing separately for builder because if they use different dataset the closest point can be different
        index_priv = closest_point(curr_param[0], interpolated_priv)
        index_fair = closest_point(curr_param[1], interpolated_fair)

        # get the gradient at the current param
        if self.algorithm == 'fairPATE':
            # regulators only change their own parameter
            curr_grad_priv = grad_builder[1][index_fair, index_priv] + self.lambda_priv * priv_step[1]
            curr_grad_fair = grad_builder[0][index_fair, index_priv] + self.lambda_fair * fair_step[0]
            
        elif self.algorithm == 'dpsgd-g-a':
            # regulators change both so can use the aggregate
            curr_grad_priv = grad_builder[1][index_fair, index_priv] + self.lambda_priv * priv_step[1] + self.lambda_fair * fair_step[1]
            curr_grad_fair = grad_builder[0][index_fair, index_priv] + self.lambda_priv * priv_step[0] + self.lambda_fair * fair_step[0]
        
        # breakpoint()
        # take a step
        step = np.multiply([curr_grad_priv, curr_grad_fair], self.step_size)

        # step size decay
        self.step_size = self.step_size * 1/self.args.step_size_decay 

        # check for nan
        if math.isnan(step[0]):
            step[0] = 0
        if math.isnan(step[1]):
            step[1] = 0
            
        # check if the params will be out of bound
        new_param = curr_param - step
        new_param[0] = max(self.priv_values.min(), new_param[0])
        new_param[1] = max(self.fair_values.min(), new_param[1])
        if self.algorithm == 'dpsgd-g-a':
            # largest tau value reported in their paper is 1, so will upper bound it at 1
            new_param[1] = min(1, new_param[1])
        
        # find the index of the closest value of the NEW param
        index_priv = closest_point(new_param[0], interpolated_priv)
        index_fair = closest_point(new_param[1], interpolated_fair)
        
        # get the losses
        curr_loss_b = interpolated_loss_b[index_fair, index_priv]
        curr_achieved_p = self.achieved_priv[index_fair, index_priv] 
        curr_achieved_f = self.achieved_fair[index_fair, index_priv]
        curr_loss_p = interpolated_loss_priv[index_fair, index_priv]
        curr_loss_f = interpolated_loss_fair[index_fair, index_priv]
        curr_loss_combined = curr_loss_b + self.lambda_priv * C_priv * curr_loss_p + self.lambda_fair * C_fair * curr_loss_f

        # set coverage if applicable
        if self.algorithm == 'fairPATE':
            curr_cov = self.cov_inter[index_fair, index_priv]
        else:
            curr_cov = 1
        # breakpoint()
        return new_param, curr_loss_combined, curr_loss_b, curr_achieved_p, curr_achieved_f, self.acc_inter[index_fair, index_priv], curr_cov

    
    def get_losses(self, curr_param, C_priv, C_fair):
        def closest_point(target, points):
            array = np.asarray(points)
            return  (np.abs(array - target)).argmin()
        
        interpolated_loss_b, interpolated_priv, interpolated_fair = self.loss_b_inter, self.priv_values_inter, self.fair_values_inter
        interpolated_loss_priv, interpolated_loss_fair = self.loss_priv_inter, self.loss_fair_inter
        
        # find the index of the closest value
        index_priv = closest_point(curr_param[0], interpolated_priv)
        index_fair = closest_point(curr_param[1], interpolated_fair)
        
        # get the losses
        curr_loss_b = interpolated_loss_b[index_fair, index_priv]
        curr_achieved_p = self.achieved_priv[index_fair, index_priv]
        curr_achieved_f = self.achieved_fair[index_fair, index_priv]
        curr_loss_p = interpolated_loss_priv[index_fair, index_priv]
        curr_loss_f = interpolated_loss_fair[index_fair, index_priv]
        curr_loss_combined = curr_loss_b + self.lambda_priv * C_priv * curr_loss_p + self.lambda_fair * C_fair * curr_loss_f
        
        if self.algorithm == 'fairPATE':
            curr_cov = self.cov_inter[index_fair, index_priv]
        else:
            curr_cov = 1
        
        return curr_loss_combined, curr_loss_b, curr_achieved_p, curr_achieved_f, self.acc_inter[index_fair, index_priv], curr_cov

        
    def update_step_size(self, round):
        '''
            updates the step size to the correct one after restarting after preemption
        '''
        self.step_size = np.array([self.args.step_size_priv, self.args.step_size_fair])
        self.step_size = self.step_size * (1/self.args.step_size_decay)**round
    
    
class ModelBuilder(Agent):
    def __init__(self, args) -> None:
        super().__init__(args, "model_builder")
        # lambda used in the weighting of coverage and accuracy
        self.builder_lambda = args.builder_lambda
    
    def choose_starting_point(self):
        '''
            Choose a starting point that has the lowest loss
        '''
        min_index = np.argmin(self.losses)
        
        return [self.priv_values[min_index], self.fair_values[min_index]]
    
    
class Regulator(Agent):
    def __init__(self, args) -> None:
        super().__init__(args, "regulators")
        self.C_priv = args.C_priv
        self.goal_priv = args.goal_priv
        self.C_fair = args.C_fair
        self.goal_fair = args.goal_fair
        
    def update_goal_fair(self, new_goal_fair):
        self.goal_fair = new_goal_fair
        
        
    def regulators_starting_point(self):
        '''
            Fairness and privacy regulators choose the starting point of the game jointly
        '''
        priv_losses = self.losses[self.pf_indices][:,1]
        fair_losses = self.losses[self.pf_indices][:,2]
        combined_losses = self.args.regulators_lambda * np.log(priv_losses/min(priv_losses)) + (1-self.args.regulators_lambda) * (fair_losses-min(fair_losses))
        
        pf_priv = self.priv_values[self.pf_indices]
        pf_fair = self.fair_values[self.pf_indices]
        
        min_index = np.argmin(combined_losses)
        
        return [pf_priv[min_index], pf_fair[min_index]]
        
        
    def best_response(self, curr_param):
        '''
            Calculates the gradient at a particular point using the loss array and
            takes a step according to the step size
            :param curr_param: current parameter. Calculates the gradient at this point
            :return: priv_step: gradient of privacy regulator at current point
            :return: fair_step: gradient of fairness regulator at current point
        '''
        
        def closest_point(target, points):
            array = np.asarray(points)
            return  (np.abs(array - target)).argmin()
        
        interpolated_priv, interpolated_fair = self.priv_values_inter, self.fair_values_inter
        interpolated_loss_priv, interpolated_loss_fair = self.loss_priv_inter, self.loss_fair_inter

        l_priv = self.C_priv * interpolated_loss_priv
        l_fair = self.C_fair * interpolated_loss_fair
        d_priv = self.priv_values_inter[1]-self.priv_values_inter[0]
        d_fair = self.fair_values_inter[1]-self.fair_values_inter[0]
        grad_priv = np.gradient(l_priv, d_fair, d_priv)
        grad_fair = np.gradient(l_fair, d_fair, d_priv)

        # find the index of the closest value
        index_priv = closest_point(curr_param[0], interpolated_priv)
        index_fair = closest_point(curr_param[1], interpolated_fair)
        
        # doing this to ensure that regulators' losses are 0 if constraints are satisified
        if curr_param[0] <= self.args.goal_priv:
            priv_step = (0, 0)
        else: 
            priv_step = (grad_priv[0][index_fair, index_priv], grad_priv[1][index_fair, index_priv])
        if curr_param[1] <= self.args.goal_fair:
            fair_step = (0, 0)
        else: 
            fair_step = (grad_fair[0][index_fair, index_priv], grad_fair[1][index_fair, index_priv])
        #breakpoint()
        return priv_step, fair_step


class GameRunner():
    """
    Entity to keep track of the game simulation
    """
    def __init__(self, args, losses, priv_values, fair_values, agents, calibration=True) -> None:
        self.args = args
        self.algorithm = args.algorithm
        self.datasets = args.dataset_list
        # all in table format
        self.losses, self.priv_values, self.fair_values= losses, priv_values, fair_values
        self.losses_reg, self.priv_values_reg, self.fair_values_reg= losses, priv_values, fair_values
        # losses:
        # acc = -acc
        # priv = priv
        # fair = fair
        # cov = -cov
        
        self.pf_indices = None
        self.agents = agents
        self.interaction_history = []
        self.results_df = None
        self.time = 0       # for logging results
        self.num_datasets = 1       # if regulators and builder use different datasets then 2
            
        if self.algorithm == 'fairPATE':
            self.fair_var = "gamma"
        else:
            self.fair_var = "tau"
            
    def set_to_two_datasets(self, losses, priv_values, fair_values):
        self.num_datasets = 2
        self.losses_reg, self.priv_values_reg, self.fair_values_reg= losses, priv_values, fair_values
    
    
    def is_pareto_efficient(self, costs, return_mask = False):
        """
            Find the pareto-efficient points
            :param costs: An (n_points, n_costs) array
            :param return_mask: True to return a mask
            :return: An array of indices of pareto-efficient points.
                If return_mask is True, this will be an (n_points, ) boolean array
                Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
    
    
    def interpolate_loss(self, losses, priv_values, fair_values):
        '''
            Interpolate the losses into a grid format
            :param priv_values: epsilon of points
            :param fair_values: gamma of points
            :param loss: loss value specific to each agent
        '''
        x = priv_values[self.pf_indices]
        y = fair_values[self.pf_indices]
        losses = losses[self.pf_indices, :]
        xi = np.linspace(x.min(), x.max(), 40)
        yi = np.linspace(y.min(), y.max(), 40)
        X,Y = np.meshgrid(xi,yi)
        interpolated_losses = []
        for i in range(4):
            z = losses[:, i].flatten()
            Z = griddata((x,y),z,(X,Y), method='linear')
            interpolated_losses.append(Z)
        return interpolated_losses, xi, yi
    
    
    def update_losses(self, results):
        '''
            Accept a result array of the new calibration model. 
            Update its cost function and input parameter list
        '''
        # game runner registers the new result
        result = results[0]
        if self.algorithm == 'fairPATE':
            self.losses = np.concatenate((self.losses, 
                                      np.array([[-1 * result['accuracy'],  
                                                result['achieved_epsilon'], 
                                                result['achieved_fairness_gaps'],
                                                -1 * result['coverage']]])), axis = 0)
            print(f"Dataset: {self.datasets[0]}; Accuracy :{result['accuracy']}; Achieved epsilon: {result['achieved_epsilon']}; Achieved fairness gap: {result['achieved_fairness_gaps']}; Coverage: {result['coverage']}", flush=True)

        elif self.algorithm == 'dpsgd-g-a':
            self.losses = np.concatenate((self.losses, 
                                      np.array([[-1 * result['accuracy'],  
                                                result['achieved_epsilon'], 
                                                result['achieved_fairness_gaps']]])), axis = 0)
            print(f"Dataset: {self.datasets[0]}; Accuracy :{result['accuracy']}; Achieved epsilon: {result['achieved_epsilon']}; Achieved fairness gap: {result['achieved_fairness_gaps']}", flush=True)
            
        self.priv_values = np.append(self.priv_values, result['epsilon'])
        self.fair_values = np.append(self.fair_values, result['fairness_gaps'])
        
        if self.num_datasets == 2:
            # only implemented for fairPATE currently
            result = results[1]
            self.losses_reg = np.concatenate((self.losses_reg, 
                                        np.array([[-1 * result['accuracy'],  
                                                    result['achieved_epsilon'], 
                                                    result['achieved_fairness_gaps'],
                                                    -1 * result['coverage']]])), axis = 0)
            print(f"Dataset: {self.datasets[1]}; Accuracy :{result['accuracy']}; Achieved epsilon: {result['achieved_epsilon']}; Achieved fairness gap: {result['achieved_fairness_gaps']}; Coverage: {result['coverage']}", flush=True)
            self.priv_values_reg = np.append(self.priv_values_reg, result['epsilon'])
            self.fair_values_reg = np.append(self.fair_values_reg, result['fairness_gaps'])
                                        
        
    def get_pf(self, losses, priv_values, fair_values):
            # select points on the PF
            pf_indices = self.is_pareto_efficient(losses)
            pf_losses = losses[pf_indices, :]
            pf_priv = priv_values[pf_indices]
            pf_fair = fair_values[pf_indices]
            
            return pf_losses, pf_priv, pf_fair, pf_indices
        
        
    def distribute_losses(self):
        '''
            Select points on the PF and distribute them to the agents
        '''
        
        pf_losses, pf_priv, pf_fair, pf_indices = self.get_pf(np.round(self.losses, decimals=2), self.priv_values, self.fair_values)
        self.pf_indices = pf_indices
        achieved_priv = pf_losses[:, 1]
        achieved_fair = pf_losses[:, 2]

        # follow the loss formulation to adjust
        # 0 if epsilon_w < epsilon, eps_w-eps otherwise
        loss_priv = np.maximum(0, pf_losses[:, 1] - self.args.goal_priv)
        # 0 if lambda_w < lambda, lambda_w < lambda otherwise
        loss_fair = np.maximum(0, pf_losses[:, 2] - self.args.goal_fair)

        # give the agents new losses
        self.agents[0].update_achieved(achieved_priv, achieved_fair, pf_priv, pf_fair)
        if self.algorithm == 'fairPATE':
            loss_builder_weighted = self.args.builder_lambda *0.01 * pf_losses[:, 0] + (1-self.args.builder_lambda) * pf_losses[:, 3]
            self.agents[0].update_loss(loss_builder_weighted, loss_priv, loss_fair, pf_priv, pf_fair, -1 * pf_losses[:, 0], -1 * pf_losses[:, 3])     # model builder
        else:
            loss_builder_weighted = 0.01 * pf_losses[:, 0]
            self.agents[0].update_loss(0.01 * pf_losses[:, 0], loss_priv, loss_fair, pf_priv, pf_fair, -1 * pf_losses[:, 0])     # model builder
        
        # give same losses to regulators
        if self.num_datasets == 1:
            if self.algorithm == 'fairPATE':
                self.agents[1].update_loss(loss_builder_weighted, loss_priv, loss_fair, pf_priv, pf_fair, -1 * pf_losses[:, 0], -1 * pf_losses[:, 3])     # model builder
            else:
                self.agents[1].update_loss(loss_builder_weighted, loss_priv, loss_fair, pf_priv, pf_fair, -1 * pf_losses[:, 0])
            self.agents[1].pf_indices = pf_indices
        # if two datasets, re-calculate pf points for second dataset and give losses to regulators
        elif self.num_datasets == 2:
            pf_losses, pf_priv, pf_fair, pf_indices = self.get_pf(self.losses_reg, self.priv_values_reg, self.fair_values_reg)
            achieved_priv = pf_losses[:, 1]
            achieved_fair = pf_losses[:, 2]
            # follow the loss formulation to adjust
            # 0 if epsilon_w < epsilon, eps_w-eps otherwise
            loss_priv = np.maximum(0, pf_losses[:, 1] - self.args.goal_priv)
            # 0 if lambda_w < lambda, lambda_w < lambda otherwise
            loss_fair = np.maximum(0, pf_losses[:, 2] - self.args.goal_fair)
            loss_builder_weighted = self.args.builder_lambda *0.01 * pf_losses[:, 0] + (1-self.args.builder_lambda) * pf_losses[:, 3]
            
            self.agents[1].update_achieved(achieved_priv, achieved_fair, pf_priv, pf_fair)
            self.agents[1].update_loss(loss_builder_weighted, loss_priv, loss_fair, pf_priv, pf_fair, -1 * pf_losses[:, 0], -1 * pf_losses[:, 3])     # model builder
            self.agents[1].pf_indices = pf_indices
    
        
    def register_interaction(self, interaction: Interaction):
        self.interaction_history.append(interaction)
        
        
    def results_to_df(self):
        '''
        Store the current round of game simulation results in a dataframe
        '''
        
        # game parameters
        if self.time == 0:
            self.results_df = pd.DataFrame(columns=["round", "dataset", "epsilon", "gamma", "agent", "loss_build_combined",
                                           "loss_build", "accuracy", "coverage", "privacy cost", "max fairness gap"])

        # get the lastest interactioin
        inter = self.interaction_history[-1]
        
        # doing this in a loop because two datasets experiments have two sets of losses
        for loss, dataset in zip(inter.losses, self.datasets):
            self.results_df = pd.concat([self.results_df, pd.DataFrame({'round': self.time, 
                                                'dataset': dataset,
                                                'epsilon': inter.round_params[0], 
                                                self.fair_var: inter.round_params[1], 
                                                'agent': 'build',
                                                'loss_build_combined': loss[0],
                                                'loss_build': loss[1], 
                                                "accuracy": loss[4], 
                                                "coverage": loss[5], 
                                                "privacy cost": loss[2], 
                                                "max fairness gap": loss[3]}, index=[0])], ignore_index=True)
        self.time += 1
            
            
    def calibration_to_df(self, results_all):
        '''
            Write the student model results to df
        '''
        # doing this in a loop because two datasets experiments have two sets of losses
        for results, dataset in zip(results_all, self.datasets):
            loss_builder_weighted = -1 * (self.args.builder_lambda * 0.01* results['accuracy'] + (1-self.args.builder_lambda) * results['coverage'])
            loss_build_combined = loss_builder_weighted + self.args.lambda_priv * self.args.C_priv * max(0, results['achieved_epsilon']-self.args.goal_priv) + self.args.lambda_fair * self.args.C_fair * max(0, (results['achieved_fairness_gaps']-self.args.goal_fair))
            self.results_df = pd.concat([self.results_df, pd.DataFrame({'round': self.time, 
                                                'dataset': dataset,
                                                'epsilon': results['epsilon'], 
                                                self.fair_var: results['fairness_gaps'], 
                                                'agent': 'calibration',
                                                'loss_build_combined': loss_build_combined,
                                                'loss_build': loss_builder_weighted, 
                                                "accuracy": results['accuracy'], 
                                                "coverage": results['coverage'], 
                                                "privacy cost": results['achieved_epsilon'], 
                                                "max fairness gap": results['achieved_fairness_gaps']}, index=[0])], ignore_index=True)
    
    def return_results_df(self):
        return self.results_df
        
        
    def train_calibration_model(self, param):
        results = []
        if self.algorithm == 'fairPATE':
            for dataset in self.datasets:
                # we need to update all the params depending on which dataset we are running on
                self.args.dataset = dataset
                self.args = set_dataset(self.args)
                results.append(train_student_governance_game(self.args, param))
                
        elif self.algorithm == 'dpsgd-g-a':
            # Currently only supports one dataset
            self.args = set_dataset(self.args)
            raise NotImplemented
            results.append(train_dpsgd_g_a(self.args, param))
        return results
    
    def sync(self, curr_time, results_df):
        # update the time and results df
        self.time = curr_time
        self.results_df = results_df
        results = []
        # get all the new student model results from df and add them to the loss, priv, and fair
        for index, row in results_df.iterrows():
            if row['agent'] == 'calibration':
                result = {'accuracy': float(row['accuracy']),
                           'coverage': float(row['coverage']),
                           'achieved_epsilon': float(row['privacy cost']),
                           'achieved_fairness_gaps': float(row['max fairness gap']),
                           'epsilon': float(row['epsilon']),
                           'fairness_gaps': float(row[self.fair_var])
                    
                }
                results.append(result)
                # if one dataset, then just update
                if self.num_datasets == 1:
                    self.update_losses(results)
                    results = []
                # if two datasets, need to also get results from the other at the current round
                elif self.num_datasets == 2:
                    if len(results) == 2:
                        self.update_losses(results)
                        results = []
            