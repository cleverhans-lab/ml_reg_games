from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pickle import FALSE
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from baselines.fairPATE.analysis import analyze_multiclass_confident_fair_gnmax
from baselines.fairPATE.utils import load_evaluation_dataloader, get_unlabeled_set

from baselines.fairPATE.models.ensemble_model import FairEnsembleModel

from baselines.fairPATE.models.private_model import get_private_model_by_id
from baselines.fairPATE.models.utils_models import get_model_name_by_id
from baselines.fairPATE.utils import eval_model
from baselines.fairPATE.utils import metric
from baselines.fairPATE.utils import train_model


def train_student_governance_game(args, param):
    """
        Train a student model using FairPATE
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # update the args according to current param
    print("Preparing to train student model with: "+str(param), flush = True)
    args.budget = param[0]
    args.max_fairness_violation = param[1]
    
    # Logs
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        1,
        "random",
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    if not os.path.exists(os.path.join(args.path, "logs")):
         os.makedirs(os.path.join(args.path, "logs"))
    file = open(os.path.join(args.path, "logs", file_name), "w")

    # get the whole unlabeled dataset
    unlabeled_dataset = get_unlabeled_set(args=args)

    # load raw votes
    filename = "model(1)-raw-votes-(mode-{})-dataset-{}.npy".format(
            "random", args.dataset
        )
    votes_path = args.prev_results_dir+"votes/"+args.dataset+"/"
    filepath = os.path.join(votes_path, filename)
    votes = np.load(filepath)
    filename = "model(1)-sensitives-(mode-{})-dataset-{}.npy".format(
        "random", args.dataset
    )
    filepath = os.path.join(votes_path, filename)
    sensitive = np.load(filepath)
    # get max num queries
    (
        max_num_query, dp_eps, _, answered, _, _, 
        _, _, _, _
        ) = analyze_multiclass_confident_fair_gnmax(votes=votes, sensitives=sensitive, \
                        threshold=args.threshold, fair_threshold=args.max_fairness_violation,\
                        sigma_threshold=args.sigma_threshold, sigma_fair_threshold=0.0, sigma_gnmax=args.sigma_gnmax,\
                        budget=args.budget, delta=args.delta, file=file,show_dp_budget='disable', \
                        args=None, num_sensitive_attributes=len(args.sensitive_group_list), num_classes=args.num_classes, 
                        minimum_group_count=args.min_group_count)
        
    ensemble_model = FairEnsembleModel(
            model_id=0, private_models=[], args=args
        )
    all_indices = list(range(0, args.num_unlabeled_samples))
    indices_queried_num = all_indices[:max_num_query]

    unlabeled_dataset = Subset(unlabeled_dataset, indices_queried_num)
    queryloader = DataLoader(
        unlabeled_dataset, batch_size=len(unlabeled_dataset), shuffle=False
    )
    # get which queries are answered and preds
    votes = votes[:len(indices_queried_num)]
    sensitive = sensitive[:len(indices_queried_num)]
    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       votes.shape[0])
    vote_counts = votes.max(axis=1)
    answered = (vote_counts + noise_threshold) > args.threshold              
    noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    votes.shape[0], votes.shape[1]))
    noisy_votes = (votes + noise_gnmax)
    preds = (noisy_votes).argmax(axis=1)

    answered = ensemble_model.apply_fairness_constraint(preds, answered, sensitive, args)

    # get the train set
    X = None
    z = None
    for data, _, sens in queryloader:
        X = data
        z = sens

    indices = np.where(answered == 1)[0]
    X = X[indices].to(torch.float32)
    y =  torch.from_numpy(preds[indices]).to(torch.float32)
    z = z[indices]

    dataset = TensorDataset(X,y,z)
    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=FALSE)
    # get the test set
    evalloader = load_evaluation_dataloader(args)
    # train
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)
    # test
    result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=True)


    mydict = {'epsilon': param[0],
              'fairness_gaps': param[1], 
              'achieved_epsilon':dp_eps[max_num_query - 1], 
              'achieved_fairness_gaps': np.amax(fairness_gaps), 
              'query_fairness_gaps': np.amax(ensemble_model.fairness_disparity_gaps), 
              'number_answered': sum(answered), 
              'accuracy':result[metric.acc],
              'auc': result[metric.auc],
              'coverage': result[metric.coverage]}
    
    return mydict

    