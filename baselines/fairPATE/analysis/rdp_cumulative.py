from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pdb
import numpy as np
import sys

from .pate import compute_logpr_answered
from .pate import compute_logq_gnmax
from .pate import compute_rdp_data_dependent_gnmax
from .pate import compute_rdp_data_dependent_threshold
from .pate import rdp_to_dp
from .pate import calculate_fairness_gaps, compute_logpr_answered_fair


def analyze_multiclass_confident_fair_gnmax(
        votes, sensitives, threshold, fair_threshold, sigma_threshold, sigma_fair_threshold, sigma_gnmax, budget, delta, file,
        show_dp_budget='disable', args=None, num_sensitive_attributes = 2, num_classes=2, minimum_group_count=10, log=print):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the Confident GNMax mechanism.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry
            corresponding to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the expected number of answered queries at a
            specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry
            corresponding to the order minimizing the privacy cost at a
            specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_threshold = rdp_eps_threshold_curr[idx]
        rdp_eps_gnmax = rdp_eps_total_curr[idx] - rdp_eps_threshold
        p = np.array([rdp_eps_threshold, rdp_eps_gnmax,
                      -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = len(votes)
    # All cumulative results
    dp_eps = np.zeros(n)
    gaps = np.zeros((n, num_sensitive_attributes), dtype=float)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)
    answered = np.zeros(n, dtype=float)
    pr_answered_per_query = np.zeros(n, dtype=float)
    # Current cumulative results
    rdp_eps_threshold_curr = np.zeros(len(orders))
    rdp_eps_total_curr = np.zeros(len(orders))
    answered_curr = 0

    sensitive_group_count = np.zeros(shape=(num_sensitive_attributes))
    # Note the shape of the positive counter. In k-class classification problem, we have shape: num_classes x num_sensitive_attributes
    per_class_pos_classified_group_count =  np.zeros(shape=(num_classes, num_sensitive_attributes))

    # Iterating over all queries
    for i in range(n):
        v = votes[i]
        sensitive = sensitives[i]

        if sigma_threshold > 0:
            # logpr - probability that the label is answered.

            # Selector one-hot vectors for the sensitive feature and the predicted class 
            sensitive_one_hot_over_sensitives = (np.arange(num_sensitive_attributes) == sensitive).astype(int)
            prediction_one_hot_over_classes = (np.arange(num_classes) == np.argmax(v)).astype(int)

            # Calculate of the new (tentative) gaps if the answered
            _per_class_pos_classified_group_count =  per_class_pos_classified_group_count + \
                                  prediction_one_hot_over_classes[:, None].dot(sensitive_one_hot_over_sensitives[:, None].T)
            _sensitive_group_count = sensitive_group_count + 1 * sensitive_one_hot_over_sensitives

            # (for comparison) calculate the probability of answering the query using only PATE analysis
            pate_logpr = compute_logpr_answered(threshold, sigma_threshold, v)

            # hard decision version (no noising)
            # if sensitive_group_count[sensitive.astype(int)] < minimum_group_count:
            #     fairpate_logpr = pate_logpr 
            # elif _group_tentative_new_gap < fair_threshold:
            #     log(f"z={sensitive}, this_group_tentative_new_gap: {_group_tentative_new_gap} < fair_threshold: {fair_threshold}")
            #     fairpate_logpr = pate_logpr
            # else:
            #     log(f"z={sensitive}, this_group_tentative_new_gap: {_group_tentative_new_gap} > fair_threshold: {fair_threshold}")
            #     fairpate_logpr = -np.inf
            # logpr = pate_logpr

            # Calculate the probability of answering using fairPATE analysis
            _new_gaps = calculate_fairness_gaps(_sensitive_group_count, _per_class_pos_classified_group_count)[np.argmax(v), :]
            # (present) group gap
            _group_tentative_new_gap = _new_gaps.dot(sensitive_one_hot_over_sensitives)
            
            # check for cold start
            if per_class_pos_classified_group_count[np.argmax(v), int(sensitive)] >= minimum_group_count:
                fairpate_logpr = compute_logpr_answered_fair(threshold, fair_threshold, sigma_threshold, sigma_fair_threshold, v, _group_tentative_new_gap)
                logpr = fairpate_logpr # or pate_logpr to disable fairPATE analysis
            else:
                logpr = pate_logpr
            '''
            if sensitive_group_count[int(sensitive)] >= minimum_group_count:
                fairpate_logpr = compute_logpr_answered_fair(threshold, fair_threshold, sigma_threshold, sigma_fair_threshold, v, _group_tentative_new_gap)
                logpr = fairpate_logpr # or pate_logpr to disable fairPATE analysis
            else:
                logpr = pate_logpr
            '''

            # useful debugging condition. probabilities should remain < 1
            if np.any(per_class_pos_classified_group_count.sum(axis=0) > sensitive_group_count + 1):
                log(per_class_pos_classified_group_count.sum(axis=0))
                log(sensitive_group_count)
                pdb.set_trace()

            # update counts (probabilistically)
            sensitive_group_count = sensitive_group_count + np.exp(logpr) * sensitive_one_hot_over_sensitives
            per_class_pos_classified_group_count = per_class_pos_classified_group_count + \
                                np.exp(logpr) * prediction_one_hot_over_classes[:, None].dot(sensitive_one_hot_over_sensitives[:, None].T)
            
            # print(sensitive_group_count)
            # re-calcualte definitive (and probabilistic) gaps
            new_gaps = calculate_fairness_gaps(sensitive_group_count, per_class_pos_classified_group_count)
            # print(new_gaps)

            #log(f"new_gaps: {new_gaps}")
            # gaps[i, :] = new_gaps
            #log(_group_tentative_new_gap, new_gaps)

            # calculate the cost of the privacy threshold mechanism
            # todo: should we add a second cost for the 2nd threshold mechanism?
            rdp_eps_threshold = compute_rdp_data_dependent_threshold(logpr, sigma_threshold, orders)
            # print(rdp_eps_threshold)

        else:
            # Do not use the Confident part of the GNMax.
            assert threshold == 0
            logpr = 0
            rdp_eps_threshold = 0

        logq = compute_logq_gnmax(v, sigma_gnmax)
        # print(logq)
        rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
            logq, sigma_gnmax, orders)
    
        # todo: do we need to add another eps for the 2nd threshold mechanism?
        rdp_eps_total = rdp_eps_threshold + np.exp(logpr) * rdp_eps_gnmax
        # print(rdp_eps_total.var())
        # Evaluate E[(rdp_eps_threshold + Bernoulli(pr) * rdp_eps_gnmax)^2]
        # Update current cumulative results.
        rdp_eps_threshold_curr += rdp_eps_threshold
        rdp_eps_total_curr += rdp_eps_total
        pr_answered = np.exp(logpr)
        pr_answered_per_query[i] = pr_answered
        answered_curr += pr_answered
        # Update all cumulative results.
        answered[i] = answered_curr
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        # print(i, dp_eps[i])
        # print(dp_eps[i])
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
        # Logs
        # if i % 100000 == 0 and i > 0:
        if show_dp_budget == 'apply':
            file = f'queries_answered_privacy_budget.txt'
            with open(file, 'a') as writer:
                if i == 0:
                    header = "queries answered,privacy budget"
                    print(header)
                    writer.write(f"{header}\n")
                info = f"{answered_curr},{dp_eps[i]}"
                print(info)
                writer.write(f"{info}\n")
                print(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                writer.write(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})\n'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                sys.stdout.flush()
                writer.flush()
        # log("\n\n")

    return max_num_query, dp_eps, partition, answered, order_opt, sensitive_group_count, per_class_pos_classified_group_count, answered_curr, gaps, pr_answered_per_query

