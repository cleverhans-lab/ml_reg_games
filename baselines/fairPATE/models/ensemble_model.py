import numpy as np
from torch import nn as nn

from .utils_models import get_model_name_by_id
from copy import deepcopy

from ..analysis.pate import calculate_fairness_gaps

class EnsembleModel(nn.Module):
    """
    Noisy ensemble of private models.
    All the models for the ensemble are pre-cached in memory.
    """

    def __init__(self, model_id: int, private_models, args):
        """

        :param model_id: id of the model (-1 denotes all private models).
        :param private_models: list of private models
        :param args: program parameters
        """
        super(EnsembleModel, self).__init__()
        self.id = model_id
        if self.id == -1:
            self.name = f"ensemble(all)"
        else:
            # This is ensemble for private model_id.
            self.name = get_model_name_by_id(id=model_id)
        self.num_classes = args.num_classes
        print("Building ensemble model '{}'!".format(self.name))
        self.ensemble = private_models

    def __len__(self):
        return len(self.ensemble)


class FairEnsembleModel(EnsembleModel):

    def __init__(self, model_id: int, private_models, args):
        super().__init__(model_id, private_models, args)
        self.subgorups = args.sensitive_group_list
        self.min_group_count = args.min_group_count
        self.max_fairness_violation = args.max_fairness_violation
        self.sensitive_group_count = np.zeros(shape=(len(args.sensitive_group_list)))
        self.per_class_pos_classified_group_count =  np.zeros(shape=(args.num_classes, len(args.sensitive_group_list)))
        
        self.fairness_disparity_gaps = None
        

    def apply_fairness_constraint(self, preds, answered, sensitive, args):
        """Ensure demographic parity fairness is within margin. Has side-effects."""         

        _answered = deepcopy(answered)

        # This is a pass-through filter. It should only block answers if they increase the fairness gap beyond `max_fairness_violation`.
        for s_id, z in enumerate(sensitive):
            z = int(z.item())
            answered = _answered[s_id]
            # Selector one-hot vectors for the sensitive feature and the predicted class 
            sensitive_one_hot_over_sensitives = (np.arange(len(args.sensitive_group_list)) == z).astype(int)
            prediction_one_hot_over_classes = (np.arange(args.num_classes) == preds[s_id]).astype(int)
            
            # Calculate of the new (tentative) gaps if the answered
            _per_class_pos_classified_group_count = self.per_class_pos_classified_group_count+ \
                prediction_one_hot_over_classes[:, None].dot(sensitive_one_hot_over_sensitives[:, None].T)
            _sensitive_group_count = self.sensitive_group_count + 1 * sensitive_one_hot_over_sensitives
            # get all the fairness gaps
            all_gaps = calculate_fairness_gaps(_sensitive_group_count, _per_class_pos_classified_group_count, rule_over_classes="all")
            _new_gaps = all_gaps[preds[s_id], :]
            # Neg. decisions maintain the gap; don;t block
            '''
            elif preds[s_id] == 0:
                _answered[s_id] = answered
            '''
            # Too few data points to estimate fairness; don't block.
            '''
            if self.sensitive_group_count[z] <  self.min_group_count:
                _answered[s_id] = answered
            '''
            if self.per_class_pos_classified_group_count[preds[s_id], z] < self.min_group_count:
                _answered[s_id] = answered
            # Pos. decisions may widen the gap, check the gap to ensure we are not over budget. If we are not, don't block.
            elif _new_gaps[z] + np.random.normal(0., 0.0) < self.max_fairness_violation:
                _answered[s_id] = answered
            else:
            # We are over budget; block.
                _answered[s_id] = False
            
            # update counters for measuring fairness 
            one_if_answered = 1 if _answered[s_id] else 0
            self.sensitive_group_count[z] += one_if_answered
            self.per_class_pos_classified_group_count[preds[s_id], z] += one_if_answered

            # update the disparity gaps 
            if one_if_answered:
                self.fairness_disparity_gaps = all_gaps

        return _answered