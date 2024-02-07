from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from enum import Enum
from typing import Dict
from typing import List


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch import Tensor
from torch.nn import DataParallel
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from .datasets.celeba.celeba_utils import get_celeba_test_set

from .datasets.fairface.fairface_utils import get_fairface_test_set
from .datasets.utkface.utkface_utils import get_utkface_test_set

from .analysis.pate import calculate_fairness_gaps




class metric(Enum):
    """
    Evaluation metrics for the models.
    """

    acc = "acc"
    acc_detailed = "acc_detailed"
    acc_detailed_avg = "acc_detailed_avg"
    balanced_acc = "balanced_acc"
    balanced_acc_detailed = "balanced_acc_detailed"
    auc = "auc"
    auc_detailed = "auc_detailed"
    f1_score = "f1_score"
    f1_score_detailed = "f1_score_detailed"
    loss = "loss"
    test_loss = "test_loss"
    train_loss = "train_loss"
    gaps_mean = "gaps_mean"
    gaps_detailed = "gaps_detailed"
    pc = "pc"
    rc = "rc"
    fc = "fc"
    po = "po"
    ro = "ro"
    fo = "fo"
    fairness_disparity_gaps = "fairness_disparity_gaps"
    coverage = "coverage"

    def __str__(self):
        return self.name


class result(Enum):
    """
    Properties of the results.
    """

    aggregated_labels = "aggregated_labels"
    indices_answered = "indices_answered"
    predictions = "predictions"
    labels_answered = "labels_answered"
    count_answered = "count_answered"
    confidence_scores = "confidence_scores"

    def __str__(self):
        return self.name


def get_device(args):
    num_devices = torch.cuda.device_count()
    device_ids = args.device_ids
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if num_devices < len(device_ids):
        raise Exception(
            "#available gpu : {} < --device_ids : {}".format(
                num_devices, len(device_ids)
            )
        )
    if args.cuda:
        device = torch.device("cuda:{}".format(device_ids[0]))
    else:
        device = torch.device("cpu")
    return device, device_ids


def get_auc(classification_type, y_true, y_pred, num_classes=None):
    """
    Compute the AUC (Area Under the receiver operator Curve).
    :param classification_type: the type of classification.
    :param y_true: the true labels.
    :param y_pred: the scores or predicted labels.
    :return: AUC score.
    """
    if classification_type == "binary":
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true, y_pred, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        # breakpoint()
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred,
                                     average="weighted")
    elif classification_type == "multiclass":
        auc = metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
            # one-vs-one, insensitive to class imbalances when average==macro
            multi_class="ovo",
            average="macro",
            labels=[x for x in range(num_classes)],
        )
    elif classification_type in ["multilabel", "multilabel_counting"]:
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true, y_pred, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred,
                                    average="weighted")
    else:
        raise Exception(
            f"Unexpected classification_type: {classification_type}.")
    return auc


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)


def get_non_train_set(args):
    """
    Getting both unlabeled set and test set
    """
    if args.dataset in ["celeba", "celebasensitive"]:
        dataset = get_celeba_test_set(args=args)
    elif args.dataset == 'fairface':
        dataset = get_fairface_test_set(args=args)
    elif args.dataset == 'utkface':
        dataset = get_utkface_test_set(args=args)
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_unlabeled_set(args):
    """
    Get the unlabeled set from nontrain set

    :param args:
    :return: only the unlabeled data.
    """

    non_trained_set = get_non_train_set(args=args)
    start = 0
    end = args.num_unlabeled_samples
    assert end > start
    subset = Subset(dataset=non_trained_set, indices=list(range(start, end)))
    assert len(subset) == args.num_unlabeled_samples
    return subset


def get_test_set(args):
    """
    Get the test set from the rest of nontrain set

    :param args:
    :return: only the test data.
    """  
    non_trained_set = get_non_train_set(args=args)
    start = args.num_unlabeled_samples
    end = len(non_trained_set)
    assert end > start
    subset = Subset(dataset=non_trained_set, indices=list(range(start, end)))
    return subset


def get_kwargs(args):
    kwargs = {"num_workers": args.num_workers,
              "pin_memory": True} if args.cuda else {}
    return kwargs


def load_evaluation_dataloader(args):
    """Load labeled data for evaluation."""
    kwargs = get_kwargs(args=args)
    dataset = get_test_set(args=args)
    evalloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return evalloader


def get_loss_criterion(model, args):
    """
    Get the loss criterion.

    :param model: model
    :param args: arguments
    :return: the loss criterion (funciton like to be called)
    """
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "BCE":
        criterion = nn.BCELoss()
    elif args.loss_type == "BCEWithLogits":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Unknown loss type: {args.loss_type}.")

    return criterion


def compute_loss(target, output, criterion, args):
    """
    Compute the loss.

    :param target: target labels
    :param output: predicted labels
    :param criterion: loss criterion
    :param weights: the weight per task / label
    :return: the computed loss
    """
    targ = F.one_hot(target.to(torch.int64), num_classes=args.num_classes).float()
    loss = criterion(output, targ)
    return loss


def train(model, trainloader, optimizer, criterion, args, public_dataloader=None):
    """Train a given model on a given dataset using a given optimizer,
    loss criterion, and other arguments, for one epoch."""
    model.train()
    losses = []
    
    for batch_id, (data, *target) in enumerate(trainloader):
        if args.has_sensitive_attribute:
            target, sensitive = target
        else:
            target = target[0]  
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            if args.has_sensitive_attribute:
                sensitive = sensitive.cuda()
        if args.loss_type in {"MSE", "BCE", "BCEWithLogits", 'CEWithDemParityLoss', 'CEWithDemParityLossPub'}:
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            if args.has_sensitive_attribute:
                sensitive = sensitive.to(torch.float32)
        else:
            target = target.to(torch.long)

        output = model(data)
        
        loss = compute_loss(
                target=target,
                output=output,
                criterion=criterion,
                args=args
            )
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = np.mean(losses)
    return train_loss


def apply_fairness_constraint(sensitive_group_count, per_class_pos_classified_group_count, preds, sensitives, not_answered, args):
    for i, (pred, z) in enumerate(zip(preds, sensitives)):
        # warm-up
        if per_class_pos_classified_group_count[int(pred), z] < args.min_group_count:
            sensitive_group_count[z] += 1
            per_class_pos_classified_group_count[int(pred), z] += 1
        else:
            # estimate new gap
            _sensitive_group_count = sensitive_group_count.copy()
            _per_class_pos_classified_group_count = per_class_pos_classified_group_count.copy()
            _sensitive_group_count[z] += 1
            _per_class_pos_classified_group_count[int(pred), z] += 1
            _new_gaps = calculate_fairness_gaps(_sensitive_group_count, _per_class_pos_classified_group_count, rule_over_classes="all")
            # change pred to -1 to not answer
            if _new_gaps[int(pred), z] + np.random.normal(0., 0.0) > args.max_fairness_violation:
                preds[i] = -1
                not_answered += 1
            else:
                sensitive_group_count[z] += 1
                per_class_pos_classified_group_count[int(pred), z] += 1

    return sensitive_group_count, per_class_pos_classified_group_count, preds, not_answered
        

def evaluate_multiclass(model, dataloader, args, sensitives=False, preprocessor=False, DPLoss=None):
    """
    Evaluation for standard multiclass classification.
    Evaluate metrics such as accuracy, detailed acc, balanced acc, auc of a given model on a given dataset.

    Accuracy detailed - evaluate the class-specific accuracy of a given model on a given dataset.

    :return:
    detailed_acc: A 1-D numpy array of length L = num-classes, containing the accuracy for each class.

    """
    device, device_ids = get_device(args=args)
    model = model.to(device)
    model.eval()
    if sensitives:
        sensitive_group_count = np.zeros(shape=(len(args.sensitive_group_list)))
        # Note the shape of the positive counter. In k-class classification problem, we have shape: num_classes x num_sensitive_attributes
        per_class_pos_classified_group_count =  np.zeros(shape=(args.num_classes, len(args.sensitive_group_list)))
    losses = []
    correct = 0
    total = 0
    correct_detailed = np.zeros(args.num_classes, dtype=np.int64)
    wrong_detailed = np.zeros(args.num_classes, dtype=np.int64)
    raw_softmax = None
    raw_logits = None
    not_answered = 0
    raw_preds = []
    raw_targets = []
    criterion = get_loss_criterion(model=model, args=args)
    total_check = 0
    with torch.no_grad():
        for data, *target in dataloader:
            if args.has_sensitive_attribute:
                target, sensitive = target
            else:
                target = target[0]  

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                if args.has_sensitive_attribute:
                    sensitive = sensitive.cuda()
            
            total += len(target)
            output = model(data)

            # breakpoint()
            # loss = criterion(input=output, target=target)
            loss = compute_loss(target=target, output=output, criterion=criterion, args=args)
            losses.append(loss.item())
            preds = output.data.argmax(axis=1)
            
            labels = target.data.view_as(preds)
            softmax_outputs = F.softmax(output, dim=1)
            softmax_outputs = softmax_outputs.detach().cpu().numpy()
            outputd = output.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds_np = preds.detach().cpu().numpy().astype(int)
            target_np = target.detach().cpu().numpy().astype(int)
            sensitive_np = sensitive.detach().cpu().numpy().astype(int)

            if raw_softmax is None:
                raw_softmax = softmax_outputs
            else:
                raw_softmax = np.append(raw_softmax, softmax_outputs, axis=0)
            if raw_logits is None:
                raw_logits = outputd
            else:
                raw_logits = np.append(raw_logits, outputd, axis=0)
            raw_targets = np.append(raw_targets, labels_np)
            raw_preds = np.append(raw_preds, preds_np)
            
            if preprocessor:
                # for queries that we do not answer, change the pred to -1
                sensitive_group_count, per_class_pos_classified_group_count, preds, not_answered = apply_fairness_constraint(sensitive_group_count, per_class_pos_classified_group_count, preds, sensitive_np, not_answered, args)

            correct += preds.eq(labels).cpu().sum().item()
            
            if sensitives and not preprocessor:
                for label, pred, z in zip(target_np, preds_np, sensitive_np):
                    if label == pred:
                        correct_detailed[label] += 1
                    else:
                        wrong_detailed[label] += 1
                    sensitive_group_count[z] += 1
                    per_class_pos_classified_group_count[pred, z] += 1
            else:
                for label, pred in zip(target_np, preds_np):
                    if label == pred:
                        correct_detailed[label] += 1
                    else:
                        if preprocessor and pred == -1:
                            pass
                        else:
                            wrong_detailed[label] += 1

    loss = np.mean(losses)
    acc = 100.0 * correct / (total-not_answered)
    coverage = (total-not_answered)/total
    balanced_acc = metrics.balanced_accuracy_score(
        y_true=raw_targets,
        y_pred=raw_preds,
    )

    if (np.round(raw_softmax.sum(axis=1)) == 1).all() and raw_targets.size > 0:
        try:
            auc = get_auc(
                classification_type=args.class_type,
                y_true=np.identity(args.num_classes)[raw_targets.astype(int)],
                y_pred=raw_softmax,
                num_classes=args.num_classes,
            )
        except ValueError as err:
            print("Error occurred: ", err)
            auc = 0
    else:
        auc = 0

    assert correct_detailed.sum() + wrong_detailed.sum() == total
    acc_detailed = 100.0 * correct_detailed / (
            correct_detailed + wrong_detailed)

    result = {
        metric.loss: loss,
        metric.acc: acc,
        metric.balanced_acc: balanced_acc,
        metric.auc: auc,
        metric.acc_detailed: acc_detailed
    }
    if sensitives:
        fairness_gaps = calculate_fairness_gaps(sensitive_group_count, per_class_pos_classified_group_count, rule_over_classes="all")
        result[metric.coverage] = coverage
        return result, fairness_gaps

    return result, None


def one_hot(indices, num_classes: int) -> Tensor:
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    onehot = torch.zeros((len(indices), num_classes))
    for i in range(len(indices)):
        onehot[i][indices[i]] = 1
    return onehot


def get_scheduler(args, optimizer, trainloader=None):
    scheduler_type = args.scheduler_type
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=args.schedule_factor,
            patience=args.schedule_patience,
        )

    elif scheduler_type == "MultiStepLR":
        milestones = args.scheduler_milestones
        if milestones is None:
            milestones = [
                int(args.num_epochs * 0.5),
                int(args.num_epochs * 0.75),
                int(args.num_epochs * 0.9),
            ]
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones,
            gamma=args.schedule_factor
        )
    elif scheduler_type == "custom":
        scheduler = None
    else:
        raise Exception("Unknown scheduler type: {}".format(scheduler_type))
    return scheduler


def get_optimizer(params, args, lr=None):
    if lr is None:
        lr = args.lr
    if args.optimizer == "SGD":
        return SGD(
            params, lr=lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adadelta":
        return Adadelta(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        return Adagrad(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        return Adam(
            params, lr=lr, weight_decay=args.weight_decay,
            amsgrad=args.adam_amsgrad
        )
    elif args.optimizer == "RMSprop":
        return RMSprop(
            params, lr=lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise Exception("Unknown optimizer : {}".format(args.optimizer))


def eval_model(args, model, dataloader, sensitives=False, preprocessor=False, DPLoss=None):
    if args.class_type in ["multiclass", "binary"]:
        return evaluate_multiclass(model=model, dataloader=dataloader,
                                     args=args, sensitives=sensitives, preprocessor=preprocessor, DPLoss=DPLoss)
    else:
        raise Exception(f"Unsupported args.class_type: {args.class_type}.")
    return result


def get_model_params(model, args):
    return model.parameters()


def train_model(args, model, trainloader, evalloader, patience=None):
    device, device_ids = get_device(args=args)
    DPLoss = None
    filename = "model-{}-{}.pth.tar".format(args.budget, args.max_fairness_violation)
    save_model_path = os.path.join(args.path, args.experiment_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)                
    filepath = os.path.join(save_model_path, filename)
     
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    model_params = get_model_params(model=model, args=args)
    optimizer = get_optimizer(params=model_params, args=args)
    scheduler = get_scheduler(args=args, optimizer=optimizer,
                              trainloader=trainloader)
    criterion = get_loss_criterion(model=model, args=args)

    if patience is not None:
        # create variables for the patience mechanism
        best_loss = None
        patience_counter = 0

    start_epoch = 0
    save_model_path = getattr(args, "save_model_path", None)
    if save_model_path is not None:
        filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
        filepath = os.path.join(save_model_path, filename)
        # Check for model checkpoints
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]
            print(
                "Restarted from checkpoint file {} at epoch {}".format(
                    filepath, start_epoch
                )
            )
    print("STARTED TRAINING")
    for epoch in range(start_epoch, args.num_epochs):
        start = time.time()
        train_loss = train(
            model=model,
            trainloader=trainloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion
        )
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        if args.scheduler_type == "OneCycleLR":
            scheduler.step()
        else:
            scheduler.step(train_loss)
        stop = time.time()
        epoch_time = stop - start
        if patience is not None:
            result_test = train_model_log(
                args=args,
                epoch=epoch,
                model=model,
                epoch_time=epoch_time,
                trainloader=trainloader,
                evalloader=evalloader,
                DPLoss = DPLoss
            )
            if result_test is None:
                raise Exception(
                    "Fatal Error, result should not be None after training model"
                )
            if best_loss is None or best_loss > result_test[metric.loss]:
                best_loss = result_test[metric.loss]
                patience_counter = 0
                # save model
                torch.save(model.state_dict(), filepath)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # load the best model
                    model.load_state_dict(torch.load(filepath))
                    break
        else:
            train_model_log(
                args=args,
                epoch=epoch,
                model=model,
                epoch_time=epoch_time,
                trainloader=trainloader,
                evalloader=evalloader,
                DPLoss = DPLoss
            )


def train_model_log(args, epoch, epoch_time, model, trainloader, evalloader, DPLoss=None):
    log_every = args.log_every_epoch
    print("EPOCH: ", epoch)
    if log_every != 0 and epoch % log_every == 0:
        start_time = time.time()
        result_train, _ = eval_model(model=model, dataloader=trainloader,
                                  args=args, DPLoss=DPLoss)
        result_test,_ = eval_model(model=model, dataloader=evalloader, args=args, DPLoss=DPLoss)
        stop_time = time.time()
        eval_time = stop_time - start_time
        if epoch == 0:
            header = [
                "epoch",
                "train_" + str(metric.loss),
                "test_" + str(metric.loss),
                "accuracy;"
                "train_" + str(metric.balanced_acc),
                "test_" + str(metric.balanced_acc),
                "train_" + str(metric.auc),
                "test_" + str(metric.auc),
                "eval_time",
                "epoch_time",
            ]
            header_str = args.sep.join(header)
            print(header_str)
            #best_loss = result_test[metric.loss]
        data = [
            epoch,
            result_train[metric.loss],
            result_test[metric.loss],
            result_train[metric.acc],
            result_train[metric.balanced_acc],
            result_test[metric.balanced_acc],
            result_train[metric.auc],
            result_test[metric.auc],
            eval_time,
            epoch_time,
        ]
        data_str = args.sep.join([str(f"{x:.4f}") for x in data])
        print(data_str)

        # Checkpoint
        save_model(args, model, result_test=result_test)

    save_model(args, model, result_test=None)

    try:
        return result_test
    except NameError:
        return eval_model(model=model, dataloader=evalloader, args=args)
    

def save_model(args, model, epoch=-1, result_test=None):
    save_model_path = getattr(args, "save_model_path", None)
    if save_model_path is not None:
        if result_test is not None:
            state = result_test
        else:
            state = {}
        raw_model = getattr(model, "module", None)
        if raw_model is None:
            raw_model = model
        state["state_dict"] = raw_model.state_dict()
        if epoch == -1:
            epoch = args.num_epochs
        state["epoch"] = epoch
        filename = "checkpoint-{}.pth.tar".format("resnet50")  # raw_model.name)
        filepath = os.path.join(save_model_path, filename)
        torch.save(state, filepath)
