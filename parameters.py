import argparse


def parse_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--experiment_name", type=str)
    
    parser.add_argument("--algorithm", default='fairPATE', help='what ML algorithm to run the game with', type=str)


    parser.add_argument("--prev_results_dir", help="path to the previous results", type=str)
    parser.add_argument("--save_path", help="path to where to save the results", type=str)
    parser.add_argument("--save", default=1, help="whether to save", type=int)
    parser.add_argument("--device_ids", nargs="+", default=[0], type=int)
    parser.add_argument("--cuda", default=1, type=int)
    
    parser.add_argument("--num_rounds", default=30, help='number of game rounds', type=int)
    parser.add_argument("--dataset_list", default=['utkface'], help="list of datasets", nargs="+", type=str)
    parser.add_argument("--calibration", default=1, help='whether to include calibration round', type=int)
    
    parser.add_argument("--init_priv", default=None, help="initial privacy budget", type=float)
    parser.add_argument("--init_fair", default=None, help="initial fairness constraint", type=float)
    parser.add_argument("--priority", help="select which agent moves first", default='regulators', type=str)
    parser.add_argument("--init_as_goal", default=1, help='whether to set initial value as goal', type=int)
    # builder
    parser.add_argument("--step_size_priv", default=10, help="step size for privacy", type=float)
    parser.add_argument("--step_size_fair", default=0.1, help="step size for fairness", type=float)
    parser.add_argument("--step_size_decay", default=1.5, type=float)
    parser.add_argument("--lambda_priv", default=0.01, help="builder weighting for privacy loss", type=float)
    parser.add_argument("--lambda_fair", default=0.3, help="builder weighting for fairness loss", type=float)
    parser.add_argument("--builder_lambda", help="weighting of the model builder", default=0.7, type=float)
    # regulators
    parser.add_argument("--C_priv", default=1, help="penalty scalar for privacy", type=float)
    parser.add_argument("--C_fair", default=1, help="penalty scalar for fairness", type=float)
    parser.add_argument("--goal_priv", help="desired privacy budget", type=float)
    parser.add_argument("--goal_fair", help="desired fairness gap", type=float)
    parser.add_argument("--regulators_lambda", help="weighting of regulators in initialization", default=0.5, type=float)

    # these are for fairPATE student model training
    # capc parameters
    parser.add_argument("--path", help="path to where to save the student models", type=str)
    parser.add_argument("--data_dir", help="path to the dataset", type=str)
    parser.add_argument("--num_models", default=100, help="number of teacher models", type=int)
    parser.add_argument("--num_epochs", default=10, help="number of trainng epochs", type=int)
    parser.add_argument("--batch_size", default=100, help="training batch size", type=int)
    parser.add_argument("--lr", default=0.001, help="learnig rate", type=float)
    parser.add_argument("--optimizer", default="SGD", help="Adam, SGD", type=str)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--patience", default=None, type=int)
    
    # variable for student model training subject to change during the game
    parser.add_argument("--epsilon", default=10, type=float)
    parser.add_argument("--max_fairness_violation", default=0.01, type=float)
    parser.add_argument("--tau", default=0.7, help="DPSGD-G-A threshold tau", type=float)

    args = parser.parse_args()
    if args.priority == 'regulators' and args.init_as_goal:
        args.goal_priv = args.init_priv
        args.goal_fair = args.init_fair
        
        
    # params related to model training but are not dataset dependent
    args.num_querying_parties = 1
    args.num_workers = 4
    args.adam_amsgrad = False
    args.schedule_factor = 0.1
    args.schedule_patience = 10
    args.inprocessing_fairness = False
    args.log_every_epoch = True
    args.sep = ";"
    args.class_type = "multiclass"
    args.has_sensitive_attribute= 'True'
    args.seed = 111
    
    # check if more than one dataset is given
    args.dataset = args.dataset_list[0]
    if len(args.dataset_list) > 1:
        # regulators will use the second dataset
        args.regulator_dataset = args.dataset_list[1]
    else:
        args.regulator_dataset = None
    
    return set_dataset(args)


def set_dataset(args):
    '''
        Set parameters that are dataset dependent
    '''
    if args.algorithm == 'fairPATE':
        if args.dataset == 'utkface':
            args.scheduler_type = 'ReduceLROnPlateau'
            args.num_models = 100
            args.architecture ='resnet50_pretrained'
            args.threshold = 50
            args.sigma_gnmax = 15
            args.sigma_threshold = 40
            args.lr = 0.000005
            args.optimizer = "Adam"
            args.momentum = 0.9
            args.weight_decay = 0.0001
            args.loss_type = 'BCEWithLogits'
            args.num_epochs = 30
            args.batch_size = 60
            
            args.num_all_samples = 23705
            args.num_val_samples = 750
            args.num_unlabeled_samples = 1500
            args.num_test_samples = 1500
            args.num_train_samples = args.num_all_samples - args.num_val_samples - args.num_unlabeled_samples - args.num_test_samples
            args.num_classes = 2 # when predicting the gender
            args.delta = 1e-6
            
            args.sensitive_group_list =  [0,1,2,3,4] 
            args.min_group_count = 20
            
        elif args.dataset == 'colormnist':
            args.dataset_path = args.data_dir
            args.scheduler_type = 'ReduceLROnPlateau'
            args.num_models = 200
            args.architecture ="ColorMnistNetPate"
            args.threshold = 120
            args.sigma_gnmax = 20
            args.sigma_threshold = 110
            args.lr = 0.05
            args.optimizer = "Adam"
            args.momentum = 0.9
            args.weight_decay = 0.0
            args.loss_type = 'CE'
            args.num_epochs = 80
            args.batch_size = 200
            
            args.num_unlabeled_samples = 1000
            args.num_val_samples = 500
            args.num_test_samples = 18000
            args.num_dev_samples = 0
            args.num_classes = 10
            # Hyper-parameter delta in (eps, delta)-DP.
            args.delta = 1e-5
            
            args.sensitive_group_list =  [0,1] 
            args.min_group_count = 15
            
        elif args.dataset == 'celebasensitive':
            args.dataset_path = args.data_dir
            args.scheduler_type = 'ReduceLROnPlateau'
            args.num_models = 150
            args.architecture ='resnet50_pretrained'
            args.threshold = 130
            args.sigma_gnmax = 10
            args.sigma_threshold = 100
            args.lr = 0.0001
            args.optimizer = "Adam"
            args.momentum = 0.99
            args.weight_decay = 1e-4
            args.loss_type = 'BCEWithLogits'
            args.num_epochs = 15
            args.batch_size = 64
            
            args.num_all_samples = 202599
            args.num_dev_samples = 0
            args.num_val_samples = 3000
            args.num_unlabeled_samples = 9000
            args.num_test_samples = 3000
            args.num_train_samples = args.num_all_samples - args.num_unlabeled_samples - args.num_test_samples - args.num_val_samples
            num_samples_per_model = args.num_train_samples / args.num_models
            args.num_classes = 2
            args.delta = 1e-6
            
            args.sensitive_group_list = [0,1] 
            args.min_group_count = 20
            
        elif args.dataset == 'fairface':
            args.dataset_path = args.data_dir
            args.scheduler_type = 'ReduceLROnPlateau'
            args.num_models = 50
            args.architecture ='resnet50_pretrained'
            args.threshold = 30
            args.sigma_gnmax = 30
            args.sigma_threshold = 10
            args.lr = 0.00005
            args.optimizer = "Adam"
            args.momentum = 0.9
            args.weight_decay = 0.0001
            args.loss_type = 'BCEWithLogits'
            args.num_epochs = 25
            args.batch_size = 64
            
            args.num_all_samples = 97698
            args.num_val_samples = 2500
            args.num_unlabeled_samples = 5000 # this is the same as the test samples
            args.num_test_samples = 5954 # that is what is left over after taking 5000 from the validation frame as validation data
            args.num_train_samples = args.num_all_samples - args.num_unlabeled_samples -args.num_val_samples - args.num_test_samples
            args.num_classes = 2 # this holds if we predict the gender as a target!
            args.delta = 1e-6
            
            args.sensitive_group_list =  [0,1,2,3,4,5,6] 
            args.min_group_count = 15
        
        args.architectures = [args.architecture]
    elif args.algorithm == 'dpsgd-g-a' or args.algorithm == 'regular':
        raise NotImplemented
        if args.dataset == 'mnist': 
            if args.algorithm == 'dpsgd-g-a':
                args.method='dpsgd-global-adapt' 
            elif args.algorithm == 'regular':
                args.method = 'regular'
            # config should be a disctionary
            args.config = []
            args.config.append('group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1')
            args.config.append('make_valid_loader=0')
            args.config.append('net=cnn')
            args.config.append('hidden_channels=32,16')
            args.config.append('train_batch_size=256')
            args.config.append('valid_batch_size=256')
            args.config.append('test_batch_size=256')
            args.config.append('max_epochs=60')
            args.config.append('delta=1e-6')
            # placeholder
            args.config.append('noise_multiplier=0')
            args.config.append('l2_norm_clip=1')
            args.config.append('strict_max_grad_norm=50')
            args.config.append('lr=0.1')
            args.config.append(f"logdir='{args.path}/{args.algorithm}/{args.experiment_name}'")
            # placeholder tau
            args.config.append('threshold=0.1')
            args.config.append('seed=0')
            args.config.append("evaluate_angles=False")
            args.config.append("evaluate_hessian=False")
            args.config.append('angle_comp_step=200')

    return args
        