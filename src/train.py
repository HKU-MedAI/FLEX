from __future__ import print_function

import argparse
import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# Ignore warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--label_num', type=int, default=0,
                    help='number of training labels (default: 0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['flex'], 
                    help='type of model')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['BRCA', 'NSCLC', 'STAD_LAUREN', 'BRCA_HER2', 'BRCA_ER', 'BRCA_PR', 
                                               'STAD_EBV', 'STAD_MSI', 'BRCA_PIK3CA', 'BRCA_CDH1', 'LUAD_EGFR', 
                                               'LUAD_STK11', 'STAD_TP53', 'STAD_MUC16', 'CRC_BRAF', 'CRC_TP53'])
# CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--test_name', type=str, help='Test name')

# Other options
parser.add_argument('--split_suffix', type=str)
parser.add_argument('--len_learnable_prompt', type=int, default=0)
parser.add_argument('--base_mil', type=str, default='abmil', choices=['abmil', 'clam_sb', 'dtfdmil', 'ilra', 'acmil'])
parser.add_argument('--slide_align', default=1)
parser.add_argument('--pretrain_epoch', type=int, default=0)
parser.add_argument('--w_infonce', type=float, default=0)
parser.add_argument('--w_kl', type=float, default=0)
args = parser.parse_args()


def main(args):
    """Main training function with cross-validation"""
    # Create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # Set fold ranges
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end

    # Initialize metrics arrays
    all_test_auc = []
    all_out_test_auc = []
    all_test_acc = []
    all_out_test_acc = []
    all_test_f1 = []
    all_out_test_f1 = []
    
    folds = np.arange(start, end)
    for i in folds:
        # Set random seed for reproducibility
        seed_torch(args.seed)
        
        # Get dataset splits for current fold
        datasets = dataset.return_splits(
            from_id=False, 
            csv_path=f'{args.split_dir}/splits_{i}.csv'
        )
        
        # Train model on current fold
        results, out_results, test_auc, out_test_auc, test_acc, out_test_acc, test_f1, out_test_f1 = train(datasets, i, args)
        
        # Store metrics
        all_test_auc.append(test_auc)
        all_out_test_auc.append(out_test_auc)
        all_test_acc.append(test_acc)
        all_out_test_acc.append(out_test_acc)
        all_test_f1.append(test_f1)
        all_out_test_f1.append(out_test_f1)
        
        # Save results for current fold
        filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)
        filename = os.path.join(args.results_dir, f'split_{i}_out_results.pkl')
        save_pkl(filename, out_results)

        # Create folds column for DataFrame
        fold_col = folds[:i+1].tolist()

        # Add summary statistics after the last fold
        if i == folds[-1]:
            # Add mean
            fold_col.append('mean')
            all_test_auc.append(np.mean(all_test_auc))
            all_out_test_auc.append(np.mean(all_out_test_auc))
            all_test_acc.append(np.mean(all_test_acc))
            all_out_test_acc.append(np.mean(all_out_test_acc))
            all_test_f1.append(np.mean(all_test_f1))
            all_out_test_f1.append(np.mean(all_out_test_f1))

            # Add standard deviation
            fold_col.append('std')
            all_test_auc.append(np.std(all_test_auc))
            all_out_test_auc.append(np.std(all_out_test_auc))
            all_test_acc.append(np.std(all_test_acc))
            all_out_test_acc.append(np.std(all_out_test_acc))
            all_test_f1.append(np.std(all_test_f1))
            all_out_test_f1.append(np.std(all_out_test_f1))

        # Create summary DataFrame and save
        final_df = pd.DataFrame({
            'folds': fold_col,
            'test_auc': all_test_auc,
            'test_f1': all_test_f1,
            'test_acc': all_test_acc,
            'out_test_auc': all_out_test_auc,
            'out_test_f1': all_out_test_f1,
            'out_test_acc': all_out_test_acc,
        })

        # Set filename based on whether we're using all folds or a subset
        if len(folds) != args.k:
            save_name = f'summary_partial_{start}_{end}.csv'
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))


def seed_torch(seed=7):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Set random seed
seed_torch(args.seed)

# Define encoding size and settings dictionary
encoding_size = 1024
settings = {
    'num_splits': args.k, 
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs, 
    'results_dir': args.results_dir, 
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'label_num': args.label_num,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    "use_drop_out": args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt
}

print('\nLoad Dataset')

# Dataset initialization based on task
task_datasets = {
    'BRCA': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
        'label_dict': {'IDC': 0, 'ILC': 1},
        'ignore': ['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']
    },
    'NSCLC': {
        'csv_path': '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
        'label_dict': {'LUAD': 0, 'LUSC': 1},
        'ignore': []
    },
    'STAD_LAUREN': {
        'csv_path': '../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv',
        'label_dict': {'Intestinal': 0, 'Diffuse': 1},
        'ignore': []
    },
    'BRCA_HER2': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label_her2.csv',
        'label_dict': {'Her2 Negative': 0, 'Her2 Positive': 1},
        'ignore': []
    },
    'BRCA_ER': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label_er.csv',
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': ['[Not Evaluated]', 'Indeterminate']
    },
    'BRCA_PR': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label_pr.csv',
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': ['[Not Evaluated]', 'Indeterminate']
    },
    'STAD_EBV': {
        'csv_path': '../Dataset/TCGA-STAD/tcga-stad_label_ebv.csv',
        'label_dict': {'Non EBV': 0, 'EBV': 1},
        'ignore': []
    },
    'STAD_MSI': {
        'csv_path': '../Dataset/TCGA-STAD/tcga-stad_label_msi.csv',
        'label_dict': {'Non MSI': 0, 'MSI': 1},
        'ignore': []
    },
    'BRCA_PIK3CA': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    },
    'BRCA_CDH1': {
        'csv_path': '../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    },
    'LUAD_EGFR': {
        'csv_path': '../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    },
    'LUAD_STK11': {
        'csv_path': '../Dataset/TCGA-NSCLC/tcga-luad_label_stk11.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    },
    'STAD_TP53': {
        'csv_path': '../Dataset/TCGA-STAD/tcga-stad_label_tp53.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': []
    },
    'STAD_MUC16': {
        'csv_path': '../Dataset/TCGA-STAD/tcga-stad_label_muc16.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': []
    },
    'CRC_BRAF': {
        'csv_path': '../Dataset/TCGA-CRC/tcga-crc_label_braf.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    },
    'CRC_TP53': {
        'csv_path': '../Dataset/TCGA-CRC/tcga-crc_label_tp53.csv',
        'label_dict': {'Not Altered': 0, 'Altered': 1},
        'ignore': [],
        'unwrap': True
    }
}

if args.task not in task_datasets:
    raise NotImplementedError(f"Task {args.task} not implemented")

# Set number of classes
args.n_classes = 2

# Create dataset
task_config = task_datasets[args.task]
dataset = Generic_MIL_Dataset(
    csv_path=task_config['csv_path'],
    data_dir=args.data_root_dir,
    shuffle=False,
    seed=args.seed,
    print_info=True,
    label_dict=task_config['label_dict'],
    patient_strat=False,
    ignore=task_config['ignore']
)

# Unwrap dataset if needed (for some tasks)
if task_config.get('unwrap', False):
    dataset = dataset[0]

# Create results directory
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Set results directory path with experiment code
args.results_dir = os.path.join(args.results_dir, str(args.exp_code), f's{args.seed}_{args.test_name}')
if os.path.exists(args.results_dir):
    shutil.rmtree(args.results_dir)
os.makedirs(args.results_dir)

# Set split directory
if args.split_dir is None:
    if args.label_num == 0:
        args.split_dir = os.path.join('site_splits', f'{args.task}_{args.split_suffix}')
    else:
        args.split_dir = os.path.join('10fold_splits', f'{args.task}_rn{args.label_num}')
else:
    args.split_dir = os.path.join('10fold_splits', args.split_dir)

print(f'split_dir: {args.split_dir}')
assert os.path.isdir(args.split_dir)

# Update settings with split directory
settings.update({'split_dir': args.split_dir})

# Save experiment settings
with open(f"{args.results_dir}/experiment_{args.exp_code}.txt", 'w') as f:
    print(settings, file=f)

# Print settings
print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")

