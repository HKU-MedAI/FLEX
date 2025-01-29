from __future__ import print_function

import argparse
import pdb
import os
import math
import shutil

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch

import pandas as pd
import numpy as np

# import wandb

import warnings
warnings.filterwarnings('ignore')

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
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['catemil'], 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['BRCA', 'NSCLC', 'STAD', 'BRCA_HER2', 'BRCA_ER', 'BRCA_PR', 'STAD_EBV', 'STAD_MSI', 'BRCA_PIK3CA', 'BRCA_CDH1', 'LUAD_EGFR', 'LUAD_STK11', 'STAD_TP53', 'STAD_MUC16', 'CRC_BRAF', 'CRC_TP53'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--test_name', type=str, help='Test name')

## A-ViT
parser.add_argument('--ponder_token_scale', default=0.0005, type=float, help="")
parser.add_argument('--pretrained', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--act_mode', default=4, type=int,
                    help='4-token act, make sure this is always 4, other modes are only used for initial method comparison and exploration')
parser.add_argument('--tensorboard', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--gate_scale', default=100., type=float, help="constant for token control gate rescale")
parser.add_argument('--gate_center', default= 3., type=float, help="constant for token control gate re-center, negatived when applied")
parser.add_argument('--warmup_epoch', default=0, type=int, help="warm up epochs for act")
parser.add_argument('--distr_prior_alpha', default=0.001, type=float, help="scaling for kl of distributional prior")

parser.add_argument('--survival_pred', action='store_true', default=False, 
                     help='survival prediction problem')

# parser.add_argument('--sites',  type=list, default=['0'])
# parser.add_argument('--sites',  type=list, default=['D8'])
parser.add_argument('--sites',  type=list, default=['A8', 'D8'])
parser.add_argument('--split_suffix',  type=str)
parser.add_argument('--len_learnable_prompt',  type=int, default=0)
parser.add_argument('--base_mil',  type=str, default='abmil', choices=['abmil', 'clam_sb', 'transmil', 'dtfdmil', 'ilra', 'dsmil', 'wikg', 'acmil'])
parser.add_argument('--slide_align', default=1)
parser.add_argument('--pretrain_epoch', type=int, default=0)
parser.add_argument('--w_infonce', type=float, default=14)
parser.add_argument('--w_kl', type=float, default=14)
# parser.add_argument('--sites', type=list, default=['A2', 'A8', 'AR', 'BH', 'D8', 'E2'],)
# parser.add_argument('--sites',  type=list, default=['22', '44'])
# parser.add_argument('--sites',  type=list, default=['22', '44', '50', '56'])
# parser.add_argument('--sites',  type=list, default=['2Z', 'A3', 'KL'])
# parser.add_argument('--sites',  type=list, default=['2Z', 'A3', 'KL', '5P', 'AK', 'KM'])
# parser.add_argument('--sites',  type=list, default=None)
# parser.add_argument('--sites',  type=list, default=['D8'])
# parser.add_argument('--sites',  type=list, default=['AR'])
# parser.add_argument('--sites',  type=list, default=['AR', 'B6'])
# parser.add_argument('--sites',  type=list, default=['V1'])
# parser.add_argument('--sites',  type=list, default=['HC', 'V1'])

args = parser.parse_args()

# wandb.init(project='CATE_v2', name=f'{args.exp_code}-{args.test_name}')

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_out_test_auc = []
    all_test_acc = []
    all_val_acc = []
    all_out_test_acc = []
    all_test_f1 = []
    all_val_f1 = []
    all_out_test_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        # import ipdb; ipdb.set_trace()
        datasets = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        # import ipdb;ipdb.set_trace()
        # datasets = (train_dataset, val_dataset, test_dataset)
        # import ipdb;ipdb.set_trace()
        # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        results, out_results, test_auc, val_auc, out_test_auc, test_acc, val_acc, out_test_acc, test_f1, val_f1, out_test_f1 = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_out_test_auc.append(out_test_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_out_test_acc.append(out_test_acc)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        all_out_test_f1.append(out_test_f1)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        filename = os.path.join(args.results_dir, 'split_{}_out_results.pkl'.format(i))
        save_pkl(filename, out_results)

        fold_col = folds[:i+1].tolist()

        if i == folds[-1]:
            fold_col.append('mean')
            all_test_auc.append(np.mean(all_test_auc))
            all_val_auc.append(np.mean(all_val_auc))
            all_out_test_auc.append(np.mean(all_out_test_auc))
            all_test_acc.append(np.mean(all_test_acc))
            all_val_acc.append(np.mean(all_val_acc))
            all_out_test_acc.append(np.mean(all_out_test_acc))
            all_test_f1.append(np.mean(all_test_f1))
            all_val_f1.append(np.mean(all_val_f1))
            all_out_test_f1.append(np.mean(all_out_test_f1))

            fold_col.append('std')
            all_test_auc.append(np.std(all_test_auc))
            all_val_auc.append(np.std(all_val_auc))
            all_out_test_auc.append(np.std(all_out_test_auc))
            all_test_acc.append(np.std(all_test_acc))
            all_val_acc.append(np.std(all_val_acc))
            all_out_test_acc.append(np.std(all_out_test_acc))
            all_test_f1.append(np.std(all_test_f1))
            all_val_f1.append(np.std(all_val_f1))
            all_out_test_f1.append(np.std(all_out_test_f1))

        final_df = pd.DataFrame({
            'folds': fold_col,
            'test_auc': all_test_auc,
            'test_f1': all_test_f1,
            'test_acc': all_test_acc,
            'out_test_auc': all_out_test_auc,
            'out_test_f1' : all_out_test_f1,
            'out_test_acc' : all_out_test_acc,
            'val_auc': all_val_auc,
            'val_f1' : all_val_f1,
            'val_acc' : all_val_acc,
            })
        
        # wandb.log({'test_auc': all_test_auc[-1], 'val_auc': all_val_auc[-1], 'out_test_auc': all_out_test_auc[-1], 'test_acc': all_test_acc[-1], 'val_acc': all_val_acc[-1], 'out_test_acc': all_out_test_acc[-1]})

        if len(folds) != args.k:
            save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
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
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

# if args.task == 'task_1_tumor_vs_normal':
#     args.n_classes=2
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
#                             shuffle = False, 
#                             seed = args.seed, 
#                             print_info = True,
#                             label_dict = {'normal_tissue':0, 'tumor_tissue':1},
#                             patient_strat=False,
#                             ignore=[])

# elif args.task == 'task_2_tumor_subtyping':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
#                             shuffle = False, 
#                             seed = args.seed, 
#                             print_info = True,
#                             label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
#                             patient_strat= False,
#                             ignore=[])

#     if args.model_type in ['clam_sb', 'clam_mb']:
#         assert args.subtyping 

if args.task == 'BRACS_MT':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/BRACS/bracs_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Type_DCIS':0, 'Type_IC':1},
                            patient_strat=False,
                            ignore=['Type_N', 'Type_PB', 'Type_UDH', 'Type_FEA', 'Type_ADH'])

elif args.task == 'ESCA_typing':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/ESCA_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Adenocarcinoma':0, 'Squamous_cell_carcinoma':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'NSCLC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])
    

elif args.task == 'NSCLC_clasite':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_sitelabel.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'49':0, '55':1},
                            patient_strat=False,
                            ignore=['05', '35', '38', '44', '4B', '50', '53', '62', '64', '67', '69', '71', '73', '75', '78', '80', '83', '86', '91', '93', '95', '97', '99', 'J2', 'L4', 'L9', 'MN', 'MP', 'NJ', 'O1', 'S2', '18', '21', '22', '33', '34', '37', '39', '43', '46', '51', '52', '56', '58', '60', '63', '66', '68', '70', '77', '85', '90', '92', '94', '96', '98', 'J1', 'L3', 'LA', 'MF', 'NC', 'NK', 'O2', 'XC'])


elif args.task == 'BRCA':
    if not args.survival_pred:
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'IDC':0, 'ILC':1},
                                patient_strat=False,
                                ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
        
    else:
        dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'IDC':0, 'ILC':1},
                                patient_strat=False,
                                ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'],
                                survival_pred=True)

elif args.task == 'BRCA_HER2':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_her2.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Her2 Negative':0, 'Her2 Positive':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'BRCA_ER':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_er.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Negative':0, 'Positive':1},
                            patient_strat=False,
                            ignore=['[Not Evaluated]', 'Indeterminate'])
    
elif args.task == 'BRCA_PR':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_pr.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Negative':0, 'Positive':1},
                            patient_strat=False,
                            ignore=['[Not Evaluated]', 'Indeterminate'])
    
elif args.task == 'BRCA_TP53':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_tp53.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_PIK3CA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_CDH1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_EGFR':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_egfr.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_BRCA1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_brca1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_BRCA2':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_brca2.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_ESR1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_esr1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'BRCA_GATA3':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_gata3.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'STAD':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/labels/ebvmsi_others.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'others':0, 'EBVMSI':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'STAD_MSI':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_msi.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Non MSI':0, 'MSI':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_EBV':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_ebv.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Non EBV':0, 'EBV':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_LAUREN':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Intestinal':0, 'Diffuse':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_TP53':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_tp53.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_MUC16':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_muc16.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_MSTATUS':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_mstatus.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'M0':0, 'M+':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'STAD_NSTATUS':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_nstatus.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'N0':0, 'N+':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'camelyon16':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/Camelyon16/camelyon16_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'RCC':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'CRC_MSI':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_msi.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'non-MSI-H':0, 'MSI-H':1},
                            patient_strat=False,
                            ignore=['Indeterminate']),
    dataset = dataset[0]

elif args.task == 'CRC_BRAF':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_braf.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'CRC_KRAS':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_kras.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'CRC_PIK3CA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_pik3ca.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'CRC_TP53':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_tp53.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_TP53':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_STK11':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_stk11.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_EGFR':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_KRAS':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_kras.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_SPTA1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_spta1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_TTN':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_ttn.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'LUAD_NTRK1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_ntrk1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'ESCA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-ESCA/tcga-esca_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Adenocarcinoma':0, 'Squamous cell carcinoma':1},
                            patient_strat=False,
                            ignore=['Basaloid squamous cell carcinoma', 'Tubular adenocarcinoma']),
    dataset = dataset[0]

elif args.task == 'PRAD':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-PRAD/tcga-prad_label_gleasongrade.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Pattern 3':0, 'Pattern 4':1, 'Pattern 5':2},
                            patient_strat=False,
                            ignore=['Pattern 2']),
    dataset = dataset[0]

elif args.task == 'SARC':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-SARC/tcga-sarc_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'MFH':0, 'LMS':1, 'MFS':2, 'DDLS':3},
                            patient_strat=False,
                            ignore=['MPNST', 'SYNS', 'DES']),
    dataset = dataset[0]

elif args.task == 'GBM_IDH1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-GBM/tcga-gbm_label_idh1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'LGG_IDH1':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-LGG/tcga-lgg_label_idh1.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Not Altered':0, 'Altered':1},
                            patient_strat=False,
                            ignore=[])

    # import ipdb;ipdb.set_trace()
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed) + '_' + args.test_name)
args.results_dir = os.path.join(args.results_dir, str(args.exp_code), 's{}'.format(args.seed) + '_' + args.test_name)
# if not os.path.isdir(args.results_dir):
if os.path.exists(args.results_dir):
    shutil.rmtree(args.results_dir)
os.makedirs(args.results_dir)

if args.split_dir is None:
    if args.label_num == 0:
        if args.sites is not None:
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}')
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}_{args.sites[1]}_{args.sites[2]}')
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}_{args.sites[1]}')
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}_{args.sites[1]}_{args.sites[2]}_{args.sites[3]}')
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}_{args.sites[1]}_{args.sites[2]}_{args.sites[3]}_{args.sites[4]}_{args.sites[5]}')
            # args.split_dir = os.path.join('site_splits', args.task + f'_{args.sites[0]}')
            # args.split_dir = os.path.join('site_splits', args.task + '_sitepre5')
            args.split_dir = os.path.join('site_splits', args.task + f'_{args.split_suffix}')
            # args.split_dir = os.path.join('site_splits', args.task)
        else:
            args.split_dir = os.path.join('10fold_splits', args.task + '_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('10fold_splits', args.task + '_rn{}'.format(args.label_num))
else:
    args.split_dir = os.path.join('10fold_splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")

