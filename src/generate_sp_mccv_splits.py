import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits, Generic_Split
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default=1,
                    help='fraction of labels (default: 1)')
parser.add_argument('--label_num', type=int, default=0)
parser.add_argument('--label_path', type=str, default='/home/yyhuang/WSI/Dataset/TCGA-BRCA/tcga-brca_label_her2.csv')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='number of splits (default: 10)')
parser.add_argument('--fold', type=int, default=3,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=[
                    'BRCA_sitepre5_fold3', 'NSCLC_sitepre5_fold3', 'STAD_LAUREN_sitepre3_fold5', 
                    'BRCA_HER2_sitepre5_fold3', 'BRCA_ER_sitepre5_fold3', 'BRCA_PR_sitepre5_fold3', 
                    'STAD_EBV_sitepre3_fold5', 'STAD_MSI_sitepre3_fold5', 'BRCA_PIK3CA_sitepre5_fold3', 
                    'BRCA_CDH1_sitepre5_fold3', 'LUAD_EGFR_sitepre3_fold5', 'LUAD_STK11_sitepre3_fold5', 
                    'STAD_TP53_sitepre3_fold5', 'STAD_MUC16_sitepre3_fold5', 'CRC_BRAF_sitepre3_fold5', 
                    'CRC_TP53_sitepre3_fold5'], 
                    default='BRCA_HER2_sitepre5_fold3')
parser.add_argument('--test_frac', type=float, default=0.3,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--val_frac', type=float, default=0,
                    help='fraction of labels for validation (default: 0, no validation split)')
parser.add_argument('--sites', type=list, default=[])
parser.add_argument('--site_split_path', type=str, default='/home/yyhuang/WSI/Dataset/TCGA-BRCA/tcga-brca_label_her2_presite5.csv')

args = parser.parse_args()

# Dictionary mapping tasks to their configurations
TASK_CONFIGS = {
    'BRCA_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'IDC': 0, 'ILC': 1},
        'ignore': ['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']
    },
    'NSCLC_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'LUAD': 0, 'LUSC': 1},
        'ignore': []
    },
    'STAD_LAUREN_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Intestinal': 0, 'Diffuse': 1},
        'ignore': []
    },
    'BRCA_HER2_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'Her2 Negative': 0, 'Her2 Positive': 1},
        'ignore': []
    },
    'BRCA_ER_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': ['[Not Evaluated]', 'Indeterminate']
    },
    'BRCA_PR_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': ['[Not Evaluated]', 'Indeterminate']
    },
    'STAD_EBV_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'EBV': 0, 'Non EBV': 1},
        'ignore': []
    },
    'STAD_MSI_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'MSI': 0, 'MSS': 1},
        'ignore': []
    },
    'BRCA_PIK3CA_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'BRCA_CDH1_sitepre5_fold3': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'LUAD_EGFR_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'LUAD_STK11_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'STAD_TP53_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'STAD_MUC16_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'CRC_BRAF_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    },
    'CRC_TP53_sitepre3_fold5': {
        'n_classes': 2,
        'label_dict': {'Negative': 0, 'Positive': 1},
        'ignore': []
    }
}

def custom_save_splits(splits, filename, boolean_style=False, out_test=None):
    """Custom function to save splits without using column_keys parameter"""
    if len(splits) == 2:  # train and test only
        column_keys = ['train', 'test']
    else:  # train, val, and test
        column_keys = ['train', 'val', 'test']
    
    split_datasets = splits
    
    splits_list = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if out_test is not None:
        out_test = pd.Series(out_test, name='out_test')
        splits_list.append(out_test)
        column_keys.append('out_test')

    if not boolean_style:
        df = pd.concat(splits_list, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits_list, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=column_keys)
    df.to_csv(filename)
    print(f"Saved to {filename}")

if __name__ == '__main__':
    label_fracs = args.label_frac
    label_num = args.label_num

    split_dir = 'site_splits/' + str(args.task)
    label_path = args.label_path
    site_split_path = args.site_split_path

    os.makedirs(split_dir, exist_ok=True)

    for i in range(args.k):
        # Get task configuration
        task_config = TASK_CONFIGS.get(args.task)
        if not task_config:
            raise ValueError(f"Task configuration for {args.task} not found")

        site_split_csv = pd.read_csv(site_split_path)
        args.n_classes = task_config['n_classes']
        args.sites = site_split_csv[site_split_csv['CV'] == i+1]['site'].unique().tolist()
        
        # Create dataset with task-specific configuration
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=label_path,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict=task_config['label_dict'],
            patient_strat=True,
            sites=args.sites,
            ignore=task_config['ignore']
        )
        
        label_df = pd.read_csv(label_path)
        label_csv = site_split_csv  # Using site_split_csv as label_csv for consistency

        # Create splits with val_num set to 0 for no validation
        num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
        
        if args.val_frac == 0:
            # No validation set case - only train and test
            val_num = np.zeros_like(num_slides_cls)
        else:
            val_num = np.round(num_slides_cls * args.val_frac).astype(int)
            
        test_num = np.round(num_slides_cls * args.test_frac).astype(int)

        dataset.create_splits(k=args.fold, val_num=val_num, test_num=test_num, label_frac=label_fracs)
        label_df['site'] = label_df['case_id'].apply(lambda x: x[5:7])
        label_df = label_df[~label_df['site'].isin(args.sites)]
        out_domain_test_ids = label_df['slide_id'].tolist()
        
        for j in range(args.fold):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            
            # Get splits from dataset
            if args.val_frac == 0:
                # For zero validation fraction, create splits with only train and test
                train_data = dataset.slide_data.loc[dataset.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, num_classes=dataset.num_classes)
                
                test_data = dataset.slide_data.loc[dataset.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, num_classes=dataset.num_classes)
                
                # Prepare splits without None values
                splits = [train_split, test_split]
                column_keys = ['train', 'test']
            else:
                # Get all three splits
                splits = dataset.return_splits(from_id=True)
                column_keys = ['train', 'val', 'test']
                
            # Save splits 
            split_filename = os.path.join(split_dir, f'splits_{i * args.fold + j}')
            
            if args.val_frac == 0:
                # Use our custom saving function for 2-split case
                custom_save_splits(splits, f'{split_filename}.csv', out_test=out_domain_test_ids)
                custom_save_splits(splits, f'{split_filename}_bool.csv', boolean_style=True)
            else:
                # Use standard save_splits function for 3-split case
                save_splits(splits, column_keys, f'{split_filename}.csv', out_test=out_domain_test_ids)
                save_splits(splits, column_keys, f'{split_filename}_bool.csv', boolean_style=True)
            
            descriptor_df.to_csv(f'{split_filename}_descriptor.csv')

            # Create and save labeled split file
            split_csv = pd.read_csv(f'{split_filename}_bool.csv')
            for k in range(len(split_csv)):
                slide_id = split_csv.loc[k, 'Unnamed: 0']
                label_value = label_csv[label_csv['slide_id'] == slide_id]['label'].values[0]
                split_csv.loc[k, 'label'] = label_value
                
            split_csv = split_csv.rename(columns={'Unnamed: 0': 'slide_id'})
            split_csv.to_csv(f'{split_filename}_bool_label.csv', index=None)
