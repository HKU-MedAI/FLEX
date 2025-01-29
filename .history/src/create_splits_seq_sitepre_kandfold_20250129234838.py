import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1,
                    help='fraction of labels (default: 1)')
parser.add_argument('--label_num', type=int, default= 0)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=3,
                    help='number of splits (default: 10)')
parser.add_argument('--fold', type=int, default=5,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['LUAD_TP53_sitepre3_fold3', 'LUAD_TP53_sitepre5_fold3', 'LUAD_STK11_sitepre3_fold5', 'CRC_MSI_sitepre3_fold5', 'CRC_BRAF_sitepre3_fold5', 'CRC_KRAS_sitepre3_fold5', 'CRC_PIK3CA_sitepre3_fold5', 'CRC_TP53_sitepre3_fold5', 'BRCA_sitepre5_fold3', 'BRCA_sitepre4_fold3', 'BRCA_sitepre5_fold3', 'BRCA_noval_sitepre5_fold3', 'BRCA_TP53_sitepre5_fold3', 'BRCA_TP53_sitepre3_fold5', 'LUAD_TP53_sitepre3_fold5', 'LUAD_EGFR_sitepre5_fold3', 'LUAD_EGFR_sitepre3_fold5', 'NSCLC_sitepre5_fold3', 'STAD_MSI_sitepre3_fold5', 'STAD_EBV_sitepre3_fold5', 'STAD_LAUREN_sitepre3_fold5', 'STAD_LAUREN_sitepre3_fold5_2', 'STAD_TP53_sitepre3_fold5', 'STAD_MUC16_sitepre3_fold5', 'BRCA_HER2_sitepre5_fold3', 'BRCA_ER_sitepre5_fold3', 'BRCA_PR_sitepre5_fold3', 'BRCA_PIK3CA_sitepre5_fold3', 'BRCA_PIK3CA_sitepre3_fold5', 'BRCA_CDH1_sitepre5_fold3', 'BRCA_CDH1_sitepre3_fold5'], default='BRCA')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.3,
                    help='fraction of labels for test (default: 0.1)')

parser.add_argument('--sites', type=list, default=[],)


args = parser.parse_args()



if __name__ == '__main__':
    # if args.label_frac > 0:
    label_fracs = args.label_frac
    label_num = args.label_num

    split_dir = 'site_splits/'+ str(args.task)

    os.makedirs(split_dir, exist_ok=True)

    for i in range(args.k):
        
        if args.task == 'BRCA_sitepre':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_cv3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'IDC':0, 'ILC':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label.csv')

        elif args.task == 'BRCA_sitepre4_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_presite4.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'IDC':0, 'ILC':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label.csv')

        elif args.task == 'BRCA_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'IDC':0, 'ILC':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label.csv')
        
        elif args.task == 'BRCA_TP53_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_tp53_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_tp53.csv')

        elif args.task == 'BRCA_TP53_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_tp53_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_tp53.csv')

        elif args.task == 'BRCA_PIK3CA_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv')

        elif args.task == 'BRCA_PIK3CA_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pik3ca.csv')

        elif args.task == 'BRCA_CDH1_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_cdh1_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv')

        elif args.task == 'BRCA_CDH1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_cdh1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_cdh1.csv')

        elif args.task == 'BRCA_EGFR_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_egfr_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_egfr.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_egfr.csv')

        elif args.task == 'BRCA_BRCA1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_brca1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_brca1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_brca1.csv')

        elif args.task == 'BRCA_BRCA2_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_brca2_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_brca2.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_brca2.csv')

        elif args.task == 'BRCA_ESR1_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_esr1_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_esr1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_esr1.csv')

        elif args.task == 'BRCA_GATA3_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_gata3_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_gata3.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_gata3.csv')

        elif args.task == 'BRCA_HER2_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_her2_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_her2.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Her2 Positive':0, 'Her2 Negative':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_her2.csv')

        elif args.task == 'BRCA_ER_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_er_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_er.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Positive':0, 'Negative':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['[Not Evaluated]', 'Indeterminate'])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_er.csv')

        elif args.task == 'BRCA_PR_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pr_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label_pr.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Positive':0, 'Negative':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['[Not Evaluated]', 'Indeterminate'])
            label_df = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label_pr.csv')

        elif args.task == 'NSCLC_sitepre5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_label_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'LUAD':0, 'LUSC':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv')

        elif args.task == 'NSCLC_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_label_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'LUAD':0, 'LUSC':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv')

        elif args.task == 'CRC_MSI_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_msi_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_msi.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'MSI-H':1, 'non-MSI-H':0},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_msi.csv')

        elif args.task == 'CRC_BRAF_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_braf_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_braf.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_braf.csv')

        elif args.task == 'CRC_KRAS_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_kras_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_kras.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_kras.csv')

        elif args.task == 'CRC_PIK3CA_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_pik3ca_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_pik3ca.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_pik3ca.csv')

        elif args.task == 'CRC_TP53_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_tp53_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_pik3ca.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_tp53.csv')

        # elif args.task == 'CRC_MSI_sitepre3':
        #     label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_msi_presite3.csv')
        #     args.n_classes=2
        #     args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
        #     # import ipdb; ipdb.set_trace()
        #     dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_msi.csv',
        #                             shuffle = False, 
        #                             seed = args.seed, 
        #                             print_info = True,
        #                             label_dict = {'MSI-H':0, 'non-MSI-H':1},
        #                             patient_strat=True,
        #                             sites = args.sites,
        #                             ignore=[])
        #     label_df = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_msi.csv')
        
        elif args.task == 'PRAD_sitepre5':
            label_csv = pd.read_csv('../Dataset/TCGA-PRAD/tcga-prad_label_gleasongrade_presite5.csv')
            args.n_classes=3
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-PRAD/tcga-prad_label_gleasongrade.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Pattern 3':0, 'Pattern 4':1, 'Pattern 5': 2},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['Pattern 2'])
            label_df = pd.read_csv('../Dataset/TCGA-PRAD/tcga-prad_label_gleasongrade.csv')

        elif args.task == 'LUAD_TP53_sitepre5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':0, 'Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv')

        elif args.task == 'LUAD_TP53_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] != i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv')

        elif args.task == 'LUAD_TP53_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_tp53.csv')

        elif args.task == 'LUAD_STK11_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_stk11_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_stk11.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_stk11.csv')

        elif args.task == 'LUAD_EGFR_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_egfr_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv')

        elif args.task == 'LUAD_EGFR_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_egfr_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_egfr.csv')

        elif args.task == 'LUAD_KRAS_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_kras_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_kras.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_kras.csv')

        elif args.task == 'LUAD_KRAS_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_kras_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_kras.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_kras.csv')

        elif args.task == 'LUAD_SPTA1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_spta1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_spta1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_spta1.csv')

        elif args.task == 'LUAD_TTN_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_ttn_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_ttn.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_ttn.csv')

        elif args.task == 'LUAD_NTRK1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_ntrk1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-luad_label_ntrk1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-luad_label_ntrk1.csv')

        elif args.task == 'RCC_sitepre':
            label_csv = pd.read_csv('../Dataset/TCGA-RCC/tcga-kidney_label_cv3.csv')
            args.n_classes=3
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-RCC/tcga-kidney_label.csv')

        elif args.task == 'STAD_MSI_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_msi_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_msi.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'MSI':0, 'Non MSI':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_msi.csv')

        elif args.task == 'STAD_EBV_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_ebv_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_ebv.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'EBV':0, 'Non EBV':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_ebv.csv')

        elif args.task == 'STAD_LAUREN_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_lauren_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Intestinal':0, 'Diffuse':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv')

        elif args.task == 'STAD_LAUREN_sitepre3_fold5_2':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_lauren_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] != i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Intestinal':0, 'Diffuse':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_lauren.csv')
        
        elif args.task == 'SARC_sitepre2_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-SARC/tcga-sarc_label_presite2.csv')
            args.n_classes=4
            args.sites = label_csv[label_csv['CV'] != i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-SARC/tcga-sarc_label.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'MFH':0, 'LMS':1, 'MFS':2, 'DDLS':3},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=['MPNST', 'SYNS', 'DES'])
            label_df = pd.read_csv('../Dataset/TCGA-SARC/tcga-sarc_label.csv')

        elif args.task == 'STAD_TP53_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_tp53_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_tp53.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_tp53.csv')

        elif args.task == 'STAD_MUC16_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_muc16_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_muc16.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':0, 'Not Altered':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_muc16.csv')

        elif args.task == 'STAD_MSTATUS_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_mstatus_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_mstatus.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'M0':0, 'M+':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_mstatus.csv')

        elif args.task == 'STAD_NSTATUS_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_nstatus_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            # import ipdb; ipdb.set_trace()
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_label_nstatus.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'N0':0, 'N+':1},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_label_nstatus.csv')

        elif args.task == 'GBM_IDH1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-GBM/tcga-gbm_label_idh1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            args.sites = [str(site) for site in args.sites]
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-GBM/tcga-gbm_label_idh1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':1, 'Not Altered':0},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-GBM/tcga-gbm_label_idh1.csv')

        elif args.task == 'LGG_IDH1_sitepre3_fold5':
            label_csv = pd.read_csv('../Dataset/TCGA-LGG/tcga-lgg_label_idh1_presite3.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            args.sites = [str(site) for site in args.sites]
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-LGG/tcga-lgg_label_idh1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Altered':1, 'Not Altered':0},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-LGG/tcga-lgg_label_idh1.csv')

        elif args.task == 'LGG_IDH1_sitepre5_fold3':
            label_csv = pd.read_csv('../Dataset/TCGA-LGG/tcga-lgg_label_idh1_presite5.csv')
            args.n_classes=2
            args.sites = label_csv[label_csv['CV'] == i+1]['site'].unique().tolist()
            args.sites = [str(site) for site in args.sites]
            dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-LGG/tcga-lgg_label_idh1.csv',
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'Not Altered':1, 'Altered':0},
                                    patient_strat=True,
                                    sites = args.sites,
                                    ignore=[])
            label_df = pd.read_csv('../Dataset/TCGA-LGG/tcga-lgg_label_idh1.csv')
        

        num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
        val_num = np.round(num_slides_cls * args.val_frac).astype(int)
        test_num = np.round(num_slides_cls * args.test_frac).astype(int)

        dataset.create_splits(k = args.fold, val_num = val_num, test_num = test_num, label_frac=label_fracs)
        label_df['site'] = label_df['case_id'].apply(lambda x: x[5:7])
        label_df = label_df[~label_df['site'].isin(args.sites)]
        out_domain_test_ids = label_df['slide_id'].tolist()
        # import ipdb; ipdb.set_trace()
        
        for j in range(args.fold):
            
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            # splits = [splits[0], splits[2]]
            # import ipdb; ipdb.set_trace()
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i * args.fold + j)), out_test=out_domain_test_ids)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i * args.fold + j)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i * args.fold + j)))

            split_csv = pd.read_csv(os.path.join(split_dir, 'splits_{}_bool.csv'.format(i * args.fold + j)))
            for k in range(len(split_csv)):
                split_csv.loc[k, 'label'] = label_csv[label_csv['slide_id'] == split_csv.loc[k, 'Unnamed: 0']]['label'].values[0]
            split_csv = split_csv.rename(columns={'Unnamed: 0':'slide_id'})
            split_csv.to_csv(os.path.join(split_dir, 'splits_{}_bool_label.csv'.format(i * args.fold + j)), index=None)
