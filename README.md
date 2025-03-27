<!-- # FLEX -->

<!-- **Knowledge-Guided Adaptation of Pathology Foundation Models Improves Cross-domain Generalization and Demographic Fairness** -->
![header](https://capsule-render.vercel.app/api?type=soft&height=120&color=gradient&text=FLEX&section=header&fontSize=60&reversal=false&textBg=false&fontAlignY=42&desc=Knowledge-Guided%20Adaptation%20of%20Pathology%20Foundation%20Models%20Improves%20Cross-domain%20Generalization%20and%20Demographic%20Fairness&descAlignY=78&descSize=13)

## Overview

The advent of foundation models has ushered in a transformative era in computational pathology, enabling the extraction of rich, transferable image features for a broad range of downstream pathology tasks. However, site-specific signatures and demographic biases persist in these features, leading to short-cut learning and unfair predictions, ultimately compromising model generalizability and fairness across diverse clinical sites and demographic groups.

This repository implements FLEX, a novel framework that enhances cross-domain generalization and demographic fairness of pathology foundation models, thus facilitating accurate diagnosis across diverse pathology tasks. FLEX employs a task-specific information bottleneck, informed by visual and textual domain knowledge, to promote:

- Generalizability across clinical settings
- Fairness across demographic groups
- Adaptability to specific pathology tasks

![FLEX Framework](fig/main.png)

## Features

- **Cross-domain generalization**: Significantly improves diagnostic performance on data from unseen sites
- **Demographic fairness**: Reduces performance gaps between demographic groups
- **Versatility**: Compatible with various vision-language models
- **Scalability**: Adaptable to varying training data sizes
- **Seamless integration**: Works with multiple instance learning frameworks

## Installation

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/HKU-MedAI/FLEX
   cd FLEX
   ```
2. Create and activate a virtual environment and install the dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate flex
   ```


## Data Preparation

### Directory Structure

Prepare your data in the following structure:
```
Dataset/
├── TCGA-BRCA/
│   ├── features/
│   │   ├── ...
│   ├── tcga-brca_label.csv
│   ├── tcga-brca_label_her2.csv
│   └── ...
├── TCGA-NSCLC/
└── ...
```

### Visual Prompts

Organize visual prompts in the following structure:
```
prompts/
├── BRCA/
│   ├── 0/
│   │   ├── image1.png
│   │   └── ...
│   └── 1/
│       ├── image1.png
│       └── ...
├── BRCA_HER2/
└── ...
```

## Usage

### Split Data into multiple site-preserved splits

```bash
python generate_sitepreserved_splits.py
```

### Generate Site-Preserved Multiple Cross-Validation (SP-MCCV) Split

To create dataset splits that preserve site information:

```bash
python generate_sp_mccv_splits.py
```

### Training FLEX with CONCH

To train the FLEX model with CONCH as the foundation model, you first need to extract the features using [CLAM](https://github.com/mahmoodlab/CLAM) or [TRIDENT](https://github.com/mahmoodlab/TRIDENT), and place the features in the `features` directory under the corresponding task folder.

Then, you can train the FLEX model with the following command:

```bash
bash ./scripts/train_flex.sh
```

For custom training configurations, you can modify the parameters in the script.

### Key Parameters

- `--task`: Task name (e.g., BRCA, NSCLC, STAD_LAUREN)
- `--data_root_dir`: Path to the data directory
- `--split_suffix`: Split suffix (e.g., sitepre5_fold3)
- `--exp_code`: Experiment code for logging and saving results
- `--model_type`: Model type (default: flex)
- `--base_mil`: Base MIL framework (default:abmil)
- `--slide_align`: Whether to align in slide level (default: 1)
- `--w_infonce`: Weight for InfoNCE loss (default: 14)
- `--w_kl`: Weight for KL loss (default: 14)
- `--len_prompt`: Number of learnable textual prompt tokens

## Evaluation Results

FLEX has been evaluated on 16 clinically relevant tasks and demonstrates:

- Improved performance on unseen clinical sites
- Reduced performance gap between seen and unseen sites
- Enhanced fairness across demographic groups

For detailed results, refer to our paper.

## License

This project is licensed under the [Apache-2.0 license](LICENSE).

## Acknowledgments

This project was built on the top of amazing works, including [CLAM](https://github.com/mahmoodlab/CLAM), [CONCH](https://github.com/mahmoodlab/CONCH), [QuiltNet](https://huggingface.co/wisdomik/QuiltNet-B-32), [PathGen-CLIP](https://huggingface.co/jamessyx/PathGen-CLIP), and [PreservedSiteCV](https://github.com/fmhoward/PreservedSiteCV). We thank the authors for their great works.