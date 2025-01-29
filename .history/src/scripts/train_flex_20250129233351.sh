#!/bin/bash
task="BRCA"
data_root_dir="../Dataset/TCGA-BRCA/patch_512/quiltnetb32"
split_suffix='sitepre5_fold3'
exp_code="brca_sitepre5_fold3_woearlystop"
model_type="catemil_quiltnetb32"
test_name="cate_quiltnetb32_abmil_1414_k100_lp8_2dirinfo_vprompt"
base_mil="abmil"
len_learnable_prompt=8
slide_align=1
pretrain_epoch=0
max_epochs=20

CUDA_VISIBLE_DEVICES=2  \
python train.py  \
--drop_out  \
--lr 2e-4   \
--k 15  \
--label_frac 1    \
--label_num 0    \
--weighted_sample   \
--bag_loss ce   \
--inst_loss svm \
--task $task   \
--log_data  \
--data_root_dir $data_root_dir   \
--exp_code $exp_code   \
--model_type $model_type    \
--test_name $test_name  \
--split_suffix $split_suffix  \
--len_learnable_prompt $len_learnable_prompt  \
--base_mil $base_mil  \
--slide_align $slide_align  \
--pretrain_epoch $pretrain_epoch  \
--max_epochs $max_epochs  \