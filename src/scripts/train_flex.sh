task="BRCA_HER2"
data_root_dir=""
split_suffix='sitepre5_fold3'
exp_code="test"
model_type="flex"
base_mil="abmil"
slide_align=1
max_epochs=20
w_infonce=14
w_kl=14
len_prompt=4
test_name=""

CUDA_VISIBLE_DEVICES=0 python train.py \
--drop_out \
--lr 1e-4 \
--k 15 \
--label_frac 1 \
--label_num 0 \
--weighted_sample \
--log_data \
--task $task \
--data_root_dir $data_root_dir \
--exp_code $exp_code \
--model_type $model_type \
--test_name $test_name \
--split_suffix $split_suffix \
--len_learnable_prompt $len_prompt \
--base_mil $base_mil \
--slide_align $slide_align \
--max_epochs $max_epochs \
--w_infonce $w_infonce \
--w_kl $w_kl