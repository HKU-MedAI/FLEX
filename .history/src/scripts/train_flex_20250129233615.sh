task="BRCA"
data_root_dir=""
split_suffix='sitepre5_fold3'
exp_code=""
model_type="flex_"
base_mil="abmil"
slide_align=1
pretrain_epoch=0
max_epochs=20
w_infonce=14
w_kl=2
len_prompt=4
test_name="${model_type}_${base_mil}_${w_infonce}${w_kl}_k100_lp${len_prompt}_2dirinfo_vprompt"

python train.py  \
--drop_out  \
--lr 1e-4   \
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
--len_learnable_prompt $len_prompt  \
--base_mil $base_mil  \
--slide_align $slide_align  \
--pretrain_epoch $pretrain_epoch  \
--max_epochs $max_epochs  \
--w_infonce $w_infonce  \
--w_kl $w_kl