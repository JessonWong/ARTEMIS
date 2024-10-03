
# attack lf
python ./attack/lf.py \
--epochs 100 --batch_size 512 --num_workers 4  \
--yaml_path ../config/attack/prototype/cifar10.yaml   \
--save_folder_name lf_attack_cifar10 \
--bd_yaml_path ../config/attack/lf/default.yaml \
--device cuda:0 --frequency_save 25 \
--pratio 0.2

# attack blended
python ./attack/blended.py --batch_size 512 --epochs 100 \
--yaml_path ../config/attack/prototype/cifar10.yaml \
--bd_yaml_path ../config/attack/blended/default.yaml \
--save_folder_name blended_attack_cifar10 \
--pratio 0.2 \
--attack_label_trans all2all \
--device cuda:1

# attack inputaware
python ./attack/inputaware.py --batch_size 512 --epochs 100 \
--yaml_path ../config/attack/prototype/cifar10.yaml \
--bd_yaml_path ../config/attack/inputaware/default.yaml \
--save_folder_name inputaware_attack_cifar10 \
--pratio 0.2

# attack bpp
python ./attack/bpp.py --batch_size 512 --epochs 100 \
--yaml_path ../config/attack/prototype/cifar10.yaml \
--bd_yaml_path ../config/attack/bpp/default.yaml \
--save_folder_name bpp_attack_cifar10 \
--pratio 0.2

# attack wanet
python ./attack/wanet.py --batch_size 512 --epochs 100 \
--yaml_path ../config/attack/prototype/cifar10.yaml \
--bd_yaml_path ../config/attack/wanet/default.yaml \
--save_folder_name wanet_attack_cifar10 \
--pratio 0.2

# attack sig
python ./attack/sig.py --batch_size 512 --epochs 100 \
--yaml_path ../config/attack/prototype/cifar10.yaml \
--bd_yaml_path ../config/attack/sig/default.yaml \
--save_folder_name sig_cifar10 \
--pratio 0.2


### defense sig
python ./defense/fp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/sig_cifar10/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/bpp_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/sig_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/sig_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /content/BackdoorBench/record/sig_cifar10/attack_result.pt \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/sig_cifar10/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml


### defense lf
python ./defense/fp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/lf_attack_cifar10/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/lf_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/lf_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/lf_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /content/BackdoorBench/record/lf_attack_cifar10/attack_result.pt \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/lf_attack_cifar10/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml

### defense wanet
python ./defense/fp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/wanet_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_wanet_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/nad_wanet_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /content/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml

### defense bpp
python ./defense/fp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /content/BackdoorBench/record/bpp_attack_cifar10/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/bpp_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/bpp_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/bpp_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /content/BackdoorBench/record/bpp_attack_cifar10/attack_result.pt \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml

### defense inputaware
python ./defense/fp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/inputaware_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/artemis.py \
--batch_size 100 --dataset_num_ratio 0.08 --epoch 100 \
--save_path ./record/defense/artemis_defense/ \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--yaml_path ./config/defense/ft/cifar10.yaml

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/inputaware_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_inputaware_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/inputaware_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/nad_inputaware_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_inputaware_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml

### defense blended
python ./defense/fp.py \
--batch_size 256 --epochs 1000 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_blended_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/blended_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/sage_wzx.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /content/BackdoorBench/record/blended_attack_cifar10/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_badnet_gtsrb_size_10_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/nad.py \
--batch_size 256 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_blended_0_1/attack_result.pt  \
--teacher_model_loc /content/BackdoorBench/record/defense/wanet_defense/wanet_defense_cifar10_L4/defense_result.pt  \
--save_path ./record/defense/sage_badnet_gtsrb_size_10_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01

python ./defense/anp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--save_path ./record/defense/anp_defense/ \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_blended_0_1/attack_result.pt  \
--yaml_path ./config/defense/anp/cifar10.yaml

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_blended_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense \
--yaml_path ./config/defense/nc/cifar10.yaml
