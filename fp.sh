python ./defense/fp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_lf_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/inputaware_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/fp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_wanet_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/inputaware_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/fp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_sig_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/inputaware_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1

python ./defense/fp.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--defense_save_path ./record/defense/fp_defense/inputaware_cifar10 \
--yaml_path ./config/defense/fp/cifar10.yaml \
--once_prune_ratio 0.2 --acc_ratio 0.1
