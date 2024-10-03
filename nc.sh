python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml \
--device cuda:0

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_lf_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml \
--device cuda:0

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_wanet_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml \
--device cuda:0

python ./defense/nc.py \
--batch_size 256 --epochs 100 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_sig_0_1/attack_result.pt  \
--save_path ./record/defense/nc_defense/ \
--yaml_path ./config/defense/nc/cifar10.yaml \
--device cuda:0

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/blended_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1