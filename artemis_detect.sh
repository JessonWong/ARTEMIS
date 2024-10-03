python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/blended_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/inputaware_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/bpp_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/lf_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/sig_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1

python ./defense/artemis_detect.py \--batch_size 256 --epochs 100 --num_workers 4 \--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/wanet_attack_cifar10/attack_result.pt  \--save_path ./record/defense/nc_defense/ \--yaml_path ./config/defense/nc/cifar10.yaml \--device cuda:1