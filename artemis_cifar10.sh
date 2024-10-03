python ./defense/artemis.py \
--batch_size 100 --dataset_num_ratio 0.2 --epoch 100 \
--save_path ./record/defense/artemis_defense/ \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/sig_cifar10/attack_result.pt  \
--yaml_path ./config/defense/ft/cifar10.yaml --device cuda:1