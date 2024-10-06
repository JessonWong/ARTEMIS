
# attack lf
python ./attack/lf.py \
--epochs 500 --batch_size 512 --num_workers 4  \
--yaml_path ../config/attack/prototype/cifar10.yaml   \
--save_folder_name lf_attack_cifar10 \
--bd_yaml_path ../config/attack/lf/default.yaml \
--device cuda:0 --frequency_save 25 \
--pratio 0.2

# # attack blended
# python ./attack/blended.py --batch_size 512 --epochs 500 \
# --yaml_path ../config/attack/prototype/cifar10.yaml \
# --bd_yaml_path ../config/attack/blended/default.yaml \
# --save_folder_name blended_attack_cifar10 \
# --pratio 0.2 \
# --attack_label_trans all2all \
# --device cuda:1

# # attack inputaware
# python ./attack/inputaware.py --batch_size 512 --epochs 100 \
# --yaml_path ../config/attack/prototype/cifar10.yaml \
# --bd_yaml_path ../config/attack/inputaware/default.yaml \
# --save_folder_name inputaware_attack_cifar10 \
# --pratio 0.2

# # attack bpp
# python ./attack/bpp.py --batch_size 512 --epochs 100 \
# --yaml_path ../config/attack/prototype/cifar10.yaml \
# --bd_yaml_path ../config/attack/bpp/default.yaml \
# --save_folder_name bpp_attack_cifar10 \
# --pratio 0.2

# # attack wanet
# python ./attack/wanet.py --batch_size 512 --epochs 100 \
# --yaml_path ../config/attack/prototype/cifar10.yaml \
# --bd_yaml_path ../config/attack/wanet/default.yaml \
# --save_folder_name wanet_attack_cifar10 \
# --pratio 0.2 --device cuda:0

# # attack sig
# python ./attack/sig.py --batch_size 512 --epochs 100 \
# --yaml_path ../config/attack/prototype/cifar10.yaml \
# --bd_yaml_path ../config/attack/sig/default.yaml \
# --save_folder_name sig_cifar10 \
# --pratio 0.2 --device cuda:0