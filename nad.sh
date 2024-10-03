# python ./defense/nad.py \
# --batch_size 32 --te_epochs 50 --num_workers 4 \
# --result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_bpp_0_1/attack_result.pt  \
# --teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
# --save_path ./record/defense/nad_bpp_defense/ \
# --yaml_path ./config/defense/nad/cifar10.yaml \
# --p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1

python ./defense/nad.py \
--batch_size 32 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_blended_0_1/attack_result.pt  \
--teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1

python ./defense/nad.py \
--batch_size 32 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_inputaware_0_1/attack_result.pt  \
--teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1

python ./defense/nad.py \
--batch_size 32 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_lf_0_1/attack_result.pt  \
--teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1

python ./defense/nad.py \
--batch_size 32 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_sig_0_1/attack_result.pt  \
--teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1

python ./defense/nad.py \
--batch_size 32 --te_epochs 50 --num_workers 4 \
--result_file /home/xuemeng/lzh/wzx/BackdoorBench/record/cifar10_vit_b_16_wanet_0_1/attack_result.pt  \
--teacher_model_loc /home/xuemeng/lzh/wzx/BackdoorBench/record/defense/anp_defense/defense_result.pt \
--save_path ./record/defense/nad_bpp_defense/ \
--yaml_path ./config/defense/nad/cifar10.yaml \
--p 2 --beta1 1000 --beta2 500 --beta3 1000 --ratio 0.4 --lr 0.01 --device cuda:1
