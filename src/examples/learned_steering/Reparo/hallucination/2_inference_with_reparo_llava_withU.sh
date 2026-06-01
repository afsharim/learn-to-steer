model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
# STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/llava_14_average_all_pope_train_-1.pt
STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/spca_models/spca_model_llava_14_all_pope_train_-1.pth
steer_model_base=$(basename "$STEER_MODEL_NAME" .pth)

reparo_subspace_U_path=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/reparo_subspace_U_llava_14_k100.pth

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_test
dataset_size=-1
max_new_tokens=100
steering_alpha_list=(4)
reparo_z_threshold_list=(-0.015 0)
reparo_z_target_list=(-0.045 -0.015)
reparo_lr_list=(5e-1)
reparo_weight_decay_list=(1e-4)
layer_list=(14)
hook_names=("reparo" "hallucination_metrics")

steering_method="reparo"

NUM_GPUS=1
FREE_MEM_THRESHOLD=0.38   # require at least 36% of memory.total free

# # Find the first GPU (0..NUM_GPUS-1) with free/total >= FREE_MEM_THRESHOLD.
# # Prints the GPU index, or nothing if none qualify.
# find_free_gpu() {
#     nvidia-smi --query-gpu=index,memory.free,memory.total \
#                --format=csv,noheader,nounits \
#         | awk -v n="$NUM_GPUS" -v thr="$FREE_MEM_THRESHOLD" -F', *' '
#             $1 < n && ($2 / $3) >= thr { print $1; exit }
#         '
# }

# # Block until some GPU has enough free memory, then print its index.
# wait_for_free_gpu() {
#     while :; do
#         local g
#         g=$(find_free_gpu)
#         if [[ -n "$g" ]]; then
#             echo "$g"
#             return
#         fi
#         sleep 10
#     done
# }

for steering_alpha in "${steering_alpha_list[@]}"; do
    for idx in "${!reparo_z_threshold_list[@]}"; do
        reparo_z_threshold=${reparo_z_threshold_list[$idx]}
        reparo_z_target=${reparo_z_target_list[$idx]}
        for reparo_lr in "${reparo_lr_list[@]}"; do
        for reparo_weight_decay in "${reparo_weight_decay_list[@]}"; do
            for split in adversarial popular random; do
                for i in "${layer_list[@]}"; do
                    shift_vector_path=${STEER_MODEL_NAME}
                    save_filename="${model}_${dataset_name}_reparo_${i}_yes_no_${split}_${steering_alpha}_${reparo_z_threshold}_${reparo_z_target}_lr${reparo_lr}_weight_decay${reparo_weight_decay}_${steer_model_base}"
                    modules_to_hook="language_model.model.layers.${i}"

                gpu_id=3 #$(wait_for_free_gpu)
                echo "[launch] gpu=$gpu_id  split=$split  lr=$reparo_lr  alpha=$steering_alpha  zthr=$reparo_z_threshold  ztgt=$reparo_z_target"

                CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
                    --model_name_or_path $model_name_or_path \
                    --save_dir $save_dir \
                    --data_dir $data_dir \
                    --split $split \
                    --dataset_size $dataset_size \
                    --dataset_name $dataset_name \
                    --hook_names "${hook_names[@]}" \
                    --modules_to_hook $modules_to_hook \
                    --generation_mode \
                    --save_filename $save_filename \
                    --save_predictions \
                    --exact_match_modules_to_hook \
                    --shift_vector_path $shift_vector_path \
                    --steering_alpha $steering_alpha \
                    --reparo_z_threshold $reparo_z_threshold \
                    --reparo_z_target $reparo_z_target \
                    --reparo_lr $reparo_lr \
                    --reparo_weight_decay $reparo_weight_decay \
                    --reparo_subspace_basis_path $reparo_subspace_U_path \
                    --individual_shift \
                    --max_new_tokens $max_new_tokens \
                    --seed 0 

                # Give the new process time to claim memory so the next
                # iteration's free-memory check sees it as occupied.
                sleep 30
            done
        done
        done
    done
done
done

# Wait for any remaining background jobs
wait




# # """
# # Saving data to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_steer_14_yes_no_adversarial_1_llava_14_average_all_pope_train_-1.json
# # Saving 643 predictions to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_steer_14_yes_no_adversarial_1_llava_14_average_all_pope_train_-1_model_prediction.json
# # """






# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct
# # cache_dir=/data/khayatan/cache/
# cache_dir=/research/hal-afsharim/cache/


# # YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
# YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# # YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
# YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
# # STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/qwen2vlinstruct_17_average_all_pope_train_-1.pt
# # STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/qwen2vlinstruct_17_average_all_pope_train_-1.pt
# # STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/0.0001_1_5e-05_last_input_average_400_17.pt
# STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/spca_models/spca_model_qwen_17_all_pope_train_-1.pth
# steer_model_base=$(basename "$STEER_MODEL_NAME" .pth)
# # steer_model_base = "reparo_17_average_all_pope_train_-1.pt"

# data_dir=${YOUR_DATA_DIR}
# save_dir=${YOUR_SAVE_DIR}


# dataset_name=pope_test
# dataset_size=-1
# max_new_tokens=100
# steering_alpha=5
# reparo_z_threshold=-0.03
# reparo_z_target=-0.06
# hook_names=("reparo" "hallucination_metrics") # should add the evaluation right here



# for split in adversarial popular random; do



#     # for i in 14; do
#     for i in 17; do
#         shift_vector_path=${STEER_MODEL_NAME}
#         save_filename="${model}_${dataset_name}_reparo_${i}_yes_no_${split}_${steering_alpha}_${reparo_z_threshold}_${reparo_z_target}_${steer_model_base}"
#         modules_to_hook="model.layers.${i}"


#         CUDA_VISIBLE_DEVICES=7 python src/save_features.py \
#             --model_name_or_path $model_name_or_path \
#             --cache_dir $cache_dir \
#             --save_dir $save_dir \
#             --data_dir $data_dir \
#             --split $split \
#             --dataset_size $dataset_size \
#             --dataset_name $dataset_name \
#             --hook_names "${hook_names[@]}" \
#             --modules_to_hook $modules_to_hook \
#             --generation_mode \
#             --save_filename $save_filename \
#             --save_predictions \
#             --exact_match_modules_to_hook \
#             --shift_vector_path $shift_vector_path \
#             --steering_alpha $steering_alpha \
#             --reparo_z_threshold $reparo_z_threshold \
#             --reparo_z_target $reparo_z_target \
#             --individual_shift \
#             --max_new_tokens $max_new_tokens \
#             --seed 0
#     done
# done



# """
# Saving data to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_steer_17_yes_no_adversarial_1_llava_17_average_all_pope_train_-1.json
# Saving 643 predictions to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_steer_17_yes_no_adversarial_1_llava_17_average_all_pope_train_-1_model_prediction.json
# """
