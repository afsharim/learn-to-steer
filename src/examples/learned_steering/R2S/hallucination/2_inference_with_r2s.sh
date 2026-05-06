# model_name_or_path=llava-hf/llava-1.5-7b-hf
# model=llava

# YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
# STEERING_VECTORS=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_all_pope_train_-1_mean.pth
# # STEERING_VECTORS=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/llava_14_model_prediction_6300_mean.pth
# steering_vectors_base_name=$(basename "$STEERING_VECTORS" .pth)

# data_dir=${YOUR_DATA_DIR}
# save_dir=${YOUR_SAVE_DIR}

# dataset_name=pope_test
# dataset_size=-1
# max_new_tokens=100

# # CHANGE 1: Use a list of small angles (in radians) instead of a massive alpha
# thetas=0.2

# hook_names=("shift_hidden_states_angular_steer" "hallucination_metrics") 

# for split in adversarial popular random; do
#     for i in 14; do
#         for theta in "${thetas[@]}"; do
#             shift_vector_path=${STEERING_VECTORS}
            
#             # CHANGE 2: Updated filename to include "angular" and "theta"
#             save_filename="${model}_${dataset_name}_angular_steer_${i}_yes_no_${split}_theta_${theta}_${steering_vectors_base_name}"
#             modules_to_hook="language_model.model.layers.${i}"

#             CUDA_VISIBLE_DEVICES=7 python src/save_features.py \
#                 --model_name_or_path $model_name_or_path \
#                 --save_dir $save_dir \
#                 --data_dir $data_dir \
#                 --split $split \
#                 --dataset_size $dataset_size \
#                 --dataset_name $dataset_name \
#                 --hook_names "${hook_names[@]}" \
#                 --modules_to_hook $modules_to_hook \
#                 --generation_mode \
#                 --save_filename $save_filename \
#                 --save_predictions \
#                 --exact_match_modules_to_hook \
#                 --shift_vector_path $shift_vector_path \
#                 --steering_alpha $theta \
#                 --individual_shift \
#                 --max_new_tokens $max_new_tokens \
#                 --seed 0
#         done
#     done
# done



########################################################
# Qwen2vlinstruct


model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct
cache_dir=/research/hal-afsharim/cache

YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination

# Pointing to the Qwen equivalent of your 100-mean replicated vector
STEERING_VECTORS=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/qwen2vlinstruct_17_average_all_pope_train_-1_mean.pth
# STEERING_VECTORS=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/qwen2vlinstruct_17_model_prediction_6300_mean.pth
steering_vectors_base_name=$(basename "$STEERING_VECTORS" .pth)

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=pope_test
dataset_size=-1
max_new_tokens=100

# Use a list of small angles/proportions
thetas=0.2

hook_names=("shift_hidden_states_angular_steer" "hallucination_metrics") 

for split in adversarial popular random; do
    for i in 17; do
        for theta in "${thetas[@]}"; do
            shift_vector_path=${STEERING_VECTORS}
            
            # Updated filename to include "angular", "theta", and Qwen details
            save_filename="${model}_${dataset_name}_angular_steer_${i}_yes_no_${split}_theta_${theta}_${steering_vectors_base_name}"
            
            # Qwen specific module targeting
            modules_to_hook="model.layers.${i}"

            CUDA_VISIBLE_DEVICES=7 python src/save_features.py \
                --model_name_or_path $model_name_or_path \
                --cache_dir $cache_dir \
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
                --steering_alpha $theta \
                --individual_shift \
                --max_new_tokens $max_new_tokens \
                --seed 0
        done
    done
done

