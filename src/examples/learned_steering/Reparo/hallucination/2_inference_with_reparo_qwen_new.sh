# model_name_or_path=llava-hf/llava-1.5-7b-hf
# model=llava

# # YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
# YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# # YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
# YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
# # STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/llava_14_average_all_pope_train_-1.pt
# STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/spca_models/spca_model_llava_14_all_pope_train_-1.pth
# steer_model_base=$(basename "$STEER_MODEL_NAME" .pth)


# data_dir=${YOUR_DATA_DIR}
# save_dir=${YOUR_SAVE_DIR}


# dataset_name=pope_test
# dataset_size=-1
# max_new_tokens=100
# steering_alpha=4
# reparo_z_threshold=-0.05
# reparo_z_target=-0.6
# hook_names=("reparo" "hallucination_metrics") # should add the evaluation right here

# steering_method="reparo"

# for split in adversarial popular random; do


#     for i in 14; do
#         shift_vector_path=${STEER_MODEL_NAME}
#         save_filename="${model}_${dataset_name}_reparo_${i}_yes_no_${split}_${steering_alpha}_${reparo_z_threshold}_${reparo_z_target}_${steer_model_base}"
#         modules_to_hook="language_model.model.layers.${i}"


#         CUDA_VISIBLE_DEVICES=5 python src/save_features.py \
#             --model_name_or_path $model_name_or_path \
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





# # """
# # Saving data to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_steer_14_yes_no_adversarial_1_llava_14_average_all_pope_train_-1.json
# # Saving 643 predictions to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_steer_14_yes_no_adversarial_1_llava_14_average_all_pope_train_-1_model_prediction.json
# # """






model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct
# cache_dir=/data/khayatan/cache/
cache_dir=/research/hal-afsharim/cache/


# YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
# STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/qwen2vlinstruct_17_average_all_pope_train_-1.pt
# STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/qwen2vlinstruct_17_average_all_pope_train_-1.pt
# STEER_MODEL_NAME=/home/khayatan/learnable_steering/xl-vlms/0.0001_1_5e-05_last_input_average_400_17.pt
STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/spca_models/spca_model_qwen_17_all_pope_train_-1.pth
steer_model_base=$(basename "$STEER_MODEL_NAME" .pth)
# steer_model_base = "reparo_17_average_all_pope_train_-1.pt"

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


dataset_name=pope_test
dataset_size=-1
max_new_tokens=100
steering_alpha_list=(10)
reparo_z_threshold_list=(-0.01)
reparo_z_target_list=(-0.03)
reparo_lr_list=(5e-3)
reparo_weight_decay_list=(0 1e-4 1e-3)
hook_names=("reparo" "hallucination_metrics") # should add the evaluation right here

steering_method="reparo"

NUM_GPUS=4
MEM_FREE_RATIO=0.36        # require this fraction of total VRAM free before launching
LAUNCH_SETTLE_SECONDS=45   # after launching, ignore this GPU for a bit so weights can load
declare -A gpu_pid         # gpu_index -> pid currently running on that GPU
declare -A gpu_launch_ts   # gpu_index -> epoch seconds at last launch

# Print free/total VRAM ratio for GPU $1 (e.g. "0.7321"). Empty string on failure.
gpu_mem_free_ratio() {
    nvidia-smi --id="$1" --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' 'NR==1 && $2>0 { printf "%.4f", $1/$2 }'
}

# Return index of a GPU that (a) isn't running one of our jobs, (b) has settled
# since its last launch, and (c) reports >= MEM_FREE_RATIO free VRAM. Empty if none.
find_free_gpu() {
    local now=$(date +%s)
    for ((g=0; g<NUM_GPUS; g++)); do
        local pid=${gpu_pid[$g]:-}
        if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
            continue  # our own job is still running on this GPU
        fi
        local last=${gpu_launch_ts[$g]:-0}
        if (( now - last < LAUNCH_SETTLE_SECONDS )); then
            continue  # job just launched; nvidia-smi hasn't caught up yet
        fi
        local ratio=$(gpu_mem_free_ratio "$g")
        [[ -z $ratio ]] && continue
        if awk -v r="$ratio" -v t="$MEM_FREE_RATIO" 'BEGIN{ exit !(r >= t) }'; then
            echo "$g"
            return 0
        fi
    done
    return 1
}


for steering_alpha in "${steering_alpha_list[@]}"; do
    for idx in "${!reparo_z_threshold_list[@]}"; do
        reparo_z_threshold=${reparo_z_threshold_list[$idx]}
        reparo_z_target=${reparo_z_target_list[$idx]}
        for reparo_lr in "${reparo_lr_list[@]}"; do
        for reparo_weight_decay in "${reparo_weight_decay_list[@]}"; do
            for split in adversarial popular random; do
                # for i in 14; do
                for i in 17; do
                    shift_vector_path=${STEER_MODEL_NAME}
                    save_filename="${model}_${dataset_name}_reparo_${i}_yes_no_${split}_${steering_alpha}_${reparo_z_threshold}_${reparo_z_target}_lr${reparo_lr}_weight_decay${reparo_weight_decay}_${steer_model_base}"
                    modules_to_hook="model.layers.${i}"

                # Block until a GPU has >= MEM_FREE_RATIO free VRAM, then grab it.
                while true; do
                    gpu_id=$(find_free_gpu) && break
                    # Prefer waiting on a job exit; fall back to polling if none alive.
                    wait -n 2>/dev/null || sleep 10
                done


                CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
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
                    --steering_alpha $steering_alpha \
                    --reparo_z_threshold $reparo_z_threshold \
                    --reparo_z_target $reparo_z_target \
                    --reparo_lr $reparo_lr \
                    --reparo_weight_decay $reparo_weight_decay \
                    --individual_shift \
                    --max_new_tokens $max_new_tokens \
                    --seed 0 &

                gpu_pid[$gpu_id]=$!
                gpu_launch_ts[$gpu_id]=$(date +%s)
            done
        done
        done
    done
done

wait

# """
# Saving data to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_steer_17_yes_no_adversarial_1_llava_17_average_all_pope_train_-1.json
# Saving 643 predictions to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_steer_17_yes_no_adversarial_1_llava_17_average_all_pope_train_-1_model_prediction.json
# """
