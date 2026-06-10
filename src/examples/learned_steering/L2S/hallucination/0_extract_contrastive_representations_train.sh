model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava


YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/train
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


save_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
dataset_name=pope_train
dataset_size=-1

max_new_tokens=100


hook_names=("save_hidden_states_for_l2s")
modules_to_hook=""



NUM_GPUS=6
FREE_MEM_THRESHOLD=0.36   # require at least 36% of memory.total free

# Find the first GPU (0..NUM_GPUS-1) with free/total >= FREE_MEM_THRESHOLD.
# Prints the GPU index, or nothing if none qualify.
find_free_gpu() {
    nvidia-smi --query-gpu=index,memory.free,memory.total \
               --format=csv,noheader,nounits \
        | awk -v n="$NUM_GPUS" -v thr="$FREE_MEM_THRESHOLD" -F', *' '
            $1 < n && ($2 / $3) >= thr { print $1; exit }
        '
}

# Block until some GPU has enough free memory, then print its index.
wait_for_free_gpu() {
    while :; do
        local g
        g=$(find_free_gpu)
        if [[ -n "$g" ]]; then
            echo "$g"
            return
        fi
        sleep 10
    done
}

# individual splits of the pope dataset adversarial popular random
for split in all; do

    for i in 14; do

        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_pos_answers_${i}_${split}_train_${dataset_size}"

        gpu_id=$(wait_for_free_gpu)

        CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --split $split \
            --annotation_file annotations.json \
            --dataset_size $dataset_size \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0 &
        sleep 20 # stagger the launches a bit to avoid contention
    done
done 



for split in all; do

    for i in 14; do

        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_neg_answers_${i}_${split}_train_${dataset_size}"

        gpu_id=$(wait_for_free_gpu)

        CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0 &
        sleep 20 # stagger the launches a bit to avoid contention
    done
done












model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct
# cache_dir=/data/khayatan/cache/
cache_dir=/research/hal-afsharim/cache


YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/train
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


save_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination
dataset_name=pope_train
dataset_size=-1

max_new_tokens=100


hook_names=("save_hidden_states_for_l2s")
modules_to_hook=""

# individual splits of the pope dataset adversarial popular random
for split in all; do

    for i in 17; do

        modules_to_hook="model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_pos_answers_${i}_${split}_train_${dataset_size}"

        gpu_id=$(wait_for_free_gpu)


        CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --split $split \
            --annotation_file annotations.json \
            --dataset_size $dataset_size \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --force_answer \
            --forced_answer_true \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0 &
        sleep 20 # stagger the launches a bit to avoid contention
    done
done 



for split in all; do

    for i in 17; do

        modules_to_hook="model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_neg_answers_${i}_${split}_train_${dataset_size}"
        gpu_id=$(wait_for_free_gpu)

        CUDA_VISIBLE_DEVICES=$gpu_id python src/save_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --data_dir $data_dir \
            --dataset_name $dataset_name \
            --dataset_size $dataset_size \
            --split $split \
            --save_dir $save_dir \
            --max_new_tokens $max_new_tokens \
            --hook_names $hook_names \
            --modules_to_hook $modules_to_hook \
            --generation_mode \
            --save_filename ${save_filename} \
            --force_answer \
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --seed 0 &
        sleep 20 # stagger the launches a bit to avoid contention

    done
done






# """
# /data/khayatan/Hallucination/POPE/hallucination/features/save_hidden_states_for_l2s_qwen2vlinstruct_pope_train_features_neg_answers_14_all_all_train_-1.pth

# """