GPU_POOL=(0 1 2 3)
declare -A PID_TO_GPU=()   # pid -> gpu id, for live background jobs

# Reap any finished background jobs so their GPU is released.
reap_finished() {
    local pid
    for pid in "${!PID_TO_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local g=${PID_TO_GPU[$pid]}
            unset 'PID_TO_GPU[$pid]'
            echo "[reap] pid $pid on GPU $g finished" >&2
        fi
    done
}

# Wait until a GPU in $GPU_POOL has >= $1 percent of memory free and is not
# currently held by one of our live background jobs. Echoes the GPU id.
pick_gpu() {
    local min_pct=$1
    while true; do
        reap_finished
        local busy=" "
        local pid
        for pid in "${!PID_TO_GPU[@]}"; do
            busy+="${PID_TO_GPU[$pid]} "
        done
        for g in "${GPU_POOL[@]}"; do
            [[ "$busy" == *" $g "* ]] && continue
            read free total < <(nvidia-smi -i "$g" \
                --query-gpu=memory.free,memory.total \
                --format=csv,noheader,nounits | tr ',' ' ')
            local pct=$(( 100 * free / total ))
            if [ "$pct" -ge "$min_pct" ]; then
                echo "$g"
                return 0
            fi
        done
        echo "[pick_gpu] no free GPU with >= ${min_pct}% memory; sleeping 30s" >&2
        sleep 30
    done
}

# Call right after launching a backgrounded job: track $! against the GPU it uses.
track_job() {
    PID_TO_GPU[$1]=$2
    echo "[launch] pid $1 on GPU $2" >&2
}



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

# llava pos
GPU=$(pick_gpu 31)
echo "[llava pos] using GPU $GPU"


# individual splits of the pope dataset adversarial popular random
for split in all; do

    for i in 16; do

        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_pos_answers_${i}_${split}_all_train_${dataset_size}"


        CUDA_VISIBLE_DEVICES=$GPU python src/save_features.py \
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
        track_job $! $GPU
    done
done

sleep 30   # let the model start loading so the next pick sees real usage


GPU=$(pick_gpu 31)
echo "[llava neg] using GPU $GPU"

for split in all; do

    for i in 16; do

        modules_to_hook="language_model.model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_neg_answers_${i}_${split}_all_train_${dataset_size}"


        CUDA_VISIBLE_DEVICES=$GPU python src/save_features.py \
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
        track_job $! $GPU
    done
done




sleep 30   # let the model start loading so the next pick sees real usage


GPU=$(pick_gpu 37)
echo "[qwen pos] using GPU $GPU"






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

    for i in 15; do

        modules_to_hook="model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_pos_answers_${i}_${split}_all_train_${dataset_size}"


        CUDA_VISIBLE_DEVICES=$GPU python src/save_features.py \
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
        track_job $! $GPU
    done
done

sleep 30   # let the model start loading so the next pick sees real usage

GPU=$(pick_gpu 37)
echo "[qwen neg] using GPU $GPU"

for split in all; do

    for i in 15; do

        modules_to_hook="model.layers.${i}"
        save_filename="${model}_${dataset_name}_features_neg_answers_${i}_${split}_all_train_${dataset_size}"


        CUDA_VISIBLE_DEVICES=$GPU python src/save_features.py \
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
        track_job $! $GPU
    done
done

# Wait for all backgrounded jobs to finish before the script exits
echo "Waiting for all jobs to finish..."
wait
echo "All done!"






# """
# /data/khayatan/Hallucination/POPE/hallucination/features/save_hidden_states_for_l2s_qwen2vlinstruct_pope_train_features_neg_answers_14_all_all_train_-1.pth

# """