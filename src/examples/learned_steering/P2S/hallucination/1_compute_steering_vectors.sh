model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava



# features_dir=/data/khayatan/Hallucination/POPE/hallucination/features
features_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/features

shift_type=average
# save_dir=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors
save_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors

analysis_name=learnable_steering

# Define once at the top of the script
select_gpu() {
    local required_free=${1:-37}
    while true; do
        while IFS=',' read -r idx free total; do
            free_pct=$((free * 100 / total))
            if [ "$free_pct" -ge "$required_free" ]; then
                echo "$idx"
                return 0
            fi
        done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
        echo "No GPU with >= ${required_free}% free. Retrying in 30s..." >&2
        sleep 30
    done
}


GPU_ID=$(select_gpu 32)
echo "Selected GPU $GPU_ID"


for split in adversarial popular random; do

    for i in 14; do

        pos_features_name=save_hidden_states_for_l2s_llava_pope_test_features_pos_answers_${i}_${split}_-1.pth
        neg_features_name=save_hidden_states_for_l2s_llava_pope_test_features_neg_answers_${i}_${split}_-1.pth


        modules_to_hook="language_model.model.layers.${i};language_model.model.layers.${i}"
        save_filename=${split}_pope_test_-1


        CUDA_VISIBLE_DEVICES=$GPU_ID python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name} \
            --seed 0
    done
done

# """
# Saving individual shift vectors in : 
# /data/khayatan/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_random_pope_test_-1.pth
# """




# for Qwen2vlinstruct


model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct
# cache_dir=/data/khayatan/cache/
cache_dir=/research/hal-afsharim/cache

# features_dir=/data/khayatan/Hallucination/POPE/hallucination/features
features_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/features
shift_type=average
# save_dir=/data/khayatan/Hallucination/POPE/hallucination/shift_vectors
save_dir=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors

analysis_name=learnable_steering

# Re-select before Qwen (in case GPU state changed)
GPU_ID=$(select_gpu 37)
echo "Selected GPU $GPU_ID"

for split in adversarial popular random; do

    for i in 17; do

        pos_features_name=save_hidden_states_for_l2s_qwen2vlinstruct_pope_test_features_pos_answers_${i}_${split}_-1.pth
        neg_features_name=save_hidden_states_for_l2s_qwen2vlinstruct_pope_test_features_neg_answers_${i}_${split}_-1.pth

        modules_to_hook="model.layers.${i};model.layers.${i}"
        save_filename=${split}_pope_test_-1


        CUDA_VISIBLE_DEVICES=$GPU_ID python src/analyse_features.py \
            --model_name_or_path $model_name_or_path \
            --cache_dir $cache_dir \
            --save_dir $save_dir \
            --analysis_name $analysis_name \
            --modules_to_hook $modules_to_hook \
            --save_filename ${save_filename} \
            --local_files_only \
            --shift_type $shift_type \
            --features_path ${features_dir}/${pos_features_name} ${features_dir}/${neg_features_name} \
            --seed 0
    done
done

