# ==========================================
# 1. LLaVA-1.5-7B Extraction (Layer 14)
# ==========================================
model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/train
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=pope_train
dataset_size=-1

# You can lower this to 10 if you want it to run faster, since we only need the 'yes/no'
max_new_tokens=100 
hook_names=("save_hidden_states_for_l2s")

for split in all; do
    for i in 14; do
        modules_to_hook="language_model.model.layers.${i}"
        
        # New naming convention: we call it a 'behavioral_harvest'
        save_filename="${model}_${dataset_name}_behavioral_harvest_${i}_${split}_all_train_${dataset_size}"

        CUDA_VISIBLE_DEVICES=2 python src/save_features.py \
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
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --save_behavioral_separation \
            --seed 0
            # REMOVED: --force_answer and --forced_answer_true
    done
done


# ==========================================
# 2. Qwen2-VL-7B-Instruct Extraction (Layer 17)
# ==========================================
model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen2vlinstruct
cache_dir=/research/hal-afsharim/cache

YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/train
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination

data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}

dataset_name=pope_train
dataset_size=-1
max_new_tokens=100
hook_names=("save_hidden_states_for_l2s")

for split in all; do
    for i in 17; do
        modules_to_hook="model.layers.${i}"
        
        save_filename="${model}_${dataset_name}_behavioral_harvest_${i}_${split}_all_train_${dataset_size}"

        CUDA_VISIBLE_DEVICES=2 python src/save_features.py \
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
            --exact_match_modules_to_hook \
            --end_special_tokens "</s>" \
            --save_behavioral_separation \
            --seed 0
    done
done