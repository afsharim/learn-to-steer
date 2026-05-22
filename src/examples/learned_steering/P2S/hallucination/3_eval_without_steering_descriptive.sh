model_name_or_path=llava-hf/llava-1.5-7b-hf
model=llava

# YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/descriptive_test
# YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination


data_dir=${YOUR_DATA_DIR}
save_dir=${YOUR_SAVE_DIR}


# save_dir=/data/khayatan/Hallucination/POPE/hallucination
dataset_name=pope_test
dataset_size=-1

max_new_tokens=512
hook_names=("hallucination_metrics")



for split in descriptive; do
    save_filename="${model}_${dataset_name}_eval_no_steer_${split}_${dataset_size}_${max_new_tokens}"
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
        --generation_mode \
        --save_filename ${save_filename} \
        --save_predictions \
        --local_files_only \
        --end_special_tokens "</s>" \
        --seed 0\
        --descriptive_answer
done


# """
# Saving data to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_eval_no_steer_random_-1.json
# Saving 643 predictions to: 
# /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_llava_pope_test_eval_no_steer_random_-1_model_prediction.json

# """




# # for Qwen2vlinstruct


# model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
# model=qwen2vlinstruct
# # cache_dir=/data/khayatan/cache/
# cache_dir=/research/hal-afsharim/cache

# # YOUR_DATA_DIR=/data/khayatan/datasets/POPE/test
# YOUR_DATA_DIR=/research/hal-afsharim/learn-to-steer/data/pope/test
# # YOUR_SAVE_DIR=/data/khayatan/Hallucination/POPE/hallucination
# YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination


# data_dir=${YOUR_DATA_DIR}
# save_dir=${YOUR_SAVE_DIR}


# # save_dir=/data/khayatan/Hallucination/POPE/hallucination
# dataset_name=pope_test
# dataset_size=-1

# max_new_tokens=100
# hook_names=("hallucination_metrics")



# for split in adversarial popular random; do
#     save_filename="${model}_${dataset_name}_eval_no_steer_${split}_${dataset_size}"
#     CUDA_VISIBLE_DEVICES=0 python src/save_features.py \
#         --model_name_or_path $model_name_or_path \
#         --cache_dir $cache_dir \
#         --data_dir $data_dir \
#         --dataset_name $dataset_name \
#         --split $split \
#         --annotation_file annotations.json \
#         --dataset_size $dataset_size \
#         --save_dir $save_dir \
#         --max_new_tokens $max_new_tokens \
#         --hook_names $hook_names \
#         --generation_mode \
#         --save_filename ${save_filename} \
#         --save_predictions \
#         --local_files_only \
#         --end_special_tokens "</s>" \
#         --seed 0
# done


# # """
# # Saving data to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_eval_no_steer_random_-1.json
# # Saving 643 predictions to: 
# # /data/khayatan/Hallucination/POPE/hallucination/hallucination_metrics_qwen2vlinstruct_pope_test_eval_no_steer_random_-1_model_prediction.json

# # """
