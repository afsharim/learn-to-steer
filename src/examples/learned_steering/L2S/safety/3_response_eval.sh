# model_name_or_path=llava-hf/llava-1.5-7b-hf
# model=llava

model_name_or_path=Qwen/Qwen2-VL-7B-Instruct
model=qwen

YOUR_SAVE_DIR=/research/hal-afsharim/learn-to-steer/id_steering/test_code/
# STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/id_steering/mmsb_steering_nets/llava_15_last_input_multi_mmsb_train_-1_test.pt
STEER_MODEL_NAME=/research/hal-afsharim/learn-to-steer/id_steering/mmsb_steering_nets/qwen2vlinstruct_14_last_input_multi_mmsb_train_-1_test.pt
steer_model_base=$(basename "$STEER_MODEL_NAME" .pt)

steering_alpha=2.2
analysis_name="safety_metrics"

#response_filename="qwen_mmsb_test_response_multi_2.2_l2s_temp.pth"
#response_filename="llava_vlguard_response_test_2.0_l2s.pth"
# response_filename="llava_mmsb_test_steer_l2s_multi_2.2_llava_15_last_input_multi_mmsb_train_-1_test.pth"
response_filename="qwen_mmsb_test_steer_l2s_multi_2.2_qwen2vlinstruct_14_last_input_multi_mmsb_train_-1_test.pth"
#response_filename="llava_mmsb_test_steer_default_multi_2.2_llava_mmsb_steering_net_v3_multi_nobias_K100.pth"
#response_filename="qwen_mmsb_test_steer_l2s_multi_2.2_qwen_14_last_input_multi_mmsb_train_-1_v2.pth"


CUDA_VISIBLE_DEVICES=0 python src/analyse_features.py \
    --model_name_or_path $model_name_or_path \
    --analysis_name $analysis_name \
    --predictions_path $response_filename \
    --save_filename False