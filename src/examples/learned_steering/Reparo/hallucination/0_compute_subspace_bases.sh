# LLaVA, layer 14 (hallucination/POPE)
CUDA_VISIBLE_DEVICES=3 python src/analysis/compute_subspace_basis_milad.py \
    --shifts_path /research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/llava_14_average_all_pope_train_-1.pth \
    --save_path   /research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/reparo_subspace_U_llava_14_k100.pth \
    --k 100

# Qwen, layer 17
CUDA_VISIBLE_DEVICES=3 python src/analysis/compute_subspace_basis_milad.py \
    --shifts_path /research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/qwen2vlinstruct_17_average_all_pope_train_-1.pth \
    --save_path   /research/hal-afsharim/learn-to-steer/Hallucination/POPE/hallucination/shift_vectors/reparo_subspace_U_qwen_17_k100.pth \
    --k 100
