import json
from src.datasets.image_text_dataset import POPE_train_Dataset, POPE_test_Dataset
from src.models import get_model_class

# -----------------------------
# CONFIG
# -----------------------------
model_name_or_path = "llava-hf/llava-1.5-7b-hf"  # or "Qwen/Qwen2-VL-7B-Instruct"
processor_name = model_name_or_path

data_dir = "/research/hal-afsharim/learn-to-steer/data/pope/train"                  # or /path/to/POPE/test
annotation_file = "annotations.json"
dataset_name = "pope_train"                       # or "pope_test"
split = "all"                                     # for train: all/adversarial/popular/random
                                                  # for test: adversarial/popular/random
dataset_size = -1
seed = 0

output_json = "pope_contrastive_full_prompts.json"

# -----------------------------
# DATASET
# -----------------------------
if dataset_name == "pope_train":
    ds = POPE_train_Dataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        split=split,
        dataset_size=dataset_size,
        seed=seed,
        dataset_name=dataset_name,
        mode="train",
        prompt_template="llava",
    )
elif dataset_name == "pope_test":
    ds = POPE_test_Dataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        split=split,
        dataset_size=dataset_size,
        seed=seed,
        dataset_name=dataset_name,
        mode="train",
        prompt_template="llava",
    )
else:
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

# -----------------------------
# MODEL WRAPPER (for chat template)
# -----------------------------
class Args:
    local_files_only = True
    cache_dir = None
    message_format = "role"

args = Args()
model_class = get_model_class(
    model_name_or_path=model_name_or_path,
    processor_name=processor_name,
    device="cuda:0",            # not used for prompt construction
    logger=None,
    args=args,
)

rows = []
for i, item in enumerate(ds.data[:3]):
    text = item["instruction"]
    gt_response = item["response"]

    # Positive (same answer)
    pos_instruction, pos_response, pos_continue = ds.construct_input(
        text=text,
        response=gt_response,
        force_answer=True,
        forced_answer_true=True,
        descriptive_answer=False,
        scenario=item.get("scenario", None),
    )

    # Negative (opposite answer)
    neg_instruction, neg_response, neg_continue = ds.construct_input(
        text=text,
        response=gt_response,
        force_answer=True,
        forced_answer_true=False,
        descriptive_answer=False,
        scenario=item.get("scenario", None),
    )

    # Build final prompt strings exactly like pipeline
    pos_prompt = model_class.preprocess_text(
        instruction=pos_instruction,
        response=pos_response,
        generation_mode=True,
        continue_final_message=pos_continue,
    )

    neg_prompt = model_class.preprocess_text(
        instruction=neg_instruction,
        response=neg_response,
        generation_mode=True,
        continue_final_message=neg_continue,
    )

    rows.append({
        "idx": i,
        "image": item["image"],
        "scenario": item.get("scenario", None),
        "original_instruction": text,
        "ground_truth_response": gt_response,
        "forced_positive_response_text": pos_response,
        "forced_negative_response_text": neg_response,
        "positive_full_prompt": pos_prompt,
        "negative_full_prompt": neg_prompt,
    })

with open(output_json, "w") as f:
    json.dump(rows, f, indent=2)

print(f"Saved {len(rows)} rows to {output_json}")