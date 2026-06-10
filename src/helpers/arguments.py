import argparse
from typing import List

__all__ = ["get_arguments"]


def parse_list_of_lists(arg):
    list_of_lists = [lst.split(",") for lst in arg.split(";")]
    return list_of_lists


def get_arguments():
    parser = argparse.ArgumentParser(description="XL-VLMs arguments parser")

    # General
    parser.add_argument("--seed", type=int, default=21, help="Global seed.")

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="The path or name of the pre-trained model.",
    )
    parser.add_argument("--cache_dir", type=str, help="Where to load the model from.")


    parser.add_argument(
        "--processor_name",
        type=str,
        default=None,
        help="Name of the processor, if different from the model_name_or_path; to be loaded from HF.",
    )

    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="Load HF models from local.",
    )

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="llava",
        help="Task prompts to be applied to each input instruction.",
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Path to data directory."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="annotations.json",
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="annotations.json",
        help="Path to the questions file.",
    )
    parser.add_argument(
        "--answer_type_to_answer",
        type=str,
        default=None,
        help="Path to the answer_type_to_answer file.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")

    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of examples to process. -1 means the full dataset.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="coco", help="Dataset name."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for data loading"
    )

    # Hooks
    parser.add_argument(
        "--hook_names",
        nargs="+",
        help="List of hook names.",
        default=None,
    )
    parser.add_argument(
        "--modules_to_hook",
        type=parse_list_of_lists,
        default=None,
        help='A list of lists. Example format: "1,2,3;4,5,6". Contains the lists of modules to hook for each hook.',
    )
    parser.add_argument(
        "--exact_match_modules_to_hook",
        default=False,
        action="store_true",
        help="Exact match when searching for modules to hook.",
    )
    parser.add_argument(
        "--hook_postprocessing_name",
        type=str,
        default="save_hidden_states",
        help="Save hook output",
    )

    # Save hidden states
    parser.add_argument(
        "--token_idx", type=int, default=None, help="Save only the token at this index."
    )
    parser.add_argument(
        "--token_start_end_idx",
        nargs="+",
        help="[start_idx, end_idx] to select the beginning and end index of tokens to be saved.",
        default=None,
    )
    parser.add_argument("--save_dir", type=str, default="results")

    parser.add_argument("--save_filename", type=str, default="results")

    parser.add_argument("--save_analysis", type=bool, default=True)

    parser.add_argument(
        "--features_path",
        nargs="+",
        help="list of paths to features.",
        default=None,
    )
    # Feature decomposition

    parser.add_argument(
        "--analysis_saving_path",
        type=str,
        help="Path to save the analysis (components, decompositions, grounding words, MAS images).",
        default="results/analysis.pth",
    )

    parser.add_argument(
        "--analysis_name",
        type=str,
        help="Analysis method.",
        default="decompose_activations",
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        help="Key for features to extract.",
        default="hidden_states",
    )
    parser.add_argument(
        "--module_to_decompose",
        type=str,
        help="Module name to prepare representation matrix for and decompose",
        default=None,
    )
    parser.add_argument(
        "--decomposition_extract_pos",
        type=int,
        default=None,
        help="Local token position for representation extraction for decomposition. None means representions will be averaged over all stored positions",
    )
    parser.add_argument(
        "--decomposition_method",
        type=str,
        help="Method for dictionary learning or decomposition (eg. pca, kmeans, snmf)",
        default=None,
    )
    parser.add_argument(
        "--num_concepts",
        type=int,
        nargs="+",
        help="Number of concepts for dictionary learning.",
        default=None,
    )
    parser.add_argument(
        "--dl_max_iter",
        type=int,
        default=5000,
        help="Max number of iterations for dictionary learning optimization.",
    )
    parser.add_argument(
        "--num_grounded_text_tokens",
        type=int,
        default=10,
        help="Number of concepts for dictionary learning.",
    )

    parser.add_argument(
        "--pre_num_top_tokens",
        type=int,
        default=50,
        help="Number of words to try to ground before filtering the top num_grounded_text_tokens.",
    )

    parser.add_argument(
        "--num_most_activating_samples",
        type=int,
        default=5,
        help="Number of most activated samples.",
    )

    # Generation mode
    parser.add_argument(
        "--generation_mode",
        default=False,
        action="store_true",
        help="True for autoregressive generation mode. False for teacher forcing.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Max numberof generated tokens."
    )

    # Saving hidden states for token of interest
    parser.add_argument(
        "--token_of_interest_num_samples",
        type=int,
        default=-1,
        help="Number of samples to save.",
    )
    parser.add_argument(
        "--select_token_of_interest_samples",
        default=False,
        action="store_true",
        help="Filtering samples containing the token of interest.",
    )
    parser.add_argument(
        "--allow_different_variations_of_token_of_interest",
        default=False,
        action="store_true",
        help="Relaxation for finding the token of interest.",
    )
    parser.add_argument(
        "--token_of_interest",
        type=str,
        default=None,
        help="The entity that you want to extract the representation of from hidden states.",
    )
    parser.add_argument(
        "--token_of_interest_key",
        type=str,
        default="response",
        help="Search for token of interest in the value of this sample key.",
    )
    parser.add_argument(
        "--token_of_interest_class",
        type=str,
        default=None,
        help="Select samples containing the words in this toi class.",
    )
    parser.add_argument(
        "--token_of_interest_idx",
        type=int,
        default=None,
        help="The tokenization of the entity that you want to extract the representation of from hidden states.",
    )
    parser.add_argument(
        "--token_of_interest_start_token",
        type=int,
        default=0,
        help="Start looking for token of interest from this index.",
    )
    parser.add_argument(
        "--end_special_tokens",
        nargs="+",
        help="Extract hidden states before this token.",
        default="",
    )
    parser.add_argument(
        "--save_only_generated_tokens",
        default=False,
        action="store_true",
        help="Consider only generated tokens when saving hidden states.",
    )
    ## Steering methods and cluster analysis
    parser.add_argument(
        "--load_matched_features",
        default=False,
        action="store_true",
        help="Load features of the same examples.",
    )
    parser.add_argument(
        "--steering_method",
        type=str,
        default="shift_of_means",
        help="Name of the steering method.",
    )
    parser.add_argument(
        "--steering_alpha",
        type=float,
        default=1,
        help="Intervention strength.",
    )
    parser.add_argument(
        "--category_of_interest",
        type=str,
        default="",
        help="Target category to intervene on.",
    )
    parser.add_argument(
        "--base_features_key",
        type=str,
        default="hidden_states",
        help="Name of base features used to compute the steering vector.",
    )
    parser.add_argument(
        "--shift_vector_path",
        nargs="+",
        help="list of paths to features.",
        default=None,
    )
    parser.add_argument(
        "--shift_vector_key",
        type=str,
        default="steering_vector",
        help="Path to steering vector.",
    )
    parser.add_argument(
        "--shift_type",
        type=str,
        default="steering_vector",
        help="Path to steering vector.",
    )
    parser.add_argument(
        "--start_prompt_token_idx_steering",
        type=int,
        default=0,
        help="Apply steering starting from this token idx of the prompt.",
    )

    # Learned steering (L2S) and P2S
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="The size of the learned steering mode that is to be loaded.",
    )
    parser.add_argument(
        "--val_features_path",
        nargs="+",
        help="Paths to val features for pope pre-split val: pos, neg, cxt (same order as --features_path).",
        default=None,
    )
    parser.add_argument(
        "--individual_shift",
        default=False,
        action="store_true",
        help="The input specific shift (P2S). When this argument is given, the given shift vector should be a list of vectors! each vector being a sample specific shift!",
    )
    parser.add_argument(
        "--reparo_encoder_path",
        type=str,
        default=None,
        help="Path to Reparo encoder checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--reparo_score_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to Reparo score before using it to modulate shift.",
    )
    parser.add_argument(
        "--reparo_score_bias",
        type=float,
        default=0.0,
        help="Additive bias applied to Reparo score after scaling.",
    )
    parser.add_argument(
        "--reparo_score_reduction",
        type=str,
        default="mean",
        choices=["mean", "l2", "first"],
        help="How to reduce encoder output z to a scalar score when it has more than one dimension.",
    )
    parser.add_argument(
        "--reparo_score_activation",
        type=str,
        default="identity",
        choices=["identity", "sigmoid", "tanh", "relu"],
        help="Activation applied to the reduced Reparo score.",
    )
    parser.add_argument(
        "--reparo_score_min",
        type=float,
        default=None,
        help="Optional lower clamp for Reparo score after activation.",
    )
    parser.add_argument(
        "--reparo_score_max",
        type=float,
        default=None,
        help="Optional upper clamp for Reparo score after activation.",
    )
    parser.add_argument(
        "--reparo_z_threshold",
        type=float,
        default=-0.05,
        help="Threshold used in Reparo optimization stop/trigger condition.",
    )
    parser.add_argument(
        "--reparo_z_target",
        type=float,
        default=-0.06,
        help="Target z value used by Reparo optimization loss.",
    )
    parser.add_argument(
        "--reparo_lr",
        type=float,
        default=5e-2,
        help="Learning rate for the Reparo AdamW optimizer.",
    )
    parser.add_argument(
        "--reparo_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the Reparo AdamW optimizer.",
    )
    parser.add_argument(
        "--reparo_subspace_basis_path",
        type=str,
        default=None,
        help="Path to a .pth with {'U': (D, k), 'mean_shift': (D,)}. When set, Reparo constrains delta = U @ c.",
    )
    parser.add_argument(
        "--delta_save_path",
        type=str,
        default=None,
        help="If set, each computed Reparo delta is appended to a list and saved as a .pt file at this path.",
    )

    parser.add_argument(
        "--force_answer",
        action="store_true",
        default=False,
        help="Whether an answer shall be forced to the assistant. In this case, the message_format should be none.",
    )
    
    parser.add_argument(
        "--forced_answer_true",
        action="store_true",
        default=False,
        help="If force_answer is true, then this answer can be the true answer, or a false one.",
    )

    parser.add_argument(
        "--descriptive_answer",
        action="store_true",
        default=False,
        help="Whether the instruction should ask for more than just yes/no answer.",
    )

    parser.add_argument(
        "--model_response",
        action="store_true",
        default=False,
        help="Append the reformulated question suffix but let the model generate freely (no forced completion).",
    )



    # Evaluation
    parser.add_argument(
        "--captioning_metrics",
        type=List[str],
        default=["CIDEr"],
        help='The cpationing metrics to compute when "captioning_metrics" is in the hook_name (CIDEr, Bleu, etc ...).',
    )
    parser.add_argument(
        "--predictions_token_of_interest",
        nargs="+",
        help="token of interests expected in the model prediction.",
        default=None,
    )
    parser.add_argument(
        "--targets_token_of_interest",
        nargs="+",
        help="token of interest expected to be in the ground truth.",
        default=None,
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Save model predictions.",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Path to model predictions.",
    )

    # Clusters analysis
    parser.add_argument(
        "--origin_model_analysis_path",
        type=str,
        help="Path where the analysis corresponding to the original model (starting point model) is saved.",
        default=None,
    )

    parser.add_argument(
        "--origin_model_feature_path",
        type=str,
        help="Path where the features corresponding to the original model (starting point model) is saved.",
        default=None,
    )

    parser.add_argument(
        "--dest_model_analysis_path",
        type=str,
        help="Path where the analysis corresponding to the destination model (starting point model) is saved.",
        default=None,
    )

    parser.add_argument(
        "--dest_model_feature_path",
        type=str,
        help="Path where the features corresponding to the destination model (starting point model) is saved.",
        default=None,
    )

    parser.add_argument(
        "--visualize_concepts",
        default=False,
        action="store_true",
        help="To visualize grounding of original, shifted, and fine-tuned concepts by their grounding.",
    )

    parser.add_argument(
        "--compute_recovery_metrics",
        default=False,
        action="store_true",
        help="Whether to compute the word and mas recovery when shifting the original concepts using the extracted shift vectors.",
    )

    parser.add_argument(
        "--compute_stat_shift_vectors",
        default=False,
        action="store_true",
        help="Whether to compute the shift vectors statistics or not.",
    )

    # Samples selected from a set of ids in a file
    parser.add_argument(
        "--select_samples_from_ids",
        default=False,
        action="store_true",
        help="Filtering samples using a given set of ids for images.",
    )

    parser.add_argument(
        "--path_to_samples_ids",
        type=str,
        default=None,
        help="Path to the file with ids of samples to be filtered.",
    )
    
    
    parser.add_argument("--save_behavioral_separation", action="store_true", 
                        help="Split saved features into Correct/Incorrect bins based on Ground Truth.")
    
    parser.add_argument("--global_center_path", type=str, 
                        default=None, help="Path to the global center tensor for Affine Rotation")

    return parser.parse_args()
