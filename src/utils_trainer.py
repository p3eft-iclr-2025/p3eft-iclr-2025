import os
from nirvana_utils import copy_out_to_snapshot
import wandb

import evaluate
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from utils_model import (
        compute_kmeans_acc, 
        compute_logreg_acc, 
        compute_spectral_auc, 
        compute_norm_auc
)

class TrainerWithMetrics(Trainer):
    # Custom class with compute_metrics function that has access to the Trainer object.

    def __init__(self, *args, task_name=None, is_regression=False, **kwargs):
        super().__init__(*args, **kwargs, compute_metrics=self.compute_metrics)
        self.is_regression = is_regression
        self.metric = self.get_metric(task_name, is_regression)

    def train(self, *args, **kwargs):
        # Run evaluation before training starts
        results = self.evaluate()
        
        # Now, you can either print the results or process them as needed
        print("results before evaluate: -----------------------")
        for key, value in results.items():
            print(f"{key}: {value}")
        print("----------------------------------------------")
        # Proceed with standard training
        return super().train(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ) :        
        if (self.state.global_step % 1000 == 0) and False:  # edit to enable
            print ('SAVING CHECKPOINT')
            save_dir = self.args.output_dir[:-6] + "/checkpoints"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_dict = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            save_path = os.path.join(save_dir, f"checkpoint_{self.state.global_step}.pt")
            print(f"Step {self.state.global_step}; Saving `checkpoint` to {save_path}")
            torch.save(save_dict, save_path)

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    @staticmethod
    def get_metric(task_name, is_regression):
        # Get the metric function
        if task_name is not None:
            metric = evaluate.load("glue", task_name)
        elif is_regression:
            metric = evaluate.load("mse")
        else:
            metric = evaluate.load("accuracy")
        return metric

    def compute_metrics(self, p: EvalPrediction):
        # original metrics from run_glue.py example
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        label_ids = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
        result = self.metric.compute(predictions=preds, references=label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        result.update(self.compute_custom_metrics(p))
        return result

    def compute_custom_metrics(self, p: EvalPrediction):
        activations_kmeans = []
        activations_cv_logreg_score = []
        activations_spectral_auc = []
        activations_norm_auc = []
        label_ids = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
        for cur_lora_activations in p.predictions[1]:
            activations_kmeans.append(
                compute_kmeans_acc(cur_lora_activations, label_ids, self.model.num_labels)
            )
            activations_cv_logreg_score.append(
                compute_logreg_acc(
                    cur_lora_activations, cur_lora_activations, label_ids, label_ids
                )[1]
            )
            if self.model.num_labels == 2:
                activations_spectral_auc.append(
                    compute_spectral_auc(cur_lora_activations, label_ids)
                )
                activations_norm_auc.append(
                    compute_norm_auc(cur_lora_activations, label_ids)
                )

        shift_exist = not (hasattr(self.model, "n_of_loras") and not self.model.n_of_loras)
        if shift_exist:
            shifts_kmeans = []
            shifts_cv_logreg_score = []
            shifts_spectral_auc = []
            shifts_norm_auc = []
            for cur_lora_shifts in p.predictions[2]:
                shifts_kmeans.append(compute_kmeans_acc(cur_lora_shifts, label_ids, self.model.num_labels))
                shifts_cv_logreg_score.append(
                    compute_logreg_acc(
                        cur_lora_shifts, cur_lora_shifts, label_ids, label_ids
                    )[1]
                )
                if self.model.num_labels == 2:
                    shifts_spectral_auc.append(
                        compute_spectral_auc(cur_lora_shifts, label_ids)
                    )
                    shifts_norm_auc.append(
                        compute_norm_auc(cur_lora_shifts, label_ids)
                    )

        custom_metrics_dict = {
            "activations_kmeans_max": np.max(activations_kmeans),
            "activations_kmeans_mean": np.mean(activations_kmeans),
            "activations_cv_logreg_max": np.max(activations_cv_logreg_score),
            "activations_cv_logreg_mean": np.mean(activations_cv_logreg_score),
        }
        if shift_exist:
            custom_metrics_dict.update({
                "shifts_kmeans_max": np.max(shifts_kmeans),
                "shifts_kmeans_mean": np.mean(shifts_kmeans),
                "shifts_cv_logreg_max": np.max(shifts_cv_logreg_score),
                "shifts_cv_logreg_mean": np.mean(shifts_cv_logreg_score),
            })
        if self.model.num_labels == 2:
            custom_metrics_dict.update({
                "activations_spectral_auc_max": np.max(activations_spectral_auc),
                "activations_spectral_auc_mean": np.mean(activations_spectral_auc),
                "activations_norm_auc_max": np.max(activations_norm_auc),
                "activations_norm_auc_mean": np.mean(activations_norm_auc),
            })
            if shift_exist:
                custom_metrics_dict.update({    
                    "shifts_spectral_auc_max": np.max(shifts_spectral_auc),
                    "shifts_spectral_auc_mean": np.mean(shifts_spectral_auc),
                    "shifts_norm_auc_max": np.max(shifts_norm_auc),
                    "shifts_norm_auc_mean": np.mean(shifts_norm_auc),
                })

        activations_privacy_leak = 1 / self.model.num_labels
        shifts_privacy_leak = 1 / self.model.num_labels
        for k, v in custom_metrics_dict.items():
            if "logreg" in k or not "max" in k:
                continue
            if "activations" in k:
                activations_privacy_leak = max(activations_privacy_leak, v)
            elif "shifts" in k:
                shifts_privacy_leak = max(shifts_privacy_leak, v)
        custom_metrics_dict["activations_privacy_leak"] = activations_privacy_leak
        custom_metrics_dict["shifts_privacy_leak"] = shifts_privacy_leak
        
        return custom_metrics_dict