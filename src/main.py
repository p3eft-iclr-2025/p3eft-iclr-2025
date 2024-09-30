#!/usr/bin/env python
# using parts borrowed from Transformers' run_glue.py example from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch.utils.checkpoint import checkpoint
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version  # , send_example_telemetry
from transformers.utils.versions import require_version

from utils_args import parse_args, task_to_keys
from utils_data import get_data, preprocess_datasets
from utils_model import (
    detect_last_checkpoint,
    get_base_model,
    get_model_multiple_loras,
    get_tokenizer,
    is_distributed_environment,
    get_pslf_model,
)
from utils_trainer import TrainerWithMetrics

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


def main():

    model_args, data_args, training_args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    raw_datasets, is_regression, label_list = get_data(
        model_args, data_args, training_args
    )

    base_model = get_base_model(
        model_args, finetuning_task=data_args.task_name, num_labels=len(label_list)
    )

    tokenizer = get_tokenizer(model_args)

    train_dataset, eval_dataset, predict_dataset, raw_datasets = preprocess_datasets(
        raw_datasets,
        data_args,
        training_args,
        base_model,
        tokenizer,
        label_list,
        is_regression,
        add_fmlg_labels=model_args.add_fmlg_labels,
        dp_eps=model_args.dp_eps,
        k_total=model_args.k_total,
    )
    
    # # TEST PSLF label flip
    # if model_args.add_fmlg_labels:
    #     print([np.sum(np.array(train_dataset['label']) == i) for i in range(2)])
    #     print([np.sum(np.array(train_dataset['flipped_labels']) == i) for i in range(2)])
    #     print(np.sum(np.sum(np.abs(np.array(train_dataset['label']) - np.array(train_dataset['flipped_labels'])))), len(train_dataset['label']) - np.sum(np.sum(np.abs(np.array(train_dataset['label']) - np.array(train_dataset['flipped_labels'])))), len(train_dataset['label']))
    #     print([np.sum(np.array(train_dataset['pseudo_labels']) == i) for i in range(model_args.k_total)])
    #     return

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    # if model_args.method_name == "pppeft":
    if not model_args.add_fmlg_labels:
        model = get_model_multiple_loras(base_model, model_args, training_args, num_labels=len(label_list))
    else:
        model = get_pslf_model(base_model, model_args, training_args, num_labels=len(label_list))

    # Initialize our Trainer
    trainer = TrainerWithMetrics(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        task_name=data_args.task_name,
        is_regression=is_regression,
    )

    last_checkpoint = detect_last_checkpoint(training_args)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        print("training_args.resume_from_checkpoint is not None")
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        print("last_checkpoint is not None:")
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print('\n'*3)

        with torch.no_grad():
            i = 0
            for name, p in trainer.model.named_parameters():
                if "ora" in name:
                    print(name, p[0][:8])
                    i += 1
                    if i == 4:
                        break

        print('\n'*3)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
#         trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            # tasks.append("mnli-mm")
            # valid_mm_dataset = raw_datasets["validation_mismatched"]
            # if data_args.max_eval_samples is not None:
            #     max_eval_samples = min(
            #         len(valid_mm_dataset), data_args.max_eval_samples
            #     )
            #     valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            # eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()