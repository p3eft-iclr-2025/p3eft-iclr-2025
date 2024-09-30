import logging
import os

import numpy as np
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from torch.utils.checkpoint import checkpoint
from transformers import (  # Trainer,; TrainingArguments,; default_data_collator,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import torch.distributed as dist
from torch.distributed.nn.functional import gather

from itertools import permutations
from copy import deepcopy

logger = logging.getLogger(__name__)


def is_distributed_environment():
    return 'LOCAL_RANK' in os.environ

def get_antisymmetric_n01_coefs(seed=0, n_of_loras=2, mult_std=0.0, 
    neck_width=1536, dtype=torch.float32):
    set_seed(seed)
    antisymmetric_noise_matrix = torch.zeros((n_of_loras, n_of_loras, neck_width), dtype=dtype)
    for i in range(n_of_loras):
        antisymmetric_noise_matrix[i, i] = 1.0 / n_of_loras
        for j in range(i + 1, n_of_loras):
            antisymmetric_noise_matrix[i, j] = torch.randn(neck_width, dtype=dtype) * 10 ** (mult_std / 2)
            antisymmetric_noise_matrix[j, i] = -antisymmetric_noise_matrix[i, j]
    return antisymmetric_noise_matrix.sum(1)  # .to(device, dtype=dtype)


def get_averaged_coefs(seed=0, n_of_loras=2, mult_std=0.0, 
    neck_width=1536, dtype=torch.float32):
    return torch.ones(n_of_loras, dtype=dtype) / n_of_loras * 10 ** (mult_std / 2) # .to(device, dtype=dtype)


coefs_name_arr = ["antisymmetric_n01", "averaged"]
name2genfunc = {
    "antisymmetric_n01": get_antisymmetric_n01_coefs,
    "averaged": get_averaged_coefs,
}


def linear_combination_of_activations(activations_arr, coefs, add_noies=0):
    for i in range(len(activations_arr)):
        if i:
            activations += activations_arr[i] * coefs[i]
        else:
            activations = activations_arr[i] * coefs[i]
    return activations

def regularizing_logreg_loss(activations, labels, logreg, 
    neck_width=1536, device="cpu", dtype=torch.float32):
    lin = nn.Linear(neck_width, 1, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(logreg.coef_))
        lin.bias.copy_(torch.from_numpy(logreg.intercept_))
    return F.binary_cross_entropy_with_logits(lin(activations).squeeze(), 1.0 - labels)


def multiclass_regularizing_logreg_loss(activations, labels, logreg, 
    neck_width=1536, device="cpu", dtype=torch.float32):
    num_labels = labels.max() + 1
    lin = nn.Linear(neck_width, num_labels, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(logreg.coef_))
        lin.bias.copy_(torch.from_numpy(logreg.intercept_))
    return F.cross_entropy(-lin(activations), labels)


def compute_kmeans_acc(activations, true_labels, num_labels):
    cluster_agreement = 0.0
    kmeans = KMeans(n_clusters=num_labels, n_init=10).fit(activations)
    new_labels = kmeans.labels_.copy()
    indexes_dict = {i: kmeans.labels_ == i for i in range(num_labels)}
    for permutation in permutations(np.arange(num_labels)):
        for i in range(num_labels):
            new_labels[indexes_dict[i]] = permutation[i]
        cluster_agreement = max(cluster_agreement, (new_labels == true_labels).mean())
    return cluster_agreement


def compute_spectral_auc(activations, true_labels):
    normalized = activations - np.mean(activations, axis=0)
    U, S, Vh = np.linalg.svd(normalized)
    scores = normalized @ Vh[0]
    auc = roc_auc_score(true_labels, scores)
    reversed_preds_auc = roc_auc_score(1 - true_labels, scores)
    return max(auc, reversed_preds_auc)


def compute_norm_auc(activations, true_labels):
    norms = np.linalg.norm(activations, axis=1)
    auc = roc_auc_score(true_labels, norms)
    reversed_preds_auc = roc_auc_score(1 - true_labels, norms)
    return max(auc, reversed_preds_auc)


def compute_logreg_acc(
    train_activations, val_activations, train_labels, val_labels, seed=0, cv=3
):
    logreg = get_fitted_logreg(train_activations, train_labels, seed=seed)
    val_acc = logreg.score(val_activations, val_labels)
    cv_acc = cross_val_score(logreg, train_activations, train_labels, cv=cv)
    return val_acc, np.mean(cv_acc)


def get_fitted_logreg(activations, labels, seed=0):
    logreg = LogisticRegression(random_state=seed, max_iter=150)
    logreg.fit(activations, labels)
    return logreg


def dc_regularizing_loss(activations, labels):
    batch_size = activations.shape[0]
    dist_matrix = torch.cdist(activations, activations)
    labels_dist_matrix = torch.cdist(labels.unsqueeze(-1).float(), 
                               labels.unsqueeze(-1).float())
    dist_matrix_normalized = (dist_matrix - dist_matrix.mean(dim=0, keepdim=True) 
                              - dist_matrix.mean(dim=1, keepdim=True) + dist_matrix.mean())
    labels_dist_matrix_normalized = (labels_dist_matrix - labels_dist_matrix.mean(dim=0, keepdim=True) - 
                                 labels_dist_matrix.mean(dim=1, keepdim=True) + labels_dist_matrix.mean())
    regularizing_loss = (((dist_matrix_normalized * labels_dist_matrix_normalized).sum() / (batch_size ** 2)) / 
        torch.sqrt(((dist_matrix_normalized * dist_matrix_normalized).sum() / (batch_size ** 2)) * 
                  ((labels_dist_matrix_normalized * labels_dist_matrix_normalized).sum() / (batch_size ** 2)))
    )
    return torch.log(regularizing_loss)


class ModelPSLF(torch.nn.Module):
    def __init__(
        self,
        base_model,
        num_labels=2,
        num_pseudo_labels=2,
        model_type="deberta",
        seed=0,
        device="cuda:0",
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        model_gradient_checkpointing=False,
    ):
        super().__init__()
        self.model_type = model_type
        if model_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
        
        def weights_init(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        if self.model_type == "deberta":
            self.model = base_model.deberta
            self.neck_width = self.model.config.pooler_hidden_size
            self.true_classifier = nn.Sequential(base_model.pooler, base_model.classifier)
            self.pseudo_classifier = nn.Sequential(deepcopy(base_model.pooler), nn.Linear(self.neck_width, num_pseudo_labels))
            self.pseudo_classifier.apply(weights_init)
            self.get_last_hidden_func = self._deberta_last_hidden
        elif self.model_type == "t5":
            self.model = base_model
            self.neck_width = self.model.config.d_model
            self.true_classifier = deepcopy(self.model.classification_head)
            self.pseudo_classifier = nn.Linear(self.neck_width, num_pseudo_labels)
            self.pseudo_classifier.apply(weights_init)
            self.model.classification_head = nn.Identity()
            self.get_last_hidden_func = self._t5_last_hidden
        elif self.model_type == "llama":
            base_model.config.pad_token_id = base_model.config.eos_token_id
            self.model = base_model.model
            self.neck_width = self.model.config.hidden_size
            self.true_classifier = base_model.score
            self.pseudo_classifier = nn.Linear(self.neck_width, num_pseudo_labels)
            self.pseudo_classifier.apply(weights_init)
            self.get_last_hidden_func = self._llama_last_hidden
        self.num_labels = num_labels
        self.num_pseudo_labels = num_pseudo_labels

        self.device = device
        self.seed = seed

        for p in self.model.parameters():
            p.requires_grad_(False)
        set_seed(0) 
        target_modules = None
        if self.model_type == "llama":
            target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
        peft_config = LoraConfig(
            peft_type=peft.PeftType.LORA,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.model.add_adapter(peft_config, adapter_name=f"lora")
        for name, p in self.model.named_parameters():
            if "lora" in name:
                p.requires_grad_(True)
        for p in self.true_classifier.parameters():
            p.requires_grad_(True)
        for p in self.pseudo_classifier.parameters():
            p.requires_grad_(True)

    def _llama_last_hidden(self, activations, input_ids):
        hidden_states = activations[0]       
        sequence_lengths = (
            torch.eq(input_ids, self.model.config.pad_token_id).long().argmax(-1) - 1
        ).to(self.device)
        return hidden_states[torch.arange(hidden_states.shape[0], device=self.device), sequence_lengths]

    def _deberta_last_hidden(self, activations, input_ids):
        return activations.last_hidden_state

    def _t5_last_hidden(self, activations, input_ids):
        return activations.logits

    def forward(self, input_ids, attention_mask, labels, pseudo_labels):

        if not self.model.training:
            with torch.random.fork_rng(
                devices=(torch.device("cpu"), self.device), enabled=True
            ):
                self.model.set_adapter([])
                baseline_activations = self.get_last_hidden_func(self.model(
                                            input_ids, attention_mask=attention_mask
                                        ), input_ids)
                self.model.set_adapter(["lora"])
                if self.model_type == "deberta":
                    baseline_activations = baseline_activations[:, 0]
            
        activations = self.get_last_hidden_func(
            self.model(input_ids, attention_mask=attention_mask), input_ids
        )

        pseudo_logits = self.pseudo_classifier(activations)
        true_logits = self.true_classifier(activations.clone().detach())
        
        loss = F.cross_entropy(pseudo_logits.view(-1, self.num_pseudo_labels), pseudo_labels.view(-1)) + F.cross_entropy(true_logits.view(-1, self.num_labels), labels.view(-1)) #* 0

        if not self.model.training:
            if self.model_type == "deberta":
                activations = activations[:, 0]
            shifts = [activations - baseline_activations]
        else:
            shifts = None

        return (
            loss,
            true_logits,
            [activations],
            shifts,
            activations,
        )


class ModelWithMultipleLoras(torch.nn.Module):
    def __init__(
        self,
        base_model,
        num_labels=2,
        model_type="deberta",
        get_head_input=linear_combination_of_activations,
        seed=0,
        device="cuda:0",
        n_of_loras=2,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        mult_std=0.0,
        method_name="averaged",
        activation_lr_rw=0.0,
        shift_lr_rw=0.0,
        activation_dc_rw=0.0,
        shift_dc_rw=0.0,
        loras_gradient_checkpointing=False,
        model_gradient_checkpointing=False,
        frozen_coefs=False,
    ):
        super().__init__()
        print(activation_lr_rw, shift_lr_rw, activation_dc_rw, shift_dc_rw)
        self.model_type = model_type
        self.n_of_loras = n_of_loras
        if method_name not in coefs_name_arr:
            raise NameError(
                f"Only options currently supported \
                are 'averaged' and 'antisymmetric_n01'."
            )
        self.method_name = method_name
        if model_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
            loras_gradient_checkpointing = True
        if self.model_type == "deberta":
            self.model = base_model.deberta
            self.classifier = nn.Sequential(base_model.pooler, base_model.classifier)
            self.neck_width = self.model.config.pooler_hidden_size
            self.get_last_hidden_func = self._deberta_last_hidden
        elif self.model_type == "t5":
            self.model = base_model
            self.classifier = self.model.classification_head
            self.model.classification_head = nn.Identity()
            self.neck_width = self.model.config.d_model
            self.get_last_hidden_func = self._t5_last_hidden
        elif self.model_type == "llama":
            base_model.config.pad_token_id = base_model.config.eos_token_id
            self.model = base_model.model
            self.neck_width = self.model.config.hidden_size
            self.classifier = base_model.score
            self.get_last_hidden_func = self._llama_last_hidden
        self.num_labels = num_labels
        print(self.num_labels)

        if self.num_labels == 2:
            self.regularizing_logreg_loss = regularizing_logreg_loss
        else:
            self.regularizing_logreg_loss = multiclass_regularizing_logreg_loss

        self.loras_gradient_checkpointing = loras_gradient_checkpointing
        self.get_head_input = get_head_input

        self.mult_std = mult_std
        self.activation_lr_rw = activation_lr_rw
        self.shift_lr_rw = shift_lr_rw
        self.activation_dc_rw = activation_dc_rw
        self.shift_dc_rw = shift_dc_rw

        self.device = device
        self.seed = seed

        self.coefs = (
            name2genfunc[self.method_name](
                seed=seed, n_of_loras=max(n_of_loras, 1), mult_std=mult_std, 
                neck_width=self.neck_width, dtype=self.model.dtype
            )
        ).to(device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        set_seed(0) 
        if self.n_of_loras:
            target_modules = None
            if self.model_type == "llama":
                target_modules = [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ]
            peft_config = LoraConfig(
                peft_type=peft.PeftType.LORA,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            for i in range(self.n_of_loras):
                self.model.add_adapter(peft_config, adapter_name=f"{self.method_name}_lora_{i}")
            for name, p in self.model.named_parameters():
                if "lora" in name:
                    p.requires_grad_(True)
            for p in self.classifier.parameters():
                p.requires_grad_(True)
        if not frozen_coefs:
            self.coefs = nn.Parameter(self.coefs)

    def _llama_last_hidden(self, activations, input_ids):
        hidden_states = activations[0]       
        sequence_lengths = (
            torch.eq(input_ids, self.model.config.pad_token_id).long().argmax(-1) - 1
        ).to(self.device)
        return hidden_states[torch.arange(hidden_states.shape[0], device=self.device), sequence_lengths]

    def _deberta_last_hidden(self, activations, input_ids):
        return activations.last_hidden_state

    def _t5_last_hidden(self, activations, input_ids):
        return activations.logits
        
    def _transformer_forward(self, input_ids, attention_mask, active_adapters):
        if active_adapters is not None:
            self.model.set_adapter(active_adapters)
        cur_lora_activations = self.model(
            input_ids, attention_mask=attention_mask
        )
        for name, p in self.model.named_parameters():
            if "lora" in name:
                p.requires_grad_(True)
        return self.get_last_hidden_func(cur_lora_activations, input_ids)

    def _choose_adapter_and_forward(self, i, input_ids, attention_mask):
        if self.n_of_loras:
            if i >= 0:
                self.active_adapters = [f"{self.method_name}_lora_{i}"]
            else:
                self.active_adapters = []
        else:
            self.active_adapters = None
        if self.loras_gradient_checkpointing:
            cur_lora_activations = checkpoint(
                self._transformer_forward,
                input_ids,
                attention_mask,
                self.active_adapters,
                use_reentrant=False,
            )
        else:
            cur_lora_activations = self._transformer_forward(
                input_ids, attention_mask, self.active_adapters
            )
        return cur_lora_activations

    def forward(self, input_ids, attention_mask, labels):
        if is_distributed_environment():
            rank = dist.get_rank()
        else:
            rank = 0
        
        if self.n_of_loras and (self.shift_lr_rw or self.shift_dc_rw or not self.model.training):
            with torch.random.fork_rng(
                devices=(torch.device("cpu"), self.device), enabled=True
            ):
                baseline_activations = self._choose_adapter_and_forward(
                    -1, input_ids, attention_mask
                )
                if self.model_type == "deberta":
                    baseline_activations = baseline_activations[:, 0]
        different_loras_activations = []
        for i in range(max(self.n_of_loras, 1)):
            if i < self.n_of_loras - 1:
                with torch.random.fork_rng(
                    devices=(torch.device("cpu"), self.device), enabled=True
                ):
                    activations = self._choose_adapter_and_forward(
                        i, input_ids, attention_mask
                    )
            else:
                activations = self._choose_adapter_and_forward(
                    i, input_ids, attention_mask
                )
            different_loras_activations.append(activations)
            
        activations = self.get_head_input(different_loras_activations, self.coefs)
        logits = self.classifier(activations)
        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        if self.model_type == "deberta":
            for i in range(len(different_loras_activations)):
                different_loras_activations[i] = different_loras_activations[i][:, 0]
        if is_distributed_environment():
            gathered_labels = gather(labels)
            labels = torch.cat(gathered_labels)
            dist.barrier()

        if self.activation_lr_rw or self.activation_dc_rw:
            for cur_activation in different_loras_activations:
                if is_distributed_environment():
                    cur_activations_list = gather(cur_activation)
                    cur_activation = torch.cat(cur_activations_list, dim=0)
                    dist.barrier()
                if not rank:
                    if self.activation_lr_rw:
                        if len(torch.unique(labels)) < self.num_labels:
                            continue
                        logreg = get_fitted_logreg(
                            cur_activation.detach().cpu().numpy(),
                            labels.cpu().numpy(),
                            seed=self.seed,
                        )
                        loss += (
                            self.regularizing_logreg_loss(
                                cur_activation,
                                labels,
                                logreg,
                                neck_width=self.neck_width,
                                device=self.device,
                                dtype=self.model.dtype
                            ) * self.activation_lr_rw
                        )
                    if self.activation_dc_rw:
                        loss += (self.activation_dc_rw 
                                * dc_regularizing_loss(cur_activation, labels))
                else:
                    loss += cur_activation.norm()
        
        different_loras_shifts = []
        if self.n_of_loras and (self.shift_lr_rw or self.shift_dc_rw or not self.model.training):
            for cur_activation in different_loras_activations:
                cur_shift = cur_activation - baseline_activations
                different_loras_shifts.append(cur_shift)
                if is_distributed_environment():
                    cur_shift_list = gather(cur_shift)
                    cur_shift = torch.cat(cur_shift_list, dim=0)
                    dist.barrier()
                if not rank:
                    if self.shift_lr_rw:
                        if len(torch.unique(labels)) < self.num_labels:
                            continue
                        logreg = get_fitted_logreg(
                            cur_shift.detach().cpu().numpy(),
                            labels.cpu().numpy(),
                            seed=self.seed,
                        )
                        loss += (
                            self.regularizing_logreg_loss(
                                cur_shift,
                                labels,
                                logreg,
                                neck_width=self.neck_width,
                                device=self.device,
                                dtype=self.model.dtype
                            ) * self.shift_lr_rw
                        )
                    if self.shift_dc_rw:
                        loss += (self.shift_dc_rw 
                                * dc_regularizing_loss(cur_shift, labels))
                else:
                    loss += cur_shift.norm()

        return (
            loss,
            logits,
            [cur_acts for cur_acts in different_loras_activations],
            different_loras_shifts,
            activations,
        )


    def update_state_dict_from_checkpoint(self, checkpoint_state_dict):
        # updates ModelWithMultipleLoras with params from checkpoint
        # only takes params present in CP, keeps rest intact
        # assuming that the CP has just the loras and head where `requires_grad==True`
        print(f"checkpoint contains {len(checkpoint_state_dict)} modules")
        sd0 = self.state_dict()
        print(f"state_dict contains {len(sd0)} modules")
        counter = 0 
        for n, p in sd0.items():    
            if n in checkpoint_state_dict.keys():
                sd0[n] = checkpoint_state_dict[n]
                counter += 1
        self.load_state_dict(sd0)
        print(f"updated {counter} modules")


def detect_last_checkpoint(training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # raise ValueError(
            #     f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            #     "Use --overwrite_output_dir to overcome."
            # )
            training_args.overwrite_output_dir = True
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            if not is_distributed_environment() or not dist.get_rank():
                print(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
    return last_checkpoint


def get_base_model(model_args, finetuning_task, num_labels):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=finetuning_task,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if "Llama" in model_args.model_name_or_path:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            quantization_config=quantization_config,

            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    print(model.config)
    print(model.dtype)
    print(model)
    print('\n'*5)
    return model


def get_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if "Llama" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_multiple_loras(base_model, model_args, training_args, num_labels):
    print(
        "grad checkpointing. loras: ",
        model_args.loras_gradient_checkpointing,
        "grad checkpointing model: ",
        model_args.model_gradient_checkpointing,
    )
    if "deberta" in model_args.model_name_or_path:
        model_type = "deberta"
    elif "Llama" in model_args.model_name_or_path:
        model_type = "llama"
    else:
        model_type = "t5"
        training_args.save_safetensors = False
    model_multiple_loras = ModelWithMultipleLoras(
        base_model=base_model,
        num_labels=num_labels,
        model_type=model_type,
        n_of_loras=model_args.n_of_loras,
        lora_rank=model_args.lora_rank,
        device=training_args.device,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        seed=training_args.seed,
        mult_std=model_args.mult_std,
        method_name=model_args.coefs_method_name,
        activation_lr_rw=model_args.activation_lr_rw,
        shift_lr_rw=model_args.shift_lr_rw,
        activation_dc_rw=model_args.activation_dc_rw,
        shift_dc_rw=model_args.shift_dc_rw,
        loras_gradient_checkpointing=model_args.loras_gradient_checkpointing,
        model_gradient_checkpointing=model_args.model_gradient_checkpointing,
        frozen_coefs=model_args.frozen_coefs,
    )
    print(model_multiple_loras)
    print('\n'*5)
    print('main model params: ',
        sum(
            p.numel()
            for p in model_multiple_loras.model.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print('classifier params: ',
        sum(
            p.numel()
            for p in model_multiple_loras.classifier.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print('total params: ',
        sum(p.numel() for p in model_multiple_loras.parameters() if p.requires_grad)
        / 1e6
    )
    print(
        model_args.lora_rank,
        model_args.lora_alpha,
        model_args.lora_dropout,
        model_args.n_of_loras,
        model_args.activation_lr_rw,
        model_args.shift_lr_rw,
        model_args.activation_dc_rw,
    )
    print("device ", training_args.device)
    print("lr ", training_args.learning_rate)
    print("n epochs ", training_args.num_train_epochs)
    print("output dir ", training_args.output_dir)
    print("fp16 ", training_args.fp16)
    print("grad acum ", training_args.gradient_accumulation_steps)
    print("eval steps ", training_args.eval_steps)
    print("save steps ", training_args.save_steps)
    print("warmup steps ", training_args.warmup_steps)
    print("wd ", training_args.weight_decay)
    print("logging steps ", training_args.logging_steps)

    return model_multiple_loras

def get_pslf_model(base_model, model_args, training_args, num_labels):
    print(
        "grad checkpointing model: ",
        model_args.model_gradient_checkpointing,
    )
    if "deberta" in model_args.model_name_or_path:
        model_type = "deberta"
    elif "Llama" in model_args.model_name_or_path:
        model_type = "llama"
    else:
        model_type = "t5"
        training_args.save_safetensors = False
    model_pslf = ModelPSLF(
        base_model=base_model,
        num_labels=num_labels,
        num_pseudo_labels=model_args.k_total,
        model_type=model_type,
        seed=training_args.seed,
        device=training_args.device,
        lora_rank=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        model_gradient_checkpointing=model_args.model_gradient_checkpointing,
    )
    print(model_pslf)
    print('\n'*5)
    print('main model params: ',
        sum(
            p.numel()
            for p in model_pslf.model.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print('true classifier params: ',
        sum(
            p.numel()
            for p in model_pslf.true_classifier.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print('pseudo classifier params: ',
        sum(
            p.numel()
            for p in model_pslf.pseudo_classifier.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print('total params: ',
        sum(p.numel() for p in model_pslf.parameters() if p.requires_grad)
        / 1e6
    )
    print(
        model_args.lora_rank,
        model_args.lora_alpha,
        model_args.lora_dropout,
    )
    print("device ", training_args.device)
    print("lr ", training_args.learning_rate)
    print("n epochs ", training_args.num_train_epochs)
    print("output dir ", training_args.output_dir)
    print("fp16 ", training_args.fp16)
    print("grad acum ", training_args.gradient_accumulation_steps)
    print("eval steps ", training_args.eval_steps)
    print("save steps ", training_args.save_steps)
    print("warmup steps ", training_args.warmup_steps)
    print("wd ", training_args.weight_decay)
    print("logging steps ", training_args.logging_steps)

    return model_pslf
