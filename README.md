# Supplementary code for Label Privacy in Split Learning for Large Models with Parameter-Efficient Training

## How to install

You just need to clone repository

```
git clone https://github.com/p3eft-iclr-2025/p3eft-iclr-2025
```

and install the requirements:

```
pip install -r requirements.txt
```

## How to run

The main folder contains 8 bash scripts that run main fine-tuning experiments from section 4.2. These scripts have the following names: pppeft_{model_name}_{task_name}.

There are three possible choices for models: [DeBERTa xxlarge (He et al., 2020)](https://arxiv.org/abs/2006.03654) (model_name == deberta_v2_xxlarge), [Flan-T5 (Chung et al., 2022)](https://arxiv.org/abs/2210.11416) and [LLaMA-2 7B (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) (model_name == llama), as well as for the datasets: SST2 (task_name == sst2), QNLI (task_name == qnli) and MNLI (task_name == mnli) from [GLUE benchmark (Wang et al., 2018)](https://arxiv.org/abs/1804.07461).

We present experiments for our method P^3EFT and for our implementations of [Distance Correlation method (Sun et al., 2022)](https://arxiv.org/abs/2203.01451) and [Private Split Learning Framework (Wan et al. (2023)](https://dl.acm.org/doi/10.1145/3583780.3615019). There are some number of important arguments.

* Common arguments for `transformers.Trainer` (such that `batch_size`, `max_seq_length`, `lr`, `lr_scheduler_type`, `n_epoch` etc.) and special arguments for [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) (`lora_rank`, `lora_alpha`, `lora_dropout`). For DeBERTa we took these arguments from the [original paper](https://arxiv.org/abs/2106.09685) with the only difference that we disabled LoRA dropout value. For Flan-T5 we adapt the exact same hyperparamters as for DeBERTa. For evaluation LLaMA we used hyperparameters from [LoRA paper](https://arxiv.org/abs/2106.09685) for GPT-3 with several changes inspired by [QLoRA Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314). Namely, we use the NF4 weight format, apply LoRA to both attention and MLP layers with rank 16. We fine-tune both tasks with maximum context length of 512 and weight decay 0.01.

* We have 2 hyperparameters for our method: `mult_std`, `activation_lr_rw`. 

    * After receiving the activations for each individual copy of adapters we mix them with specific coefficients. The magnitude of the norm of these coefficients can be changed via `mult_std`. More specifically, the random vectors used hereafter to obtain the coefficients are initially generated from a normal distribution with variance `1` and then multiplied by `10 ^ (mult_std / 2)`. In almost all experiments we set this value to `0`.
    
    * `activation_lr_rw` adjusts the weights of the corresponding regularization loss functions.

    * Note, that we use `n_of_loras == 2` in our method to insert two different copies of the adapters into the model. 

* There are also a few hyperparameters for switching to and starting baselines.

    * `activation_dc_rw` has a similar meaning to the `activation_lr_rw`, it is reguralizing weight for the [Distance Correlation method (Sun et al., 2022)](https://arxiv.org/abs/2203.01451).

    * `coefs_method_name` is an auxiliary parameter which can take 2 values: 'antisymmetric_n01' and 'averaged'. The first value is the main one for experiments with our method, and the second one is needed to run the baselines with regularization (together with the value `n_of_loras = 1`).

    * `add_fmlg_labels` enables [PSLF baseline (Wan et al. (2023)](https://dl.acm.org/doi/10.1145/3583780.3615019).

    * `dp_eps` is the differential privacy budget for PSLF.
    
    * `k_total` number of pseudo labels for PSLF method.

The scripts have default values for regular fune-tuning with one LoRA copy and without any regularization. To run baseline without LoRAs set `n_of_loras` to 0.

* In order to reduce the amount of GPU RAM you can use 3 additional flags:

    * Above all, we advise to turn on `fp16` for DeBERTa or `bf16` for Flan-T5 and LLaMA-2 7B.

    * `loras_gradient_checkpointing` creates checkpoints between adapter set switches. Thus the amount of memory consumed will be the same as in the case of a single adapter set.

    * `model_gradient_checkpointing` turns on regular `transformers.Trainer` chechpoints. Cannot be used without `loras_gradient_checkpointing`, since the regular checkpointing system does not switch adapter sets, which is essential to run our method.

    * We don't advise to use `gradient_accumulation`, since then the loss is computed separately on each minibatch, while both privacy protection methods assume the computation of the regularization loss over the entire batch.

    * Our code can be run on multiple GPUs using `torch.nn.parallel.DistributedDataParallel`. To do this, specify the `num_gpus` value.

## Additional experiments

Notebooks `vis_1_get_acts_grads.ipynb` and `vis_2_draw_charts.ipynb` can be used to reproduce Figure 3.