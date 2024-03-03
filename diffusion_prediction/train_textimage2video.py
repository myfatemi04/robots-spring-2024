#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
from pathlib import Path
import datetime

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig # type: ignore
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel # type: ignore
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr, cast_training_params
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import check_min_version, deprecate, is_wandb_available # type: ignore
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from rt1_dataset_wrapper import RT1Dataset
from parse_args import parse_args
from unet_utilities import get_add_time_ids
from model_cards import save_model_card


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# https://github.com/huggingface/accelerate/issues/1221
# Store to file. DEPRECATED: Replaced with the `tee` command in a training script.
# However, if tee doesn't work, we can use this as a backup...
logging_dir = "logs"
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
logfilename = f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_py.log"
logfile = os.path.join(logging_dir, logfilename)
file_handler = logging.FileHandler(logfile)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(formatter)
logger.logger.addHandler(file_handler)

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

class DistributedDiffusionTrainer:
    def __init__(self, args, accelerator: Accelerator, accelerator_project_config: ProjectConfiguration):
        self.args = args
        self.accelerator = accelerator
        self.accelerator_project_config = accelerator_project_config

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        self.repo_id = None
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

            if args.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
                ).repo_id

        # Load models
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # type: ignore
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.pretrained_text_encoder_model_name_or_path) # type: ignore

        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKLTemporalDecoder will not enjoy the parameter sharding
        # across multiple gpus and only UNetSpatioTemporalConditionModel will get ZeRO sharded.
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            # Again, text encoder is from a separate model; it is not part of Stable Diffusion Video.
            text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained( # type: ignore
                args.pretrained_text_encoder_model_name_or_path, variant=args.variant
                # subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
            self.text_encoder: CLIPTextModel = text_encoder # type: ignore
            vae: AutoencoderKLTemporalDecoder = AutoencoderKLTemporalDecoder.from_pretrained( # type: ignore
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
            )
            self.vae: AutoencoderKLTemporalDecoder = vae # type: ignore
            # Taken from line ~119 of `pipeline_stable_video_diffusion.py`
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) # type: ignore
            self.vae_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        unet: UNetSpatioTemporalConditionModel = UNetSpatioTemporalConditionModel.from_pretrained( # type: ignore
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision,
        )
        self.unet: UNetSpatioTemporalConditionModel = unet # type: ignore
        self.unet_original_config = self.unet.config
        trainable_parameters = self.inject_lora()

        # Freeze vae and text_encoder and set unet to trainable
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train()

        # Create EMA for the unet.
        if args.use_ema:
            ema_unet: UNetSpatioTemporalConditionModel = UNetSpatioTemporalConditionModel.from_pretrained( # type: ignore
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
            )
            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=ema_unet.config)
        else:
            self.ema_unet = None

        self.register_checkpointing_hooks()

        if args.enable_xformers_memory_efficient_attention:
            self.enable_xformers_memory_efficient_attention()

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb # type: ignore
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        #region Dataset
        self.dataset = RT1Dataset(args.rt1_dataset_root)
        # Allow a maximum number of training samples for quick debugging.
        # Question: Why is `accelerator.main_process_first()` used here?
        with accelerator.main_process_first():
            if args.max_train_samples:
                # Create random torch.utils.data.Subset of original dataset.
                # indices_to_keep = torch.randperm(len(train_dataset))[:args.max_train_samples]
                # train_dataset = torch.utils.data.Subset(train_dataset, indices_to_keep)
                self.dataset = torch.utils.data.Subset(self.dataset, list(range(args.max_train_samples)))

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            collate_fn=self.collate_fn
        )
        #endregion

        #region LR scheduler
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        #endregion

        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, self.dataloader, self.lr_scheduler = accelerator.prepare(
            self.unet, self.optimizer, self.dataloader, self.lr_scheduler
        )
        # For type checking.
        unet: UNetSpatioTemporalConditionModel = unet  # type: ignore

        if args.use_ema:
            assert self.ema_unet, "args.use_ema is true, but ema_unet was None"
            self.ema_unet.to(accelerator.device)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision
        self.weight_dtype = weight_dtype

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts")
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def collate_fn(self, batch):
        args = self.args

        # input: (text, image_sequence)[]
        # return: (text batch, text attention masks, text sequence lengths, images)
        text_batch = [text for (text, imgseq) in batch]
        tokenization = self.tokenizer(text_batch, padding='longest', return_tensors='pt')
        text_tokens = tokenization['input_ids']
        text_attention_masks = tokenization['attention_mask']

        imgseqs = [self.vae_image_processor.preprocess(imgseq, height=args.image_height, width=args.image_width) for (_, imgseq) in batch]
        minseqlen = min([len(x) for x in imgseqs])
        minseqlen = min(25, minseqlen)

        # Randomly select a start and end time consistent with minseqlen.
        imgseqs_selected = []
        for imgseq in imgseqs:
            # for torch.randint:
            # low: least integer
            # high: one above the highest integer
            start_index = torch.randint(0, len(imgseq) - minseqlen + 1, ())
            imgseqs_selected.append(imgseq[start_index:start_index + minseqlen])

        return (text_tokens, text_attention_masks, torch.stack(imgseqs_selected))

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def inject_lora(self):
        # https://github.com/huggingface/peft/blob/main/docs/source/task_guides/lora_based_methods.md
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
        args = self.args
        unet = self.unet
        if args.rank:
            # Disable grad for original U-net parameters
            unet.requires_grad_(False)
            for param in unet.parameters():
                param.requires_grad_(False)

            unet_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            unet.add_adapter(unet_lora_config)
            if args.mixed_precision == "fp16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(unet, dtype=torch.float32)

            trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
            
            logging.info("Using LoRA: Rank = %s", args.rank)
            logging.info("Number of trainable parameters: %d", sum(p.numel() for p in trainable_parameters))
        else:
            trainable_parameters = unet.parameters()

        return trainable_parameters

    def register_checkpointing_hooks(self):
        args = self.args
        accelerator = self.accelerator
        ema_unet = self.ema_unet
        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        assert ema_unet, "args.use_ema is true, but ema_unet was None"

                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    assert ema_unet, "args.use_ema is true, but ema_unet was None"

                    load_model: EMAModel = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel) # type: ignore
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet") # type: ignore
                    load_model: UNetSpatioTemporalConditionModel
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            import xformers # type: ignore

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def _load_checkpoint(self):
        args = self.args
        accelerator = self.accelerator

        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(self.args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // self.num_update_steps_per_epoch
            
        return (initial_global_step, first_epoch)

    def encode_latent_sequences(self, image_sequences):
        # Initialize the latent embeddings in a flat manner.
        # Infer target latent embeddings.
        # This could lowkey be pre-computed or computed in parallel.
        # I guess it depends on if we also want to train the VAE (but I'm assuming
        # that the VAE is trained *before* the diffusion step)
        torch.cuda.empty_cache()
        target_image_sequences = image_sequences
        batch_size = target_image_sequences.shape[0]
        n_images = target_image_sequences.shape[1]
        target_image_sequences_flat = target_image_sequences.reshape(batch_size * n_images, *target_image_sequences.shape[2:])
        with torch.no_grad():
            # do this in batches
            batches = []
            target_image_counter = 0
            target_image_encode_batch_size = 8
            while target_image_counter < len(target_image_sequences_flat):
                images = target_image_sequences_flat[target_image_counter:target_image_counter + target_image_encode_batch_size]
                encoding = self.vae.encode(images)
                sample = encoding.latent_dist.sample() * self.vae.config['scaling_factor'] # type: ignore
                batches.append(sample)
                target_image_counter += target_image_encode_batch_size
            target_latent_sequences_flat = torch.cat(batches, dim=0)
        # Reshape the flat latent variables to original
        target_latent_sequences = target_latent_sequences_flat.reshape(batch_size, n_images, *target_latent_sequences_flat.shape[1:])

        return target_latent_sequences

    def train_loop(self):
        args = self.args
        unet = self.unet
        vae = self.vae
        ema_unet = self.ema_unet
        accelerator = self.accelerator
        noise_scheduler = self.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        text_encoder = self.text_encoder
        weight_dtype = self.weight_dtype
        tokenizer = self.tokenizer

        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
        # Preconditioning functions that are based on noise level.
        # sigma_values is an array representing the noise level at
        # each time step.
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        sigma_values = sqrt_one_minus_alphas_cumprod.to(device=accelerator.device, dtype=weight_dtype)

        c_skip = 1 / (sigma_values ** 2 + 1)
        c_out = -sigma_values * torch.rsqrt(sigma_values ** 2 + 1)
        c_in = torch.rsqrt(sigma_values ** 2 + 1)
        c_noise = 0.25 * torch.log(sigma_values)
        # This is how much to weight the MSE loss at each time step.
        lambda_values = (1 + sigma_values ** 2) / (sigma_values ** 2)

        global_step = 0
        initial_global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            initial_global_step, first_epoch = self._load_checkpoint()

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(self.dataloader):
                with accelerator.accumulate(unet):
                    # 1. Get image pixel values
                    # 2. Encode with VAE
                    # Text: List of token sequences
                    """
                    To consider:
                    - precomputing VAE embeddings for image sequences
                    - attention masks for training with ragged sequences
                    image_sequences: [batch_size, n_images, channels, height, width]
                    """
                    (texts, text_attention_masks, image_sequences) = batch

                    batch_size = image_sequences.shape[0]
                    n_images = image_sequences.shape[1]

                    # Convert dataloaded stuff to accelerator
                    texts = texts.to(device=accelerator.device, dtype=torch.long)
                    text_attention_masks = text_attention_masks.to(device=accelerator.device, dtype=torch.bool)
                    image_sequences = image_sequences.to(device=accelerator.device, dtype=self.weight_dtype)

                    # Question: Do I need to scale this?
                    initial_images = image_sequences[:, 0, :]
                    # Note: VAE encoder is single-image, VAE decoder is temporal
                    initial_image_latents = self.vae.encode(initial_images).latent_dist.mode() # type: ignore

                    target_latent_sequences = self.encode_latent_sequences(
                        image_sequences.to(device=accelerator.device, dtype=self.weight_dtype)
                    )

                    # Sample a random timestep for each sequence.
                    # All images in the same sequence must have the
                    # same timestep.
                    # Make sure the device matches the latents
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, # type: ignore
                        (batch_size,),
                        device=target_latent_sequences.device
                    )
                    timesteps = timesteps.long()

                    # Sample noise that we'll add to the latents.
                    noise = torch.randn_like(target_latent_sequences)

                    unsqueeze_to_batch = lambda x: x.view(len(timesteps), *([1] * (len(target_latent_sequences.shape) - 1)))

                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (target_latent_sequences.shape[0], target_latent_sequences.shape[1], 1, 1),
                            device=target_latent_sequences.device
                        )

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                        noisy_latents = noise_scheduler.add_noise(target_latent_sequences, new_noise, timesteps) # type: ignore
                    else:
                        noisy_latents = noise_scheduler.add_noise(target_latent_sequences, noise, timesteps) # type: ignore

                    USE_EDM_FORMULATION = True

                    if not USE_EDM_FORMULATION:
                        # Get the target for loss depending on the prediction type
                        if args.prediction_type is not None:
                            # set prediction_type of scheduler if defined
                            noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                        # epsilon = directly predict the total noise that has been added to the original latent vector
                        noise_scheduler_prediction_type = noise_scheduler.config.prediction_type # type: ignore
                        if noise_scheduler_prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler_prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(target_latent_sequences, noise, timesteps) # type: ignore
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler_prediction_type}")
                    else:
                        noise = noise * unsqueeze_to_batch(sigma_values[timesteps])
                        noisy_latents = unsqueeze_to_batch(c_in[timesteps]) * (target_latent_sequences + noise)
                        target = target_latent_sequences

                    # Get the text embedding sequences for conditioning.
                    # Right now, just use pooler output; but at some point, would like to condition on all input tokens.
                    # This is a tensor of [bsz, d_model] --unsqueeze-> [bsz, 1, d_model]
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder.forward(
                            input_ids=texts,
                            attention_mask=text_attention_masks
                        ).pooler_output.unsqueeze(1) # type: ignore

                        # Randomly set 10% of the hidden states to 0.
                        # This is for classifier-free guidance.
                        classifier_free_mask = torch.rand(encoder_hidden_states.shape[0]) < 0.1
                        encoder_hidden_states[classifier_free_mask] = 0

                    # Predict the noise residual and compute loss
                    # Everything should be the same otherwise
                    added_time_ids = get_add_time_ids(
                        self.unet,
                        self.unet_original_config,
                        fps=7,
                        motion_bucket_id=0,
                        noise_aug_strength=0,
                        dtype=target_latent_sequences.dtype,
                        batch_size=batch_size,
                        num_videos_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    added_time_ids = added_time_ids.to(device=accelerator.device)
                    # (bsz, nframes, d_model) -> (bsz, nframes, d_model * 2)
                    noisy_latents_concatenated_with_initial_image_latents = torch.cat([noisy_latents, initial_image_latents.unsqueeze(1).repeat(1, n_images, 1, 1, 1)], dim=2)

                    model_pred = unet(
                        noisy_latents_concatenated_with_initial_image_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_time_ids,
                        return_dict=False
                    )[0]

                    if USE_EDM_FORMULATION:
                        denoiser_prediction = (
                            unsqueeze_to_batch(c_skip[timesteps]) * noisy_latents +
                            unsqueeze_to_batch(c_out[timesteps]) * model_pred
                        )
                        loss = (
                            # Scale MSE according to lambda(sigma)
                            ((denoiser_prediction.float() - target.float()) ** 2) *
                            unsqueeze_to_batch(lambda_values[timesteps])
                        ).mean()
                    elif args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        noise_scheduler_prediction_type = noise_scheduler.config.prediction_type # type: ignore
                        if noise_scheduler_prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler_prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean() # type: ignore
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        assert ema_unet, "args.use_ema is true, but ema_unet was None"
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            unet = self.unwrap_model(unet)
                            if args.use_ema:
                                assert ema_unet, "args.use_ema is true, but ema_unet was None"
                                ema_unet.copy_to(unet.parameters())

                            # Run `save_pretrained`
                            # At some point, also save optimizer state...
                            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unet,
                                revision=args.revision,
                                variant=args.variant,
                            )
                            save_path = os.path.join(args.checkpoint_dir, f"checkpoint-{global_step}")
                            pipeline.save_pretrained(save_path)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_main_process:
                if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                    if args.use_ema:
                        assert ema_unet, "args.use_ema is true, but ema_unet was None"
                        
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

                    if args.use_ema:
                        assert ema_unet, "args.use_ema is true, but ema_unet was None"

                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

    def finalize(self):
        args = self.args
        unet = self.unet
        ema_unet = self.ema_unet
        accelerator = self.accelerator
        weight_dtype = self.weight_dtype

        unet = self.unwrap_model(unet)
        if args.use_ema:
            assert ema_unet, "args.use_ema is true, but ema_unet was None"

            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # Run a final round of inference.
        images = []
        if args.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            for i in range(len(args.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

        if args.push_to_hub:
            assert self.repo_id, "repo_id is None, but should have been initialized because args.push_to_hub is True"

            save_model_card(args, self.repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=self.repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

def main():
    args = parse_args()
    # Check args
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            (
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # Set up logger and accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    trainer = DistributedDiffusionTrainer(args, accelerator, accelerator_project_config)

    # Train!
    trainer.train_loop()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.finalize()

    accelerator.end_training()


if __name__ == "__main__":
    main()
