from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid, _append_dims, StableVideoDiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image

from typing import Optional, Union, List, Callable, Dict

class VisualTrajectorySynthesizer(StableVideoDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPProcessor,
        text_encoder: CLIPTextModelWithProjection,
    ):
        super().__init__(vae, image_encoder, unet, scheduler, feature_extractor)

        self.register_modules(text_encoder=text_encoder)

        self.text_encoder = text_encoder

    @staticmethod
    def from_stable_video_diffusion_pipeline(pipeline: StableVideoDiffusionPipeline, text_encoder: CLIPTextModelWithProjection):
        return VisualTrajectorySynthesizer(
            pipeline.vae,
            pipeline.image_encoder,
            pipeline.unet,
            pipeline.scheduler,
            pipeline.feature_extractor,
            text_encoder,
        )

    def _encode_text(self, text, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(text, torch.Tensor):
            tokens = self.feature_extractor(text=text, return_tensors='pt').input_ids
        else:
            tokens = text

        tokens = tokens.to(device=device)

        # convert to single-token sequences
        encoded = self.text_encoder(input_ids=tokens)
        print(encoded)
        print(encoded.pooler_output.shape)
        text_embeddings = encoded.pooler_output.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_text_embeddings = torch.zeros_like(text_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])

        return text_embeddings
    
    @torch.no_grad()
    def custom_call(self,
                    image: PIL.Image.Image,
                    text: str, 
                    height: int = 576,
                    width: int = 1024,
                    num_frames: Optional[int] = None,
                    num_inference_steps: int = 25,
                    min_guidance_scale: float = 1.0,
                    max_guidance_scale: float = 3.0,
                    fps: int = 7,
                    motion_bucket_id: int = 127,
                    noise_aug_strength: float = 0.02,
                    decode_chunk_size: Optional[int] = None,
                    num_videos_per_prompt: Optional[int] = 1,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    latents: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                    return_dict: bool = True):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image. This is the conditioning for the model.
        # This is used as an encoder hidden state.
        # Will try to incorporate joint text/image conditioning too.
        # Will do classifier + classifier-free guidance for the conditioning image and conditioning text separately.
        # Will augment the dataset of conditioning text as much as possible (i.e. by generating several variations
        # with GPT-4).
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        text_embeddings = self._encode_text(text, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # print(image_embeddings.shape, text_embeddings.shape)

        # return

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE. This is because the conditioning image also gets concatenated with the noise.
        # This is supposed to improve the model's ability to learn quickly.
        # Also, side note: I am surprised that Stability decided to do frame-by-frame spatial tokens instead of spatiotemporal
        # tokens.
        # Would be nice to convert the denoising U-Net to a transformer, on the low, to make it compatible with longer contexts.
        #  - That is, we disable denoising for a couple of the layers, and treat them solely as conditioning layers.
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # Conditioning image gets concatenated with the noise
        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        # Sample a range of time steps
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings, # image_embeddings, # lol most hacky thing ever
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        return frames
