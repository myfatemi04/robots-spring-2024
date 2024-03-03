"""
Might be useful for debugging.
"""

import os

from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
import tqdm
import torch

from rt1_dataset_wrapper import RT1Dataset

# Load VAE.
vae: AutoencoderKLTemporalDecoder = AutoencoderKLTemporalDecoder.from_pretrained( # type: ignore
    "stabilityai/stable-video-diffusion-img2vid-xt", subfolder="vae"
)
vae.to('cuda')
vae.requires_grad_(False)

vae_scale_factor = 2 ** (len(vae.config['block_out_channels']) - 1)
vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

dataset = RT1Dataset("/scratch/gsk6me/WORLDMODELS/datasets/rt-1-data-release")
out_dir = "/scratch/gsk6me/WORLDMODELS/datasets/rt-1-data-release-vae-embeddings"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

chunk_id = 0
chunk_size = 1024
chunk = []
sequence_embedding_batch_size = 128
image_height = 256
image_width = 320

with torch.no_grad():
    for i in tqdm.tqdm(range(len(dataset))):
        (instructions, image_sequence) = dataset[i]

        sequence = []
        sequence_counter = 0
        while sequence_counter < len(image_sequence):
            batch = image_sequence[sequence_counter:sequence_counter + sequence_embedding_batch_size]
            batch = vae_image_processor.preprocess(batch, height=image_height, width=image_width)
            batch = batch.to(dtype=torch.float, device='cuda')
            latent_dist_parameters = vae.encode(batch).latent_dist.parameters # type: ignore
            sequence.append(latent_dist_parameters)
            sequence_counter += sequence_embedding_batch_size

        sequence = torch.cat(sequence, dim=0)
        chunk.append(sequence)

        if len(chunk) == chunk_size:
            torch.save(chunk, f"{out_dir}/chunk_{chunk_id:06}.pt")
            chunk_id += 1
            chunk = []

    if len(chunk) > 0:
        torch.save(chunk, f"{out_dir}/chunk_{chunk_id:06}.pt")
        chunk_id += 1
        chunk = []
