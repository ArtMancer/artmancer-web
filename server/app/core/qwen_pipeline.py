from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.outputs import BaseOutput
from PIL import Image
from torch import nn

logger = logging.getLogger(__name__)

# Constants from QwenImageEditPlusModel
CONDITION_IMAGE_SIZE = 512  # Default condition image size
VAE_IMAGE_SIZE = 512  # Default VAE image size
PATCH_SIZE = 2  # Patch size for QwenImageEditPlus


class QwenImageEditPlusPipelineOutput(BaseOutput):
    """Output class for QwenImageEditPlus pipeline."""
    images: List[Image.Image]


class QwenImageEditPlusPipeline:
    """
    Custom pipeline wrapper for QwenImageEditPlusModel.
    
    This pipeline handles:
    - Encoding conditional images through VAE
    - Packing/unpacking latents with patch size 2
    - Concatenating control latents with main latents
    - Proper device/dtype consistency
    """
    
    def __init__(
        self,
        model: Any,  # QwenImageEditPlusModel instance
        vae: Any,  # VAE encoder/decoder
        text_encoder: Any,  # Text encoder
        scheduler: Any,  # Scheduler
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        
        # Move components to device with proper dtype handling
        self.model = self.model.to(device)
        if hasattr(self.model, 'to'):
            try:
                self.model = self.model.to(dtype)
            except (TypeError, AttributeError):
                # Some models don't support direct dtype conversion
                pass
        
        # VAE encoder/decoder
        if hasattr(self.vae, 'encoder'):
            self.vae.encoder = self.vae.encoder.to(device)
            if hasattr(self.vae.encoder, 'to'):
                try:
                    self.vae.encoder = self.vae.encoder.to(dtype)
                except (TypeError, AttributeError):
                    pass
        if hasattr(self.vae, 'decoder'):
            self.vae.decoder = self.vae.decoder.to(device)
            if hasattr(self.vae.decoder, 'to'):
                try:
                    self.vae.decoder = self.vae.decoder.to(dtype)
                except (TypeError, AttributeError):
                    pass
        
        # Text encoder
        self.text_encoder = self.text_encoder.to(device)
        if hasattr(self.text_encoder, 'to'):
            try:
                self.text_encoder = self.text_encoder.to(dtype)
            except (TypeError, AttributeError):
                pass
        
        logger.info(f"✅ QwenImageEditPlusPipeline initialized on {device} with dtype {dtype}")
    
    def _resize_image_to_condition_size(
        self, 
        image: Image.Image, 
        target_size: int = CONDITION_IMAGE_SIZE
    ) -> Image.Image:
        """Resize image to condition size, rounding to multiple of 32."""
        # Round to nearest multiple of 32
        rounded_size = ((target_size + 31) // 32) * 32
        
        # Resize maintaining aspect ratio, then pad to square
        width, height = image.size
        scale = min(rounded_size / width, rounded_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Round to multiple of 32
        new_width = ((new_width + 31) // 32) * 32
        new_height = ((new_height + 31) // 32) * 32
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Pad to square if needed
        if new_width != new_height:
            max_size = max(new_width, new_height)
            padded = Image.new("RGB", (max_size, max_size), (0, 0, 0))
            x_offset = (max_size - new_width) // 2
            y_offset = (max_size - new_height) // 2
            padded.paste(resized, (x_offset, y_offset))
            return padded
        
        return resized
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor (0-1 scale, shape [1, 3, H, W])."""
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device).to(self.dtype)
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image through VAE encoder."""
        # Resize to VAE image size
        vae_size = ((VAE_IMAGE_SIZE + 31) // 32) * 32
        if image.size != (vae_size, vae_size):
            image = image.resize((vae_size, vae_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = self._pil_to_tensor(image)
        
        # Ensure tensor is on correct device and dtype
        image_tensor = image_tensor.to(self.device).to(self.dtype)
        
        # Scale from [0, 1] to [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Encode through VAE
        with torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            # Ensure latent is on correct device and dtype
            latent = latent.to(self.device).to(self.dtype)
        
        return latent
    
    def _pack_latents(self, latents: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
        """
        Pack latents for concatenation with control latents.
        
        Args:
            latents: Tensor of shape [B, C, H, W]
            patch_size: Patch size (default 2 for QwenImageEditPlus)
        
        Returns:
            Packed latents of shape [B, H//patch_size * W//patch_size, C * patch_size^2]
        """
        batch, channels, height, width = latents.shape
        
        # Ensure dimensions are divisible by patch_size
        h_padded = ((height + patch_size - 1) // patch_size) * patch_size
        w_padded = ((width + patch_size - 1) // patch_size) * patch_size
        
        if h_padded != height or w_padded != width:
            # Pad if needed (pad right and bottom)
            pad_h = h_padded - height
            pad_w = w_padded - width
            latents = torch.nn.functional.pad(
                latents, 
                (0, pad_w, 0, pad_h), 
                mode='constant', 
                value=0
            )
            height, width = h_padded, w_padded
        
        # Verify dimensions are divisible
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Latent dimensions ({height}, {width}) must be divisible by patch_size ({patch_size})"
            )
        
        # Pack: view as [B, C, H//patch_size, patch_size, W//patch_size, patch_size]
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        try:
            packed = latents.view(
                batch, channels, h_patches, patch_size, w_patches, patch_size
            )
            
            # Permute to [B, H//patch_size, W//patch_size, C, patch_size, patch_size]
            packed = packed.permute(0, 2, 4, 1, 3, 5)
            
            # Reshape to [B, H//patch_size * W//patch_size, C * patch_size^2]
            packed = packed.reshape(batch, h_patches * w_patches, channels * patch_size * patch_size)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to pack latents with shape {latents.shape} and patch_size {patch_size}: {e}"
            )
        
        return packed
    
    def _unpack_latents(self, packed: torch.Tensor, original_shape: Tuple[int, int, int, int], patch_size: int = PATCH_SIZE) -> torch.Tensor:
        """
        Unpack latents from packed format.
        
        Args:
            packed: Packed latents of shape [B, H//patch_size * W//patch_size, C * patch_size^2]
            original_shape: Original shape [B, C, H, W]
            patch_size: Patch size (default 2)
        
        Returns:
            Unpacked latents of shape [B, C, H, W]
        """
        batch, channels, height, width = original_shape
        
        # Ensure dimensions are divisible by patch_size
        h_padded = ((height + patch_size - 1) // patch_size) * patch_size
        w_padded = ((width + patch_size - 1) // patch_size) * patch_size
        
        h_patches = h_padded // patch_size
        w_patches = w_padded // patch_size
        
        # Verify packed shape matches expected shape
        expected_packed_shape = (batch, h_patches * w_patches, channels * patch_size * patch_size)
        if packed.shape != expected_packed_shape:
            raise ValueError(
                f"Packed tensor shape {packed.shape} does not match expected shape {expected_packed_shape}"
            )
        
        try:
            # Reshape from [B, H//patch_size * W//patch_size, C * patch_size^2]
            # to [B, H//patch_size, W//patch_size, C, patch_size, patch_size]
            unpacked = packed.reshape(batch, h_patches, w_patches, channels, patch_size, patch_size)
            
            # Permute back to [B, C, H//patch_size, patch_size, W//patch_size, patch_size]
            unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
            
            # Reshape to [B, C, H_padded, W_padded]
            unpacked = unpacked.reshape(batch, channels, h_padded, w_padded)
            
            # Crop to original dimensions if padding was applied
            if h_padded != height or w_padded != width:
                unpacked = unpacked[:, :, :height, :width]
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to unpack latents with packed shape {packed.shape} and original_shape {original_shape}: {e}"
            )
        
        return unpacked
    
    def _encode_conditional_images(
        self, 
        conditional_images: List[Image.Image]
    ) -> List[torch.Tensor]:
        """
        Encode conditional images through VAE.
        
        According to code sample, control images should be passed as PIL Images
        to model's get_noise_prediction, which will handle encoding, resizing, and packing.
        However, for compatibility, we can also pre-encode them here.
        
        Returns:
            List of encoded control images as tensors (0-1 scale, shape [1, 3, H, W])
        """
        encoded_conditionals = []
        
        for cond_img in conditional_images:
            # Convert PIL Image to tensor (0-1 scale, shape [1, 3, H, W])
            # This matches the format expected by model's get_noise_prediction
            import torch.nn.functional as F
            import math
            
            # Convert PIL to numpy then to tensor
            img_array = np.array(cond_img, dtype=np.float32) / 255.0
            # Convert to tensor and permute (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            # Ensure on correct device and dtype
            img_tensor = img_tensor.to(self.device).to(self.dtype)
            
            encoded_conditionals.append(img_tensor)
        
        return encoded_conditionals
    
    def _encode_prompt(
        self,
        prompt: str,
        conditional_images: Optional[List[Image.Image]] = None,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt with optional conditional images.
        
        For QwenImageEditPlus, conditional images should be encoded and included
        in the prompt embeddings if the model supports it.
        
        Returns:
            prompt_embeds: Encoded prompt embeddings
            prompt_embeds_mask: Mask for prompt embeddings
        """
        # Try to use model's encode_prompt if available (QwenImageEditPlusModel)
        if hasattr(self.model, 'encode_prompt'):
            try:
                # Use model's encode_prompt method which may handle conditional images
                result = self.model.encode_prompt(
                    prompt=prompt,
                    conditional_images=conditional_images,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                )
                
                # Handle different return formats
                if isinstance(result, tuple):
                    prompt_embeds, prompt_embeds_mask = result
                else:
                    # Single return value, create default mask
                    prompt_embeds = result
                    prompt_embeds_mask = torch.ones(
                        prompt_embeds.shape[:2],
                        device=self.device,
                        dtype=torch.bool
                    )
                
                # Ensure correct device and dtype
                prompt_embeds = prompt_embeds.to(self.device).to(self.dtype)
                prompt_embeds_mask = prompt_embeds_mask.to(self.device)
                
                return prompt_embeds, prompt_embeds_mask
            except Exception as e:
                logger.warning(f"⚠️ Model's encode_prompt failed, falling back to standard encoding: {e}")
        
        # Fallback to standard text encoding
        # Encode prompt
        if not hasattr(self.text_encoder, 'tokenizer'):
            raise ValueError("Text encoder must have a tokenizer attribute")
        
        text_inputs = self.text_encoder.tokenizer(
            prompt,
            padding="max_length",
            max_length=getattr(
                self.text_encoder.config, 
                'max_position_embeddings', 
                77  # Default CLIP max length
            ),
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode through text encoder
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Ensure prompt_embeds are on correct device and dtype
        prompt_embeds = prompt_embeds.to(self.device).to(self.dtype)
        
        # Expand for num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        
        # Create mask (all ones for valid tokens)
        # The mask indicates which tokens are valid (not padding)
        prompt_embeds_mask = (text_input_ids != self.text_encoder.tokenizer.pad_token_id).to(self.device)
        # Expand mask to match prompt_embeds shape
        if prompt_embeds_mask.dim() == 2:
            prompt_embeds_mask = prompt_embeds_mask.unsqueeze(-1).expand_as(prompt_embeds[:, :, :prompt_embeds_mask.shape[1]])
        else:
            # Create default mask if shape doesn't match
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2],
                device=self.device,
                dtype=torch.bool
            )
        
        return prompt_embeds, prompt_embeds_mask
    
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        conditional_images: List[Image.Image],
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> QwenImageEditPlusPipelineOutput:
        """
        Generate image using QwenImageEditPlusModel.
        
        Args:
            prompt: Text prompt
            image: Original input image
            mask_image: Mask image (white where to edit)
            conditional_images: List of conditional images [mask, background, object, mae, ...]
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale (if supported)
            negative_prompt: Negative prompt
            generator: Random generator
            width: Output width
            height: Output height
        
        Returns:
            QwenImageEditPlusPipelineOutput with generated images
        """
        logger.info("🎨 Starting QwenImageEditPlus generation")
        
        # Use image dimensions if width/height not specified
        if width is None:
            width = image.width
        if height is None:
            height = image.height
        
        # Round to multiple of 8
        width = ((width + 7) // 8) * 8
        height = ((height + 7) // 8) * 8
        
        # Resize image and mask to target size
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        if mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Encode main image
        logger.info("📥 Encoding main image through VAE...")
        main_latent = self._encode_image(image)
        main_latent_shape = main_latent.shape
        
        # Pack main latents
        packed_main = self._pack_latents(main_latent)
        
        # Prepare conditional images as tensors (0-1 scale, shape [1, 3, H, W])
        # According to code sample, control images are passed as tensors to get_noise_prediction
        # which will handle encoding, resizing, packing, and concat internally
        logger.info(f"📥 Preparing {len(conditional_images)} conditional images...")
        control_tensor_list = self._encode_conditional_images(conditional_images)
        
        logger.info(f"📊 Control images prepared: {len(control_tensor_list)} images")
        for i, ctrl in enumerate(control_tensor_list):
            logger.info(f"   Control {i} shape: {ctrl.shape}")
        
        # Encode prompt
        logger.info("📥 Encoding prompt...")
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            prompt=prompt,
            conditional_images=conditional_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
        )
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Initialize noise
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        noise = torch.randn(
            (1, main_latent_shape[1], main_latent_shape[2], main_latent_shape[3]),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Ensure noise is on correct device and dtype
        noise = noise.to(self.device).to(self.dtype)
        
        # Start with noise
        latents = noise
        
        # Denoising loop
        logger.info(f"🔄 Running {num_inference_steps} denoising steps...")
        for i, t in enumerate(timesteps):
            # Pack current latents
            packed_latents = self._pack_latents(latents)
            
            # Get noise prediction from model
            if hasattr(self.model, 'get_noise_prediction'):
                # Use model's get_noise_prediction method
                # According to code sample, control images should be in batch.control_tensor_list
                # as a list of lists (one list per batch item)
                # Each control image in the list will be encoded, resized, packed, and concat with latents
                
                # Create batch object with control_tensor_list
                # Format: batch.control_tensor_list is a list of lists
                # For single batch: [[control_img1, control_img2, ...]]
                class Batch:
                    def __init__(self, control_tensor_list):
                        # control_tensor_list should be list of lists (one per batch item)
                        # For single batch, wrap in outer list
                        if control_tensor_list and isinstance(control_tensor_list[0], torch.Tensor):
                            # If it's a list of tensors, wrap in outer list for single batch
                            self.control_tensor_list = [control_tensor_list]
                        else:
                            # Already in correct format (list of lists)
                            self.control_tensor_list = control_tensor_list
                
                # Pass control_tensor_list (list of encoded conditionals)
                # Model's get_noise_prediction will handle encoding, resizing, packing, and concat
                batch = Batch(control_tensor_list)
                
                # Convert timestep from 0-1000 scale to 0-1 scale (as in code sample)
                timestep_scaled = t / 1000.0 if t > 1.0 else t
                
                # Create PromptEmbeds-like object for text_embeddings
                class PromptEmbeds:
                    def __init__(self, text_embeds, attention_mask):
                        self.text_embeds = text_embeds
                        self.attention_mask = attention_mask
                
                text_embeddings = PromptEmbeds(prompt_embeds, prompt_embeds_mask)
                
                # Model expects unpacked latents (B, C, H, W)
                # Convert main_latent_shape (torch.Size) to tuple with 4 elements
                shape_tuple: Tuple[int, int, int, int] = (
                    int(main_latent_shape[0]),
                    int(main_latent_shape[1]),
                    int(main_latent_shape[2]),
                    int(main_latent_shape[3]),
                )
                unpacked_latents = self._unpack_latents(packed_latents, shape_tuple)
                
                noise_pred = self.model.get_noise_prediction(
                    latent_model_input=unpacked_latents,
                    timestep=timestep_scaled,
                    text_embeddings=text_embeddings,
                    batch=batch,
                )
            else:
                # Fallback: use standard UNet forward
                # This is a simplified version - actual implementation may differ
                noise_pred = self.model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                ).sample
            
            # Apply guidance scale if > 1.0
            if guidance_scale > 1.0:
                # For CFG, we'd need negative prompt embeddings
                # Simplified for now
                pass
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        # Decode latents
        logger.info("📥 Decoding latents through VAE...")
        latents = 1 / self.vae.config.scaling_factor * latents
        
        with torch.no_grad():
            image_tensor = self.vae.decode(latents).sample
        
        # Convert to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
        image_tensor = (image_tensor * 255).round().astype("uint8")
        
        images = [Image.fromarray(img) for img in image_tensor]
        
        # Resize to original dimensions if needed
        if images[0].size != (width, height):
            images = [img.resize((width, height), Image.Resampling.LANCZOS) for img in images]
        
        logger.info("✅ Generation complete")
        
        return QwenImageEditPlusPipelineOutput(images=images)

