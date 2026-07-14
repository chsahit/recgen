from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class RecGenPipeline(Pipeline):
    """
    Pipeline for inferring RecGen image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
        pose_normalizer: Optional pose normalizer for denormalizing predicted poses.
        pose_representation (str): The pose representation used (e.g., '6d_translation_scale').
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
        pose_normalizer = None,
        pose_representation: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = None
        self.pose_normalizer = pose_normalizer
        self.pose_representation = pose_representation
        self._init_image_cond_model(image_cond_model)

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            # Upload as uint8 and scale on the GPU: 4x less PCIe traffic than
            # uploading float32, and the divide stops being a CPU-side pass.
            image = np.stack([np.asarray(i.convert('RGB')) for i in image])
            image = torch.from_numpy(image).to(self.device).permute(0, 3, 1, 2).float().div_(255)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        # Return only patch tokens, strip CLS (1) + registers (4) to match training
        return patchtokens[:, 5:]
    
    @torch.no_grad()
    def encode_pointmap(self, pointmap: Union[torch.Tensor, list[Image.Image]], model_key: str = None) -> torch.Tensor:
        """
        Encode the pointmap using the specified model's encoder.

        Args:
            pointmap (Union[torch.Tensor, list[Image.Image]]): The pointmap to encode
            model_key (str): Which model to use for encoding. If None, tries available models in order.

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(pointmap, torch.Tensor):
            assert pointmap.ndim == 4, "Pointmap tensor should be batched (B, C, H, W)"
            pointmap = pointmap.to(self.device)
        elif isinstance(pointmap, list):
            assert all(isinstance(i, Image.Image) for i in pointmap), "Pointmap list should be list of PIL images"
            pointmap = [i.resize((518, 518), Image.NEAREST) for i in pointmap]
            pointmap = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in pointmap]
            pointmap = [torch.from_numpy(i).permute(2, 0, 1).float() for i in pointmap]
            pointmap = torch.stack(pointmap).to(self.device)
        else:
            raise ValueError(f"Unsupported type of pointmap: {type(pointmap)}")
        
        # Select which model to use for encoding
        assert model_key is not None, "model_key must be provided"
        # model_key = 'sparse_structure_pose_flow_model'
        model = self.models.get(model_key)
        print(f"[DEBUG encode_pointmap] Model {model_key}")
        if model is None or not hasattr(model, 'encode_pointmap'):
            raise ValueError(f"Model {model_key} doesn't support pointmap encoding")

        patchtokens = model.encode_pointmap(pointmap)
        return patchtokens

    @torch.no_grad()
    def encode_mask(self, mask: Union[torch.Tensor, list[Image.Image]], model_key: str = None) -> torch.Tensor:
        """
        Encode the mask using the specified model's encoder.

        Args:
            mask (Union[torch.Tensor, list[Image.Image]]): Mask tensor/list to encode.
            model_key (str): Which model to use for encoding. If None, tries available models in order.

        Returns:
            torch.Tensor: Encoded mask features.
        """
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()
        elif isinstance(mask, list):
            if not all(isinstance(m, Image.Image) for m in mask):
                raise ValueError(f"Unsupported type in mask list: {type(mask[0])}")
            processed_masks = []
            for m in mask:
                m = m.resize((518, 518), Image.Resampling.NEAREST)
                mask_np = np.array(m)
                mask_tensor = torch.tensor(mask_np).float() / 255.0
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                elif mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.permute(2, 0, 1)
                else:
                    raise ValueError(f"Unsupported mask dimensions: {mask_tensor.shape}")
                processed_masks.append(mask_tensor)
            mask_tensor = torch.stack(processed_masks)
            print(f"[DEBUG encode_mask] Mask tensor shape: {mask_tensor.shape}, min: {mask_tensor.min()}, max: {mask_tensor.max()}")
        else:
            raise ValueError(f"Unsupported type of mask: {type(mask)}")

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        elif mask_tensor.ndim != 4:
            raise ValueError(f"Mask tensor should have 4 dimensions, got {mask_tensor.ndim}")

        if mask_tensor.shape[1] != 1:
            mask_tensor = mask_tensor[:, :1]

        mask_tensor = mask_tensor.to(self.device)

        # Select which model to use for encoding
        assert model_key is not None, "model_key must be provided"
        print(f"[DEBUG encode_mask] Model {model_key}")
        model = self.models.get(model_key)
        if model is None or not hasattr(model, 'encode_mask'):
            raise ValueError(f"Model {model_key} doesn't support mask encoding")


        features = model.encode_mask(mask_tensor)
        return features

    def get_cond(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        pointmap: Union[torch.Tensor, list[Image.Image]] = None,
        mask: Union[torch.Tensor, list[Image.Image]] = None,
        model_key: str = None,
    ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.
            pointmap: Optional pointmap conditioning.
            mask: Optional mask conditioning.
            model_key: Which model to use for encoding pointmap/mask. If None, uses default order.

        Returns:
            dict: The conditioning information with 'cond' and 'neg_cond' keys.
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        if pointmap is not None:
            cond_pointmap = self.encode_pointmap(pointmap, model_key=model_key)
            cond += cond_pointmap
            # Pointmap should NOT be added to neg_cond (matches training behavior)
        if mask is not None:
            cond_mask = self.encode_mask(mask, model_key=model_key)
            cond += cond_mask
            # Mask should NOT be added to neg_cond (matches training behavior)

        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def _convert_pose_representation(self, pose: torch.Tensor, from_repr: str, to_repr: str) -> torch.Tensor:
        """Convert between pose representations (operates on denormalized poses)."""
        if pose.ndim == 3:
            return torch.stack([
                self._convert_pose_representation(pose[:, i], from_repr, to_repr)
                for i in range(pose.shape[1])
            ], dim=1)

        from ..utils.pose_utils import get_pose_dimensions
        rot_dim_from, trans_start_from, scale_idx_from, _ = get_pose_dimensions(from_repr)

        rot = pose[:, :rot_dim_from]
        trans = pose[:, trans_start_from:trans_start_from+3]
        scale = pose[:, scale_idx_from:scale_idx_from+1]

        if from_repr == '6d_translation_scale' and to_repr == '9d_translation_scale':
            a1 = rot[:, :3]
            a2 = rot[:, 3:6]
            b1 = F.normalize(a1, dim=1)
            b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
            b2 = F.normalize(b2, dim=1)
            b3 = torch.cross(b1, b2, dim=1)
            rot_9d = torch.cat([b1, b2, b3], dim=1)
        else:
            raise NotImplementedError(f"Conversion {from_repr} -> {to_repr} not implemented")

        return torch.cat([rot_9d, trans, scale], dim=1)

    def _normalize_pose_2d(self, normalizer, pose: torch.Tensor, representation: str, mode: str = 'normalize') -> torch.Tensor:
        """
        Normalize or denormalize pose, handling both (B, D) and (B, K, D) shapes.
        PoseNormalizer only handles (B, D), so we flatten/unflatten for multi-pose.
        """
        if pose.ndim == 2:
            if mode == 'normalize':
                return normalizer.normalize(pose, representation)
            else:
                return normalizer.denormalize(pose, representation)
        elif pose.ndim == 3:
            B, K, D = pose.shape
            flat = pose.reshape(B * K, D)
            if mode == 'normalize':
                flat_out = normalizer.normalize(flat, representation)
            else:
                flat_out = normalizer.denormalize(flat, representation)
            return flat_out.reshape(B, K, -1)
        else:
            raise ValueError(f"Expected 2D or 3D pose tensor, got {pose.ndim}D")

    def _prepare_pose_for_slat(self, pose_raw_normalized: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized pose from SS model to SLAT conditioning token(s).

        Args:
            pose_raw_normalized: Raw pose from SS sampler (normalized, in SS representation).
                                 Shape: (B, D) for single pose or (B, K, D) for K poses.

        Returns:
            Pose conditioning tokens (B, K, C) ready to concat to SLAT cond.
        """
        slat_model = self.models['slat_flow_model']
        slat_repr = getattr(self, 'slat_pose_representation', None)
        ss_repr = self.pose_representation
        # Use SLAT's own normalizer if available (trained on different dataset),
        # otherwise fall back to SS normalizer
        slat_normalizer = getattr(self, 'slat_pose_normalizer', None) or self.pose_normalizer

        if ss_repr == slat_repr and slat_normalizer is self.pose_normalizer:
            # Same representation and same normalizer — pass through
            pose_for_slat = pose_raw_normalized
        else:
            # 1. Denormalize from SS model's normalized space (using SS normalizer)
            #    If SS normalizer is None (no-norm training), pose is already in real space
            if self.pose_normalizer is not None:
                pose_denorm = self._normalize_pose_2d(self.pose_normalizer, pose_raw_normalized, ss_repr, mode='denormalize')
            else:
                pose_denorm = pose_raw_normalized
            # 2. Convert representation if needed (e.g. 10D -> 13D)
            if ss_repr != slat_repr:
                pose_converted = self._convert_pose_representation(pose_denorm, ss_repr, slat_repr)
            else:
                pose_converted = pose_denorm
            # 3. Re-normalize using SLAT's normalizer (may have different stats)
            #    If SLAT normalizer is also None, pass through
            if slat_normalizer is not None:
                pose_for_slat = self._normalize_pose_2d(slat_normalizer, pose_converted, slat_repr, mode='normalize')
            else:
                pose_for_slat = pose_converted

        return slat_model.encode_pose(pose_for_slat)

    def _add_pose_cond_to_slat(self, cond_slat: dict, pose_raw_normalized: torch.Tensor) -> dict:
        """Add pose conditioning to SLAT cond dict if SLAT supports it."""
        if not getattr(self, 'slat_use_pose', False) or pose_raw_normalized is None:
            return cond_slat

        pose_cond = self._prepare_pose_for_slat(pose_raw_normalized)
        cond_slat['cond'] = torch.cat([cond_slat['cond'], pose_cond], dim=1)
        zero_pose_cond = torch.zeros_like(pose_cond)
        cond_slat['neg_cond'] = torch.cat([cond_slat['neg_cond'], zero_pose_cond], dim=1)
        print(f"  SLAT pose conditioning: added {pose_cond.shape[1]} token(s)")
        return cond_slat

    def sample_sparse_structure_pose(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample sparse structures with pose information using the given conditioning.
        Aligned with structured latent flow model training pattern.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.

        Returns:
            Tuple of (coords, pose_normalized, pose_anchor):
                - coords: Sparse structure coordinates
                - pose_normalized: Full pose from sampler (normalized space, for SLAT conditioning).
                  Shape (B, D) or (B, K, D) for multi-pose tokens.
                - pose_anchor: Anchor (first) pose, denormalized, always (B, D) for downstream eval.
        """
        # Sample occupancy latent with pose following structured latent flow pattern
        flow_model = self.models['sparse_structure_pose_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        # Initialize pose noise matching the model's num_pose_tokens
        num_pose_tokens = getattr(flow_model, 'num_pose_tokens', 1)
        if num_pose_tokens > 1:
            pose_noise = torch.randn(num_samples, num_pose_tokens, flow_model.pose_channels).to(self.device)
        else:
            pose_noise = torch.randn(num_samples, flow_model.pose_channels).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        # Sample with pose - the sampler's _inference_model already handles pose
        # Just pass pose in kwargs and it will be threaded through correctly
        result = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            pose=pose_noise,  # Pass initial pose, sampler will update it
            verbose=True
        )

        z_s = result.samples
        # Extract final pose from result (sampler returns it as 'final_pose')
        pose_normalized = result.get('final_pose', pose_noise)

        # Denormalize pose for output/metrics
        # For multi-pose (B, K, D), denormalize all and also extract anchor (first) pose
        pose_denormalized = pose_normalized
        if self.pose_normalizer is not None and self.pose_representation is not None:
            pose_denormalized = self._normalize_pose_2d(
                self.pose_normalizer, pose_normalized, self.pose_representation, mode='denormalize'
            )
        # For backward compat: return anchor (first) pose as (B, D) for output['pose']
        if pose_denormalized.ndim == 3:
            pose_anchor = pose_denormalized[:, 0]  # (B, D) — first pose token
        else:
            pose_anchor = pose_denormalized  # already (B, D)

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords, pose_normalized, pose_anchor, pose_denormalized
    

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        # Apply denormalization if normalization parameters are provided
        if self.slat_normalization is not None:
            std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
            slat = slat * std + mean
            print(f"  Applied denormalization (std/mean)")
        else:
            print(f"  Skipped denormalization (slat_normalization is None)")
        
        print(f"  slat.feats.shape: {slat.feats.shape}")
        print(f"[DEBUG sample_slat] Done\n")
        return slat

    @torch.no_grad()
    def run_pointmap(
        self,
        image: Image.Image,
        pointmap: Union[Image.Image, torch.Tensor] = None,
        mask: Image.Image = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if pointmap is not None:
            if isinstance(pointmap, torch.Tensor):
                if pointmap.ndim == 3:
                    pointmap_arg = pointmap.unsqueeze(0)
                elif pointmap.ndim == 4:
                    pointmap_arg = pointmap
                else:
                    raise ValueError(f"Unsupported tensor pointmap shape: {pointmap.shape}")
            else:
                pointmap_arg = [pointmap]
        else:
            pointmap_arg = None

        mask_arg = [mask] if mask is not None else None        
        # Get conditioning for sparse structure using sparse model's encoder
        cond_sparse = self.get_cond([image], pointmap=pointmap_arg, mask=mask_arg, 
                                     model_key='sparse_structure_pose_flow_model')
        torch.manual_seed(seed)
        coords, pose_normalized, pose, _all_poses = self.sample_sparse_structure_pose(cond_sparse, num_samples, sparse_structure_sampler_params)

        # Get conditioning for SLaT using SLaT model's encoder
        cond_slat = self.get_cond([image], pointmap=pointmap_arg, mask=mask_arg,
                                   model_key='slat_flow_model')
        cond_slat = self._add_pose_cond_to_slat(cond_slat, pose_normalized)
        slat = self.sample_slat(cond_slat, coords, slat_sampler_params)
        output = self.decode_slat(slat, formats)
        output['pose'] = pose
        output['coords'] = coords
        return output

    @torch.no_grad()
    def run_pointmap_coarse(
        self,
        image: Image.Image,
        pointmap: Union[Image.Image, torch.Tensor] = None,
        mask: Image.Image = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
    ) -> dict:
        """Run only the coarse (sparse-structure) stage of the pipeline.

        This mirrors the first half of :meth:`run_pointmap` — image/pointmap/mask
        conditioning followed by ``sample_sparse_structure_pose`` — but stops before
        the (expensive) SLAT sampling + decoding. It returns the occupancy voxel
        ``coords`` and the estimated ``pose`` (the SLAT stage does not refine pose,
        so this is the same pose the full pipeline reports).

        Returns:
            dict with keys ``coords`` (occupancy voxel indices, resolution =
            ``slat_flow_model.resolution``), ``pose`` (denormalized anchor pose),
            and ``pose_normalized`` (SLAT-conditioning pose).
        """
        if pointmap is not None:
            if isinstance(pointmap, torch.Tensor):
                if pointmap.ndim == 3:
                    pointmap_arg = pointmap.unsqueeze(0)
                elif pointmap.ndim == 4:
                    pointmap_arg = pointmap
                else:
                    raise ValueError(f"Unsupported tensor pointmap shape: {pointmap.shape}")
            else:
                pointmap_arg = [pointmap]
        else:
            pointmap_arg = None

        mask_arg = [mask] if mask is not None else None
        cond_sparse = self.get_cond([image], pointmap=pointmap_arg, mask=mask_arg,
                                    model_key='sparse_structure_pose_flow_model')
        torch.manual_seed(seed)
        coords, pose_normalized, pose, _all_poses = self.sample_sparse_structure_pose(
            cond_sparse, num_samples, sparse_structure_sampler_params
        )
        return {'coords': coords, 'pose': pose, 'pose_normalized': pose_normalized}

    @torch.no_grad()
    def get_cond_multiview(
        self,
        images: List[Image.Image],
        pointmaps: List[torch.Tensor] = None,
        masks: List[Image.Image] = None,
        model_key: str = None,
    ) -> dict:
        """
        Get conditioning for multi-view input.

        Encodes N images and concatenates their features along the sequence dimension.
        This matches the training behavior in MultiImageConditionedMixin.

        Args:
            images: List of N PIL images
            pointmaps: List of N pointmap tensors (3, H, W) or None
            masks: List of N PIL mask images or None
            model_key: Which model to use for encoding

        Returns:
            dict: Conditioning with 'cond' and 'neg_cond' keys
        """
        num_views = len(images)
        print(f"[DEBUG get_cond_multiview] Processing {num_views} views")

        # Encode all images and concatenate along sequence dimension
        # encode_image now returns only patch tokens (CLS/registers already stripped)
        all_image_features = []
        for img in images:
            img_features = self.encode_image([img])  # (1, seq_len, dim) - no CLS/registers
            all_image_features.append(img_features)

        # Apply per-frame token-type embeddings if the model supports it (matching training behavior)
        if num_views > 1 and model_key is not None:
            model = self.models.get(model_key)
            if model is not None and getattr(model, 'use_frame_token_embedder', False):
                S = all_image_features[0].shape[1]
                stacked = torch.stack(all_image_features, dim=1)  # (1, N, seq_len, dim)
                cond = model.encode_frame_tokens(stacked)  # (1, N*seq_len, dim)
            else:
                cond = torch.cat(all_image_features, dim=1)
        else:
            # Concatenate along sequence dimension: (1, N*seq_len, dim)
            cond = torch.cat(all_image_features, dim=1)

        neg_cond = torch.zeros_like(cond)

        # Encode pointmaps if provided
        if pointmaps is not None:
            all_pointmap_features = []
            for pm in pointmaps:
                if pm.ndim == 3:
                    pm = pm.unsqueeze(0)  # (1, 3, H, W)
                pm_features = self.encode_pointmap(pm, model_key=model_key)  # (1, seq_len, dim)
                all_pointmap_features.append(pm_features)

            # Concatenate and add directly - both have same structure now (no CLS/registers)
            pointmap_cond = torch.cat(all_pointmap_features, dim=1)
            cond += pointmap_cond

        # Encode masks if provided
        if masks is not None:
            all_mask_features = []
            for m in masks:
                m_features = self.encode_mask([m], model_key=model_key)  # (1, seq_len, dim)
                all_mask_features.append(m_features)

            # Concatenate and add directly - both have same structure now (no CLS/registers)
            mask_cond = torch.cat(all_mask_features, dim=1)
            cond += mask_cond

        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @torch.no_grad()
    def run_pointmap_multiview(
        self,
        images: List[Image.Image],
        pointmaps: List[torch.Tensor] = None,
        masks: List[Image.Image] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        sparse: bool = False,
        status_callback=None,
        slat_single_view: bool = False,
    ) -> dict:
        """
        Run multi-view inference.

        Processes N views and generates a single 3D output using the multi-view
        SS checkpoint. By default SLAT also uses all views (matching training).
        Set slat_single_view=True for legacy behavior (first view only).

        Args:
            images: List of N PIL images
            pointmaps: List of N pointmap tensors (3, H, W) or None
            masks: List of N PIL mask images or None
            num_samples: Number of samples to generate
            seed: Random seed
            sparse_structure_sampler_params: Additional sampler params
            slat_sampler_params: Additional sampler params
            formats: Output formats

        Returns:
            dict: Output with 'mesh', 'gaussian', 'pose', 'coords' keys
        """
        num_views = len(images)
        slat_mode = "single-view" if slat_single_view else f"multi-view ({num_views})"
        print(f"\n[run_pointmap_multiview] Processing {num_views} views, SLAT mode: {slat_mode}")

        # Get multi-view conditioning for sparse structure
        cond_sparse = self.get_cond_multiview(
            images=images,
            pointmaps=pointmaps,
            masks=masks,
            model_key='sparse_structure_pose_flow_model'
        )

        torch.manual_seed(seed)
        coords, pose_normalized, pose, all_poses_denormalized = self.sample_sparse_structure_pose(
            cond_sparse, num_samples, sparse_structure_sampler_params
        )

        if sparse:
            return {'pose': pose, 'coords': coords}

        if slat_single_view:
            # Legacy: SLAT uses single-view (first view) only
            first_image = images[0]
            first_pointmap = pointmaps[0] if pointmaps is not None else None
            first_mask = masks[0] if masks is not None else None

            if first_pointmap is not None:
                if first_pointmap.ndim == 3:
                    first_pointmap = first_pointmap.unsqueeze(0)

            cond_slat = self.get_cond(
                [first_image],
                pointmap=first_pointmap,
                mask=[first_mask] if first_mask is not None else None,
                model_key='slat_flow_model'
            )
        else:
            # Multi-view SLAT: pass all views (matches training behavior)
            cond_slat = self.get_cond_multiview(
                images=images,
                pointmaps=pointmaps,
                masks=masks,
                model_key='slat_flow_model'
            )
        cond_slat = self._add_pose_cond_to_slat(cond_slat, pose_normalized)

        slat = self.sample_slat(cond_slat, coords, slat_sampler_params)
        output = self.decode_slat(slat, formats)
        output['pose'] = pose
        output['all_poses'] = all_poses_denormalized  # (B, K, D) all pose tokens
        output['coords'] = coords
        return output
