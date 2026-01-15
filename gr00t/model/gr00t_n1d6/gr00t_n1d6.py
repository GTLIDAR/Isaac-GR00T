from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Force/torque embedding configuration.
        self.force_embedding_mode = config.force_embedding_mode
        self.force_token_mode = config.force_token_mode
        self.force_history_length = config.force_history_length
        self.max_force_dim = config.max_force_dim
        self.force_history_dim = max(0, self.force_history_length * self.max_force_dim)
        self.force_encoder_adapter = None
        self.force_decoder_adapter = None
        self.state_force_encoder = None

        if self.force_embedding_mode != "none":
            if self.max_force_dim <= 0:
                raise ValueError("max_force_dim must be > 0 when force embedding is enabled")
            if self.force_history_length <= 0:
                raise ValueError("force_history_length must be > 0 when force embedding is enabled")
            if self.force_token_mode not in ("single", "frame"):
                raise ValueError(
                    f"force_token_mode must be 'single' or 'frame', got {self.force_token_mode}"
                )

            force_adapter_input_dim = (
                self.max_force_dim
                if self.force_token_mode == "frame"
                else self.force_history_dim
            )
            if self.force_embedding_mode == "encoder":
                self.force_encoder_adapter = CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=force_adapter_input_dim,
                    hidden_dim=self.hidden_size,
                    output_dim=config.backbone_embedding_dim,
                )
            elif self.force_embedding_mode == "decoder_post":
                self.force_decoder_adapter = CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=force_adapter_input_dim,
                    hidden_dim=self.hidden_size,
                    output_dim=self.input_embedding_dim,
                )
            elif self.force_embedding_mode == "decoder_pre":
                # For pre-concat we flatten the (history, dim) force sequence into one vector.
                self.state_force_encoder = CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=config.max_state_dim + self.force_history_dim,
                    hidden_dim=self.hidden_size,
                    output_dim=self.input_embedding_dim,
                )
            else:
                raise ValueError(f"Unknown force_embedding_mode: {self.force_embedding_mode}")

        # Joint action-torque diffusion objective.
        self.force_objective = config.force_objective
        self.force_objective_weight = config.force_objective_weight
        self.joint_action_dim = self.action_dim
        self.joint_action_encoder = None
        self.joint_action_decoder = None
        if self.force_objective:
            if self.max_force_dim <= 0:
                raise ValueError("max_force_dim must be > 0 when force objective is enabled")
            self.joint_action_dim = self.action_dim + self.max_force_dim
            self.joint_action_encoder = MultiEmbodimentActionEncoder(
                action_dim=self.joint_action_dim,
                hidden_size=self.input_embedding_dim,
                num_embodiments=config.max_num_embodiments,
            )
            self.joint_action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.joint_action_dim,
            )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.joint_action_encoder is not None:
                self.joint_action_encoder.requires_grad_(False)
            if self.joint_action_decoder is not None:
                self.joint_action_decoder.requires_grad_(False)
            if self.force_encoder_adapter is not None:
                self.force_encoder_adapter.requires_grad_(False)
            if self.force_decoder_adapter is not None:
                self.force_decoder_adapter.requires_grad_(False)
            if self.state_force_encoder is not None:
                self.state_force_encoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.joint_action_encoder is not None:
                    self.joint_action_encoder.eval()
                if self.joint_action_decoder is not None:
                    self.joint_action_decoder.eval()
                if self.force_encoder_adapter is not None:
                    self.force_encoder_adapter.eval()
                if self.force_decoder_adapter is not None:
                    self.force_decoder_adapter.eval()
                if self.state_force_encoder is not None:
                    self.state_force_encoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _get_action_input_value(self, action_input: BatchFeature, key: str):
        if hasattr(action_input, key):
            return getattr(action_input, key)
        if isinstance(action_input, dict):
            return action_input.get(key)
        try:
            return action_input[key]
        except Exception:
            return None

    def _prepare_force_inputs(self, action_input: BatchFeature):
        if self.force_embedding_mode == "none":
            return None, None

        force = self._get_action_input_value(action_input, "force")
        if force is None:
            force = self._get_action_input_value(action_input, "torque")
        if force is None:
            return None, None

        if force.dim() == 1:
            force = force.view(force.shape[0], 1, 1)
        elif force.dim() == 2:
            force = force.unsqueeze(1)
        elif force.dim() != 3:
            raise ValueError(f"Expected force tensor to be 1D, 2D, or 3D, got {force.shape}")

        force_mask = self._get_action_input_value(action_input, "force_mask")
        if force_mask is None:
            force_mask = self._get_action_input_value(action_input, "torque_mask")
        if force_mask is not None:
            if force_mask.dim() == 3:
                force_mask = force_mask.any(dim=-1)
            elif force_mask.dim() == 1:
                force_mask = force_mask.unsqueeze(1)
            elif force_mask.dim() != 2:
                raise ValueError(
                    f"Expected force_mask tensor to be 1D, 2D, or 3D, got {force_mask.shape}"
                )
            force_mask = force_mask.to(device=force.device)
            if force_mask.dtype != torch.bool:
                force_mask = force_mask != 0

        batch_size, history_len, force_dim = force.shape
        target_force_dim = self.max_force_dim
        if force_dim < target_force_dim:
            pad = torch.zeros(
                batch_size,
                history_len,
                target_force_dim - force_dim,
                device=force.device,
                dtype=force.dtype,
            )
            force = torch.cat([force, pad], dim=-1)
        elif force_dim > target_force_dim:
            force = force[:, :, :target_force_dim]

        if force_mask is None:
            force_mask = torch.ones(
                (batch_size, history_len), device=force.device, dtype=torch.bool
            )

        target_history_len = self.force_history_length
        if history_len > target_history_len:
            force = force[:, -target_history_len:, :]
            force_mask = force_mask[:, -target_history_len:]
        elif history_len < target_history_len:
            pad_len = target_history_len - history_len
            force_pad = torch.zeros(
                batch_size,
                pad_len,
                target_force_dim,
                device=force.device,
                dtype=force.dtype,
            )
            mask_pad = torch.zeros(
                batch_size, pad_len, device=force.device, dtype=torch.bool
            )
            force = torch.cat([force_pad, force], dim=1)
            force_mask = torch.cat([mask_pad, force_mask], dim=1)

        force = force * force_mask.unsqueeze(-1).to(dtype=force.dtype)
        return force, force_mask

    def _prepare_force_targets(self, action_input: BatchFeature, target_horizon: int):
        if not self.force_objective:
            return None, None

        force = self._get_action_input_value(action_input, "force_target")
        if force is None:
            force = self._get_action_input_value(action_input, "torque_target")
        if force is None:
            candidate = self._get_action_input_value(action_input, "force")
            if candidate is None:
                candidate = self._get_action_input_value(action_input, "torque")
            if candidate is not None:
                if candidate.dim() == 3 and candidate.shape[1] == target_horizon:
                    force = candidate
                elif candidate.dim() == 2 and target_horizon == 1:
                    force = candidate

        if force is None:
            return None, None

        if force.dim() == 1:
            force = force.view(force.shape[0], 1, 1)
        elif force.dim() == 2:
            force = force.unsqueeze(1)
        elif force.dim() != 3:
            raise ValueError(f"Expected force target tensor to be 1D, 2D, or 3D, got {force.shape}")

        force_mask = self._get_action_input_value(action_input, "force_target_mask")
        if force_mask is None:
            force_mask = self._get_action_input_value(action_input, "torque_target_mask")
        if force_mask is None:
            force_mask = self._get_action_input_value(action_input, "force_mask")
        if force_mask is None:
            force_mask = self._get_action_input_value(action_input, "torque_mask")

        batch_size, history_len, force_dim = force.shape
        if force_mask is None:
            force_mask = torch.ones(
                (batch_size, history_len, force_dim), device=force.device, dtype=torch.bool
            )
        else:
            if force_mask.dim() == 3:
                if force_mask.shape[-1] == 1 and force_dim != 1:
                    force_mask = force_mask.expand(-1, -1, force_dim)
                elif force_mask.shape[-1] != force_dim:
                    raise ValueError(
                        "force_target_mask last dimension must match force target dimension"
                    )
            elif force_mask.dim() == 2:
                force_mask = force_mask.unsqueeze(-1).expand(-1, -1, force_dim)
            elif force_mask.dim() == 1:
                force_mask = force_mask.view(batch_size, 1, 1).expand(-1, 1, force_dim)
            else:
                raise ValueError(
                    f"Expected force target mask tensor to be 1D, 2D, or 3D, got {force_mask.shape}"
                )
            force_mask = force_mask.to(device=force.device)
            if force_mask.dtype != torch.bool:
                force_mask = force_mask != 0

        target_force_dim = self.max_force_dim
        if force_dim < target_force_dim:
            pad = torch.zeros(
                batch_size,
                history_len,
                target_force_dim - force_dim,
                device=force.device,
                dtype=force.dtype,
            )
            mask_pad = torch.zeros(
                batch_size,
                history_len,
                target_force_dim - force_dim,
                device=force.device,
                dtype=torch.bool,
            )
            force = torch.cat([force, pad], dim=-1)
            force_mask = torch.cat([force_mask, mask_pad], dim=-1)
        elif force_dim > target_force_dim:
            force = force[:, :, :target_force_dim]
            force_mask = force_mask[:, :, :target_force_dim]

        if history_len > target_horizon:
            force = force[:, -target_horizon:, :]
            force_mask = force_mask[:, -target_horizon:, :]
        elif history_len < target_horizon:
            pad_len = target_horizon - history_len
            force_pad = torch.zeros(
                batch_size,
                pad_len,
                target_force_dim,
                device=force.device,
                dtype=force.dtype,
            )
            mask_pad = torch.zeros(
                batch_size, pad_len, target_force_dim, device=force.device, dtype=torch.bool
            )
            force = torch.cat([force_pad, force], dim=1)
            force_mask = torch.cat([mask_pad, force_mask], dim=1)

        force = force * force_mask.to(dtype=force.dtype)
        return force, force_mask

    def _force_token_mask(self, force_mask: torch.Tensor | None) -> torch.Tensor | None:
        if force_mask is None:
            return None
        if self.force_token_mode == "frame":
            return force_mask
        return force_mask.any(dim=1, keepdim=True)

    def _embed_force_tokens(
        self, force: torch.Tensor, embodiment_id: torch.Tensor, adapter: nn.Module
    ) -> torch.Tensor:
        if self.force_token_mode == "frame":
            return adapter(force, embodiment_id)
        batch_size, history_len, force_dim = force.shape
        flat_force = force.reshape(batch_size, 1, history_len * force_dim)
        return adapter(flat_force, embodiment_id)

    def _append_force_to_encoder(
        self,
        vl_embeds: torch.Tensor,
        vl_attn_mask: torch.Tensor,
        image_mask: torch.Tensor | None,
        force_tokens: torch.Tensor,
        force_token_mask: torch.Tensor | None,
    ):
        if force_token_mask is not None:
            force_tokens = force_tokens * force_token_mask.unsqueeze(-1).to(force_tokens.dtype)
            force_attn_mask = force_token_mask.to(dtype=vl_attn_mask.dtype)
        else:
            force_attn_mask = torch.ones(
                force_tokens.shape[0],
                force_tokens.shape[1],
                device=force_tokens.device,
                dtype=vl_attn_mask.dtype,
            )

        vl_embeds = torch.cat([vl_embeds, force_tokens], dim=1)
        vl_attn_mask = torch.cat([vl_attn_mask, force_attn_mask], dim=1)
        if image_mask is not None:
            image_pad = torch.zeros(
                force_attn_mask.shape, device=force_attn_mask.device, dtype=image_mask.dtype
            )
            image_mask = torch.cat([image_mask, image_pad], dim=1)
        return vl_embeds, vl_attn_mask, image_mask

    def _encode_state_features(
        self,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        force: torch.Tensor | None,
        force_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.force_embedding_mode != "decoder_pre" or force is None:
            return self.state_encoder(state, embodiment_id)

        force_token_mask = self._force_token_mask(force_mask)
        if force_token_mask is not None:
            force = force * force_token_mask.unsqueeze(-1).to(dtype=force.dtype)

        batch_size, history_len, force_dim = force.shape
        flat_force = force.reshape(batch_size, 1, history_len * force_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if flat_force.shape[1] == 1 and state.shape[1] > 1:
            flat_force = flat_force.expand(-1, state.shape[1], -1)

        state_with_force = torch.cat([state, flat_force], dim=-1)
        return self.state_force_encoder(state_with_force, embodiment_id)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]
                - force/torque: [B, force_horizon, force_dim] (optional)
                - force_mask/torque_mask: [B, force_horizon] (optional)
                - force_target/torque_target: [B, action_horizon, force_dim] (optional)
                - force_target_mask/torque_target_mask: [B, action_horizon] (optional)

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        force_inputs, force_mask = self._prepare_force_inputs(action_input)
        state_features = self._encode_state_features(
            action_input.state, embodiment_id, force_inputs, force_mask
        )

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Optionally prepend force tokens to the decoder inputs.
        if self.force_embedding_mode == "decoder_post" and force_inputs is not None:
            force_tokens = self._embed_force_tokens(
                force_inputs, embodiment_id, self.force_decoder_adapter
            )
            force_token_mask = self._force_token_mask(force_mask)
            if force_token_mask is not None:
                force_tokens = force_tokens * force_token_mask.unsqueeze(-1).to(
                    force_tokens.dtype
                )
            state_features = torch.cat((force_tokens, state_features), dim=1)

        # Embed noised action (or action-torque) trajectory.
        actions = action_input.action
        force_targets, force_target_mask = self._prepare_force_targets(
            action_input, actions.shape[1]
        )
        use_force_objective = self.force_objective and force_targets is not None
        if self.force_objective and force_targets is None:
            raise ValueError(
                "force_objective is enabled but no force_target/torque_target was provided"
            )

        if use_force_objective:
            joint_clean = torch.cat([actions, force_targets], dim=-1)
        else:
            joint_clean = actions

        noise = torch.randn(joint_clean.shape, device=joint_clean.device, dtype=joint_clean.dtype)
        t = self.sample_time(joint_clean.shape[0], device=joint_clean.device, dtype=joint_clean.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * joint_clean
        velocity = joint_clean - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        if use_force_objective:
            action_features = self.joint_action_encoder(
                noisy_trajectory, t_discretized, embodiment_id
            )
        else:
            action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        # Optionally append force tokens to the encoder context.
        image_mask = backbone_output.image_mask if self.config.use_alternate_vl_dit else None
        if self.force_embedding_mode == "encoder" and force_inputs is not None:
            force_tokens = self._embed_force_tokens(
                force_inputs, embodiment_id, self.force_encoder_adapter
            )
            force_token_mask = self._force_token_mask(force_mask)
            vl_embeds, vl_attn_mask, image_mask = self._append_force_to_encoder(
                vl_embeds, vl_attn_mask, image_mask, force_tokens, force_token_mask
            )

        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=vl_attn_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        if use_force_objective:
            pred = self.joint_action_decoder(model_output, embodiment_id)
        else:
            pred = self.action_decoder(model_output, embodiment_id)
        pred_joint = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        pred_actions = pred_joint[..., : self.action_dim]
        velocity_actions = velocity[..., : self.action_dim]
        action_loss = F.mse_loss(pred_actions, velocity_actions, reduction="none") * action_mask
        action_loss_value = action_loss.sum() / (action_mask.sum() + 1e-6)

        output = {
            "loss": action_loss_value,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        if use_force_objective:
            pred_forces = pred_joint[..., self.action_dim :]
            velocity_forces = velocity[..., self.action_dim :]
            force_mask = force_target_mask.to(device=pred_forces.device, dtype=pred_forces.dtype)
            force_loss = (
                F.mse_loss(pred_forces, velocity_forces, reduction="none") * force_mask
            )
            force_loss_value = force_loss.sum() / (force_mask.sum() + 1e-6)
            output["loss"] = action_loss_value + self.force_objective_weight * force_loss_value
            output["force_loss"] = force_loss
            output["force_mask"] = force_mask

        return output

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)
                - force/torque: [B, force_horizon, force_dim] (optional)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        force_inputs, force_mask = self._prepare_force_inputs(action_input)
        state_features = self._encode_state_features(
            action_input.state, embodiment_id, force_inputs, force_mask
        )

        if self.force_embedding_mode == "decoder_post" and force_inputs is not None:
            force_tokens = self._embed_force_tokens(
                force_inputs, embodiment_id, self.force_decoder_adapter
            )
            force_token_mask = self._force_token_mask(force_mask)
            if force_token_mask is not None:
                force_tokens = force_tokens * force_token_mask.unsqueeze(-1).to(
                    force_tokens.dtype
                )
            state_features = torch.cat((force_tokens, state_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask
        image_mask = backbone_output.image_mask if self.config.use_alternate_vl_dit else None
        if self.force_embedding_mode == "encoder" and force_inputs is not None:
            force_tokens = self._embed_force_tokens(
                force_inputs, embodiment_id, self.force_encoder_adapter
            )
            force_token_mask = self._force_token_mask(force_mask)
            vl_embeds, vl_attn_mask, image_mask = self._append_force_to_encoder(
                vl_embeds, vl_attn_mask, image_mask, force_tokens, force_token_mask
            )

        return BatchFeature(
            data={
                "backbone_features": vl_embeds,
                "backbone_attention_mask": vl_attn_mask,
                "image_mask": image_mask,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
                - force_pred: [B, action_horizon, force_dim] predicted torques (optional)
        """
        vl_embeds = backbone_features
        if backbone_attention_mask is None and backbone_output is not None:
            backbone_attention_mask = backbone_output.backbone_attention_mask
        if image_mask is None and backbone_output is not None and self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        diffusion_action_dim = (
            self.joint_action_dim if self.force_objective else self.action_dim
        )
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, diffusion_action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            if self.force_objective:
                action_features = self.joint_action_encoder(
                    actions, timesteps_tensor, embodiment_id
                )
            else:
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=backbone_attention_mask,
                    timestep=timesteps_tensor,
                )
            if self.force_objective:
                pred = self.joint_action_decoder(model_output, embodiment_id)
            else:
                pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :, : actions.shape[-1]]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        output = {
            "action_pred": actions[..., : self.action_dim],
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if self.force_objective:
            output["force_pred"] = actions[..., self.action_dim :]
        return BatchFeature(data=output)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_attention_mask=features.backbone_attention_mask,
            image_mask=features.image_mask,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
