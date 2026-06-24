import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae.data_handling.data_transforms import create_input_condition

# Constants to prevent posterior collapse and numerical instability
LOGVAR_CLAMP_MIN = -20.0
LOGVAR_CLAMP_MAX = 10.0


def _build_activation(activation: str) -> nn.Module:
    """
    Centralize activation instantiation to ensure consistent non-linearities
    across all encoder and decoder blocks without redundant conditional logic.
    """
    activation_name = activation.lower()
    if activation_name == "relu":
        return nn.ReLU(inplace=True)
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unsupported activation: {activation}")


def _build_pool_layer(pool_type: str) -> nn.Module:
    """
    Provide interchangeable spatial downsampling strategies.
    """
    if pool_type == "max":
        return nn.MaxPool1d(kernel_size=2, stride=2)
    return nn.AvgPool1d(kernel_size=2, stride=2)


class ResidualBlock1D(nn.Module):
    """
    One-dimensional residual block employing pre-normalization.

    Pre-norm configuration (Norm -> Activation -> Conv) is utilized to stabilize
    gradients in deeper networks, preventing vanishing/exploding gradients during
    the optimization of long stochastic trajectories.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dilation: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.activation_fn = _build_activation(activation)
        gn_groups = groups if channels % groups == 0 else 1

        self.block = nn.Sequential(
            nn.GroupNorm(gn_groups, channels),
            self.activation_fn,
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                padding_mode="circular",
            ),
            nn.GroupNorm(gn_groups, channels),
            self.activation_fn,
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                padding_mode="circular",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block with an identity skip connection."""
        return x + self.block(x)


class SkipDropout(nn.Module):
    """
    Channel-wise dropout applied exclusively to encoder skip connections.

    Forces the decoder to rely more heavily on the compressed latent space `z`
    rather than simply copying high-frequency spatial details from the conditional
    skip features, mitigating posterior collapse in the VAE.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = float(p)
        self.channel_dropout = nn.Dropout1d(self.p)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply channel dropout during active training."""
        if not self.training or self.p <= 0.0:
            return feats
        return [self.channel_dropout(f) for f in feats]


class VariationalAutoencoder_FullyConv(nn.Module):
    """
    Fully convolutional Conditional Variational Autoencoder (CVAE).

    Translates discrete surface kinetics into a continuous latent space distribution,
    conditioned on local surface slopes and environmental temperatures, to predict
    stochastic structural evolution.
    """

    def __init__(
        self,
        max_negative_change: int,
        max_positive_change: int,
        in_channels: int = 1,
        cond_channels: int = 1,
        latent_channels: int = 16,
        skip_dropout: float = 0.0,
        n_layers: int = 4,
        kernel_size: int = 3,
        base_channels: int = 32,
        groups: int = 8,
        padding_mode: str = "circular",
        activation: str = "relu",
        use_pooling: bool = True,
        pool_type: str = "max",
        temp_channel: bool = False,
        temp_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(max_negative_change + max_positive_change + 1)
        self.cond_channels = int(cond_channels)
        self.latent_channels = int(latent_channels)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        self.base_channels = int(base_channels)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.activation = activation
        self.use_pooling = use_pooling
        self.pool_type = pool_type.lower()
        self.temp_channel = temp_channel
        self.temp_range = temp_range
        self.max_negative_change = int(max_negative_change)
        self.max_positive_change = int(max_positive_change)

        if self.pool_type not in ["max", "mean"]:
            raise ValueError(f"pool_type must be 'max' or 'mean', got {pool_type}")

        self.activation_fn = _build_activation(self.activation)
        self.skip_dropout = SkipDropout(p=skip_dropout)

        self.enc_x = nn.ModuleList()
        self.enc_c = nn.ModuleList()
        pad = self.kernel_size // 2

        for i in range(self.n_layers):
            in_ch = self.in_channels if i == 0 else self.base_channels * (2 ** (i - 1))
            out_ch = self.base_channels * (2**i)
            gn_groups = self.groups if out_ch % self.groups == 0 else 1

            enc_x_layers = [
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=pad,
                    padding_mode=self.padding_mode,
                ),
                nn.GroupNorm(gn_groups, out_ch),
                self.activation_fn,
                ResidualBlock1D(out_ch, groups=gn_groups, activation=self.activation),
            ]
            if self.use_pooling:
                enc_x_layers.append(_build_pool_layer(self.pool_type))

            self.enc_x.append(nn.Sequential(*enc_x_layers))

            cond_in_ch = self.cond_channels if i == 0 else in_ch
            enc_c_layers = [
                nn.Conv1d(
                    cond_in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=pad,
                    padding_mode=self.padding_mode,
                ),
                nn.GroupNorm(gn_groups, out_ch),
                self.activation_fn,
                ResidualBlock1D(out_ch, groups=gn_groups, activation=self.activation),
            ]
            if self.use_pooling:
                enc_c_layers.append(_build_pool_layer(self.pool_type))

            self.enc_c.append(nn.Sequential(*enc_c_layers))

        self.enc_channels = self.base_channels * (2 ** (self.n_layers - 1))
        bottleneck_ch = self.enc_channels * 2
        bn_groups = self.groups if bottleneck_ch % self.groups == 0 else 1

        self.bottleneck_norm = nn.GroupNorm(bn_groups, bottleneck_ch)
        self.to_mu = nn.Conv1d(bottleneck_ch, self.latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(bottleneck_ch, self.latent_channels, kernel_size=1)
        self.from_latent = nn.Conv1d(
            self.latent_channels, self.enc_channels, kernel_size=1
        )

        self.conv_ups = nn.ModuleList()
        self.res_ups = nn.ModuleList()
        self.level_channels = [
            self.base_channels * (2**i) for i in range(self.n_layers)
        ]

        current_h = self.enc_channels
        for lvl in reversed(range(self.n_layers)):
            cond_ch = self.level_channels[lvl]
            in_ch = current_h + cond_ch
            out_ch = max(current_h // 2, self.base_channels)
            gn_groups = self.groups if out_ch % self.groups == 0 else 1

            self.conv_ups.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=pad,
                    padding_mode=self.padding_mode,
                )
            )
            self.res_ups.append(
                ResidualBlock1D(out_ch, groups=gn_groups, activation=self.activation)
            )
            current_h = out_ch

        self.final_channels = current_h
        self.conv_out = nn.Conv1d(
            self.final_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            padding_mode=self.padding_mode,
        )

        self.receptive_field = self._calculate_receptive_field()
        self.trainable_parameters = self._calculate_trainable_parameters()

    def _calculate_receptive_field(self) -> int:
        """Compute the spatial receptive field to ensure long-range surface interactions are captured."""
        r = 1
        j = 1
        for _ in range(self.n_layers):
            r += (self.kernel_size - 1) * j
            r += (3 - 1) * j
            r += (3 - 1) * j
            if self.use_pooling:
                r += (2 - 1) * j
                j *= 2
        return r

    def _calculate_trainable_parameters(self) -> int:
        """Count the active gradients for capacity logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_x(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode the primary target lattice sequence."""
        feats = []
        h = x
        for block in self.enc_x:
            h = block(h)
            feats.append(h)
        return feats

    def encode_cond(self, cond: torch.Tensor) -> List[torch.Tensor]:
        """Encode the environmental and boundary condition tensor."""
        feats = []
        h = cond
        for block in self.enc_c:
            h = block(h)
            feats.append(h)
        return feats

    def encode(
        self, x: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Map the input and conditions into the Gaussian latent space parameters.

        Clamping the log-variance stabilizes the KL-divergence loss term, avoiding
        NaN values if the variance attempts to collapse to zero or explode.
        """
        if cond is None:
            cond = torch.zeros_like(x)

        ex = self.encode_x(x)
        ec = self.encode_cond(cond)

        h_combined = torch.cat([ex[-1], ec[-1]], dim=1)
        h_combined = self.bottleneck_norm(h_combined)

        mu = self.to_mu(h_combined)
        logvar = self.to_logvar(h_combined)
        logvar = torch.clamp(logvar, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        return mu, logvar, ec

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Execute the reparameterization trick (z = mu + epsilon * sigma) to allow
        backpropagation through the stochastic sampling node.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        cond_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Reconstruct the probability distribution of lattice changes from the latent code.
        """
        cond_feats = self.skip_dropout(cond_feats)
        h = self.from_latent(z)

        for idx, (conv, res) in enumerate(zip(self.conv_ups, self.res_ups)):
            level = self.n_layers - 1 - idx
            c = cond_feats[level]

            h = torch.cat([h, c], dim=1)
            h = conv(h)
            h = res(h)

            if self.use_pooling:
                h = F.interpolate(h, scale_factor=2, mode="nearest")

        return self.conv_out(h)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the full autoencoding forward pass returning reconstruction and distribution parameters."""
        mu, logvar, cond_feats = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, cond_feats)

        return x_rec, mu, logvar

    @torch.no_grad()
    def decode_only(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Directly decode an explicit latent vector, bypassing the stochastic encoder."""
        cond_feats = self.encode_cond(cond)
        return self.decode(z, cond_feats)

    def _apply_nucleus_sampling(
        self, logits: torch.Tensor, tau: float, top_p: float, p_min: float
    ) -> torch.Tensor:
        """
        Apply nucleus (top-p) and threshold (p-min) filtering to the output logits.

        Truncating the long tail of the probability distribution prevents the model
        from predicting physically impossible structural artifacts (like massive,
        instantaneous isolated towers) caused by low-probability neural noise.
        """
        B_total, C_logits, L_logits = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C_logits)
        logits_flat = logits_flat / tau

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits_flat, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits_flat[indices_to_remove] = -float("Inf")

        if p_min > 0.0:
            probs_current = F.softmax(logits_flat, dim=-1)
            low_prob_mask = probs_current < p_min
            logits_flat[low_prob_mask] = -float("Inf")

        probs_flat = F.softmax(logits_flat, dim=-1)
        samples_flat = torch.multinomial(probs_flat, num_samples=1)

        return samples_flat.reshape(B_total, L_logits)

    def _generate_latent_noise(
        self,
        z: Optional[Union[torch.Tensor, int, float]],
        batch_total: int,
        latent_spatial_dim: int,
        dev: torch.device,
        seed: Optional[int],
    ) -> torch.Tensor:
        """
        Initialize the latent noise tensor `z` for generative sampling.
        """
        if z is None:
            if seed is not None:
                torch.manual_seed(seed)
            return torch.randn(
                batch_total, self.latent_channels, latent_spatial_dim, device=dev
            )

        if isinstance(z, (int, float)) and z == 0:
            return torch.zeros(
                batch_total, self.latent_channels, latent_spatial_dim, device=dev
            )

        return z

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        input_heights_int: torch.Tensor,
        z: Optional[Union[torch.Tensor, int, float]] = None,
        argmax: bool = False,
        tau: float = 1.0,
        top_p: float = 1.0,
        p_min: float = 0.0,
        temperature_kelvin: Optional[Union[torch.Tensor, float]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate an autoregressive surface evolution step by sampling from the learned distribution.
        """
        dev = device if device is not None else next(self.parameters()).device

        if temperature_kelvin is not None and not isinstance(
            temperature_kelvin, torch.Tensor
        ):
            temperature_kelvin = torch.tensor([temperature_kelvin], device=dev)

        cond = create_input_condition(
            start_lattice=input_heights_int,
            max_negative_change=self.max_negative_change,
            max_positive_change=self.max_positive_change,
            temp=self.temp_channel,
            temperature_kelvin=temperature_kelvin,
            temp_range=self.temp_range if self.temp_channel else None,
        )

        cond = cond.to(dev)

        N, C, L = cond.shape
        input_ref = input_heights_int

        if input_ref.dim() == 2:
            input_ref = input_ref.unsqueeze(1)

        if n_samples > 1:
            cond = cond.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, C, L)
            input_ref = input_ref.expand(N, n_samples, L)

        cond_feats = self.encode_cond(cond)
        deepest_feat_shape = cond_feats[-1].shape
        batch_total = deepest_feat_shape[0]
        latent_spatial_dim = deepest_feat_shape[2]

        z = self._generate_latent_noise(z, batch_total, latent_spatial_dim, dev, seed)
        logits = self.decode(z, cond_feats)

        if argmax:
            samples = torch.argmax(logits, dim=1)
        else:
            samples = self._apply_nucleus_sampling(logits, tau, top_p, p_min)

        change = samples - self.max_negative_change
        change = change.view(N, n_samples, -1)

        res_samples = input_ref.to(dev) + change

        # Enforce physical boundary condition: material depth cannot fall below the substrate (0).
        res_samples = torch.clamp(res_samples, min=0)

        return res_samples, F.softmax(logits, dim=1), change


def load_model_from_folder(model_folder: str) -> VariationalAutoencoder_FullyConv:
    """
    Load a trained CVAE model from the specified directory, reconstructing the architecture
    and loading the saved state dict.

    The model configuration is expected to be stored in a 'config.json' file within the folder,
    containing all necessary hyperparameters to instantiate the architecture.
    """
    config_path = os.path.join(model_folder, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model = VariationalAutoencoder_FullyConv(**config)
    state_dict_path = os.path.join(model_folder, "best_model.pt")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"Model state dict not found at {state_dict_path}")

    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode by default

    return model
