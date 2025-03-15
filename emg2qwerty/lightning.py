# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    CrossSensorAttention,
    CrossWristAttention,
    MultiBandRotationInvariantMLP,
    SinusoidalPositionalEncoding,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSLSTMEncoder
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                # WindowedEMGDataset(
                #     hdf5_path,
                #     transform=self.test_transform,
                #     # Feed the entire session at once without windowing/padding
                #     # at test time for more realism
                #     window_length=None,
                #     padding=(0, 0),
                #     jitter=False,
                # )
                # for hdf5_path in self.test_sessions

                # NORMALLY AT TEST TIME WE WOULD FEED IN THE WHOLE THING FOR REALISM
                # WE DID NOT TRAIN ON DIFFERENT SEQUENCE LENGTHS FOR POSITIONAL ENCODING MODELS, ONLY 10,000 SAMPLES
                # AS A RESULT, THE EASY SOLUTION IS TO CONTINUE USING WINDOW LENGTH AND PADDING

                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class OGTDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),



            # TDSConvEncoder(
            #     num_features=num_features,
            #     block_channels=block_channels,
            #     kernel_width=kernel_width,
            # ),

            TDSLSTMEncoder(
                num_features=num_features,
                lstm_hidden_size=128,
                num_lstm_layers=4,
            ),



            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvCTCModule(pl.LightningModule): #TDS Conv + LSTM encoder+ CTC
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 4,
        dropout: float = 0.2,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()


        
        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        self.model = nn.Sequential(
            # Spectrogram normalization
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            
            # Feature extraction
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            
            # Flatten features
            nn.Flatten(start_dim=2),
            
            # Enhanced LSTM encoder
            TDSLSTMEncoder(
                num_features=num_features,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout
            ),
            
            # Output projection
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        
        
        # ... rest of the initialization code remains the same ...
            # TDSConvEncoder(
            #     num_features=num_features,
            #     block_channels=block_channels,
            #     kernel_width=kernel_width,
            # ),



            # # (T, N, num_classes)
            # nn.Linear(num_features, charset().num_classes),
            # nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvLstmCTCModule(pl.LightningModule): #double encoder model, goated so far
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            
            # TDS Convolutional layers
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            
            # LSTM layers
            TDSLSTMEncoder(
                num_features=num_features,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
            ),
            
            # Output layer
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    # Reuse the same methods as TDSConvCTCModule
    forward = TDSConvCTCModule.forward
    _step = TDSConvCTCModule._step
    _epoch_end = TDSConvCTCModule._epoch_end
    training_step = TDSConvCTCModule.training_step
    validation_step = TDSConvCTCModule.validation_step
    test_step = TDSConvCTCModule.test_step
    on_train_epoch_end = TDSConvCTCModule.on_train_epoch_end
    on_validation_epoch_end = TDSConvCTCModule.on_validation_epoch_end
    on_test_epoch_end = TDSConvCTCModule.on_test_epoch_end
    configure_optimizers = TDSConvCTCModule.configure_optimizers


class TransformerCTCModule(pl.LightningModule):
    """Transformer-based model for predicting keystrokes from EMG signals
    with options for cross-wrist and cross-sensor attention.
    """
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cross_wrist_attention: bool,
        use_cross_sensor_attention: bool,
        loss_type: str,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.use_cross_wrist_attention = use_cross_wrist_attention
        self.use_cross_sensor_attention = use_cross_sensor_attention

        # Feature extraction
        self.spectrogram_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )

        # Cross-Sensor Attention (optional)
        if use_cross_sensor_attention:
            self.cross_sensor_attention = nn.ModuleList([
                CrossSensorAttention(
                    d_model=mlp_features[-1],
                    num_channels=self.ELECTRODE_CHANNELS,
                    nhead=nhead,
                    dropout=dropout
                ) for _ in range(self.NUM_BANDS)
            ])

        # CrossWristAttention (optional)
        if use_cross_wrist_attention:
            self.cross_wrist_attention = CrossWristAttention(
                d_model=mlp_features[-1], 
                nhead=nhead, 
                dropout=dropout
            )
            # After cross-wrist attention, feature dim remains mlp_features[-1]
            transformer_input_dim = mlp_features[-1]
        else:
            # Without cross-wrist attention, we flatten the bands
            transformer_input_dim = num_features

        # Positional encoding (sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=transformer_input_dim,
            dropout=dropout,
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Keep time first (T, N, F)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(transformer_input_dim, charset().num_classes),
            nn.LogSoftmax(dim=-1) if loss_type == 'ctc' else nn.Identity(),
        )

        # Loss function
        self.loss_type = loss_type
        if loss_type == 'ctc':
            self.loss_fn = nn.CTCLoss(blank=charset().null_class)
        elif loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.spectrogram_norm(inputs)  # (T, N, bands=2, C=16, freq)
        x = self.mlp(x)  # (T, N, bands=2, features)
        
        # Apply Cross-Sensor Attention if enabled
        if self.use_cross_sensor_attention:
            # Process each band separately through cross-sensor attention
            bands_list = []
            for band_idx in range(self.NUM_BANDS):
                band_data = x[:, :, band_idx]  # (T, N, features)
                band_data = self.cross_sensor_attention[band_idx](band_data)
                bands_list.append(band_data)
            
            # Recombine bands
            x = torch.stack(bands_list, dim=2)  # (T, N, bands=2, features)
        
        # Apply CrossWristAttention if enabled
        if self.use_cross_wrist_attention:
            x = self.cross_wrist_attention(x)  # (T, N, features)
        else:
            # Flatten the bands dimension
            x = x.flatten(start_dim=2)  # (T, N, bands*features)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)
        
        # Apply output layer
        return self.output_layer(transformer_output)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        if self.loss_type == 'ctc':
            loss = self.loss_fn(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )
        else:  # cross_entropy
            # Flatten time and batch dimensions for CE loss
            pred_flat = emissions.view(-1, emissions.size(-1))
            target_flat = targets.view(-1)
            loss = self.loss_fn(pred_flat, target_flat)

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    


class TDSConvTransformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
        # New parameters layer normalization
        use_layer_norm: bool = False,
        transformer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.use_layer_norm = use_layer_norm

        num_features = self.NUM_BANDS * mlp_features[-1]
        
        # Spectrogram normalization layer
        self.spectrogram_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # MLP for feature extraction
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        
        # Layer normalization layers (if enabled)
        if self.use_layer_norm:
            self.post_mlp_norm = nn.LayerNorm(num_features)
            self.post_conv_norm = nn.LayerNorm(num_features)
            self.pre_output_norm = nn.LayerNorm(num_features)
        
        # TDS Convolutional layers
        self.conv_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )
        
        # Transformer Encoder components
        # Project to d_model if needed
        self.input_projection = nn.Linear(num_features, d_model) if num_features != d_model else nn.Identity()
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, dropout)
        
        # Transformer encoder with optional norm_first
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Time dimension first
            norm_first=transformer_norm_first,  # Option to apply normalization first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Output projection if needed
        self.output_projection = nn.Linear(d_model, num_features) if num_features != d_model else nn.Identity()
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        
        # Loss type (default to CTC)
        self.loss_type = 'ctc'
        self.loss_fn = self.ctc_loss

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.spectrogram_norm(inputs)  # (T, N, bands=2, C=16, freq)
        x = self.mlp(x)  # (T, N, bands=2, features)
        
        # Flatten the bands dimension
        x = x.flatten(start_dim=2)  # (T, N, bands*features)
        
        # Apply layer norm after MLP if enabled
        if self.use_layer_norm:
            x = self.post_mlp_norm(x)
        
        # Apply TDS Conv Encoder
        x = self.conv_encoder(x)  # (T, N, features)
        
        # Apply layer norm after conv encoder if enabled
        if self.use_layer_norm:
            x = self.post_conv_norm(x)
        
        # Project to transformer dimension
        x = self.input_projection(x)  # (T, N, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)  # (T, N, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (T, N, d_model)
        
        # Project back to original feature dimension
        x = self.output_projection(x)  # (T, N, features)
        
        # Apply layer norm before output layer if enabled
        if self.use_layer_norm:
            x = self.pre_output_norm(x)
        
        # Apply output layer
        x = self.output_layer(x)  # (T, N, num_classes)
        
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

    