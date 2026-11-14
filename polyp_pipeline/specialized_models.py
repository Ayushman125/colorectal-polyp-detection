from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PraNetStyle(nn.Module):
    """ResNet reverse-attention segmentation network with multi-scale heads."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise RuntimeError(
                "Install segmentation-models-pytorch to build the PraNet-style model."
            ) from error

        self.encoder = smp.encoders.get_encoder(
            "resnet34", in_channels=3, depth=5,
            weights="imagenet" if pretrained else None,
        )
        channels = self.encoder.out_channels
        self.context = _conv_block(channels[5], 128)
        self.initial_logit = nn.Conv2d(128, 1, kernel_size=1)
        self.feature_projections = nn.ModuleList(
            nn.Conv2d(channels[index], 128, kernel_size=1)
            for index in (4, 3, 2)
        )
        self.refinement = nn.ModuleList(_conv_block(128, 128) for _ in range(3))
        self.delta_heads = nn.ModuleList(
            nn.Conv2d(128, 1, kernel_size=1) for _ in range(3)
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.encoder(image)
        current = self.context(features[5])
        logit = self.initial_logit(current)
        scale_logits = []
        for feature, projection, refine, delta_head in zip(
            (features[4], features[3], features[2]),
            self.feature_projections,
            self.refinement,
            self.delta_heads,
        ):
            logit = F.interpolate(
                logit, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
            projected = projection(feature)
            reverse_attention = 1.0 - torch.sigmoid(logit)
            current = refine(projected + projected * reverse_attention)
            logit = logit + delta_head(current)
            scale_logits.append(logit)
        final = F.interpolate(
            scale_logits[-1], size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        auxiliaries = [
            F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
            for value in reversed(scale_logits[:-1])
        ]
        return (final, *auxiliaries)


class TransFuseStyle(nn.Module):
    """CNN/Swin feature fusion network with multi-scale segmentation heads."""

    def __init__(self, pretrained: bool = True, image_size: int = 352) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("Install timm to build the TransFuse-style model.") from error

        self.cnn_encoder = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.transformer_encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=image_size,
        )
        cnn_channels = self.cnn_encoder.feature_info.channels()
        transformer_channels = self.transformer_encoder.feature_info.channels()
        self.cnn_projection = nn.ModuleList(
            nn.Conv2d(channels, 64, kernel_size=1) for channels in cnn_channels
        )
        self.transformer_projection = nn.ModuleList(
            nn.Conv2d(channels, 64, kernel_size=1)
            for channels in transformer_channels
        )
        self.cnn_gates = nn.ModuleList(
            nn.Conv2d(128, 64, kernel_size=1) for _ in cnn_channels
        )
        self.transformer_gates = nn.ModuleList(
            nn.Conv2d(128, 64, kernel_size=1) for _ in cnn_channels
        )
        self.fusion = nn.ModuleList(_conv_block(128, 64) for _ in cnn_channels)
        self.decoder = nn.ModuleList(_conv_block(64, 64) for _ in range(3))
        self.auxiliary_heads = nn.ModuleList(
            nn.Conv2d(64, 1, kernel_size=1) for _ in range(2)
        )
        self.final_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cnn_features = self.cnn_encoder(image)
        transformer_features = self.transformer_encoder(image)
        fused = []
        for cnn, transformer, cnn_projection, transformer_projection, cnn_gate, transformer_gate, fusion in zip(
            cnn_features,
            transformer_features,
            self.cnn_projection,
            self.transformer_projection,
            self.cnn_gates,
            self.transformer_gates,
            self.fusion,
        ):
            cnn = cnn_projection(cnn)
            if transformer.shape[1] != transformer_projection.in_channels:
                transformer = transformer.permute(0, 3, 1, 2).contiguous()
            transformer = transformer_projection(transformer)
            transformer = F.interpolate(
                transformer, size=cnn.shape[-2:], mode="bilinear", align_corners=False
            )
            joint = torch.cat((cnn, transformer), dim=1)
            cnn_fused = cnn + torch.sigmoid(cnn_gate(joint)) * transformer
            transformer_fused = transformer + torch.sigmoid(
                transformer_gate(joint)
            ) * cnn
            fused.append(fusion(torch.cat((cnn_fused, transformer_fused), dim=1)))

        decoded = fused[-1]
        for decoder, skip in zip(self.decoder, reversed(fused[:-1])):
            decoded = F.interpolate(
                decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = decoder(decoded + skip)
        scale_logits = [
            self.auxiliary_heads[0](fused[1]),
            self.auxiliary_heads[1](fused[0]),
            self.final_head(decoded),
        ]
        return tuple(
            F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
            for value in reversed(scale_logits)
        )
