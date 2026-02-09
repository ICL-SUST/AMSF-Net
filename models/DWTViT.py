import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.vision_transformer import VisionTransformer


class DirectionalGating(nn.Module):

    def __init__(self, feature_dim=768):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, lh, hl, hh):
        B, C, H, W = lh.shape

        lh_pool = self.gap(lh).view(B, 1)
        hl_pool = self.gap(hl).view(B, 1)
        hh_pool = self.gap(hh).view(B, 1)


        features = torch.cat([lh_pool, hl_pool, hh_pool], dim=1)  # [B, 3]


        weights = self.mlp(features)
        alpha, beta, gamma = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]


        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        hf = alpha * lh + beta * hl + gamma * hh

        return hf, weights


class ScaleGating(nn.Module):


    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_scales, num_scales * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_scales * 2, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, hf_list):

        B = hf_list[0].shape[0]

        features = []
        for hf in hf_list:
            hf_pool = self.gap(hf).view(B, 1)
            features.append(hf_pool)

        features = torch.cat(features, dim=1)


        weights = self.mlp(features)


        hf_fused = torch.zeros_like(hf_list[0])
        for i, hf in enumerate(hf_list):
            weight = weights[:, i:i + 1].unsqueeze(-1).unsqueeze(-1)
            hf_fused = hf_fused + weight * hf

        return hf_fused, weights


class GlobalLocalFeatureInteraction(nn.Module):


    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads


        self.multihead_attn = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=0.1,
            batch_first=True
        )


        self.context_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )


        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, global_features, local_features):

        B, N, D = local_features.shape


        global_expanded = global_features.unsqueeze(1).expand(B, N, D)  # [B, N, D]


        weighted_global, attn_weights = self.multihead_attn(
            query=local_features,
            key=global_expanded,
            value=global_expanded
        )

        gate_input = torch.cat([local_features, weighted_global], dim=-1)
        gate = self.context_gate(gate_input)


        local_interacted = local_features - gate * weighted_global


        local_interacted = self.norm(local_interacted)

        return local_interacted


class CrossDomainAttention(nn.Module):


    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):

        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]


        q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaptiveDomainFusion(nn.Module):


    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        self.norm = nn.LayerNorm(dim)


class MultiScaleDWT_ViT(nn.Module):


    def __init__(self,
                 model_name='vit_base_patch16_224.augreg_in21k',
                 pretrained=True,
                 num_classes=4,
                 dwt_levels=3,
                 drop_rate=0.05,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 fusion_layer=11,
                 use_context_refinement=True):

        super().__init__()

        self.num_classes = num_classes
        self.dwt_levels = dwt_levels
        self.fusion_layer = fusion_layer
        self.in_chans = 1 + 3 * dwt_levels
        self.use_context_refinement = use_context_refinement


        self.directional_gates = nn.ModuleList([
            DirectionalGating() for _ in range(dwt_levels)
        ])


        self.scale_gate = ScaleGating(num_scales=dwt_levels)


        self.spatial_input_proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1),
            nn.BatchNorm2d(2)
        )


        self.freq_input_proj = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )


        self.vit_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )


        self.embed_dim = self.vit_backbone.embed_dim
        self.num_patches = self.vit_backbone.patch_embed.num_patches


        self.cross_attn_s2f = CrossDomainAttention(
            self.embed_dim,
            num_heads=8,
            attn_drop=attn_drop_rate
        )
        self.cross_attn_f2s = CrossDomainAttention(
            self.embed_dim,
            num_heads=8,
            attn_drop=attn_drop_rate
        )


        self.domain_fusion = AdaptiveDomainFusion(self.embed_dim)


        if self.use_context_refinement:
            self.feature_interaction = GlobalLocalFeatureInteraction(
                self.embed_dim,
                num_heads=8
            )


        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.feature_dropout = nn.Dropout(drop_rate)


        self._init_weights()


        self.register_buffer('cached_band_weights', None, persistent=False)
        self.register_buffer('cached_scale_weights', None, persistent=False)
        self.register_buffer('cached_domain_weights', None, persistent=False)
        self.cached_activations = None

    def _init_weights(self):

        for m in [self.spatial_input_proj, self.freq_input_proj]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)


        for gate in self.directional_gates:
            for layer in gate.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward_features(self, x):

        B, C, H, W = x.shape
        L = self.dwt_levels


        LL = x[:, 0:1, :, :]
        bands = x[:, 1:, :, :].view(B, L, 3, H, W)


        hf_list = []
        band_weights_list = []

        for l in range(L):
            lh = bands[:, l, 0:1, :, :]
            hl = bands[:, l, 1:2, :, :]
            hh = bands[:, l, 2:3, :, :]


            hf, weights = self.directional_gates[l](lh, hl, hh)
            hf_list.append(hf)
            band_weights_list.append(weights)


        HF, scale_weights = self.scale_gate(hf_list)


        spatial_channels = self.spatial_input_proj(LL)
        freq_channels = self.freq_input_proj(HF)


        x_combined = torch.cat([spatial_channels, freq_channels], dim=1)


        x = self.vit_backbone.patch_embed(x_combined)


        cls_tokens = self.vit_backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit_backbone.pos_embed
        x = self.vit_backbone.pos_drop(x)


        num_blocks = len(self.vit_backbone.blocks)
        fusion_idx = min(self.fusion_layer, num_blocks - 1)

        for i in range(fusion_idx):
            x = self.vit_backbone.blocks[i](x)


        N = x.shape[1] - 1
        spatial_end = int(N * 2 / 3)

        cls_token = x[:, 0:1, :]
        spatial_tokens = x[:, 1:spatial_end + 1, :]
        freq_tokens = x[:, spatial_end + 1:, :]


        spatial_enhanced = spatial_tokens + self.cross_attn_s2f(spatial_tokens, freq_tokens, freq_tokens)
        freq_enhanced = freq_tokens + self.cross_attn_f2s(freq_tokens, spatial_tokens, spatial_tokens)


        x_fused = torch.cat([cls_token, spatial_enhanced, freq_enhanced], dim=1)


        spatial_pool = torch.cat([cls_token, spatial_enhanced], dim=1).mean(dim=1)
        freq_pool = torch.cat([cls_token, freq_enhanced], dim=1).mean(dim=1)


        features_concat = torch.cat([spatial_pool, freq_pool], dim=1)
        domain_weights = self.domain_fusion.mlp(features_concat)
        ws, wf = domain_weights[:, 0:1], domain_weights[:, 1:2]


        spatial_part = torch.cat([cls_token, spatial_enhanced], dim=1)
        freq_part = torch.cat([cls_token, freq_enhanced], dim=1)

        ws_expanded = ws.unsqueeze(1)
        wf_expanded = wf.unsqueeze(1)


        weighted_spatial = ws_expanded * spatial_part[:, :1 + spatial_end, :]
        weighted_freq = wf_expanded * freq_part[:, -freq_tokens.shape[1]:, :]


        x_fused[:, :1 + spatial_end, :] = weighted_spatial
        x_fused[:, 1 + spatial_end:, :] = weighted_freq


        x_fused = self.domain_fusion.norm(x_fused)


        for i in range(fusion_idx, num_blocks):
            x_fused = self.vit_backbone.blocks[i](x_fused)


        x_fused = self.vit_backbone.norm(x_fused)


        global_features = x_fused[:, 0]  # [B, D]
        local_features = x_fused[:, 1:]  # [B, N, D]


        if self.use_context_refinement:
            local_interacted = self.feature_interaction(global_features, local_features)
        else:
            local_interacted = local_features


        global_features = self.final_norm(global_features)
        local_features = self.final_norm(local_features)
        local_interacted = self.final_norm(local_interacted)
        local_interacted = self.feature_dropout(local_interacted)


        self.cached_band_weights = torch.stack(band_weights_list, dim=1) if band_weights_list else None
        self.cached_scale_weights = scale_weights
        self.cached_domain_weights = domain_weights
        self.cached_activations = {
            'LL': LL,
            'bands': bands,
            'HF': HF,
            'spatial_channels': spatial_channels,
            'freq_channels': freq_channels
        }

        return {
            'global_features': global_features,
            'local_features': local_features,
            'local_interacted': local_interacted,
            'feature_set': {
                'global': global_features,
                'local': local_features,
                'interacted': local_interacted
            },
            'domain_info': {
                'band_weights': self.cached_band_weights,
                'scale_weights': self.cached_scale_weights,
                'domain_weights': domain_weights,
                'spatial_weight': domain_weights[:, 0].mean().item(),
                'freq_weight': domain_weights[:, 1].mean().item()
            }
        }

    def forward(self, x):

        return self.forward_features(x)

    def get_domain_contribution(self, x):

        with torch.no_grad():
            features = self.forward_features(x)

            if self.cached_band_weights is None or self.cached_scale_weights is None:
                return {
                    'domain_weights': {'spatial': 0.5, 'frequency': 0.5},
                    'band_weights': None,
                    'scale_weights': None,
                    'contributions': {'LL': 0.25, 'LH': 0.25, 'HL': 0.25, 'HH': 0.25},
                    'activations': {'LL': 0, 'band_activations': []}
                }


            band_weights = self.cached_band_weights
            scale_weights = self.cached_scale_weights
            domain_weights = self.cached_domain_weights


            ws = domain_weights[:, 0].mean().item()
            wf = domain_weights[:, 1].mean().item()


            C_LL = ws
            C_LH_total = 0
            C_HL_total = 0
            C_HH_total = 0

            for l in range(self.dwt_levels):
                pi_l = scale_weights[:, l].mean().item()
                alpha_l = band_weights[:, l, 0].mean().item()
                beta_l = band_weights[:, l, 1].mean().item()
                gamma_l = band_weights[:, l, 2].mean().item()

                C_LH_total += wf * pi_l * alpha_l
                C_HL_total += wf * pi_l * beta_l
                C_HH_total += wf * pi_l * gamma_l


            total = C_LL + C_LH_total + C_HL_total + C_HH_total
            if total > 1e-8:
                C_LL = C_LL / total
                C_LH = C_LH_total / total
                C_HL = C_HL_total / total
                C_HH = C_HH_total / total
            else:
                C_LL = 0.25
                C_LH = 0.25
                C_HL = 0.25
                C_HH = 0.25


            bands = self.cached_activations['bands']
            band_activations = []

            for l in range(self.dwt_levels):
                lh_act = torch.std(bands[:, l, 0]).item()
                hl_act = torch.std(bands[:, l, 1]).item()
                hh_act = torch.std(bands[:, l, 2]).item()
                band_activations.append([lh_act, hl_act, hh_act])

            LL_activation = torch.std(self.cached_activations['LL']).item()
            HF_activation = torch.std(self.cached_activations['HF']).item()

            return {
                'domain_weights': {
                    'spatial': ws,
                    'frequency': wf
                },
                'band_weights': band_weights.cpu().numpy(),
                'scale_weights': scale_weights.cpu().numpy(),
                'contributions': {
                    'LL': C_LL,
                    'LH': C_LH,
                    'HL': C_HL,
                    'HH': C_HH
                },
                'activations': {
                    'LL': LL_activation,
                    'HF': HF_activation,
                    'band_activations': band_activations
                }
            }


def dwt_vit_multiscale(model_name='vit_base_patch16_224.augreg_in21k',
                       pretrained=True,
                       num_classes=4,
                       dwt_levels=3,
                       use_context_refinement=True,
                       **kwargs):

    drop_rate = kwargs.pop('drop_rate', 0.1)
    attn_drop_rate = kwargs.pop('attn_drop_rate', 0.1)
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)

    model = MultiScaleDWT_ViT(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dwt_levels=dwt_levels,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_context_refinement=use_context_refinement,
        **kwargs
    )

    print(f"\n=== 创建多尺度DWT-ViT模型 ===")
    print(f"使用timm模型: {model_name}")
    print(f"预训练: {pretrained}")
    print(f"DWT分解层数: {dwt_levels}")
    print(f"输入通道数: {model.in_chans}")
    print(f"嵌入维度: {model.embed_dim}")
    print(f"上下文细化: {use_context_refinement}")
    print("=== 模型创建完成 ===\n")

    return model


def dwt_vit(model_name='vit_base_patch16_224.augreg_in21k',
            pretrained=True,
            num_classes=4,
            **kwargs):


    drop_rate = kwargs.pop('drop_rate', 0.1)
    attn_drop_rate = kwargs.pop('attn_drop_rate', 0.1)
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)

    model = MultiScaleDWT_ViT(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dwt_levels=1,  # 单尺度
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_context_refinement=False,
        **kwargs
    )

    print(f"\n=== 创建单尺度DWT-ViT模型 ===")
    print(f"使用timm模型: {model_name}")
    print(f"预训练: {pretrained}")
    print(f"嵌入维度: {model.embed_dim}")
    print("=== 模型创建完成 ===\n")

    return model