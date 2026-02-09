import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .backbones import Conv_4, ResNet
from .DWTViT import dwt_vit, dwt_vit_multiscale


class FRN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False, num_cat=None,
                 use_dwt_vit=False, dwt_vit_config=None, use_multi_scale=False,
                 use_context_refinement=True):
        super().__init__()

        self.use_dwt_vit = use_dwt_vit
        self.use_multi_scale = use_multi_scale
        self.use_context_refinement = use_context_refinement

        if use_dwt_vit:

            if dwt_vit_config is None:
                dwt_vit_config = {}


            model_name = dwt_vit_config.pop('model_name', 'vit_small_patch16_224')
            pretrained = dwt_vit_config.pop('pretrained', True)
            num_classes = dwt_vit_config.pop('num_classes', 4)
            dwt_levels = dwt_vit_config.pop('dwt_levels', 3)


            if use_multi_scale:

                self.dwt_vit = dwt_vit_multiscale(
                    model_name=model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    dwt_levels=dwt_levels,
                    use_context_refinement=use_context_refinement,
                    **dwt_vit_config
                )
            else:

                self.dwt_vit = dwt_vit(
                    model_name=model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    **dwt_vit_config
                )


            vit_dim = self.dwt_vit.embed_dim


            self.num_patches = self.dwt_vit.num_patches


            self.feature_proj = nn.Sequential(
                nn.Linear(vit_dim, 640),
                nn.LayerNorm(640),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05)
            )


            self.d = 640


            self.spatial_proj = nn.Sequential(
                nn.LayerNorm(self.d),
                nn.Linear(self.d, self.d),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05)
            )


            self.feature_norm = nn.LayerNorm(self.d)

            num_channel = self.d


            self.resolution = 196


            init_scale = 15.0

        else:
            if resnet:
                num_channel = 640
                self.feature_extractor = ResNet.resnet12()
            else:
                num_channel = 64
                self.feature_extractor = Conv_4.BackBone(num_channel)

            self.d = num_channel
            self.resolution = 25
            init_scale = 1.0

        self.shots = shots
        self.way = way
        self.resnet = resnet


        self.scale = nn.Parameter(torch.FloatTensor([init_scale]), requires_grad=True)


        self.r = nn.Parameter(torch.FloatTensor([0.15, 0.15]), requires_grad=not is_pretraining)


        self.register_buffer('scale_min', torch.tensor(0.1))
        self.register_buffer('scale_max', torch.tensor(100.0))

        if is_pretraining:
            self.num_cat = num_cat

            self.cat_mat = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)
            nn.init.xavier_uniform_(self.cat_mat)

    def get_feature_map(self, inp):

        batch_size = inp.size(0)

        if self.use_dwt_vit:

            features = self.dwt_vit(inp)


            if self.use_context_refinement and 'local_interacted' in features:

                local_features = features['local_interacted']
                print(f"Using context-refined features") if not hasattr(self, '_logged_feature_type') else None
            else:
                local_features = features['local_features']
                print(f"Using original local features") if not hasattr(self, '_logged_feature_type') else None

            if not hasattr(self, '_logged_feature_type'):
                self._logged_feature_type = True

            B, N, D = local_features.shape

            assert N == 196, f"Expected 196 patches, got {N}"

            selected_features = local_features.view(B * N, D)
            selected_features = self.feature_proj(selected_features)

            feature_map = self.spatial_proj(selected_features)
            feature_map = self.feature_norm(feature_map)
            feature_map = feature_map.view(B, N, self.d)

        else:
            feature_map = self.feature_extractor(inp)

            if self.resnet:
                B, C, H, W = feature_map.shape
                feature_map = feature_map.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]

                feature_map = F.normalize(feature_map, p=2, dim=-1)

                feature_map = feature_map * math.sqrt(C)
            else:
                feature_map = feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()

        return feature_map

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):

        reg_base = self.resolution / self.d

        lam_scale = 10
        lam_eps = 1e-2
        lam = F.softplus(alpha) * lam_scale + lam_eps

        rho = 1.0 + torch.sigmoid(beta)

        st = support.permute(0, 2, 1)

        if Woodbury:

            sts = st.matmul(support)
            I = torch.eye(sts.size(-1), device=sts.device, dtype=sts.dtype).unsqueeze(0)

            rhs = sts
            lhs = sts + I * lam

            hat = torch.linalg.solve(lhs, rhs)

        else:

            sst = support.matmul(st)
            I = torch.eye(sst.size(-1), device=sst.device, dtype=sst.dtype).unsqueeze(0)

            m_inv = torch.linalg.solve(sst + I * lam, I)
            hat = st.matmul(m_inv).matmul(support)

        Q_bar = query.matmul(hat).mul(rho)

        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)

        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):

        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)

        support = feature_map[:way * shot].view(way, shot, resolution, d).mean(1)
        query = feature_map[way * shot:].view(way * query_shot * resolution, d)

        support = F.normalize(support, p=2, dim=-1)
        query = F.normalize(query, p=2, dim=-1)

        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha, beta=beta)
        neg_l2_dist = recon_dist.neg().view(way * query_shot, resolution, way).mean(1)

        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward_pretrain(self, inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size * self.resolution, self.d)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat, alpha=alpha, beta=beta)

        neg_l2_dist = recon_dist.neg().view(batch_size, self.resolution, self.num_cat).mean(1)

        scale = torch.clamp(self.scale, self.scale_min, self.scale_max)

        logits = neg_l2_dist * scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction

    def forward(self, inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        scale = torch.clamp(self.scale, self.scale_min, self.scale_max)

        logits = neg_l2_dist * scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, support

    def get_interpretability_info(self, inp):

        if not self.use_multi_scale:
            return None

        return self.dwt_vit.get_domain_contribution(inp)


def create_frn_with_multiscale_dwt_vit(way=5, shots=[5, 15], num_classes=4,
                                       model_name='vit_small_patch16_224',
                                       dwt_levels=3, drop_rate=0.05,
                                       attn_drop_rate=0.05, drop_path_rate=0.05,
                                       use_context_refinement=True):  # 新增参数

    dwt_vit_config = {
        'model_name': model_name,
        'pretrained': True,
        'num_classes': num_classes,
        'dwt_levels': dwt_levels,
        'drop_rate': drop_rate,
        'attn_drop_rate': attn_drop_rate,
        'drop_path_rate': drop_path_rate,
        'fusion_layer': 11
    }

    model = FRN(
        way=way,
        shots=shots,
        use_dwt_vit=True,
        dwt_vit_config=dwt_vit_config,
        use_multi_scale=True,
        use_context_refinement=use_context_refinement  # 传递上下文细化参数
    )

    return model


if __name__ == "__main__":
    model = create_frn_with_multiscale_dwt_vit(
        way=4,
        shots=[5, 15],
        num_classes=4,
        model_name='vit_small_patch16_224',
        dwt_levels=3,
        use_context_refinement=True
    )

    if torch.cuda.is_available():
        model = model.cuda()


    batch_size = 80
    x = torch.randn(batch_size, 10, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()

    model.train()
    log_pred, support = model(x)

    print(f"\n模型输出:")
    print(f"Log prediction shape: {log_pred.shape}")
    print(f"Support shape: {support.shape}")

    interp_info = model.get_interpretability_info(x)
    if interp_info:
        print(f"\n可解释性分析:")
        print(f"空域权重: {interp_info['domain_weights']['spatial']:.3f}")
        print(f"频域权重: {interp_info['domain_weights']['frequency']:.3f}")
        print(f"各子带贡献:")
        print(f"  LL: {interp_info['contributions']['LL']:.3f}")
        print(f"  LH: {interp_info['contributions']['LH']:.3f}")
        print(f"  HL: {interp_info['contributions']['HL']:.3f}")
        print(f"  HH: {interp_info['contributions']['HH']:.3f}")

    print(f"\n使用上下文细化: {model.use_context_refinement}")