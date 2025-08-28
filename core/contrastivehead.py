
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import segutils
import core.denseaffinity as dautils
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from torchvision.transforms import GaussianBlur, ColorJitter
identity_mapping = lambda x, *args, **kwargs: x

class ContrastiveConfig:
    def __init__(self, config=None):
        if config is None:
            self._data = {
                'aug': {
                    'n_transformed_imgs': 2,
                    'blurkernelsize': [1],
                    'maxjitter': 0.0,
                    'maxangle': 0,
                    'maxscale': 1.0,
                    'maxshear': 20,
                    'randomhflip': False,
                    'apply_affine': True,
                    'debug': False
                },
                'model': {
                    'out_channels': 64,
                    'kernel_size': 1,
                    'prepend_relu': False,
                    'append_normalize': False,
                    'debug': False
                },
                'fitting': {
                    'lr': 1e-3,
                    'optimizer': torch.optim.SGD,
                    'num_epochs': 25,
                    'nce': {
                        'temperature': 0.5,
                        'debug': False
                    },
                    'normalize_after_fwd_pass': True,
                    'q_nceloss': True,
                    's_nceloss': True,
                    'protoloss': False,
                    'keepvarloss': True,
                    'symmetricloss': False,
                    'selfattentionloss': False,
                    'o_t_contr_proto_loss': True,
                    'debug': False
                },
                'featext': {
                    'l0': 3,  # the first resnet bottleneck id to consider (0,1,2,3,4,5...15)
                    'fit_every_episode': False
                }
            }
        else:
            self._data = config
    def __getattr__(self, key):
        # Try to get '_data' without causing a recursive call to __getattr__
        _data = super().__getattribute__('_data') if '_data' in self.__dict__ else None
        if _data is not None and key in _data:
            if isinstance(_data[key], dict):
                return ContrastiveConfig(_data[key])
            return _data[key]
        raise AttributeError(f"No setting named {key}")
    def __setattr__(self, key, value):
        if key == '_data':
            super().__setattr__(key, value)
        else:
            _data = super().__getattribute__('_data') if '_data' in self.__dict__ else None

            if _data is not None:
                _data[key] = value
            else:
                raise AttributeError("Unexpected")
    def __repr__(self):
        return str(self._data)

def dense_info_nce_loss(original_features, transformed_features, config_nce):
    B, C, H, W = transformed_features.shape
    o_features = original_features.expand(B, C, H, W).permute(0, 2, 3, 1).view(B, H * W, C)
    t_features = transformed_features.permute(0, 2, 3, 1).view(B, H * W, C)
    positive_logits = torch.einsum('bik,bik->bi', o_features, t_features) / config_nce.temperature
    all_logits = torch.einsum('bik,bjk->bij', o_features, t_features) / config_nce.temperature
    if config_nce.debug: print('pos/neg:', positive_logits.mean().detach(), all_logits.mean().detach())
    max_logits = torch.max(all_logits, dim=-1, keepdim=True).values
    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(all_logits - max_logits), dim=-1, keepdim=True))
    loss = - (positive_logits - log_sum_exp.squeeze())
    return loss.mean()  # [B=k*aug] or [B=k] -> scalar

def ssim(a, b):
    return torch.nn.CosineSimilarity()(a, b)

def augwise_proto(feat_vol, mask, k, aug):
    k, aug, c, h, w = k, aug, *feat_vol.shape[-3:]
    feature_vectors_augwise = torch.cat(feat_vol.view(k, aug, c, h * w).unbind(0), dim=-1)
    mask_augwise = torch.cat(segutils.downsample_mask(mask, h, w).view(k, aug, h * w).unbind(0), dim=-1)
    assert feature_vectors_augwise.shape == (aug, c, k * h * w) and mask_augwise.shape == (
        aug, k * h * w), "of transformed"
    fg_proto, bg_proto = segutils.fg_bg_proto(feature_vectors_augwise, mask_augwise)
    assert fg_proto.shape == bg_proto.shape == (aug, c)
    return fg_proto, bg_proto

def calc_q_pred_coarse_nodetach(qft, sft, s_mask, l0=3):
    bsz, c, hq, wq = qft.shape
    hs, ws = sft.shape[-2:]
    sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
    smasks_downsampled = [segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)]
    smask_row = torch.cat(smasks_downsampled, -1)
    damat = dautils.buildDenseAffinityMat(qft, sft_row)
    filtered = dautils.filterDenseAffinityMap(damat, smask_row)
    q_pred_coarse = filtered.view(bsz, hq, wq)
    return q_pred_coarse

def cal_pred_mask(f_base, f_transformed, mask_transformed, k, aug):
    c, h, w = f_base.shape[-3:]
    pseudoquery = torch.cat(f_base.view(-1, aug, c, h, w).unbind(0), -1)  # shape aug,c,h,w*k
    pseudosupport = f_transformed.view(k, aug, c, h, w).transpose(0, 1)  # shape bsz,k,c,h,w
    pseudosupportmask = mask_transformed.view(k, aug, h, w).transpose(0, 1)  # shape bsz,k,h,w
    pred_map = calc_q_pred_coarse_nodetach(pseudoquery, pseudosupport, pseudosupportmask, l0=0)
    return pred_map

def self_attention_loss(f_base, f_transformed, mask_base, mask_transformed, k, aug):
    c, h, w = f_base.shape[-3:]
    pseudoquery = torch.cat(f_base.view(-1, aug, c, h, w).unbind(0), -1)  # shape aug,c,h,w*k
    pseudoquerymask = torch.cat(mask_base.view(-1, aug, h, w).unbind(0), -1)  # shape aug,h,w*k
    pseudosupport = f_transformed.view(k, aug, c, h, w).transpose(0, 1)  # shape bsz,k,c,h,w
    pseudosupportmask = mask_transformed.view(k, aug, h, w).transpose(0, 1)  # shape bsz,k,h,w
    pred_map = calc_q_pred_coarse_nodetach(pseudoquery, pseudosupport, pseudosupportmask, l0=0)
    loss = torch.nn.BCELoss()(torch.sigmoid(pred_map).float(),pseudoquerymask.float())
    return loss.mean()

def mask_loss(pred, groundTrue):
    loss = torch.nn.CrossEntropyLoss()(pred, groundTrue.long())
    return loss

def mask_loss1(pred1, pred2):
    loss = torch.nn.CrossEntropyLoss()(pred1, pred2)
    return loss

def ctrstive_prototype_loss(base, transformed, mask_base, mask_transformed, k, aug):
    assert transformed.shape == base.shape, ".."
    b, c, h, w = base.shape
    assert b == k * aug, 'provide correct k and aug such that dim0=k*aug'
    assert mask_base.shape == mask_transformed.shape == (b, h, w), ".."
    fg_proto_o, bg_proto_o = augwise_proto(base, mask_base, k, aug)
    fg_proto_t, bg_proto_t = augwise_proto(transformed, mask_transformed, k, aug)
    enumer = torch.exp(
        ssim(fg_proto_o, fg_proto_t))  # 5vs5 (augvsaug), but in 5-shot: 25vs25, no, you want also augvsaug
    denom = torch.exp(ssim(fg_proto_o, fg_proto_t)) + torch.exp(ssim(fg_proto_o, bg_proto_t))
    assert enumer.shape == denom.shape == torch.Size([aug]), 'you want to calculate one prototype for each augmentation'
    loss = -torch.log(enumer / denom)  # [bsz]
    return loss.mean()

def opposite_proto_sim_in_aug(transformed_features, mapped_s_masks, k, aug):
    fg_proto_t, bg_proto_t = augwise_proto(transformed_features, mapped_s_masks, k, aug)
    fg_bg_sim_t = ssim(fg_proto_t, bg_proto_t)
    return fg_bg_sim_t.mean()

def proto_align_val_measure(original_features, transformed_features, mapped_s_masks, k, aug):
    fg_proto_o, _ = augwise_proto(original_features, mapped_s_masks, k, aug)
    fg_proto_t, _ = augwise_proto(transformed_features, mapped_s_masks, k, aug)
    fg_proto_sim = ssim(fg_proto_o, fg_proto_t)
    return fg_proto_sim.mean()

def atest():
    k, aug, c, h, w = 2, 5, 8, 20, 20
    f_base = torch.rand(k * aug, c, h, w).float()
    f_base.requires_grad = True
    f_transformed = torch.rand(k * aug, c, h, w).float()
    mask_base = torch.randint(0, 2, (k * aug, h, w)).float()
    mask_transformed = torch.randint(0, 2, (k * aug, h, w)).float()
    return self_attention_loss(f_base, f_transformed, mask_base, mask_transformed, k, aug)

def keep_var_loss(original_features, transformed_features):
    meandiff = original_features.mean((-2, -1)) - transformed_features.mean((-2, -1))
    vardiff = original_features.var((-2, -1)) - transformed_features.var((-2, -1))
    keepvarloss = torch.abs(meandiff).mean() + torch.abs(vardiff).mean()
    return keepvarloss

class GlobalLocalLoss(torch.nn.Module):
    def __init__(self, lambda_global=1.0, lambda_local=1.0, num_patches=2):
        super(GlobalLocalLoss, self).__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.num_patches = num_patches

    def forward(self, S_o, S_aug_o):
        L_global = keep_var_loss(S_o, S_aug_o)
        patches_o = self.split_into_patches(S_o)
        patches_aug_o = self.split_into_patches(S_aug_o)
        L_local = sum(keep_var_loss(patch_o, patch_aug_o)
                      for patch_o, patch_aug_o in zip(patches_o, patches_aug_o))
        L_global_local = self.lambda_global * L_global + self.lambda_local * L_local
        return L_global_local

    def split_into_patches(self, feature_map):
        bsz, c, h, w = feature_map.size()
        patch_size = h // self.num_patches
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w:
                    patch = feature_map[:, :, i:i + patch_size, j:j + patch_size]
                    patches.append(patch)
        return patches

class BatchChannelPreturbeAttention(nn.Module):
    def __init__(self):
        super(BatchChannelPreturbeAttention, self).__init__()
        self.bs = 512
        self.hidden_bs = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(self.bs,self.hidden_bs,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_bs,self.bs,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        B,C,H,W = x.shape
        G = C//self.bs
        result = []
        for i in range(G):
            x_crop = x[:, (0 + i) * self.bs:(1 + i) * self.bs, :, :]
            avg = self.avg_pool(x_crop).view([B,self.bs,1,1])
            fc = self.fc(avg).view([B,self.bs])
            result.append(fc)
        result =torch.cat(result,dim=1)
        result = result.view(B, C, 1, 1)
        return x*result

def split_into_patches1(feature_map):
    bsz, c, h, w = feature_map.size()
    patch_size = h // 2
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, h, patch_size):
            patch = feature_map[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def split_into_patches11(feature_map):
    c, h, w = feature_map.size()
    patch_size = h // 2
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = feature_map[ :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def augwise_proto1(feat_vol, mask, k, aug):
    k, aug, c, h, w = k, aug, *feat_vol.shape[-3:]
    feature_vectors_augwise = torch.cat(feat_vol.reshape(k, aug, c, h * w).unbind(0), dim=-1)
    mask_augwise = torch.cat(segutils.downsample_mask(mask, h, w).reshape(k, aug, h * w).unbind(0), dim=-1)
    assert feature_vectors_augwise.shape == (aug, c, k * h * w) and mask_augwise.shape == (
        aug, k * h * w), "of transformed"
    fg_proto, bg_proto = segutils.fg_bg_proto(feature_vectors_augwise, mask_augwise)
    assert fg_proto.shape == bg_proto.shape == (aug, c)
    return fg_proto, bg_proto

def ctrstive_prototype_loss1(base, transformed, mask_base, mask_transformed, k, aug):
    assert transformed.shape == base.shape, ".."
    b, c, h, w = base.shape
    assert b == k * aug, 'provide correct k and aug such that dim0=k*aug'
    assert mask_base.shape == mask_transformed.shape == (b, h, w), ".."
    fg_proto_o, bg_proto_o = augwise_proto1(base, mask_base, k, aug)
    fg_proto_t, bg_proto_t = augwise_proto1(transformed, mask_transformed, k, aug)
    enumer = torch.exp(
        ssim(fg_proto_o, fg_proto_t))  # 5vs5 (augvsaug), but in 5-shot: 25vs25, no, you want also augvsaug
    denom = torch.exp(ssim(fg_proto_o, fg_proto_t)) + torch.exp(ssim(fg_proto_o, bg_proto_t))
    assert enumer.shape == denom.shape == torch.Size([aug]), 'you want to calculate one prototype for each augmentation'
    loss = -torch.log(enumer / denom)  # [bsz]
    return loss.mean()

class ContrastiveFeatureTransformer(nn.Module):
    def __init__(self, in_channels, config_model):
        super(ContrastiveFeatureTransformer, self).__init__()
        self.damat_comp = dautils.DAMatComparison()
        kernel_size = config_model.kernel_size
        self.out_channels = config_model.out_channels
        self.in_channels = in_channels
        self.referenceLayer = nn.Linear(self.out_channels, 2, bias=True)
        nn.init.kaiming_normal_(self.referenceLayer.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.referenceLayer.bias, 0)
        self.rank = 16
        self.alpha = 32
        self.ConvLora = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.lora_A = nn.Parameter(self.ConvLora.weight.new_zeros((self.rank * 1, self.out_channels * 1)))
        self.lora_B = nn.Parameter(self.ConvLora.weight.new_zeros((self.out_channels * 1, self.rank * 1)))
        self.scaling = self.alpha / self.rank
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.linear = nn.Conv2d(self.out_channels, self.out_channels, 1)
        self.mask_conv = nn.Conv2d(self.out_channels, 2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.prepend_relu = config_model.prepend_relu  # annotation: 'prepend_relu': False
        self.append_normalize = config_model.append_normalize  # annotation: 'append_normalize': False
        self.global_local_loss_fn = GlobalLocalLoss(lambda_global=0.5, lambda_local=0.5)
        self.BatchChannelPreturbeAttention = BatchChannelPreturbeAttention()

    def forward(self, x, sfeat, smask, fit=False):
        h, w = sfeat.shape[-2], sfeat.shape[-1]
        smask = torch.squeeze(smask, dim=0)
        segutils.downsample_mask(smask, h, w)
        x = self.BatchChannelPreturbeAttention(x)
        x = self.conv(x)
        x = self.bn(x)
        x = nn.ReLU()(x)
        x = self.linear(x)
        if fit:
            mask = self.mask_conv(x)
            return x, mask
        return x

    def fit(self, mapped_qfeat_vol, aug_qfeat_vols, sfeat, smask, mapped_sfeat_vol, aug_sfeat_vols, q_pascal, s_pascal,qfeat,augmented_smasks, config_fit):
        f_norm = F.normalize if config_fit.normalize_after_fwd_pass else identity_mapping
        optimizer = config_fit.optimizer(self.parameters(), lr=config_fit.lr)
        s_pascal0 = s_pascal[0].cuda()
        q_pascal = q_pascal.cuda()
        if s_pascal0.shape[1]!=1:
            s_pascal0 = s_pascal0.view(-1, 1,s_pascal0.shape[-2],s_pascal0.shape[-1])
            q_pascal = q_pascal.view(1, -1,q_pascal.shape[-2],q_pascal.shape[-1])
        for epoch in range(config_fit.num_epochs):

            original_qfeatures, origial_qmasks = self(mapped_qfeat_vol, sfeat, smask, fit=True)
            transformed_qfeatures, transformed_qmasks = self(aug_qfeat_vols, sfeat, smask, fit=True)
            original_qfeatures = f_norm(original_qfeatures, p=2, dim=1)  # fwd pass non-augmented but affined
            transformed_qfeatures = f_norm(transformed_qfeatures, p=2, dim=1)  # fwd pass augmented
            qloss = dense_info_nce_loss(original_qfeatures, transformed_qfeatures,
                                        config_fit.nce) if config_fit.q_nceloss else 0
            original_sfeatures, original_spred = self(mapped_sfeat_vol, sfeat, smask, fit=True)
            transformed_sfeatures, transformed_spred = self(aug_sfeat_vols, sfeat, smask, fit=True)
            original_sfeatures = f_norm(original_sfeatures, p=2, dim=1)
            transformed_sfeatures = f_norm(transformed_sfeatures, p=2, dim=1)
            sfeatures, spred = self(sfeat, sfeat, smask, fit=True)
            qfeatures, qpred = self(qfeat, sfeat, smask, fit=True)
            resized_q_pascal = F.interpolate(q_pascal, size=qpred.shape[-2:], mode='bilinear', align_corners=False)
            resized_s_pascal0 = F.interpolate(s_pascal0, size=spred.shape[-2:], mode='bilinear', align_corners=False)
            ce_loss_q = F.cross_entropy(qpred, resized_q_pascal.squeeze(1).long())  # 转换成整数类别
            ce_loss_s = F.cross_entropy(spred, resized_s_pascal0.squeeze(1).long())
            sloss = dense_info_nce_loss(original_sfeatures, transformed_sfeatures,
                                        config_fit.nce) if config_fit.s_nceloss else 0
            global_local_loss = self.global_local_loss_fn(original_sfeatures, transformed_sfeatures)
            if config_fit.keepvarloss:
                sloss += keep_var_loss(original_sfeatures, transformed_sfeatures)
            kaug, c, h, w = transformed_sfeatures.shape
            aug = aug_qfeat_vols.shape[0]
            k = kaug // aug
            openFSSLoss = False
            original_sfeatures_split = split_into_patches1(original_sfeatures)
            transformed_sfeatures_split = split_into_patches1(transformed_sfeatures)
            augmented_smasks_split = split_into_patches11(augmented_smasks)
            split_loss = 0
            for i in range(4):
                loss = ctrstive_prototype_loss1(original_sfeatures_split[i], transformed_sfeatures_split[i],
                                                  augmented_smasks_split[i], augmented_smasks_split[i], k, aug)
                split_loss = split_loss + loss
            split_loss = split_loss + 0
            if config_fit.maskloss:
                h, w = spred.shape[-2], sfeat.shape[-1]
                smask_temp = smask.clone()
                smask_temp = segutils.downsample_mask(smask_temp.squeeze(dim=0), h, w)
                maskLosss = mask_loss(spred, smask_temp) + mask_loss(transformed_spred, augmented_smasks)
                _, spred = spred.max(dim=1)
                acc = (spred == smask_temp).sum() / (spred.shape[-1] * spred.shape[-1])
                if acc > 0.6:
                    openFSSLoss = True
            else:
                maskLosss = 0
            if config_fit.proto_loss:
                assert not config_fit.o_t_contr_proto_loss, 'only one of the proto losses should be used'
                opposite_proto_sim = opposite_proto_sim_in_aug(transformed_sfeatures, augmented_smasks, k, aug)
                if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0): print(
                    'proto-sim intER-class transf<->transf', opposite_proto_sim.item())
                proto_loss = opposite_proto_sim
            elif config_fit.triplet_loss:
                proto_loss = ctrstive_prototype_loss(original_sfeatures, transformed_sfeatures,
                                                      augmented_smasks, augmented_smasks, k, aug)
            elif config_fit.selfattention_loss:
                if openFSSLoss:
                    _, transformed_qmasks = transformed_qmasks.max(dim=1)
                    _, origial_qmasks = origial_qmasks.max(dim=1)
                    proto_loss = self_attention_loss(transformed_qfeatures, original_sfeatures, transformed_qmasks,
                                                     augmented_smasks, k, aug)
                else:
                    proto_loss = self_attention_loss(original_sfeatures, transformed_sfeatures, augmented_smasks,
                                                     augmented_smasks, k, aug)
                if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0): print(
                    'self-att non-transf<->transformed bce', proto_loss.item())
            else:
                proto_loss = 0
            q_s_loss_diff = torch.abs(qloss - sloss) if config_fit.symmetricloss else 0
            loss = qloss + sloss + q_s_loss_diff + 2*proto_loss + maskLosss + global_local_loss+ split_loss + ce_loss_q + ce_loss_s
            assert loss.isfinite().all(), f"invalid contrastive loss:{loss}"
            if config_fit.debug and (epoch == config_fit.num_epochs - 1 or epoch == 0):
                def gradient_magnitude(loss_term):
                    optimizer.zero_grad()
                    loss_term.backward(retain_graph=True)
                    magn = torch.abs(self.conv.weight.grad.mean()) + torch.abs(self.linear.weight.grad.mean())
                    return magn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config_fit.debug and epoch % 10 == 0: print('loss', loss.detach())

    def count_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

class AffineProxy:
    def __init__(self, angle, translate, scale, shear):
        self.affine_params = {
            'angle': angle,
            'translate': translate,
            'scale': scale,
            'shear': shear
        }
    def apply(self, img):
        return affine(img, angle=self.affine_params['angle'], translate=self.affine_params['translate'],
                      scale=self.affine_params['scale'], shear=self.affine_params['shear'])

class Augmen:
    def __init__(self, config_aug):
        # annotation: read config file
        self.config = config_aug
        self.blurs, self.jitters, self.affines = self.setup_augmentations()
    def setup_augmentations(self):
        blurkernelsize = self.config.blurkernelsize
        maxjitter = self.config.maxjitter
        maxangle = self.config.maxangle
        translate = (0, 0)
        maxscale = self.config.maxscale
        maxshear = self.config.maxshear
        blurs = []
        jitters = []
        affine_trans = []
        for i in range(self.config.n_transformed_imgs):
            kernel_size = np.random.choice(torch.tensor(blurkernelsize), (1,)).item()
            blur = GaussianBlur(kernel_size)
            blurs.append(blur)
            brightness_val = torch.rand(1).item() * maxjitter  # up to <maxjitter> change
            contrast_val = torch.rand(1).item() * maxjitter
            saturation_val = torch.rand(1).item() * maxjitter
            jitter = ColorJitter(brightness=brightness_val, contrast=contrast_val, saturation=saturation_val)
            jitters.append(jitter)
            angle = torch.randint(-maxangle, maxangle + 1, (1,)).item()
            shear = [torch.randint(-maxshear, maxshear + 1, (1,)).item() for _ in range(2)]
            scale = torch.rand(1).item() * (1 - maxscale) + maxscale
            affine_trans.append(AffineProxy(angle=angle, translate=translate, scale=scale, shear=shear))
        return (blurs, jitters, affine_trans)  # tuple of lists
    def augment(self, original_image, orignal_mask):
        transformed_imgs = []
        transformed_masks = []
        for blur, jitter, affine_trans in zip(self.blurs, self.jitters, self.affines):
            t_img = blur(original_image)
            t_img = jitter(t_img)
            t_mask = orignal_mask.clone()
            if self.config.apply_affine:
                t_img = affine_trans.apply(t_img)
                t_mask = affine_trans.apply(t_mask)
            transformed_imgs.append(t_img)
            transformed_masks.append(t_mask)
        return torch.stack(transformed_imgs, dim=1), torch.stack(transformed_masks, dim=1)
    def applyAffines(self, feat_vol):
        return torch.stack([trans.apply(feat_vol) for trans in self.affines], dim=1)

class CTrBuilder:
    def __init__(self, config, augmentator=None):
        if augmentator is None:
            augmentator = Augmen(config.aug)
        self.augmentator = augmentator
        self.augimgs = self.AugImgStack(augmentator)
        self.hasfit = False
        self.config = config

    class AugImgStack():
        def __init__(self, augmentator):
            self.augmentator = augmentator
            self.q, self.s, self.s_mask = None, None, None

        def init(self, s_img):
            bsz, k, aug, c, h, w = *s_img.shape[:2], self.augmentator.config.n_transformed_imgs, *s_img.shape[-3:]
            self.q = torch.empty(bsz, aug, c, h, w).to(s_img.device)
            self.s = torch.empty(bsz, k, aug, c, h, w).to(s_img.device)
            self.s_mask = torch.empty(bsz, k, aug, h, w).to(s_img.device)

        def show(self):
            bsz_, k_, aug_ = self.s.shape[:3]
            for b in range(bsz_):
                for k in range(k_):
                    print('k=', k, ' aug x (s, smask):')

    def makeAugmented(self, q_img, s_img, s_mask):
        self.augimgs.init(s_img)
        self.augimgs.q, _ = self.augmentator.augment(q_img, s_mask)
        for k in range(s_img.shape[1]):
            s_aug_imgs, s_aug_masks = self.augmentator.augment(s_img[:, k], s_mask[:, k])
            self.augimgs.s[:, k] = s_aug_imgs
            self.augimgs.s_mask[:, k] = s_aug_masks
        if self.config.aug.debug: self.augimgs.show()

    def build_and_fit(self, q_feat, s_feat, q_feataug, s_feataug,q_pascal,s_pascal, s_maskaug=None, s_mask=None):
        if s_maskaug is None: s_maskaug = self.augimgs.s_mask
        self.ctrs = self.buildContrastiveTransformers(q_feat, s_feat, q_feataug, s_feataug,q_pascal,s_pascal, s_maskaug, s_mask)
        self.hasfit = True

    def buildContrastiveTransformers(self, qfeat_alllayers, sfeat_alllayers, query_feats_aug, support_feats_aug,q_pascal,s_pascal,
                                     supp_aug_mask, smask=None):
        contrastive_transformers = []
        l0 = self.config.featext.l0
        s_aug_mask = supp_aug_mask.view(-1, *supp_aug_mask.shape[-2:])
        for (qfeat, sfeat, qfeataug, sfeataug) in zip(qfeat_alllayers[l0:], sfeat_alllayers[l0:], query_feats_aug[l0:], support_feats_aug[l0:]):
            bsz, k, aug, ch, h, w = sfeataug.shape
            assert bsz == 1, "bsz should be 1"
            assert supp_aug_mask.shape[1] == sfeat.shape[1] == k, f'augmented support shot-dimension mismatch:{s_aug_mask.shape[1]=},{sfeat.shape[1]=},(bsz,k,aug,ch,h,w)={bsz, k, aug, ch, h, w}'
            assert supp_aug_mask.shape[2] == qfeataug.shape[1] == aug, 'augmented shot-dimension mismatch'
            qfeat = qfeat.view(-1, *qfeat.shape[-3:])
            sfeat = sfeat.view(-1, *sfeat.shape[-3:])
            qfeataug = qfeataug.view(-1, *qfeataug.shape[-3:])
            sfeataug = sfeataug.view(-1, *qfeataug.shape[-3:])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            contrastive_head = ContrastiveFeatureTransformer(in_channels=ch, config_model=self.config.model).to(device)
            num_params = contrastive_head.count_trainable_params()
            mapped_qfeat = self.augmentator.applyAffines(qfeat)
            assert mapped_qfeat.shape[1] == aug, "should be 1,aug,c,h,w"
            mapped_qfeat = mapped_qfeat.view(-1, *qfeat.shape[-3:] )
            mapped_sfeat = self.augmentator.applyAffines(sfeat)
            assert mapped_sfeat.shape[1] == aug and mapped_sfeat.shape[0] == k, "should be k,aug,c,h,w"
            mapped_sfeat = mapped_sfeat.view(-1, *sfeat.shape[-3:])
            contrastive_head.fit(mapped_qfeat, qfeataug, sfeat, smask, mapped_sfeat, sfeataug, q_pascal, s_pascal, qfeat,
                                 segutils.downsample_mask(s_aug_mask, h, w), self.config.fitting)
            contrastive_transformers.append(contrastive_head)
        return contrastive_transformers

    def getTaskAdaptedFeats(self, layerwise_feats, s_feats, s_mask):
        if (self.ctrs == None): print("error: call buildContrastiveTransformers() first")
        task_adapted_feats = []
        for idx in range(len(layerwise_feats)):
            if idx < self.config.featext.l0:
                task_adapted_feats.append(None)
            else:
                input_shape = layerwise_feats[idx].shape
                idxth_feat = layerwise_feats[idx].view(-1, *input_shape[-3:])
                s_feat = s_feats[idx].view(-1, *input_shape[-3:])
                s_mask = s_mask.view(-1, s_feat.shape[0], s_mask.shape[-2], s_mask.shape[-2])
                forward_pass_res = self.ctrs[idx - self.config.featext.l0](idxth_feat, s_feat, s_mask)
                target_shape = *input_shape[:-3], *forward_pass_res.shape[-3:]
                task_adapted_feats.append(forward_pass_res.view(target_shape))
        return task_adapted_feats



class FeatureMaker:
    def __init__(self, feat_extraction_method, class_ids, config=ContrastiveConfig()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.featextractor = feat_extraction_method
        self.c_trs = {ctr: CTrBuilder(config) for ctr in class_ids}
        self.config = config  # 配置
        self.norm_bb_feats = False

    def extract_bb_feats(self, img):
        with torch.no_grad():
            return self.featextractor(img)

    def create_and_fit(self, c_tr, q_img, s_img, s_mask, q_feat, s_feat,q_pascal,s_pascal):
        if self.config.model.debug: print('contrastive adaption')
        c_tr.makeAugmented(q_img, s_img, s_mask)
        bsz, k, c, h, w = s_img.shape
        aug = c_tr.augmentator.config.n_transformed_imgs
        q_feataug = self.extract_bb_feats(c_tr.augimgs.q.view(-1, c, h, w))
        q_feataug = [l.view(bsz, aug, *l.shape[1:]) for l in q_feataug]
        s_feataug = self.extract_bb_feats(c_tr.augimgs.s.view(-1, c, h, w))
        s_feataug = [l.view(bsz, k, aug, *l.shape[1:]) for l in s_feataug]
        c_tr.build_and_fit(q_feat, s_feat, q_feataug, s_feataug,q_pascal,s_pascal, s_maskaug=None, s_mask=s_mask)

    def taskAdapt(self, q_img, s_img, s_mask, class_id,q_pascal,s_pascal):
        ch_norm = lambda t: t / torch.linalg.norm(t, dim=1)
        q_feat = self.extract_bb_feats(q_img)
        bsz, k, c, h, w = s_img.shape
        s_feat = self.extract_bb_feats(s_img.view(-1, c, h, w))
        if self.norm_bb_feats:
            q_feat = [ch_norm(l) for l in q_feat]
            s_feat = [ch_norm(l) for l in s_feat]
        s_feat = [l.view(bsz, k, *l.shape[1:]) for l in s_feat]
        c_tr = self.c_trs[class_id]
        if c_tr.hasfit is False or c_tr.config.featext.fit_every_episode:
            self.create_and_fit(c_tr, q_img, s_img, s_mask, q_feat, s_feat,q_pascal,s_pascal)
        q_feat_t, s_feat_t = c_tr.getTaskAdaptedFeats(q_feat, s_feat, s_mask), c_tr.getTaskAdaptedFeats(s_feat, s_feat, s_mask)
        return q_feat_t, s_feat_t

