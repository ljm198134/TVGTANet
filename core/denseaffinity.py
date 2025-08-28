import torch
import torch.nn.functional as F
import math
from utils import segutils


def buildHyperCol(feat_pyram):
    # concatenate along channel dim
    # upsample spatial size to largest feat vol space available
    target_size = feat_pyram[0].shape[-2:]
    upsampled = []
    for layer in feat_pyram:
        # if idx < self.stack_ids[0]: continue
        upsampled.append(F.interpolate(layer, size=target_size, mode='bilinear', align_corners=False))
    return torch.cat(upsampled, dim=1)

def paste_supports_together(supports):
    return torch.cat(supports.unbind(dim=1), dim=-1)

def buildDenseAffinityMat(qfeat_volume, sfeat_volume, softmax_arg2=True):  # bsz,C,H,W
    qfeat_volume, sfeat_volume = qfeat_volume.permute(0, 2, 3, 1), sfeat_volume.permute(0, 2, 3, 1)
    bsz, H, Wq, C = qfeat_volume.shape
    Ws = sfeat_volume.shape[2]
    # [px,C][C,px]=[px,px]
    dense_affinity_mat = torch.matmul(qfeat_volume.view(bsz, H * Wq, C),
                                      sfeat_volume.view(bsz, H * Ws, C).transpose(1, 2))
    if softmax_arg2 is False: return dense_affinity_mat
    dense_affinity_mat_softmax = (dense_affinity_mat / math.sqrt(C)).softmax(
        dim=-1)  # each query pixel's affinities sum up to 1 over support pxls
    return dense_affinity_mat_softmax


# filter with support mask following DAM
def filterDenseAffinityMap(dense_affinity_mat, downsampled_smask):
    # for each query pixel, aggregate all correlations where the support mask ==1
    # [px,px][px,1]=[px,1]
    bsz, HWq, HWs = dense_affinity_mat.shape
    # let mean(V)=1 -> sum(V)=len(V) -> d_mask / mean(d_mask)
    # downsampled_smask_norm = downsampled_smask / downsampled_smask.mean()
    q_coarse = torch.matmul(dense_affinity_mat, downsampled_smask.view(bsz, HWs, 1))
    return q_coarse.view(bsz, HWq)


def upsample(volume, h, w):
    return F.interpolate(volume, size=(h, w), mode='bilinear', align_corners=False)


def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.view(tensor1.size(0), -1)
    tensor2 = tensor2.view(tensor2.size(0), -1)
    return F.cosine_similarity(tensor1, tensor2)
def masked_average_pooling(features, mask):
    mask_expanded = mask.unsqueeze(1).expand_as(features)
    masked_features = features * mask_expanded
    return masked_features.sum(dim=[2, 3]) / mask_expanded.sum(dim=[2, 3])
def threshold_filter(similarity_map, threshold_fg, threshold_bg):
    mask_fg = (similarity_map > threshold_fg).float()
    mask_bg = (similarity_map < threshold_bg).float()
    return mask_fg, mask_bg
def adaptive_threshold_adjustment(similarity_fg, similarity_bg, similarity_query, sim_lost):
    avg_similarity_fg = similarity_fg.mean()
    avg_similarity_bg = similarity_bg.mean()
    avg_similarity_query = similarity_query.mean()
    delta = 0.2 * (avg_similarity_query - avg_similarity_fg) / (avg_similarity_query + 1e-5)
    return 0.7 + delta
def refine_self_matching(support_features, query_features):
    threshold_fg = 0.8
    threshold_bg = 0.6
    for i in range(3, len(support_features)):
        if support_features[i] is None or query_features[i] is None:
            continue
        similarity_map = cosine_similarity(support_features[i], query_features[i])
        mask_fg, mask_bg = threshold_filter(similarity_map, threshold_fg, threshold_bg)
        support_prototype = masked_average_pooling(support_features[i], mask_fg)
        query_prototype = masked_average_pooling(query_features[i], mask_fg)
        similarity_fg = cosine_similarity(support_prototype.unsqueeze(0), query_prototype.unsqueeze(0))
        new_threshold_fg = adaptive_threshold_adjustment(similarity_fg, mask_bg, similarity_map, mask_fg)
        threshold_fg = new_threshold_fg
        threshold_bg = threshold_bg - 0.05
        mask_fg, mask_bg = threshold_filter(similarity_map, threshold_fg, threshold_bg)
        refined_support_prototype = masked_average_pooling(support_features[i], mask_fg)
        refined_query_prototype = masked_average_pooling(query_features[i], mask_fg)
        out1 = cosine_similarity(refined_support_prototype.unsqueeze(0), refined_query_prototype.unsqueeze(0))
        prediction = F.softmax(out1, dim=1)
        if i > 3 and torch.abs(similarity_fg.mean() - prediction.mean()) < 1e-3:
            break
    return prediction


class DAMatComparison:

    def algo_mean(self, q_pred_coarses_t, s_mask=None):
        # print("q_pred_coarses_t.shape:",q_pred_coarses_t.shape)
        # print("q_pred_coarses_t.mean(1).shape:", q_pred_coarses_t.mean(1).shape)
        return q_pred_coarses_t.mean(1)

    #annotation: calculate the query's coarse predication
    def calc_q_pred_coarses(self, q_feat_t, s_feat_t, s_mask, l0=3):
        # print(s_mask.shape)
        q_pred_coarses = []
        # 1024 * 25 * 25 = c , h0 , w0
        h0, w0 = q_feat_t[l0].shape[-2:]
        for (qft, sft) in zip(q_feat_t[l0:], s_feat_t[l0:]):
            # print("sft.shape:",sft.shape)
            # print("qft.shape:",qft.shape)
            qft, sft = qft.detach(), sft.detach()
            bsz, c, hq, wq = qft.shape
            # 1024*25*25 or 2048*13*13
            hs, ws = sft.shape[-2:]

            sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
            smasks_downsampled = [segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)]
            smask_row = torch.cat(smasks_downsampled, -1)

            damat = buildDenseAffinityMat(qft, sft_row)
            filtered = filterDenseAffinityMap(damat, smask_row)
            q_pred_coarse = upsample(filtered.view(bsz, 1, hq, wq), h0, w0).squeeze(1)
            q_pred_coarses.append(q_pred_coarse)
        return torch.stack(q_pred_coarses, dim=1)

    def forward(self, q_feat_t, s_feat_t, s_mask, upsample=True, debug=False):
        q_pred_coarses_t = self.calc_q_pred_coarses(q_feat_t, s_feat_t, s_mask)
        # print(q_pred_coarses_t.shape)
        # if debug: display(segutils.pilImageRow(*q_pred_coarses_t.unbind(1), q_pred_coarses_t.mean(1)))

        # select the algorithm
        postprocessing_algorithm = self.algo_mean
        # do the postprocessing q_pred_coarses_t.mean(1)
        logit_mask = postprocessing_algorithm(q_pred_coarses_t, s_mask)
        # print(logit_mask.shape)
        if upsample:  # if query and support have different shape, then you must do upsampling yourself afterwards
            logit_mask = segutils.downsample_mask(logit_mask, *s_mask.shape[-2:])
            # print(logit_mask.shape)

        return logit_mask
