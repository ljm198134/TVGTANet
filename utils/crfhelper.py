import utils.segutils as segutils
import core.denseaffinity as dautils

def fwd_pass_support_imgs(sseval, supp_imgs,supp_masks):
    bsz, k, colorc, h, w = supp_imgs.shape
    s_feat = sseval.feat_maker.extract_bb_feats(supp_imgs.view(-1, colorc, h, w))
    s_feat = [l.view(bsz, k, *l.shape[1:]) for l in s_feat]
    c_tr = sseval.feat_maker.c_trs[sseval.class_id]
    s_feat_t = c_tr.getTaskAdaptedFeats(s_feat,s_feat,supp_masks)
    return s_feat_t

def get_augmentations_and_features(sseval):
    augimg_stack = sseval.feat_maker.c_trs[sseval.class_id].augimgs
    s_aug_imgs, s_aug_masks = augimg_stack.s, augimg_stack.s_mask[0]  # drop batch dim
    s_feat_aug = fwd_pass_support_imgs(sseval, s_aug_imgs[0],s_aug_masks)
    s_feat_aug = [f for f in s_feat_aug if f is not None]
    return s_feat_aug, s_aug_masks

def crf_is_good(sseval):
    debug = sseval.verbosity > 0
    if debug: print('estimating whether to postprocess')
    s_feat = [f[0] for f in sseval.task_adapted[1] if f is not None]
    s_mask = sseval.s_mask[0]
    s_feataug, s_maskaug = get_augmentations_and_features(sseval)
    pseudoquery = s_feat
    pseudosupport = s_feataug
    pred_maps = dautils.DAMatComparison().forward(pseudoquery, pseudosupport, s_maskaug)
    crf_yes_votes = 0
    criterion = lambda: crf_yes_votes > len(pred_maps) / 3
    for i in range(len(pred_maps)):
        pred_map = pred_maps[i:i + 1]
        fthresh = segutils.thresh_fn(sseval.thresh_method)
        thresh = fthresh(pred_map, s_maskaug)[0]
        pred_mask = (pred_map > thresh).float()
        pred_mask_crf = apply_crf(sseval.s_img[0], pred_map, fthresh, iterations=1).to(sseval.device)
        iou = segutils.iou(pred_mask, s_mask[i:i + 1])
        iou_crf = segutils.iou(pred_mask_crf, s_mask[i:i + 1])
        if debug: print('vote:', iou_crf > iou)
        if iou_crf > iou: crf_yes_votes += 1
        if criterion():
            break
        if len(pred_maps) - i - 1 < len(pred_maps) / 3 and crf_yes_votes == 0:
            break
    return criterion()

def apply_crf(rgb_img, fg_pred, fthresh, iterations=5):  # 5 on deployment, 1 on support-aug test for speedup
    crf = segutils.CRF(gaussian_stdxy=(1, 1), gaussian_compat=2,
                       bilateral_stdxy=(35, 35), bilateral_compat=1, stdrgb=(13, 13, 13))
    q = crf.iterrefine(iterations, rgb_img, fg_pred, fthresh)
    return q.argmax(1).float()