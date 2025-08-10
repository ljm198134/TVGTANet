import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F
import pandas as pd

norm = lambda t: (t - t.min()) / (t.max() - t.min())
denorm = lambda t, min_, max_: t * (max_ - min_) + min_

percentilerange = lambda t, perc: t.min() + perc * (t.max() - t.min())
midrange = lambda t: percentilerange(t, .5)

downsample_mask = lambda mask, H, W: F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear',
                                                   align_corners=False).squeeze(1)


def fg_bg_proto(sfeat_volume, downsampled_smask):
    B, C, vecs = sfeat_volume.shape
    reshaped_mask = downsampled_smask.expand(B, vecs).unsqueeze(1)  # ->[B,1,vecs]

    masked_fg = reshaped_mask * sfeat_volume
    fg_proto = torch.sum(masked_fg, dim=-1) / (torch.sum(reshaped_mask, dim=-1) + 1e-8)

    masked_bg = (1 - reshaped_mask) * sfeat_volume
    bg_proto = torch.sum(masked_bg, dim=-1) / (torch.sum(1 - reshaped_mask, dim=-1) + 1e-8)
    assert fg_proto.shape == (B, C), ":o"
    return fg_proto, bg_proto


intersection = lambda pred, target: (pred * target).float().sum()
union = lambda pred, target: (pred + target).clamp(0, 1).float().sum()

def iou(pred, target):  # binary only, input bsz,h,w
    i, u = intersection(pred, target), union(pred, target)
    iou = (i + 1e-8) / (u + 1e-8)
    return iou.item()

def otsus(batched_tensor_image, drop_least=0.05, mode='ordinary'):
    bsz = batched_tensor_image.size(0)
    binary_tensors = []
    thresholds = []

    for i in range(bsz):
        # Convert the tensor to numpy array
        numpy_image = batched_tensor_image[i].cpu().numpy()

        # Rescale to [0, 255] and convert to uint8 type for OpenCV compatibility
        npmin, npmax = numpy_image.min(), numpy_image.max()
        numpy_image = (norm(numpy_image) * 255).astype(np.uint8)

        # Drop values that are in the lowest percentiles
        truncated_vals = numpy_image[numpy_image >= int(255 * drop_least)]

        # Apply Otsu's thresholding
        if mode == 'via_triclass':
            thresh_value, _ = iterative_triclass_thresholding(truncated_vals)
        else:
            thresh_value, _ = cv2.threshold(truncated_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply the computed threshold on the original image
        binary_image = (numpy_image > thresh_value).astype(np.uint8) * 255
        binary_tensors.append(torch.from_numpy(binary_image).float() / 255)

        thresholds.append(torch.tensor(denorm(thresh_value / 255, npmin, npmax)) \
                          .to(batched_tensor_image.device, dtype=batched_tensor_image.dtype))
    binary_tensor_batch = torch.stack(binary_tensors, dim=0)
    thresh_batch = torch.stack(thresholds, dim=0)
    return thresh_batch, binary_tensor_batch


def iterative_otsus(probab_mask, s_mask, maxiters=5, mode='ordinary',
                    debug=False):  # verify that it works correctly when batch_size >1
    it = 1
    otsuthresh = 0
    assert probab_mask.min() >= 0 and probab_mask.max() <= 1, 'you should pass probabilites'
    while True:
        clipped = torch.where(probab_mask < otsuthresh, 0, probab_mask)
        otsuthresh, newmask = otsus(clipped.detach(), drop_least=.02, mode=mode)
        if otsuthresh >= s_mask.mean():
            return otsuthresh.to(probab_mask.device), newmask.to(probab_mask.device)
        if it >= maxiters:
            if debug:
                print('reached maxiter:', it, 'with thresh', otsuthresh.item(), \
                      'removed', int(((clipped == 0).sum() / clipped.numel()).item() * 10000) / 100, \
                      '% at lower and and new min,max is', clipped[clipped > 0].min().item(), clipped.max().item())
            return s_mask.mean(), (probab_mask > s_mask.mean()).float()  # otsuthresh
        it += 1


def calcthresh(fused_pred, s_masks, method='otsus'):

    if method == 'otsus':
        thresh = otsus(fused_pred)[0]
        return thresh
    elif method == 'pred_mean':
        otsu_thresh = otsus(fused_pred)[0]
        thresh = torch.max(otsu_thresh, fused_pred.mean())
    return thresh


def thresh_fn(method):
    def inner(fused_pred, s_masks=None):
        return calcthresh(fused_pred, s_masks, method)

    return inner

def install_pydensecrf():
    os.system('pip install git+https://github.com/08-401/TVSeg.git')

class CRF:
    def __init__(self, gaussian_stdxy=(3, 3), gaussian_compat=3,
                 bilateral_stdxy=(80, 80), bilateral_compat=10, stdrgb=(13, 13, 13)):
        self.gaussian_stdxy = gaussian_stdxy
        self.gaussian_compat = gaussian_compat
        self.bilateral_stdxy = bilateral_stdxy
        self.bilateral_compat = bilateral_compat
        self.stdrgb = stdrgb
        self.iters = 5
        self.debug = False

    def refine(self, image_tensor, fg_probs, soft_thresh=None, T=1):

        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
        except ImportError as e:
            print("pydensecrf not found. Installing...")
            install_pydensecrf()  # Ensure this function installs pydensecrf and handles any potential errors during installation.

        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
        except ImportError as e:
            print("Failed to import after installation. Please check the installation of pydensecrf.")
            raise  # This will raise the last exception that was handled by the except block

        if soft_thresh is None:
            soft_thresh, _ = otsus(fg_probs)
        image_tensor, fg_probs, soft_thresh = image_tensor.cpu(), fg_probs.cpu(), soft_thresh.cpu()
        fg_probs = torch.sigmoid(T * (fg_probs - soft_thresh))
        probs = torch.stack([1 - fg_probs, fg_probs], dim=1)  # crf expects both classes as input
        if self.debug:
            print('softthresh', soft_thresh)
            print('fg_probs min max', fg_probs.min(), fg_probs.max())
        bsz, C, H, W = probs.shape
        refined_masks = []
        image_numpy = np.ascontiguousarray( \
            (255 * image_tensor.permute(0, 2, 3, 1)).numpy().astype(np.uint8))
        probs_numpy = probs.numpy()
        for (image, prob) in zip(image_numpy, probs_numpy):
            # Unary potentials
            unary = np.ascontiguousarray(unary_from_softmax(prob))
            d = dcrf.DenseCRF2D(W, H, C)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=self.gaussian_stdxy, compat=self.gaussian_compat)
            d.addPairwiseBilateral(sxy=self.bilateral_stdxy, srgb=self.stdrgb,
                                   rgbim=image, compat=self.bilateral_compat)
            Q = d.inference(self.iters)
            if self.debug:
                print('Q:', np.array(Q).shape, np.array(Q)[0].mean(), np.array(Q).mean())
            result = np.reshape(Q, (2, H, W))  # np.argmax(Q, axis=0).reshape((H, W))
            refined_masks.append(result)

        return torch.from_numpy(np.stack(refined_masks, axis=0))

    def iterrefine(self, iters, q_img, fg_probs, thresh_fn, debug=False):
        pred = fg_probs.unsqueeze(1).expand(1, 2, *fg_probs.shape[-2:])
        for it in range(iters):
            thresh = thresh_fn(pred[:, 1])[0]

            pred = self.refine(q_img, pred[:, 1], soft_thresh=thresh)
        return pred

to_pil = lambda t: transforms.ToPILImage()(t) if t.shape[-1] > 4 else transforms.ToPILImage()(t.permute(2, 0, 1))

def pilImageRow(*imgs, maxwidth=800, bordercolor=0x000000):
    imgs = [to_pil(im.float()) for im in imgs]
    dst = Image.new('RGB', (sum(im.width for im in imgs), imgs[0].height))
    for i, im in enumerate(imgs):
        loc = [x0, y0, x1, y1] = [i * im.width, 0, (i + 1) * im.width, im.height]
        dst.paste(im, (x0, y0))
        ImageDraw.Draw(dst).rectangle(loc, width=2, outline=bordercolor)
    factorToBig = dst.width / maxwidth
    dst = dst.resize((int(dst.width / factorToBig), int(dst.height / factorToBig)))
    return dst

def tensor_table(**kwargs):
    tensor_overview = {}
    for name, tensor in kwargs.items():
        if callable(tensor):
            print(name, [tensor(t) for _, t in kwargs.items() if isinstance(t, torch.Tensor)])
        else:
            tensor_overview[name] = {
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'shape': tensor.shape,
            }
    return pd.DataFrame.from_dict(tensor_overview, orient='index')

