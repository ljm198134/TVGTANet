# from data.dataset import FSSDataset
# from core.backbone import Backbone
# from eval.logger import Logger, AverageMeter
# from eval.evaluation import Evaluator
# from utils import commonutils as utils
# import utils.segutils as segutils
# import utils.crfhelper as crfutils
# import core.contrastivehead as ctrutils
# import core.denseaffinity as dautils
# import torch
#
# import torch.nn.functional as F
#
# import numpy as np
# def set_args(_args):
#     global args
#     # _args should write benchmark, datapath, nshot, adapt-to, postprocessing, logpath, verbosity
#     args = _args
#     args.backbone = 'resnet50'
#     args.nworker = 0
#     args.bsz = 1 # the method works on a single task, hence bsz=1
#     args.fold = 0
#
# # def makeDataloader():
# #     FSSDataset.initialize(img_size=400, datapath=args.datapath)
# #     dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
# #
# #
# #     return dataloader
#
# def makeDataloader():
#     try:
#         # 初始化数据集
#         FSSDataset.initialize(img_size=400, datapath=args.datapath)
#
#         # 构建数据加载器
#         dataloader = FSSDataset.build_dataloader(
#             args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot
#         )
#
#         # 检查数据加载器是否为空
#         if dataloader is None:
#             raise ValueError("DataLoader is None - failed to load dataset")
#
#         # 检查数据加载器中是否有数据
#         try:
#             sample = next(iter(dataloader))
#             print("Successfully loaded dataset with sample shape:", sample.shape)
#         except StopIteration:
#             raise ValueError("DataLoader is empty - no data was loaded")
#         except Exception as e:
#             print("Warning: Could not verify sample data shape:", str(e))
#
#         return dataloader
#
#     except Exception as e:
#         print(f"Error loading dataset: {str(e)}")
#         # 根据你的需求，可以选择返回None或重新抛出异常
#         return None
#
# def makeConfig():
#     config = ctrutils.ContrastiveConfig()
#
#     config.fitting.keepvarloss = True
#
#     config.fitting.maskloss = True
#
#     # 1 in 3
#     config.fitting.triplet_loss = True
#     config.fitting.proto_loss = False
#     config.fitting.selfattention_loss = False
#     config.fitting.o_t_contr_proto_loss = False
#
#     config.fitting.symmetricloss = False
#     config.fitting.q_nceloss = True
#     config.fitting.s_nceloss = True
#     config.fitting.split = True
#
#     config.fitting.num_epochs = 25
#     config.fitting.lr = 1e-2
#     config.fitting.debug = args.verbosity > 2
#
#     # used in ContrastiveFeatureTransformer
#     config.model.out_channels = 64
#     config.model.debug = args.verbosity > 0
#     config.featext.fit_every_episode = False
#
#     # Augmen(config.aug)
#     config.aug.blurkernelsize = [1]
#     config.aug.n_transformed_imgs = 2
#     config.aug.maxjitter = 0.0
#     config.aug.maxangle = 0
#     config.aug.maxscale = 1
#     config.aug.maxshear = 20
#     config.aug.apply_affine = True
#     config.aug.debug = args.verbosity > 2
#     return config
#
#
# def makeFeatureMaker(dataset, config, device='cpu', randseed=2, feat_extr_method=None):
#     utils.fix_randseed(randseed)
#     if feat_extr_method is None:
#         # annotation: get different resnet50 layers' features
#         feat_extr_method = Backbone(args.backbone).to(device).extract_feats
#     feat_maker = ctrutils.FeatureMaker(feat_extr_method, dataset.class_ids, config)
#     utils.fix_randseed(randseed)
#     feat_maker.norm_bb_feats = False
#     return feat_maker
#
#
#
#
#
# class SingleSampleEval:
#     def __init__(self, batch, feat_maker):
#         self.damat_comp = dautils.DAMatComparison()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.batch = batch
#         self.feat_maker = feat_maker
#         self.thresh_method = 'pred_mean'
#         self.post_proc_method = 'off'
#         self.verbosity = args.verbosity
#
#
#
#     def taskAdapt(self, detach=True):
#         b = self.batch
#         if self.device.type == 'cuda': b = utils.to_cuda(b)
#         # annotation: query_images support_images support_masks and support_class
#         self.q_img, self.s_img, self.s_mask, self.q_pascal,self.s_pascal ,self.class_id= b['query_img'], b['support_imgs'], b['support_masks'],b['query_pascal_label'],b['support_pascal_labels'], b[
#             'class_id'].item()
#
#         # # do taskAdapt
#         self.task_adapted = self.feat_maker.taskAdapt(self.q_img, self.s_img, self.s_mask, self.class_id,self.q_pascal, self.s_pascal)
#
#
#     def compare_feats(self):
#         if self.task_adapted is None:
#             print("error, do task adaption first")
#             return None
#         # print(self.task_adapted[1][5].shape)
#         self.logit_mask = self.damat_comp.forward(self.task_adapted[0], self.task_adapted[1], self.s_mask)
#         # self.logit_mask2 = refine_self_matching(self.task_adapted[1], self.task_adapted[0])
#         # visualize_result(self.logit_mask2)
#         return self.logit_mask
#
#     def threshold(self, method=None):
#         if self.logit_mask is None:
#             print("error, calculate logit mask first (do forward pass)")
#         if method is None:
#             method = self.thresh_method
#         self.thresh = segutils.calcthresh(self.logit_mask, self.s_mask, method)
#         self.pred_mask = (self.logit_mask > self.thresh).float()
#         return self.thresh, self.pred_mask
#
#     def postprocess(self):
#         if self.post_proc_method == 'off':
#             apply = False
#         elif self.post_proc_method == 'always':
#             apply = True
#         elif self.post_proc_method == 'dynamic':
#             apply = crfutils.crf_is_good(self)
#         else:
#             apply = False
#             print(f'Unknown postproc method: {self.post_proc_method=}')
#         return crfutils.apply_crf(self.q_img, self.logit_mask, segutils.thresh_fn(self.thresh_method)).to(self.device) if apply else self.pred_mask
#
#     # this method calls above components sequentially
#     def forward(self):
#         self.taskAdapt()
#
#         self.logit_mask = self.compare_feats()
#
#         self.thresh, self.pred_mask = self.threshold()
#
#         self.pred_mask = self.postprocess()
#         # visualize_result22(self.q_img)
#         # visualize_result(self.pred_mask)
#
#         return self.logit_mask, self.pred_mask
#
#     def calc_metrics(self):
#         self.area_inter, self.area_union = Evaluator.classify_prediction(self.pred_mask, self.batch)
#         self.fgratio_pred = self.pred_mask.float().mean()
#         self.fgratio_gt = self.batch['query_mask'].float().mean()
#         return self.area_inter[1] / self.area_union[1]  # fg-iou
#
# class AverageMeterWrapper:
#     def __init__(self, dataloader, device='cpu', initlogger=True):
#         if initlogger: Logger.initialize(args, training=False)
#         self.average_meter = AverageMeter(dataloader.dataset, device)
#         self.device = device
#         self.dataloader = dataloader
#         self.write_batch_idx = 50
#
#     def update(self, sseval):
#         self.average_meter.update(sseval.area_inter, sseval.area_union, torch.tensor(sseval.class_id).to(self.device),
#                                   loss=None)
#
#
#     def write(self, i):
#         self.average_meter.write_process(i, len(self.dataloader), 0, self.write_batch_idx)
#


from data.dataset import FSSDataset
from core.backbone import Backbone
from eval.logger import Logger, AverageMeter
from eval.evaluation import Evaluator
from utils import commonutils as utils
import utils.segutils as segutils
import utils.crfhelper as crfutils
import core.contrastivehead as ctrutils
import core.denseaffinity as dautils
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import time
import psutil

def set_args(_args):
    global args
    args = _args
    args.backbone = 'resnet50'
    args.nworker = 0
    args.bsz = 1
    args.fold = 0

def makeDataloader():
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    return dataloader

def makeConfig():
    config = ctrutils.ContrastiveConfig()
    config.fitting.keepvarloss = True
    config.fitting.maskloss = True
    config.fitting.triplet_loss = True
    config.fitting.proto_loss = False
    config.fitting.selfattention_loss = False
    config.fitting.o_t_contr_proto_loss = False
    config.fitting.symmetricloss = False
    config.fitting.q_nceloss = True
    config.fitting.s_nceloss = True
    config.fitting.split = True
    config.fitting.num_epochs = 25
    config.fitting.lr = 1e-2
    config.fitting.debug = args.verbosity > 2
    config.model.out_channels = 64
    config.model.debug = args.verbosity > 0
    config.featext.fit_every_episode = False
    config.aug.blurkernelsize = [1]
    config.aug.n_transformed_imgs = 2
    config.aug.maxjitter = 0.0
    config.aug.maxangle = 0
    config.aug.maxscale = 1
    config.aug.maxshear = 20
    config.aug.apply_affine = True
    config.aug.debug = args.verbosity > 2
    return config

def makeFeatureMaker(dataset, config, device='cpu', randseed=2, feat_extr_method=None):
    utils.fix_randseed(randseed)
    if feat_extr_method is None:
        feat_extr_method = Backbone(args.backbone).to(device).extract_feats
    feat_maker = ctrutils.FeatureMaker(feat_extr_method, dataset.class_ids, config)
    utils.fix_randseed(randseed)
    feat_maker.norm_bb_feats = False
    return feat_maker

def visualize_result(result):
    for idx, tensor in enumerate(result):
        if tensor is None:
            print(f"Element {idx}: None")
        else:
            img_np = result.detach().squeeze(0).cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np, cmap='gray')
            plt.title(f'Element {idx}')
            plt.axis('off')
            plt.show()

def visualize_result22(result):
    for idx, tensor in enumerate(result):
        if tensor is None:
            print(f"Element {idx}: None")
        else:
            result = result.squeeze(0)
            result = result.view(3, 400, 400)
            result = torch.mean(result, dim=0)
            print(result.shape)
            img_np = result.detach().squeeze(0).squeeze(0).cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np)
            plt.title(f'Element {idx}')
            plt.axis('off')
            plt.show()

class SingleSampleEval:
    def __init__(self, batch, feat_maker):
        self.damat_comp = dautils.DAMatComparison()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.feat_maker = feat_maker
        self.thresh_method = 'pred_mean'
        self.post_proc_method = 'off'
        self.verbosity = args.verbosity
        self.inference_time = None
        self.memory_usage = None
        self._ps_process = psutil.Process() if psutil else None

    def taskAdapt(self, detach=True):
        b = self.batch
        if self.device.type == 'cuda': b = utils.to_cuda(b)
        self.q_img, self.s_img, self.s_mask, self.q_pascal, self.s_pascal, self.class_id = \
            b['query_img'], b['support_imgs'], b['support_masks'], b['query_pascal_label'], b['support_pascal_labels'], b['class_id'].item()
        self.task_adapted = self.feat_maker.taskAdapt(self.q_img, self.s_img, self.s_mask, self.class_id, self.q_pascal, self.s_pascal)

    def compare_feats(self):
        if self.task_adapted is None:
            print("error, do task adaption first")
            return None
        self.logit_mask = self.damat_comp.forward(self.task_adapted[0], self.task_adapted[1], self.s_mask)
        return self.logit_mask

    def threshold(self, method=None):
        if self.logit_mask is None:
            print("error, calculate logit mask first (do forward pass)")
        if method is None:
            method = self.thresh_method
        self.thresh = segutils.calcthresh(self.logit_mask, self.s_mask, method)
        self.pred_mask = (self.logit_mask > self.thresh).float()
        return self.thresh, self.pred_mask

    def postprocess(self):
        if self.post_proc_method == 'off':
            apply = False
        elif self.post_proc_method == 'always':
            apply = True
        elif self.post_proc_method == 'dynamic':
            apply = crfutils.crf_is_good(self)
        else:
            apply = False
            print(f'Unknown postproc method: {self.post_proc_method=}')
        return crfutils.apply_crf(self.q_img, self.logit_mask, segutils.thresh_fn(self.thresh_method)).to(self.device) if apply else self.pred_mask

    def forward(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.taskAdapt()
        start_time = time.time()
        self.logit_mask = self.compare_feats()
        self.thresh, self.pred_mask = self.threshold()
        self.pred_mask = self.postprocess()
        end_time = time.time()
        self.inference_time = end_time - start_time
        if self.device.type == 'cuda':
            self.memory_usage = torch.cuda.max_memory_allocated(self.device)
        elif self._ps_process is not None:
            self.memory_usage = self._ps_process.memory_info().rss
        return self.logit_mask, self.pred_mask

    def calc_metrics(self):
        self.area_inter, self.area_union = Evaluator.classify_prediction(self.pred_mask, self.batch)
        self.fgratio_pred = self.pred_mask.float().mean()
        self.fgratio_gt = self.batch['query_mask'].float().mean()
        return self.area_inter[1] / self.area_union[1]

class AverageMeterWrapper:
    def __init__(self, dataloader, device='cpu', initlogger=True):
        if initlogger: Logger.initialize(args, training=False)
        self.average_meter = AverageMeter(dataloader.dataset, device)
        self.device = device
        self.dataloader = dataloader
        self.write_batch_idx = 1
        self.total_time = 0.0
        self.max_memory = 0
        self.count = 0

    def update(self, sseval):
        self.average_meter.update(sseval.area_inter, sseval.area_union, torch.tensor(sseval.class_id).to(self.device), loss=None)
        if sseval.inference_time is not None:
            self.total_time += sseval.inference_time
        if sseval.memory_usage is not None:
            self.max_memory = max(self.max_memory, sseval.memory_usage)
        self.count += 1

    def write(self, i):
        self.average_meter.write_process(i, len(self.dataloader), 0, self.write_batch_idx)
        if (i + 1) % self.write_batch_idx == 0 and self.count > 0:
            avg_time = self.total_time / self.count
            mem_str = f"{self.max_memory / 1024**2:.2f} MB"
            print(f"[AverageMeterWrapper] After {i+1} samples: avg inference time: {avg_time * 1000:.1f} ms, max memory: {mem_str}")
