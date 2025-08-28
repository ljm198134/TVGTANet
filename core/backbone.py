from functools import reduce
from operator import add
import torch
import torch.nn as nn
from torchvision.models import resnet

class Backbone(nn.Module):

    def __init__(self, typestr):
        super(Backbone, self).__init__()

        self.backbone = typestr

        # feature extractor initialization
        if typestr == 'resnet50':
            self.feature_extractor = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        else:
            raise Exception('Unavailable backbone: %s' % typestr)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []
        bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
        # Layer 0
        feat = self.feature_extractor.conv1.forward(img)
        feat = self.feature_extractor.bn1.forward(feat)
        feat = self.feature_extractor.relu.forward(feat)
        feat = self.feature_extractor.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
            res = feat
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.feat_ids:
                feats.append(feat.clone())

            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

#
# # 这一部分是使用clip 的resnet50
# from functools import reduce
# from operator import add
#
# import clip
# import torch
# import torch.nn as nn
#
#
# class Backbone(nn.Module):
#
#     def __init__(self, typestr):
#         super(Backbone, self).__init__()
#
#         self.backbone = typestr
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
#         # feature extractor initialization
#         if typestr =='resnet50':
#             self.feature_extractor,_ = clip.load("RN50", device=self.device)
#             # self.feature_extractor, _ = clip.load("RN50", device=self.device)
#             self.feature_extractor = self.feature_extractor.visual.float()
#             self.feat_channels = [256, 512, 1024, 2048]
#             self.nlayers = [3, 4, 6, 3]
#             self.feat_ids = list(range(0, 17))
#         else:
#             raise Exception('Unavailable backbone: %s' % typestr)
#         self.feature_extractor.eval()
#
#         # define model
#         self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
#         print("self.lids",self.lids)
#         self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
#
#         self.cross_entropy_loss = nn.CrossEntropyLoss()
#
#     def extract_feats(self, img):
#         # print("img.shape",img.shape)
#         r""" Extract input image features """
#         # img = img.half()
#         feats = []
#         bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
#
#         # Layer 0
#         feat = self.feature_extractor.conv1.forward(img).to(self.device)
#         feat = self.feature_extractor.bn1.forward(feat).to(self.device)
#         feat = self.feature_extractor.relu1.forward(feat).to(self.device)
#
#         feat = self.feature_extractor.conv2.forward(feat).to(self.device)
#         feat = self.feature_extractor.bn2.forward(feat).to(self.device)
#         feat = self.feature_extractor.relu2.forward(feat).to(self.device)
#
#         feat = self.feature_extractor.conv3.forward(feat).to(self.device)
#         feat = self.feature_extractor.bn3.forward(feat).to(self.device)
#         feat = self.feature_extractor.relu3.forward(feat).to(self.device)
#         feat = self.feature_extractor.avgpool.forward(feat).to(self.device)
#
#         # Layer 1-4
#         for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
#             # print(hid, (bid, lid))
#             res = feat
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu1.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu2.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].avgpool.forward(feat)
#
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
#             # feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu3.forward(feat)
#
#             if bid == 0:
#                 res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
#             feat += res
#
#             if hid + 1 in self.feat_ids:
#                 feats.append(feat.clone())
#
#             feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu3.forward(feat)
#
#         return feats
#
#
# class Backbone2(nn.Module):
#
#     def __init__(self, typestr):
#         super(Backbone2, self).__init__()
#
#         self.backbone = typestr
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # feature extractor initialization
#         if typestr == 'vit32':
#             self.feature_extractor, _ = clip.load("ViT-B/32", device=self.device)
#             self.feature_extractor = self.feature_extractor.visual.float()
#             self.feat_channels = [768]  # ViT-B/32的输出通道数
#             self.nlayers = [12]  # ViT-B/32有12层
#             self.feat_ids = list(range(0, 12))
#         else:
#             raise Exception('Unavailable backbone: %s' % typestr)
#
#         self.feature_extractor.eval()
#
#     def extract_feats2(self, img):
#         r""" Extract input image features using ViT-B/32 """
#         # img = img.half()  # 如果需要，可以启用半精度
#         img = img.to(self.device)  # 将图像转移到设备
#
#         # 使用ViT-B/32模型进行特征提取
#         with torch.no_grad():
#             feats = self.feature_extractor.encode_image(img)
#
#         # 将特征维度从[bsz, 768]转换为更适合后续处理的形式
#         feats = feats.view(feats.shape[0], -1)  # Flatten the feature if necessary
#
#         return feats
#
# def main():
#     # Initialize the Backbone with 'resnet50'
#     model = Backbone(typestr='clip')
#
#     # Set the model to evaluation mode
#     model.eval()
#
#     # Generate a random input image tensor
#     # Image tensor shape [batch_size, channels, height, width]
#     img = torch.randn(1, 3, 400, 400).cuda()
#
#     # Extract features
#     feats = model.extract_feats(img)
#     # feats: list,存放16个feat
#
#
#     # Print extracted feature shapes
#     for idx, feat in enumerate(feats):
#         print(f"Feature {idx} shape: {feat.shape}")
#
# if __name__ == "__main__":
#     main()


