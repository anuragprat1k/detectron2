# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List
import torch
import numpy as np
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, cat, interpolate, get_norm
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

logger = logging.getLogger(__name__)

_TOTAL_SKIPPED = 0

ROI_PROPERTY_HEAD_REGISTRY = Registry("ROI_PROPERTY_HEAD")
ROI_PROPERTY_HEAD_REGISTRY.__doc__ = """
Registry for properties heads, which make property predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_properties_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_PROPERTY_HEAD.NAME`.
    """
    # logger.info("cfg.MODEL {}".format(cfg.MODEL))
    name = cfg.MODEL.ROI_PROPERTY_HEAD.NAME
    return ROI_PROPERTY_HEAD_REGISTRY.get(name)(cfg, input_shape)


def property_rcnn_loss(pred_logits, instances, num_classes):
    """
    Arguments:
        pred_logits (Tensor): A tensor of shape (N, C) where N is the total number
            of instances in the batch, C are number of property classes.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_props` field

    Returns a scalar tensor containing the loss.
    """

    # https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573

    gt_prop_labels = []
    for i in instances:
        # logger.info("instance gt_props {}".format(i.gt_props))
        gt_props = torch.zeros(i.gt_props.size(0), num_classes, device=i.gt_props.device).scatter_(1, i.gt_props, 1)
        for x in gt_props:
            x[0] = 0
        # logger.info("multi-hot {}".format(gt_props))
        gt_prop_labels.append(gt_props)

    target = cat(gt_prop_labels, dim=0)
    # logger.info("devices {}".format(gt_prop_labels.device))
    # logger.info("catted gt_prop_labels {}".format(gt_prop_labels.shape))
    # mult-hot encoding 
    # target = torch.zeros(gt_prop_labels.size(0), num_classes, device=gt_prop_labels.device).scatter_(1, gt_prop_labels, 1.)
    # logger.info("target {} logits {}".format(target.shape, pred_logits.shape))
    # sigmoid on logits 
    # pred_prob = pred_logits.sigmoid()

    pos_weight = torch.tensor([   0.0000, 1475.2000,   33.9908, 7376.0000,  409.7778,  819.5555,
        3688.0000, 1229.3334, 1053.7142, 1844.0000, 3688.0000,  819.5555,
         737.6000, 7376.0000, 3688.0000, 1844.0000, 3688.0000, 3688.0000,
        1844.0000, 1844.0000, 1229.3334, 2458.6667, 1844.0000, 7376.0000,
        1844.0000, 2458.6667, 1229.3334, 3688.0000, 3688.0000, 3688.0000,
        2458.6667, 3688.0000, 2458.6667, 1844.0000, 1229.3334, 3688.0000,
          22.2840,  819.5555, 3688.0000, 2458.6667, 1844.0000,  368.8000,
         819.5555, 1844.0000, 3688.0000,   14.9010, 3688.0000, 7376.0000,
        7376.0000,  433.8824,  167.6364, 2458.6667, 7376.0000, 3688.0000,
        3688.0000,   20.8952, 3688.0000, 7376.0000, 1844.0000,  351.2381,
        7376.0000, 3688.0000, 3688.0000, 3688.0000, 1475.2000, 3688.0000,
         737.6000, 3688.0000, 3688.0000, 7376.0000, 7376.0000, 7376.0000,
         106.8986, 3688.0000, 3688.0000, 3688.0000,  433.8824,  922.0000,
        7376.0000, 1475.2000, 1475.2000, 1229.3334, 1229.3334,  461.0000,
         737.6000, 3688.0000, 7376.0000, 1844.0000,  922.0000, 3688.0000,
        1229.3334, 3688.0000, 2458.6667, 3688.0000,  461.0000, 3688.0000,
        1844.0000,  254.3448, 3688.0000, 1475.2000, 3688.0000,  216.9412,
        3688.0000, 3688.0000,  567.3846, 1844.0000, 1844.0000, 3688.0000,
        3688.0000, 2458.6667, 3688.0000,  737.6000, 7376.0000, 3688.0000,
         368.8000, 2458.6667, 7376.0000, 7376.0000, 3688.0000, 3688.0000,
        3688.0000, 3688.0000, 7376.0000, 1475.2000,  368.8000,   14.7226,
          13.8908,    9.7955, 1844.0000, 3688.0000,  461.0000, 1229.3334,
         567.3846, 1844.0000, 1844.0000,   28.2605, 3688.0000, 3688.0000,
        3688.0000, 1844.0000, 2458.6667, 3688.0000,  922.0000,  150.5306,
        7376.0000, 7376.0000, 1844.0000,  388.2105, 3688.0000, 2458.6667,
        1053.7142, 3688.0000,  737.6000, 1844.0000, 7376.0000, 2458.6667,
        3688.0000, 3688.0000, 1844.0000, 1844.0000, 2458.6667, 1053.7142,
        7376.0000,  819.5555, 1844.0000,  461.0000, 3688.0000, 1475.2000,
        3688.0000,  273.1852,   76.8333, 1844.0000, 1053.7142,  922.0000,
        1844.0000, 1844.0000,  737.6000, 2458.6667, 3688.0000, 3688.0000,
         409.7778,  491.7333,  204.8889,  307.3333,  670.5455,  670.5455,
        3688.0000,  491.7333, 7376.0000, 7376.0000, 3688.0000, 3688.0000,
         614.6667, 7376.0000, 7376.0000, 3688.0000,  670.5455, 3688.0000,
        3688.0000, 1475.2000, 3688.0000,   25.3471, 3688.0000, 7376.0000,
        7376.0000, 1053.7142, 3688.0000, 2458.6667, 1229.3334, 3688.0000,
        7376.0000, 3688.0000, 1844.0000, 7376.0000,   14.0228, 7376.0000,
        3688.0000,  737.6000, 3688.0000, 1844.0000, 3688.0000, 1053.7142,
         388.2105,  737.6000,  614.6667, 1475.2000, 7376.0000, 2458.6667,
        7376.0000, 3688.0000, 7376.0000, 2458.6667, 1229.3334, 3688.0000,
        7376.0000, 7376.0000, 1844.0000,  737.6000, 3688.0000, 3688.0000,
        3688.0000, 3688.0000, 1475.2000, 7376.0000, 3688.0000,  922.0000,
          22.0838, 7376.0000, 1844.0000, 1475.2000, 1844.0000, 7376.0000,
        1229.3334,  153.6667, 3688.0000,  295.0400, 1475.2000, 3688.0000,
         922.0000,  163.9111, 3688.0000, 1053.7142, 1053.7142, 3688.0000,
        7376.0000, 7376.0000, 7376.0000, 1229.3334, 2458.6667, 3688.0000,
          21.5044, 3688.0000,    0.0000,  320.6956, 7376.0000,  922.0000,
        1844.0000, 3688.0000, 3688.0000, 3688.0000, 1475.2000,  670.5455,
         819.5555, 3688.0000, 3688.0000, 1844.0000,  320.6956,   69.5849,
        1053.7142,   22.5566,   28.8125])

    weights = torch.tensor([ 0.0000,  5.0694,  0.1168, 25.3471,  1.4082,  2.8163, 12.6735,  4.2245,
         3.6210,  6.3368, 12.6735,  2.8163,  2.5347, 25.3471, 12.6735,  6.3368,
        12.6735, 12.6735,  6.3368,  6.3368,  4.2245,  8.4490,  6.3368, 25.3471,
         6.3368,  8.4490,  4.2245, 12.6735, 12.6735, 12.6735,  8.4490, 12.6735,
         8.4490,  6.3368,  4.2245, 12.6735,  0.0766,  2.8163, 12.6735,  8.4490,
         6.3368,  1.2674,  2.8163,  6.3368, 12.6735,  0.0512, 12.6735, 25.3471,
        25.3471,  1.4910,  0.5761,  8.4490, 25.3471, 12.6735, 12.6735,  0.0718,
        12.6735, 25.3471,  6.3368,  1.2070, 25.3471, 12.6735, 12.6735, 12.6735,
         5.0694, 12.6735,  2.5347, 12.6735, 12.6735, 25.3471, 25.3471, 25.3471,
         0.3673, 12.6735, 12.6735, 12.6735,  1.4910,  3.1684, 25.3471,  5.0694,
         5.0694,  4.2245,  4.2245,  1.5842,  2.5347, 12.6735, 25.3471,  6.3368,
         3.1684, 12.6735,  4.2245, 12.6735,  8.4490, 12.6735,  1.5842, 12.6735,
         6.3368,  0.8740, 12.6735,  5.0694, 12.6735,  0.7455, 12.6735, 12.6735,
         1.9498,  6.3368,  6.3368, 12.6735, 12.6735,  8.4490, 12.6735,  2.5347,
        25.3471, 12.6735,  1.2674,  8.4490, 25.3471, 25.3471, 12.6735, 12.6735,
        12.6735, 12.6735, 25.3471,  5.0694,  1.2674,  0.0506,  0.0477,  0.0337,
         6.3368, 12.6735,  1.5842,  4.2245,  1.9498,  6.3368,  6.3368,  0.0971,
        12.6735, 12.6735, 12.6735,  6.3368,  8.4490, 12.6735,  3.1684,  0.5173,
        25.3471, 25.3471,  6.3368,  1.3341, 12.6735,  8.4490,  3.6210, 12.6735,
         2.5347,  6.3368, 25.3471,  8.4490, 12.6735, 12.6735,  6.3368,  6.3368,
         8.4490,  3.6210, 25.3471,  2.8163,  6.3368,  1.5842, 12.6735,  5.0694,
        12.6735,  0.9388,  0.2640,  6.3368,  3.6210,  3.1684,  6.3368,  6.3368,
         2.5347,  8.4490, 12.6735, 12.6735,  1.4082,  1.6898,  0.7041,  1.0561,
         2.3043,  2.3043, 12.6735,  1.6898, 25.3471, 25.3471, 12.6735, 12.6735,
         2.1123, 25.3471, 25.3471, 12.6735,  2.3043, 12.6735, 12.6735,  5.0694,
        12.6735,  0.0871, 12.6735, 25.3471, 25.3471,  3.6210, 12.6735,  8.4490,
         4.2245, 12.6735, 25.3471, 12.6735,  6.3368, 25.3471,  0.0482, 25.3471,
        12.6735,  2.5347, 12.6735,  6.3368, 12.6735,  3.6210,  1.3341,  2.5347,
         2.1123,  5.0694, 25.3471,  8.4490, 25.3471, 12.6735, 25.3471,  8.4490,
         4.2245, 12.6735, 25.3471, 25.3471,  6.3368,  2.5347, 12.6735, 12.6735,
        12.6735, 12.6735,  5.0694, 25.3471, 12.6735,  3.1684,  0.0759, 25.3471,
         6.3368,  5.0694,  6.3368, 25.3471,  4.2245,  0.5281, 12.6735,  1.0139,
         5.0694, 12.6735,  3.1684,  0.5633, 12.6735,  3.6210,  3.6210, 12.6735,
        25.3471, 25.3471, 25.3471,  4.2245,  8.4490, 12.6735,  0.0739, 12.6735,
         0.0000,  1.1020, 25.3471,  3.1684,  6.3368, 12.6735, 12.6735, 12.6735,
         5.0694,  2.3043,  2.8163, 12.6735, 12.6735,  6.3368,  1.1020,  0.2391,
         3.6210,  0.0775,  0.0990], device=pred_logits.device)

    # pos_weight = torch.full([num_classes], 3)
    # pos_weight[0] = 0
    pos_weight = pos_weight.to(device=pred_logits.device)
    # logger.info("{}, {}, {}".format(pos_weight.device, pred_prob.device, target.device))
    # bcewithcrossentropy
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_logits, target)
    # loss = (loss * pos_weight).mean()
    return {"loss_prop_v2": loss}


def property_rcnn_inference(pred_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_logits (Tensor): A tensor of shape (predicted_boxes, C) 
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    # logger.info("pred_logits {} \n pred_instances {}".format(pred_logits.shape, pred_instances[0].pred_classes.shape))
    
    pred_prob = pred_logits.sigmoid()

    # for each instance, pick all indices above the properties threshold
    # masks - for each bbox there is one m*n
    # keypoints - for each bbox, there is a list of at most n keypoints. This is what I want too.
    # sol 1 - just append indices of all 
    pred_instance = pred_instances[0]

    topk = []
    max_props = 3

    for x in pred_prob:
        curk = []
        for i in range(x.shape[0]):
            if x[i].item() > 0.5:
                curk.append((x[i].item(), i))
        curk.sort(key = lambda t: t[0])
        ik = [k[1] for k in curk[-min(max_props,len(curk)):]]
        if not ik:
            ik = [0]
        topk.append(ik)
    
    # logger.info("topk {}".format(topk))
    # logger.info("pred_instance {}".format(pred_instances))
    pred_instance.pred_props = torch.tensor(topk)
    
    # pred_instance.pred_props = (pred_prob > 0.7).int()

    # pick all indices above a certain threshold as predictions 

    # cls_agnostic_mask = pred_mask_logits.size(1) == 1

    # if cls_agnostic_mask:
    #     mask_probs_pred = pred_mask_logits.sigmoid()
    # else:
    #     # Select masks corresponding to the predicted classes
    #     num_masks = pred_mask_logits.shape[0]
    #     class_pred = cat([i.pred_classes for i in pred_instances])
    #     indices = torch.arange(num_masks, device=class_pred.device)
    #     mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    # num_boxes_per_image = [len(i) for i in pred_instances]
    # mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    # for prob, instances in zip(mask_probs_pred, pred_instances):
    #     instances.pred_masks = prob  # (1, Hmask, Wmask)



@ROI_PROPERTY_HEAD_REGISTRY.register()
class BasicPropertiesRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES #cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.num_classes = cfg.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # self.fcs.append(Linear(fc_dim, num_classes+1))

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x)) # Apparently [Batch Size] * [Num classes] 
        # logger.info("properties logits shape {}".format(x.shape))
        if self.training:
            return property_rcnn_loss(x, instances, self.num_classes)
        else:    
            property_rcnn_inference(x, instances)
            return instances

    @property
    def output_size(self):
        return self._output_size