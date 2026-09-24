# -*- coding: utf-8 -*-
"""Balanced BCE variant of PointPillarDepthLoss for the QCar encoder-pretrain
stage. New file, HEAL's own point_pillar_loss.py / point_pillar_depth_loss.py
are NOT modified.

Why: this project already found (Sept 3-6 session) that SigmoidFocalLoss
collapsed classification to all-negative on a small/imbalanced QCar
detection task, and that plain balanced BCEWithLogitsLoss with uniformly
sampled negatives is what recovered real detections. HEAL's own cls_weights
scheme (positives*pos_cls_weight + negatives, normalized by pos_normalizer)
already does per-anchor balancing -- the only change here is dropping the
focal (1-p)^gamma modulation term, i.e. plain weighted BCE instead of
weighted focal BCE. Regression/dir/iou/depth terms are untouched, copied
verbatim from the base classes so behavior stays identical except for the
one line that matters.
"""
import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import weighted_smooth_l1_loss


class PointPillarDepthBalancedBceLoss(PointPillarDepthLoss):
    def _cls_and_reg_loss(self, output_dict, target_dict, suffix=""):
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1, 1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1, 1) > 0
        pos_normalizer = positives.sum(1, keepdim=True).float()

        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
        if f'dm{suffix}' in output_dict:
            output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']

        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
            .view(batch_size, -1, 1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        # The one real change: plain weighted BCE, no focal (1-p)^gamma term.
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds, cls_labls.float(), reduction='none') * cls_weights
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

        total_loss = reg_loss + cls_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                                'reg_loss': reg_loss.item(),
                                'cls_loss': cls_loss.item()})
        return total_loss

    def forward(self, output_dict, target_dict, suffix=""):
        # Bypass PointPillarLoss.forward (focal cls loss) entirely, then run
        # PointPillarDepthLoss's depth-supervision addition the same way it
        # would run on top of the base class -- by calling it directly with
        # our own total_loss instead of super().forward()'s.
        total_loss = self._cls_and_reg_loss(output_dict, target_dict, suffix)

        all_depth_loss = 0
        depth_items_list = [x for x in output_dict.keys() if x.startswith(f"depth_items{suffix}")]
        for depth_item_name in depth_items_list:
            depth_item = output_dict[depth_item_name]
            depth_logit, depth_gt_indices = depth_item[0], depth_item[1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices)
            if self.use_fg_mask:
                fg_mask = depth_item[-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss *= weight_mask
            depth_loss = depth_loss.mean() * self.depth_weight
            all_depth_loss += depth_loss

        total_loss += all_depth_loss
        self.loss_dict.update({'depth_loss': all_depth_loss})
        return total_loss
