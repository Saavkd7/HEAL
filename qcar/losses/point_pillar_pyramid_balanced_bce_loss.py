# -*- coding: utf-8 -*-
"""Pyramid Fusion loss with the QCar balanced-BCE detection head.

Why: HEAL's PointPillarPyramidLoss computes its main detection loss through
PointPillarDepthLoss -> PointPillarLoss, i.e. SigmoidFocalLoss -- the same
focal classification term that collapsed to all-negative on QCar and was
replaced by balanced BCE for attfuse (see point_pillar_depth_balanced_bce_loss.py).
This class gives Pyramid that same correction and nothing else.

How: pure method-resolution order, no copied code. The MRO is
    PointPillarPyramidBalancedBceLoss -> PointPillarPyramidLoss
    -> PointPillarDepthBalancedBceLoss -> PointPillarDepthLoss -> PointPillarLoss
so every `super().forward(...)` inside PointPillarPyramidLoss (the detection
term in forward_collab / forward_single) now lands on the balanced-BCE
forward, while Pyramid's own per-level occupancy loss (calc_occ_loss, the
"_single" supervision) is left exactly as HEAL wrote it.
"""
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss

from qcar.losses.point_pillar_depth_balanced_bce_loss import PointPillarDepthBalancedBceLoss


class PointPillarPyramidBalancedBceLoss(PointPillarPyramidLoss, PointPillarDepthBalancedBceLoss):
    pass
