import torch
from .wrapped_triplet_loss import WrappedTripletLoss
from .singular_loss import SingularLoss

import os


def SingularTripletLoss(num_classes: int, use_gpu: bool, args, param_controller) -> 'func':

    xent_loss = SingularLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth, penalty_position=args.penalty_position)
    htri_loss = WrappedTripletLoss(num_classes, use_gpu, args, param_controller, htri_only=True)

    def _loss(x, pids):

        _, y, v, features_dict = x

        from .sa_loss import sa_loss

        sa_loss_value = sa_loss(features_dict)

        loss = (
            args.lambda_xent * xent_loss(x, pids) +
            args.lambda_htri * htri_loss(x, pids) * param_controller.get_value()
        )

        return loss + sa_loss_value

    return _loss
