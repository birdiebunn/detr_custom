import sre_compile
from torch import nn
import torch
import torch.nn.functional as F
from utils.box_ops import generalized_box_iou, convert_cxcywh_to_xyxy
from utils.misc import nested_tensor_from_tensor_list, interpolate, sigmoid_focal_loss, dice_loss

class SetCriterion(nn.Module):
    """
        This class SetCriterion computes the loss for DeTR model
        The process contains 2 steps:
            1. compute hungarian assignment between the ground-truth & the output of the model
            2. supervise each pair of matched ground-truth / prediction (superivse class and bbox)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1]  =self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def get_loss(self, loss, outputs, targets, indices, num_bboxes, **kwargs):
        loss_map = {
            'label': self.loss_label,
            'cardinality': self.loss_cardinality,
            'bbox': self.loss_bbox,
            'mask': self.loss_mask
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss'
        return loss_map[loss](outputs, targets, indices, num_bboxes, **kwargs)

    def get_loss_label(self, outputs, targets, indices, num_bboxes, log=True):
        """
            Classification loss (NLL - Negative Log Loss)
            Note: targets dicts must contain key 'label'
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self.get_src_permutation_idx(indices)
        target_classes_o = torch.cat([target['label'][i] for target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits.transpose(1, 2), target_classes_o)[0]
        return losses

    def get_loss_cardinality(self, outputs, targets, indices, num_bboxes):
        """
            Cardinality error.
            It is intended for logging purposes only. It does not propagate/affect gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        target_lens = torch.as_tensor([len(v['label'] for v in targets)], device=device)

        # number of predictions that are NOT 'no-object'
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] -1).sum(1)
        card_err = F.l1_loss(card_pred.float(), target_lens.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_loss_bbox(self, outputs, targets, indices, num_bboxes):
        """
            Compute the losses related to the bounding box.
            L1 regression loss & GIoU loss
        """
        assert 'pred_boxes' in outputs
        idx = self.get_src_permutation_idx(indices)
        src_bbox = outputs['pred_boxes'][idx]
        target_bbox = torch.cat([target['bbox'][i] for target, (_, i) in zip(targets, indices)])

        loss_bbox = F.l1_loss(src_bbox, target_bbox, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            convert_cxcywh_to_xyxy(src_bbox),
            convert_cxcywh_to_xyxy(target_bbox)
        ))
        
        losses = {
            'loss_bbox': loss_bbox.sum() / num_bboxes,
            'loss_giou': loss_giou.sum() / num_bboxes
        }
        return losses

    def get_loss_mask(self, outputs, targets, indices, num_bboxes):
        """
            compute the losses related to the mask
            Focal loss & dice loss
        """
        assert 'pred_masks' in outputs
        src_idx = self.get_src_permutation_idx(indices)
        target_idx = self.get_target_permutation_idx(indices)

        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [target['mask'] for target in targets]

        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        losses = {
            'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_bboxes),
            'loss_dice': dice_loss(src_masks, target_masks, num_bboxes)
        }
        return losses

    def get_src_permutation_idx(self, indices):
        # perform permutation predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx
    
    def get_target_permutation_idx(self, indices):
        # perform permutation targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
        Compute the precision@k for the specified values of k
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    
    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct =  pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result