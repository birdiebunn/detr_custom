from models.criterion import SetCriterion
from models.postprocess import PostProcess, PostProcessSegm
import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone
from .transformer import build_transformer
from utils.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list
)
from .detr_segm import DETRsegm
from .matcher import build_matcher

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
    
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """
        Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n,k in zip([input_dim] + h, [output_dim] + h))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x    

def build(args, num_classes):
    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    # build DeTR base model (without segmentation head)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss 
    )

    # add in segmentation head if the 'masks' parameter is set to True
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # building matcher
    matcher = build_matcher(args)

    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_mask': args.mask_loss_coef,
        'loss_dice': args.dice_loss_coef,
    }
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

    losses = ['label', 'bbox', 'cardinality']
    if args.masks:
        losses += ['mask']
    
    # setup criterion
    criterion = SetCriterion(num_classes, matcher, weight_dict,
                            args.eos_coef, losses)
    criterion.to(device)

    # setup PostProcess
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
    
    return model, criterion, postprocessors
