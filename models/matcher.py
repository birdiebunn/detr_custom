import torch
from utils.box_ops import generalized_box_iou, convert_cxcywh_to_xyxy
from torch import nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cannot be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]

        # Compute Cost Matrices for a batch
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)

        # 
        target_ids = torch.cat([v['labels'] for v in targets])
        target_bbox = torch.cat([v['boxes'] for v in targets])

        # classification cost
        cost_class = - out_prob[: target_ids]
        
        # L1 cost between bbox-es
        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)

        # giou cost between bbox-es
        converted_out_bbox = convert_cxcywh_to_xyxy(out_bbox)
        converted_target_bbox = convert_cxcywh_to_xyxy(target_bbox) 
        cost_giou = -generalized_box_iou(converted_out_bbox, converted_target_bbox)

        # Final cost matrix
        total_cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        total_cost = total_cost.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(cost[id]) for id, cost in enumerate(total_cost.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)