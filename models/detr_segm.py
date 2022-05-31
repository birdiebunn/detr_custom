from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, GroupNorm
from torch import Tensor
from typing import List, Optional

from utils.misc import NestedTensor, nested_tensor_from_tensor_list

class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)
        
        hidden_dim = detr.transformer.d_model
        num_head = detr.transformer.num_head
        self.bbox_attn = MultiHeadAttentionMap(hidden_dim, hidden_dim, num_head, dropout=0.0)
        self.mask_head = MaskHeadSmallconv(hidden_dim + num_head, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        feats, pos = self.detr.backbone(samples)

        bs = feats[-1].tensors.shape[0]

        src, mask = feats[-1].decompose()
        assert mask is not None
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.quuery_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            output['aux_outputs'] = self.detr.set_aux_loss(outputs_class, outputs_coord)
        
        bbox_mask = self.bbox_attn(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [feats[2].tensors, feats[1].tensors, feats[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        output['pred_masks'] = outputs_seg_masks
        return output


class MultiHeadAttentionMap(nn.Module):
    def __init__(self, query_dim, hidden_dim, num_head, dropout=0.0, bias=True):
        super().__init__()
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # initialize weight as bias for q&k Linear layer
        nn.init.zeros_(self.q_linear.bias)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.xavier_uniform(self.q_linear.weight)
        nn.init.xavier_uniform(self.k_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_head) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)

        qh = q.view(q.shape[0], q.shape[1], self.num_head, self.hidden_dim // self.num_head)
        kh = k.view(k.shape[0], self.num_head, self.hidden_dim // self.num_head, k.shape[-2], k.shape[-1])
        weights = torch.einsum('bqnc,bnchw->bqnhw', qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill(mask.unsqeeze(1).unsqueeze(1), float('-inf'))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights
    
class MaskHeadSmallconv(nn.Module):
    def __init__(self, dim, fpn_dim, context_dim):
        super().__init__()

        inter_dim = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        
        self.conv1 = Conv2d(dim, dim, 3, padding=1)
        self.gn1 = GroupNorm(8, dim)

        self.conv2 = Conv2d(dim, inter_dim[1], 3, padding=1)
        self.gn2 = GroupNorm(8, inter_dim[1])

        self.conv3 = Conv2d(inter_dim[1], inter_dim[2], 3, padding=1)
        self.gn3 = GroupNorm(8, inter_dim[2])

        self.conv4 = Conv2d(inter_dim[2], inter_dim[3], 3, padding=1)
        self.gn4 = GroupNorm(8, inter_dim[3])
        
        self.conv5 = Conv2d(inter_dim[3], inter_dim[4], 3, padding=1)
        self.gn5 = GroupNorm(8, inter_dim[4])

        self.conv6 = Conv2d(inter_dim[4], 1, 3, padding=1)

        # adapter
        self.adp1 = Conv2d(fpn_dim[0], inter_dim[1], 1)
        self.adp2 = Conv2d(fpn_dim[1], inter_dim[2], 1)
        self.adp3 = Conv2d(fpn_dim[2], inter_dim[3], 1)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x:Tensor, bbox_mask: Tensor, fpn_list: List[Tensor]):
        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.con1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        adp_list = [self.adp1, self.adp2, self.adp3]
        conv_list = [self.conv3, self.conv4, self.conv5]
        gn_list = [self.gn3, self.gn4, self.gn5]

        for i in range(3):
            cur_fpn = adp_list[i](fpn_list[i])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
            
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
            x = conv_list[i](x)
            x = gn_list[i](x)
            x = F.relu(x)
        
        x = self.conv6(x)
        return x

def expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
