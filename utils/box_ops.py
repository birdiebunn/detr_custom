import torch
from torchvision.ops.boxes import box_area

def convert_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h
    ]
    return torch.stack(b, dim=-1)

def convert_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        x1 - x0,
        y1 - y1
    ]
    return torch.stack(b, dim=-1)

def bbox_iou(bbox1, bbox2):
    """
        Modify the script from torchvision to return both iou & union
    """
    area1 = box_area(bbox1)
    area2 = box_area(bbox2)

    lt = torch.max(bbox1[:, None, :2], bbox2[:, :2])
    rb = torch.min(bbox1[:, None, 2:], bbox2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(bbox1, bbox2):
    """
        bbox should be in [x0, y0, x1, y1] format
    """

    assert(bbox1[:, 2:] >= bbox1[:, :2]).all()
    assert(bbox2[:, 2:] >= bbox2[:, :2]).all()

    iou, union = bbox_iou(bbox1, bbox2)

    lt = torch.min(bbox1[:, None, :2], bbox2[:, :2])
    rb = torch.max(bbox1[:, None, 2:], bbox2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
