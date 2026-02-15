import torch
import torch.nn as nn


class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Complete IoU (CIoU) Loss for bounding box regression.
        Considers: overlapping area, central point distance, and aspect ratio.
        
        Args:
            pred_boxes: (N, 4) in [x1, y1, x2, y2] format
            target_boxes: (N, 4) in [x1, y1, x2, y2] format
        """
        pred_cxcy = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        pred_wh = pred_boxes[:, 2:] - pred_boxes[:, :2]
        
        target_cxcy = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        target_wh = target_boxes[:, 2:] - target_boxes[:, :2]

        pred_area = pred_wh[:, 0] * pred_wh[:, 1]
        target_area = target_wh[:, 0] * target_wh[:, 1]

        x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)

        x1_union = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_union = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_union = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_union = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)

        rho2 = (pred_cxcy - target_cxcy).pow(2).sum(dim=1)
        c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        c2 = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2)

        pred_w_ratio = pred_wh[:, 0] / (pred_wh[:, 1] + 1e-7)
        target_w_ratio = target_wh[:, 0] / (target_wh[:, 1] + 1e-7)
        v = (4 / (torch.pi ** 2)) * (torch.atan(target_w_ratio) - torch.atan(pred_w_ratio)).pow(2)
        
        alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - (rho2 / c2 + v * alpha)

        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
