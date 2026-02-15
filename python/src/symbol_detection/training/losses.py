import torch
import torch.nn as nn

class CIoULoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps

        pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=eps)
        target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=eps)

        pred_cxcy = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_cxcy = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2

        pred_area = pred_wh[:, 0] * pred_wh[:, 1]
        target_area = target_wh[:, 0] * target_wh[:, 1]

        x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)

        union_area = (pred_area + target_area - inter_area).clamp(min=eps)
        iou = inter_area / union_area

        c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        c2 = ((c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2)).clamp(min=eps)
        rho2 = (pred_cxcy - target_cxcy).pow(2).sum(dim=1)

        pred_ratio = pred_wh[:, 0] / pred_wh[:, 1]
        target_ratio = target_wh[:, 0] / target_wh[:, 1]
        v = (4.0 / (torch.pi ** 2)) * (torch.atan(target_ratio) - torch.atan(pred_ratio)).pow(2)

        alpha = v / (1 - iou.detach() + v + eps)
        ciou = iou - (rho2 / c2 + alpha * v)

        loss = 1 - ciou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
