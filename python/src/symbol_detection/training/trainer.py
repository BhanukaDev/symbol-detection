import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

from symbol_detection.training.data import COCODetectionDataset, collate_fn
from symbol_detection.training.losses import CIoULoss


class Trainer:
    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        num_classes: int = 7,
        batch_size: int = 4,
        learning_rate: float = 0.005,
        num_epochs: int = 50,
        device: Optional[str] = None,
        use_ciou_loss: bool = True,
        use_pretrained: bool = False,
        eval_every_n: int = 10,
        save_every_n: int = 20,
        enable_ap_eval: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_ciou_loss = use_ciou_loss
        self.use_pretrained = use_pretrained
        self.eval_every_n = eval_every_n
        self.save_every_n = save_every_n
        self.enable_ap_eval = enable_ap_eval

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.model = self._build_model()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0005,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        if self.use_ciou_loss:
            self.ciou_loss = CIoULoss(reduction='mean')
            self._setup_ciou_hooks()

        self.train_losses = []
        self.val_losses = []
        self.ap_history = []
        self.start_epoch = 0

    def _build_model(self):
        if self.use_pretrained:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            # Replace detection head for our number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
            print("Model: FasterRCNN ResNet50+FPN (COCO pretrained, head replaced)")
        else:
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                num_classes=self.num_classes + 1,
                trainable_backbone_layers=3,
            )
            print("Model: FasterRCNN ResNet50+FPN (random init)")
        return model.to(self.device)

    def _setup_ciou_hooks(self):
        """Capture proposals, labels, regression targets and box regression for CIoU loss."""
        self._ciou_capture = {}
        capture = self._ciou_capture

        # Wrap select_training_samples to capture sampled proposals and targets
        original_select = self.model.roi_heads.select_training_samples

        def wrapped_select(proposals, targets):
            result = original_select(proposals, targets)
            # result = (proposals, matched_idxs, labels, regression_targets)
            capture['proposals'] = result[0]
            capture['labels'] = result[2]
            capture['regression_targets'] = result[3]
            return result

        self.model.roi_heads.select_training_samples = wrapped_select

        # Hook on box_predictor to capture box_regression output
        def box_pred_hook(module, input, output):
            capture['box_regression'] = output[1]

        self.model.roi_heads.box_predictor.register_forward_hook(box_pred_hook)

    def _compute_ciou_loss(self, loss_dict: dict) -> dict:
        """Replace loss_box_reg with CIoU loss computed on decoded absolute boxes."""
        try:
            c = self._ciou_capture
            if not c.get('proposals') or c.get('box_regression') is None:
                return loss_dict

            proposals = torch.cat(c['proposals'])           # (N, 4)
            labels    = torch.cat(c['labels'])              # (N,)
            reg_tgts  = torch.cat(c['regression_targets'])  # (N, 4)
            box_reg   = c['box_regression']                 # (N, num_classes*4)

            pos_inds = torch.where(labels > 0)[0]
            if pos_inds.numel() == 0:
                return loss_dict

            labels_pos   = labels[pos_inds]
            box_reg_r    = box_reg.reshape(box_reg.shape[0], -1, 4)
            pred_deltas  = box_reg_r[pos_inds, labels_pos]   # (P, 4)
            target_deltas = reg_tgts[pos_inds]                # (P, 4)
            proposals_pos = proposals[pos_inds]               # (P, 4)

            # Decode encoded deltas → absolute boxes
            box_coder = self.model.roi_heads.box_coder
            pred_boxes   = box_coder.decode(pred_deltas,   [proposals_pos]).reshape(-1, 4)
            target_boxes = box_coder.decode(target_deltas, [proposals_pos]).reshape(-1, 4)

            loss_dict['loss_box_reg'] = self.ciou_loss(pred_boxes, target_boxes)
        except Exception as e:
            pass  # Keep default loss_box_reg if anything goes wrong

        return loss_dict

    def load_coco_annotations(self):
        coco_json = self.dataset_dir / 'annotations.json'
        if not coco_json.exists():
            raise FileNotFoundError(f"Annotations file not found: {coco_json}")

        with open(coco_json, 'r') as f:
            coco_data = json.load(f)

        image_ids = [img['id'] for img in coco_data['images']]

        # Fixed seed so train/val split is reproducible across runs
        random.seed(42)
        random.shuffle(image_ids)

        train_split = int(0.8 * len(image_ids))
        train_image_ids = set(image_ids[:train_split])
        val_image_ids   = set(image_ids[train_split:])

        train_coco = {
            'images':      [img for img in coco_data['images']       if img['id']      in train_image_ids],
            'annotations': [ann for ann in coco_data['annotations']   if ann['image_id'] in train_image_ids],
            'categories':  coco_data['categories'],
        }
        val_coco = {
            'images':      [img for img in coco_data['images']       if img['id']      in val_image_ids],
            'annotations': [ann for ann in coco_data['annotations']   if ann['image_id'] in val_image_ids],
            'categories':  coco_data['categories'],
        }

        train_json = self.dataset_dir / 'train_annotations.json'
        val_json   = self.dataset_dir / 'val_annotations.json'

        with open(train_json, 'w') as f:
            json.dump(train_coco, f)
        with open(val_json, 'w') as f:
            json.dump(val_coco, f)

        return str(train_json), str(val_json)

    def get_dataloaders(self):
        train_json, val_json = self.load_coco_annotations()

        train_dataset = COCODetectionDataset(
            coco_json_path=train_json,
            images_dir=self.dataset_dir / 'images',
            augment=True,
        )
        val_dataset = COCODetectionDataset(
            coco_json_path=val_json,
            images_dir=self.dataset_dir / 'images',
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images  = [img.to(self.device) for img in images]
            targets = [{'boxes': t['boxes'].to(self.device), 'labels': t['labels'].to(self.device)} for t in targets]

            self.optimizer.zero_grad()

            loss_dict = self.model(images, targets)

            if self.use_ciou_loss:
                loss_dict = self._compute_ciou_loss(loss_dict)

            losses = torch.stack(list(loss_dict.values())).sum()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        self.model.train()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(self.device) for img in images]
                targets = [{'boxes': t['boxes'].to(self.device), 'labels': t['labels'].to(self.device)} for t in targets]

                loss_dict = self.model(images, targets)
                losses = torch.stack(list(loss_dict.values())).sum()
                total_loss += losses.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def evaluate_ap(self, val_loader):
        self.model.eval()
        metric = MeanAveragePrecision(iou_type='bbox')

        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(self.device) for img in images]
                targets = [{'boxes': t['boxes'].to(self.device), 'labels': t['labels'].to(self.device)} for t in targets]

                preds = self.model(images)
                metric.update(
                    [{'boxes': p['boxes'].cpu(), 'scores': p['scores'].cpu(), 'labels': p['labels'].cpu()} for p in preds],
                    [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets],
                )

        results = metric.compute()
        return {
            'mAP':  results['map'].item(),
            'AP50': results['map_50'].item(),
            'AP75': results['map_75'].item(),
            'mAR':  results['mar_100'].item(),
        }

    def train(self):
        train_loader, val_loader = self.get_dataloaders()

        if self.start_epoch >= self.num_epochs:
            print(f"Already at epoch {self.start_epoch}, target is {self.num_epochs}. Increase TARGET_EPOCHS.")
            return

        print(f"Training epochs {self.start_epoch + 1} → {self.num_epochs}")
        print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")
        print(f"CIoU loss: {self.use_ciou_loss} | Pretrained: {self.use_pretrained}\n")

        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                print(f"Epoch {epoch+1}/{self.num_epochs} ...", end='\r', flush=True)

                train_loss = self.train_epoch(train_loader)
                val_loss   = self.validate(val_loader)
                self.scheduler.step()

                print(f"Epoch {epoch+1}/{self.num_epochs} — Train: {train_loss:.4f}, Val: {val_loss:.4f}", end='')

                if self.enable_ap_eval and (epoch + 1) % self.eval_every_n == 0:
                    print("\n  AP eval...", end=' ')
                    ap = self.evaluate_ap(val_loader)
                    self.ap_history.append({'epoch': epoch + 1, **ap})
                    print(f"mAP: {ap['mAP']:.3f}  AP50: {ap['AP50']:.3f}  AP75: {ap['AP75']:.3f}  mAR: {ap['mAR']:.3f}")
                else:
                    print()

                if (epoch + 1) % self.save_every_n == 0:
                    self.save_checkpoint(epoch + 1)

        except KeyboardInterrupt:
            print(f"\nInterrupted — saving checkpoint at epoch {epoch + 1}...")
            self.save_checkpoint(epoch + 1)
            self.save_metrics()
            return

        if self.enable_ap_eval:
            print("\nFinal AP eval...")
            ap = self.evaluate_ap(val_loader)
            self.ap_history.append({'epoch': 'final', **ap})
            print(f"mAP: {ap['mAP']:.3f}  AP50: {ap['AP50']:.3f}  AP75: {ap['AP75']:.3f}  mAR: {ap['mAR']:.3f}")

        self.save_checkpoint('final')
        self.save_metrics()

    def save_checkpoint(self, epoch):
        path = self.output_dir / f'model_epoch_{epoch}.pth'
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'use_pretrained':       self.use_pretrained,
            'train_loss':           self.train_losses[-1] if self.train_losses else None,
            'val_loss':             self.val_losses[-1]   if self.val_losses   else None,
        }, path)
        print(f"  Saved: {path.name}")

    def save_metrics(self):
        path = self.output_dir / 'metrics.json'
        with open(path, 'w') as f:
            json.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses, 'ap_history': self.ap_history}, f, indent=2)
        print(f"  Saved: {path.name}")

    def load_checkpoint(self, checkpoint_path: str | Path, resume_training: bool = False):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Old checkpoint — fast-forward scheduler to correct position on cosine curve
            saved_epoch = checkpoint.get('epoch', 0)
            if isinstance(saved_epoch, int):
                for _ in range(saved_epoch):
                    self.scheduler.step()

        if resume_training:
            self.start_epoch = checkpoint.get('epoch', 0)
            if isinstance(self.start_epoch, str):
                self.start_epoch = 0

            metrics_path = self.output_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                self.train_losses = m.get('train_losses', [])
                self.val_losses   = m.get('val_losses',   [])
                self.ap_history   = m.get('ap_history',   [])

        print(f"Loaded: {Path(checkpoint_path).name}  (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint
