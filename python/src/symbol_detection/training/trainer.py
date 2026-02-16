import json
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
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
        eval_every_n: int = 10,
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
        self.eval_every_n = eval_every_n
        self.enable_ap_eval = enable_ap_eval
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Enable cuDNN autotune for faster convolutions after warmup
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
        
        self.train_losses = []
        self.val_losses = []
        self.ap_history = []
        self.start_epoch = 0

    def _build_model(self):
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=self.num_classes + 1,
            trainable_backbone_layers=3,
        )
        model = model.to(self.device)
        return model

    def load_coco_annotations(self):
        coco_json = self.dataset_dir / 'annotations.json'
        if not coco_json.exists():
            raise FileNotFoundError(f"Annotations file not found: {coco_json}")
        
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        num_images = len(coco_data.get('images', []))
        train_split = int(0.8 * num_images)
        
        image_ids = [img['id'] for img in coco_data['images']]
        
        train_image_ids = set(image_ids[:train_split])
        val_image_ids = set(image_ids[train_split:])
        
        train_coco = {
            'images': [img for img in coco_data['images'] if img['id'] in train_image_ids],
            'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids],
            'categories': coco_data['categories'],
        }
        
        val_coco = {
            'images': [img for img in coco_data['images'] if img['id'] in val_image_ids],
            'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids],
            'categories': coco_data['categories'],
        }
        
        train_json = self.dataset_dir / 'train_annotations.json'
        val_json = self.dataset_dir / 'val_annotations.json'
        
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
        )
        val_dataset = COCODetectionDataset(
            coco_json_path=val_json,
            images_dir=self.dataset_dir / 'images',
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for images, targets in train_loader:
            images = [img.to(self.device) for img in images]
            targets = [{
                'boxes': t['boxes'].to(self.device),
                'labels': t['labels'].to(self.device),
            } for t in targets]
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model(images, targets)
            losses = sum(loss_dict.values(), torch.tensor(0.0, requires_grad=True, device=self.device))
            
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
        
        avg_loss = total_loss / len(train_loader)  # type: ignore
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        self.model.train()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{
                    'boxes': t['boxes'].to(self.device),
                    'labels': t['labels'].to(self.device),
                } for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss_dict.values(), torch.tensor(0.0, device=self.device))
                
                total_loss += losses.item()
        
        avg_loss = total_loss / len(val_loader)  # type: ignore
        self.val_losses.append(avg_loss)
        return avg_loss

    def evaluate_ap(self, val_loader):
        """Compute COCO-style Average Precision metrics (mAP, AP50, AP75)."""
        self.model.eval()
        metric = MeanAveragePrecision(iou_type='bbox')
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{
                    'boxes': t['boxes'].to(self.device),
                    'labels': t['labels'].to(self.device),
                } for t in targets]
                
                # Get predictions
                predictions = self.model(images)
                
                # Format for torchmetrics (needs CPU tensors)
                preds = [{
                    'boxes': p['boxes'].cpu(),
                    'scores': p['scores'].cpu(),
                    'labels': p['labels'].cpu(),
                } for p in predictions]
                
                gts = [{
                    'boxes': t['boxes'].cpu(),
                    'labels': t['labels'].cpu(),
                } for t in targets]
                
                metric.update(preds, gts)
        
        # Compute metrics
        results = metric.compute()
        
        return {
            'mAP': results['map'].item(),  # Mean AP across IoU 0.5:0.95
            'AP50': results['map_50'].item(),  # AP at IoU=0.5
            'AP75': results['map_75'].item(),  # AP at IoU=0.75
            'mAR': results['mar_100'].item(),  # Mean Average Recall
        }

    def train(self):
        train_loader, val_loader = self.get_dataloaders()
        
        train_size = len(train_loader.dataset)  # type: ignore
        val_size = len(val_loader.dataset)  # type: ignore
        
        if self.start_epoch >= self.num_epochs:
            print(f"Already trained for {self.start_epoch} epochs. Target is {self.num_epochs}.")
            print("Increase num_epochs to continue training.")
            return

        print(f"Training for {self.num_epochs} epochs (starting from epoch {self.start_epoch + 1})...")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        if self.enable_ap_eval:
            print(f"AP evaluation every {self.eval_every_n} epochs\n")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs} [Training...]", end='\r', flush=True)
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            # Clear line and print result
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end='')
            
            # Periodic AP evaluation
            if self.enable_ap_eval and (epoch + 1) % self.eval_every_n == 0:
                print("\n  Evaluating AP metrics...", end=' ')
                ap_results = self.evaluate_ap(val_loader)
                self.ap_history.append({
                    'epoch': epoch + 1,
                    **ap_results
                })
                print(f"mAP: {ap_results['mAP']:.3f}, AP50: {ap_results['AP50']:.3f}, "
                      f"AP75: {ap_results['AP75']:.3f}, mAR: {ap_results['mAR']:.3f}")
            else:
                print()  # Newline
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final evaluation logic remains same...
        
        # Final evaluation
        if self.enable_ap_eval:
            print("\nFinal evaluation...")
            final_ap = self.evaluate_ap(val_loader)
            self.ap_history.append({
                'epoch': 'final',
                **final_ap
            })
            print(f"Final - mAP: {final_ap['mAP']:.3f}, AP50: {final_ap['AP50']:.3f}, "
                  f"AP75: {final_ap['AP75']:.3f}, mAR: {final_ap['mAR']:.3f}")
        
        self.save_checkpoint('final')
        self.save_metrics()

    def save_checkpoint(self, epoch):
        checkpoint_path = self.output_dir / f'model_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_metrics(self):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ap_history': self.ap_history,
        }
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    def load_checkpoint(self, checkpoint_path: str | Path, resume_training: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if resume_training:
            self.start_epoch = checkpoint.get('epoch', 0)
            if isinstance(self.start_epoch, str): # Handle 'final' case
                 self.start_epoch = 0
            
            # Try to load metric history if available
            metrics_path = self.output_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.train_losses = metrics.get('train_losses', [])
                    self.val_losses = metrics.get('val_losses', [])
                    self.ap_history = metrics.get('ap_history', [])
            
        print(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
