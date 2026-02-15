import argparse
from pathlib import Path

from symbol_detection.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train symbol detection model on COCO dataset"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=str(Path(__file__).parent.parent.parent.parent / 'dataset'),
        help='Path to dataset directory with images and annotations.json',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).parent.parent.parent.parent / 'checkpoints'),
        help='Path to save trained models and metrics',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.005,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=7,
        help='Number of object classes (excluding background)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (cuda/cpu). Auto-detects if not specified',
    )
    parser.add_argument(
        '--no-ciou-loss',
        action='store_true',
        help='Disable CIoU loss for bounding box regression',
    )

    args = parser.parse_args()

    trainer = Trainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        use_ciou_loss=not args.no_ciou_loss,
    )

    trainer.train()


if __name__ == '__main__':
    main()
