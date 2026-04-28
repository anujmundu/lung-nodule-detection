# train_faster_rcnn.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
from collections import defaultdict
import time
import copy
import math
import datetime          # <-- ADDED for timedelta
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # Background + nodule
BATCH_SIZE = 4  # Adjust based on your GPU memory (start with 4)
NUM_EPOCHS = 25  # Number of epochs for fine-tuning
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_LR_SIZE = 8  # StepLR decay step
GAMMA_LR = 0.1  # StepLR decay factor

# Paths to your patch dataset
TRAIN_IMG_DIR = 'data/processed_patches/images'
TRAIN_ANNO_PATH = 'data/processed_patches/train_coco.json'
VAL_IMG_DIR = 'data/processed_patches/images'
VAL_ANNO_PATH = 'data/processed_patches/val_coco.json'
TEST_IMG_DIR = 'data/processed_patches/images'
TEST_ANNO_PATH = 'data/processed_patches/test_coco.json'

# --- COCO Dataset Loader ---
class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] + 1)  # COCO categories are 1-indexed, our background is 0

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([image_id])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros((len(annotations),), dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target

# --- Data Transforms ---
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return T.ToTensor()(image), target

# --- Model Initialization ---
def get_model(num_classes):
    # Load a pre-trained model for classification training
    backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
    backbone.out_channels = 1280
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# --- Training Function ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

# --- Evaluation Function ---
@torch.inference_mode()
def evaluate(model, data_loader, device, coco_gt=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    
    # Collect predictions and ground truths for COCO evaluation
    all_predictions = []
    all_targets = []
    
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time
        evaluator_time = time.time()
        
        # Store predictions and ground truths
        all_predictions.extend(outputs)
        all_targets.extend(targets)
        
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    
    # If a COCO ground truth object is provided, compute mAP
    if coco_gt is not None:
        from pycocotools.cocoeval import COCOeval
        
        # Convert predictions to COCO format
        coco_dt = []
        for pred, target in zip(all_predictions, all_targets):
            img_id = target['image_id'].item()
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                # Convert xyxy to xywh
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                coco_dt.append({
                    'image_id': img_id,
                    'bbox': [x1, y1, w, h],
                    'score': float(score),
                    'category_id': int(label) - 1  # subtract 1 to match original class ID (0 for nodule)
                })
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(coco_dt), iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        print("COCO Evaluation Results:")
        coco_eval.summarize()
        mAP_50 = coco_eval.stats[1]  # mAP@0.5
        mAP_50_95 = coco_eval.stats[0]  # mAP@0.5:0.95
        print(f"mAP@0.5: {mAP_50:.4f}")
        print(f"mAP@0.5:0.95: {mAP_50_95:.4f}")
        return mAP_50, mAP_50_95
    
    return metric_logger

# --- Helper classes for logging (from torchvision references) ---
class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        self.total += value * n
        self.count += n

    def synchronize_between_processes(self):
        return

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattribute__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')

# --- Main Training Loop ---
if __name__ == '__main__':
    # Create datasets and dataloaders
    dataset_train = CocoDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANNO_PATH, transforms=Compose([ToTensor()]))
    dataset_val = CocoDetectionDataset(VAL_IMG_DIR, VAL_ANNO_PATH, transforms=Compose([ToTensor()]))
    dataset_test = CocoDetectionDataset(TEST_IMG_DIR, TEST_ANNO_PATH, transforms=Compose([ToTensor()]))
    
    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_SIZE, gamma=GAMMA_LR)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, optimizer, data_loader_train, DEVICE, epoch, print_freq=50)
        lr_scheduler.step()
        # Evaluate on validation set every epoch (without mAP computation, just logging)
        evaluate(model, data_loader_val, DEVICE)
    
    # Final evaluation on test set with mAP
    print("Evaluating on test set...")
    from pycocotools.coco import COCO
    coco_gt = COCO(TEST_ANNO_PATH)  # load ground truth annotations
    mAP50, mAP50_95 = evaluate(model, data_loader_test, DEVICE, coco_gt=coco_gt)
    print(f"Final mAP@0.5: {mAP50:.4f}")
    torch.save(model.state_dict(), 'faster_rcnn_patch_model.pth')
    print("Model saved as faster_rcnn_patch_model.pth")