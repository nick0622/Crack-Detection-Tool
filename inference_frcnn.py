#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from preprocess import enhance_image

def check_dependencies():
    missing = []
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import PIL
    except ImportError:
        missing.append("pillow")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print("\n💡 To install them, run:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install all requirements with:")
        print("   pip install -r requirements.txt")
        return False
    return True

class FasterRCNNDetector:
    def __init__(self, enhance_images=False, use_tta=False):
        """
        Initialize Faster R-CNN crack detector
        
        Args:
            enhance_images: Whether to apply image enhancement
            use_tta: Whether to use Test Time Augmentation
        """
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = {}
        self.model_loaded = False
        self.enhance_images = enhance_images
        self.use_tta = use_tta
        self.class_confidences = {}  # Per-class confidence thresholds
        
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.target_size = 1024  # 長邊目標尺寸
        self.nms_threshold = 0.2  # NMS IoU 閾值
        
        self.load_model()
    
    def get_model_path(self):
        """Get the Faster R-CNN ONNX model path"""
        model_folder = Path("model")
        model_filename = "frcnn.onnx"  # Faster R-CNN ONNX 模型
        model_path = model_folder / model_filename
        
        if not model_path.exists():
            print(f"⚠️ Model file not found: {model_path}")
            print(f"   Looking for alternative .onnx models...")
            # Fallback to any available onnx model
            onnx_files = [f for f in model_folder.glob("*.onnx") 
                         if "yolo" not in f.name.lower()]
            if onnx_files:
                print(f"   Using: {onnx_files[0].name}")
                return onnx_files[0]
            return None
        
        return model_path
    
    def load_model(self):
        """Load the Faster R-CNN ONNX model"""
        try:
            import onnxruntime as ort
            
            onnx_path = self.get_model_path()
            
            if onnx_path is None:
                print("❌ No Faster R-CNN model file found in ./model/ directory")
                return False
            
            print(f"🤖 Model Type: Faster R-CNN (ONNX)")
            print(f"🔍 Loading model: {onnx_path.name}")
            
            providers = ['CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
                print("🚀 GPU acceleration available")
            elif 'CoreMLExecutionProvider' in available_providers:
                providers.insert(0, 'CoreMLExecutionProvider')
                print("🚀 CoreML acceleration available")
            
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"🎯 Model outputs: {len(self.output_names)} ({', '.join(self.output_names)})")
            
            self.load_classes()
            
            print("✅ Faster R-CNN model loaded successfully")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_classes(self):
        """Load class names - Faster R-CNN 只有 4 類別"""
        self.class_names = {
            "0": "transverse", 
            "1": "longitudinal", 
            "2": "joint", 
            "3": "alligator"
        }
        print(f"📋 Loaded {len(self.class_names)} crack type classes")
    
    def set_class_confidences(self, class_confidences):
        """
        Set per-class confidence thresholds
        
        Args:
            class_confidences: dict mapping class_id (int or str) to confidence threshold (float)
                              Example: {0: 0.3, 1: 0.25, 2: 0.2, 3: 0.35}
        """
        self.class_confidences = {}
        for key, value in class_confidences.items():
            if isinstance(key, str) and not key.isdigit():
                # Convert class name to class id
                for class_id, class_name in self.class_names.items():
                    if class_name.lower() == key.lower():
                        self.class_confidences[int(class_id)] = float(value)
                        break
            else:
                self.class_confidences[int(key)] = float(value)
        
        if self.class_confidences:
            print("🎯 Per-class confidence thresholds:")
            for class_id, conf in self.class_confidences.items():
                class_name = self.class_names.get(str(class_id), f"class_{class_id}")
                print(f"   • {class_name.title()}: {conf:.2f}")
    
    def get_class_color(self, class_id):
        """Get BGR color for each crack type (for OpenCV)"""
        class_colors = {
            0: (255, 0, 0),      # Blue for transverse
            1: (255, 220, 100),  # Light blue for longitudinal
            2: (255, 255, 255),  # White for joint
            3: (240, 250, 100),  # Light cyan for alligator
        }
        return class_colors.get(class_id, (128, 128, 128))   # Default gray
    
    def get_class_color_name(self, class_id):
        """Get color name for display"""
        color_names = {
            0: "blue",              # Transverse
            1: "Blue",              # Longitudinal  
            2: "white",             # Joint
            3: "lightseagreen",     # Alligator
        }
        return color_names.get(class_id, "Gray")
    
    def show_color_legend(self):
        """Display color coding for crack types"""
        print("\n🎨 Crack Type Color Coding:")
        for class_id, class_name in self.class_names.items():
            color_name = self.get_class_color_name(int(class_id))
            print(f"   • {class_name.title()} - {color_name}")
        print()
    
    def preprocess_image(self, image):
        """
        Prepare image for Faster R-CNN ONNX model
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply enhancement if enabled
        image = enhance_image(image, self.enhance_images)
        
        h, w = image.shape[:2]
        
        # 計算縮放比例（保持長寬比，長邊縮放到 target_size）
        max_side = max(h, w)
        scale = self.target_size / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 調整大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to divisible by 32
        pad_h = int(np.ceil(new_h / 32) * 32)
        pad_w = int(np.ceil(new_w / 32) * 32)
        
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # 正規化 (減均值除標準差)
        normalized = rgb.astype(np.float32)
        normalized = (normalized - self.mean) / self.std
        
        # 轉換為 NCHW 格式
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale, (h, w)
    
    def apply_tta_transforms(self, input_image):
        """
        Apply Test Time Augmentation transforms
        Returns list of (transformed_image, transform_info) tuples
        """
        transforms = []
        
        # Original image
        transforms.append((input_image, {'type': 'original'}))
        
        # Horizontal flip
        flipped_h = input_image[:, :, :, ::-1].copy()
        transforms.append((flipped_h, {'type': 'flip_h'}))
        
        # Vertical flip
        flipped_v = input_image[:, :, ::-1, :].copy()
        transforms.append((flipped_v, {'type': 'flip_v'}))
        
        # Both flips (180 degree rotation)
        flipped_both = input_image[:, :, ::-1, ::-1].copy()
        transforms.append((flipped_both, {'type': 'flip_both'}))
        
        return transforms
    
    def reverse_tta_transform(self, detections, transform_info, img_width, img_height):
        """
        Reverse the TTA transform applied to bounding boxes
        """
        transform_type = transform_info['type']
        
        if transform_type == 'original':
            return detections
        
        reversed_detections = []
        for det in detections:
            bbox = det['bbox'].copy()
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            if transform_type == 'flip_h':
                # Reverse horizontal flip
                bbox['x1'] = img_width - x2
                bbox['x2'] = img_width - x1
            
            elif transform_type == 'flip_v':
                # Reverse vertical flip
                bbox['y1'] = img_height - y2
                bbox['y2'] = img_height - y1
            
            elif transform_type == 'flip_both':
                # Reverse both flips
                bbox['x1'] = img_width - x2
                bbox['x2'] = img_width - x1
                bbox['y1'] = img_height - y2
                bbox['y2'] = img_height - y1
            
            reversed_det = det.copy()
            reversed_det['bbox'] = bbox
            reversed_detections.append(reversed_det)
        
        return reversed_detections
    
    def merge_tta_detections(self, all_detections, iou_threshold=0.5):
        """
        Merge detections from multiple TTA transforms using NMS
        """
        if not all_detections:
            return []
        
        # Convert to format for NMS: [x1, y1, x2, y2]
        boxes = []
        scores = []
        labels = []
        
        for det in all_detections:
            bbox = det['bbox']
            boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            scores.append(det['confidence'])
            labels.append(det['class_id'])
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Group by class
        unique_labels = np.unique(labels)
        merged_detections = []
        
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            if len(class_boxes) == 0:
                continue
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                class_boxes.tolist(),
                class_scores.tolist(),
                0.0,  # No score threshold here (already filtered)
                iou_threshold
            )
            
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                if hasattr(indices, 'flatten'):
                    indices = indices.flatten()
                
                for i in indices:
                    class_name = self.class_names.get(str(int(label)), f"class_{int(label)}")
                    
                    detection = {
                        "bbox": {
                            "x1": float(class_boxes[i][0]),
                            "y1": float(class_boxes[i][1]),
                            "x2": float(class_boxes[i][2]),
                            "y2": float(class_boxes[i][3])
                        },
                        "confidence": float(class_scores[i]),
                        "class_id": int(label),
                        "class_name": class_name
                    }
                    merged_detections.append(detection)
        
        return merged_detections
    
    def postprocess_results(self, outputs, scale, orig_shape, conf_threshold=0.15):
        """
        後處理 Faster R-CNN ONNX 模型輸出
        
        Args:
            outputs: 模型輸出 [dets, labels]
                    dets: (batch, num_dets, 5) -> [x1, y1, x2, y2, score]
                    labels: (batch, num_dets) -> [class_id]
            scale: 縮放比例
            orig_shape: 原始圖像形狀 (h, w)
            conf_threshold: 全局置信度閾值
        
        Returns:
            detections: 檢測結果列表
        """
        # outputs[0]: dets (batch, num_dets, 5) -> [x1, y1, x2, y2, score]
        # outputs[1]: labels (batch, num_dets) -> [class_id]
        
        dets = outputs[0][0]  # 取出 batch=0
        labels = outputs[1][0]  # 取出 batch=0
        
        detections = []
        orig_h, orig_w = orig_shape
        
        for det, label in zip(dets, labels):
            x1, y1, x2, y2, score = det
            class_id = int(label)
            
            # 應用 per-class 置信度閾值
            if self.class_confidences:
                class_threshold = self.class_confidences.get(class_id, conf_threshold)
            else:
                class_threshold = conf_threshold
            
            # 過濾低置信度
            if score < class_threshold:
                continue
            
            # 將座標縮放回原始圖像尺寸
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
            
            # 限制在圖像邊界內
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            class_name = self.class_names.get(str(class_id), f"class_{class_id}")
            
            detection = {
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                },
                "confidence": float(score),
                "class_id": class_id,
                "class_name": class_name
            }
            detections.append(detection)
        
        # Apply NMS
        if len(detections) > 0:
            detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections):
        """應用 NMS 去除重疊框"""
        if len(detections) == 0:
            return []
        
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            bbox = det['bbox']
            boxes.append([int(bbox['x1']), int(bbox['y1']), 
                         int(bbox['x2'] - bbox['x1']), int(bbox['y2'] - bbox['y1'])])
            confidences.append(float(det['confidence']))
            class_ids.append(int(det['class_id']))
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            score_threshold=0.0,  # 已經在 postprocess 過濾過
            nms_threshold=self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        filtered_detections = []
        for i in indices.flatten():
            filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def detect_cracks(self, image_path, confidence=0.15, save_results=False):
        """Detect cracks in an image using Faster R-CNN"""
        if not self.model_loaded:
            print("❌ Model not loaded. Cannot perform detection.")
            return []
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Could not load image: {image_path}")
                    return []
                display_path = image_path
            else:
                image = image_path
                display_path = "image"
            
            print(f"\n🔍 Analyzing: {Path(display_path).name}")
            
            orig_shape = image.shape[:2]  # (h, w)
            input_tensor, scale, _ = self.preprocess_image(image)
            
            start_time = time.time()
            
            if self.use_tta:
                # Get TTA transforms
                transforms = self.apply_tta_transforms(input_tensor)
                all_detections = []
                
                # Run inference on each transform
                for idx, (transformed_img, transform_info) in enumerate(transforms):
                    outputs = self.session.run(self.output_names, {self.input_name: transformed_img})
                    detections = self.postprocess_results(outputs, scale, orig_shape, confidence)
                    
                    # Reverse transform on bounding boxes
                    reversed_detections = self.reverse_tta_transform(
                        detections, transform_info, orig_shape[1], orig_shape[0]
                    )
                    all_detections.extend(reversed_detections)
                
                # Merge detections from all transforms
                detections = self.merge_tta_detections(all_detections, iou_threshold=0.5)
            else:
                # Standard inference without TTA
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                detections = self.postprocess_results(outputs, scale, orig_shape, confidence)
            
            detection_time = time.time() - start_time
            
            print(f"⚡ Detection completed in {detection_time:.3f} seconds")
            
            if detections:
                print(f"🚨 Found {len(detections)} crack(s):")
                for i, crack in enumerate(detections, 1):
                    bbox = crack["bbox"]
                    conf_percent = crack["confidence"] * 100
                    crack_type = crack["class_name"]
                    class_id = crack["class_id"]
                    color_name = self.get_class_color_name(class_id)
                    print(f"   {i}. {crack_type.title()} crack - {conf_percent:.1f}% confidence")
                    print(f"       Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
            else:
                print("✅ No cracks detected in this image")
            
            if save_results and isinstance(image_path, str):
                self.save_detection_results(image_path, image, detections)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_detection_results(self, image_path, image, detections):
        """
        Save detection results in LabelMe format and annotated image
        """
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True)
        
        image_name = Path(image_path).name
        
        # Save LabelMe JSON
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": image.shape[0],
            "imageWidth": image.shape[1]
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            
            points = [
                [bbox["x1"], bbox["y1"]],
                [bbox["x2"], bbox["y2"]]
            ]
            
            shape = {
                "label": class_name,
                "points": points,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "description": f"Confidence: {detection['confidence']:.2f}"
            }
            labelme_data["shapes"].append(shape)
        
        json_path = results_folder / f"{Path(image_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
        
        # Draw annotations on image
        result_image = image.copy()
        img_height, img_width = result_image.shape[:2]
        
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            conf = detection["confidence"]
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            
            color = self.get_class_color(class_id)
            
            x1 = int(np.clip(x1, 0, img_width - 1))
            y1 = int(np.clip(y1, 0, img_height - 1))
            x2 = int(np.clip(x2, 0, img_width - 1))
            y2 = int(np.clip(y2, 0, img_height - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Adjust visualization parameters based on image size
            if image.shape[0] <= 2500:
                line_width = 2
                font_scale = 2
                font_thickness = 2
                padding = 20
            else:
                line_width = 16
                font_scale = 6
                font_thickness = 8
                padding = 40
            
            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_width)
            
            # Draw label
            label = f"{class_name.title()} {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            text_x = x1 + padding
            text_y = y1 - padding
            
            if text_y - label_height < 0:
                text_y = y1 + label_height + 30
                if text_y > y2:
                    text_y = y2 + label_height + 30
            
            if text_x < 0:
                text_x = padding
            
            if text_x + label_width > img_width:
                text_x = max(padding, img_width - label_width - padding)
            
            if text_y > img_height:
                text_y = img_height - padding
            
            text_x = int(np.clip(text_x, 0, img_width - 1))
            text_y = int(np.clip(text_y, label_height, img_height - 1))
            
            cv2.putText(result_image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        # Save annotated image
        image_path_result = results_folder / f"{Path(image_path).stem}.jpg"
        cv2.imwrite(str(image_path_result), result_image)
        
        print(f"💾 Results saved:")
        print(f"   • LabelMe data: {json_path}")
        print(f"   • Marked image: {image_path_result}")

def parse_class_confidences(conf_str):
    """
    Parse class-specific confidence string
    Format: "0:0.3,1:0.25,2:0.2,3:0.35" or "transverse:0.3,longitudinal:0.25"
    """
    if not conf_str:
        return None
    
    class_confs = {}
    try:
        pairs = conf_str.split(',')
        for pair in pairs:
            key, value = pair.split(':')
            key = key.strip()
            value = float(value.strip())
            class_confs[key] = value
        return class_confs
    except Exception as e:
        print(f"⚠️ Error parsing class confidences: {e}")
        print(f"   Expected format: '0:0.3,1:0.25' or 'transverse:0.3,longitudinal:0.25'")
        return None

def main():
    print("🔧 Crack Detection Tool - Faster R-CNN")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description="Detect cracks using Faster R-CNN")
    parser.add_argument("input", nargs='?', help="Image file or folder path")
    parser.add_argument("--confidence", "-c", type=float, default=0.15, 
                        help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--class-confidences", type=str, default=None,
                        help="Per-class confidence thresholds. Format: '0:0.15,1:0.15,2:0.15,3:0.15'")
    parser.add_argument("--save", "-s", action="store_true", 
                        help="Save detection results")
    parser.add_argument("--enhance", "-e", action="store_true",
                        help="Apply image enhancement preprocessing")
    parser.add_argument("--model", "-m", type=str, default="faster_rcnn",
                        help="Model type (kept for compatibility)")
    parser.add_argument("--tta", "-t", action="store_true",
                        help="Use Test Time Augmentation (slower but more accurate)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("\n📁 Please provide an image file or folder:")
        print("   Example: python inference_frcnn.py image.jpg")
        print("   Example: python inference_frcnn.py photos/ --enhance --save --tta")
        print("   Example with per-class confidences: python inference_frcnn.py image.jpg --class-confidences '0:0.15,1:0.15,2:0.15,3:0.15'")
        args.input = input("\nEnter path: ").strip().strip('"')
    
    detector = FasterRCNNDetector(enhance_images=args.enhance, use_tta=args.tta)
    
    # Set per-class confidences if provided
    if args.class_confidences:
        class_confs = parse_class_confidences(args.class_confidences)
        if class_confs:
            detector.set_class_confidences(class_confs)
    
    if args.enhance:
        print("🔧 Image enhancement: ENABLED")
    else:
        print("🔧 Image enhancement: DISABLED")
        print("   • Use --enhance to enable preprocessing")
    
    if args.tta:
        print("🔄 Test Time Augmentation: ENABLED")
        print("   • Using 4 transforms (original + flips)")
    else:
        print("🔄 Test Time Augmentation: DISABLED")
        print("   • Use --tta to enable TTA for better accuracy")
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            detector.detect_cracks(str(input_path), args.confidence, args.save)
        else:
            print(f"❌ Unsupported file format: {input_path.suffix}")
    
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {input_path}")
        else:
            print(f"📂 Processing {len(image_files)} images...")
            
            total_cracks = 0
            images_with_cracks = 0
            
            for image_file in sorted(image_files):
                detections = detector.detect_cracks(str(image_file), args.confidence, args.save)
                if detections:
                    total_cracks += len(detections)
                    images_with_cracks += 1
            
            print(f"\n📊 Summary:")
            print(f"   • Images processed: {len(image_files)}")
            print(f"   • Images with cracks: {images_with_cracks}")
            print(f"   • Total cracks found: {total_cracks}")
            if len(image_files) > 0:
                print(f"   • Average cracks per image: {total_cracks/len(image_files):.1f}")
    
    else:
        print(f"❌ Path not found: {input_path}")
    
    print("\n✅ Detection completed!")

if __name__ == "__main__":
    main()