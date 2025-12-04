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
from tqdm import tqdm

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

class CrackDetector:
    def __init__(self, model_type="yolov8_single", enhance_images=False, use_tta=False):
        """
        Initialize crack detector
        
        Args:
            model_type: Type of model to use
                - "yolov8_single": YOLOv8 Single Class
                - "yolov8_4classes": YOLOv8 4 Classes
            enhance_images: Whether to apply image enhancement
            use_tta: Whether to use Test Time Augmentation (multi-scale)
        """
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.class_names = {}
        self.model_loaded = False
        self.enhance_images = enhance_images
        self.model_type = model_type
        self.use_tta = use_tta
        self.class_confidences = {}
        self.load_model()
    
    def get_model_path(self):
        """Get the correct model path based on model_type"""
        model_folder = Path("model")
        
        model_mapping = {
            "yolov8_single": "yolov8_single.onnx",
            "yolov8_4classes": "yolov8_multi.onnx",  # Dynamic size model
        }
        
        model_filename = model_mapping.get(self.model_type)
        
        if model_filename is None:
            print(f"⚠️ Unknown model type: {self.model_type}")
            print(f"   Available types: {', '.join(model_mapping.keys())}")
            onnx_files = list(model_folder.glob("*.onnx"))
            if onnx_files:
                return onnx_files[0]
            return None
        
        model_path = model_folder / model_filename
        
        if not model_path.exists():
            print(f"⚠️ Model file not found: {model_path}")
            print(f"   Looking for alternative models...")
            onnx_files = list(model_folder.glob("*.onnx"))
            if onnx_files:
                print(f"   Using: {onnx_files[0].name}")
                return onnx_files[0]
            return None
        
        return model_path
    
    def load_model(self):
        """Load the crack detection model"""
        try:
            import onnxruntime as ort
            
            onnx_path = self.get_model_path()
            
            if onnx_path is None:
                print("❌ No model file found in ./model/ directory")
                return False
            
            print(f"🤖 Model Type: {self.model_type}")
            # print(f"🔍 Loading model: {onnx_path.name}")
            
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
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # print(f"📏 Model input shape: {self.input_shape}")
            # print(f"🎯 Model outputs: {len(self.output_names)}")
            
            self.load_classes()
            
            # print("✅ Crack detection model loaded successfully")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_classes(self):
        """Load class names from classes.json or use defaults based on model type"""
        classes_path = Path("model/classes.json")
        
        if self.model_type == "yolov8_single":
            self.class_names = {"0": "crack"}
            print("📋 Using single class: crack")
        else:
            if classes_path.exists():
                try:
                    with open(classes_path, 'r') as f:
                        self.class_names = json.load(f)
                    # print(f"📋 Loaded {len(self.class_names)} class names from classes.json")
                except Exception as e:
                    print(f"⚠️ Could not load classes.json: {e}")
                    self.class_names = {
                        "0": "transverse", 
                        "1": "longitudinal", 
                        "2": "joint", 
                        "3": "alligator"
                    }
                    print("📋 Using default 4 crack type classes")
            else:
                self.class_names = {
                    "0": "transverse", 
                    "1": "longitudinal", 
                    "2": "joint", 
                    "3": "alligator"
                }
                print("📋 Using default 4 crack type classes")
    
    def set_class_confidences(self, class_confidences):
        """Set per-class confidence thresholds"""
        self.class_confidences = {}
        for key, value in class_confidences.items():
            if isinstance(key, str) and not key.isdigit():
                for class_id, class_name in self.class_names.items():
                    if class_name.lower() == key.lower():
                        self.class_confidences[int(class_id)] = float(value)
                        break
            else:
                self.class_confidences[int(key)] = float(value)
    def get_class_color(self, class_id):
        """Get BGR color for each crack type"""
        class_colors = {
            0: (255, 0, 0),
            1: (255, 220, 100),
            2: (255, 255, 255),
            3: (240, 250, 100),
        }
        return class_colors.get(class_id, (128, 128, 128))
    
    def get_class_color_name(self, class_id):
        """Get color name for display"""
        color_names = {
            0: "blue",
            1: "Blue",
            2: "white",
            3: "lightseagreen",
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
        Prepare image for YOLOv8 ONNX model with dynamic size
        Matches Ultralytics preprocessing
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Optional image enhancement
        image = enhance_image(image, self.enhance_images)
        
        original_height, original_width = image.shape[:2]
        
        # Dynamic sizing with stride alignment
        imgsz = 1024
        stride = 32
        scale = min(imgsz / original_height, imgsz / original_width)
        
        # Calculate aligned dimensions
        new_width = int(np.ceil(original_width * scale / stride) * stride)
        new_height = int(np.ceil(original_height * scale / stride) * stride)
        
        # Resize
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        input_image = resized_rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        input_image = np.transpose(input_image, (2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        # No padding offsets for dynamic sizing
        return input_image, scale, 0, 0
    
    def postprocess_results(self, outputs, scale, x_offset, y_offset, conf_threshold=0.2):
        detections = []
        
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        if output.shape[0] < output.shape[1]:
            output = output.T  # Now [num_detections, 4+num_classes]
        
        boxes = output[:, :4]  # xywh (model input size)
        scores = output[:, 4:]  # class scores
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Apply per-class confidence thresholds
        # (此處會保留所有通過各自閾值的框，讓 NMS 處理最終的重疊)
        if self.class_confidences:
            valid_detections = np.zeros(len(confidences), dtype=bool)
            for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                class_threshold = self.class_confidences.get(int(class_id), conf_threshold)
                valid_detections[i] = conf > class_threshold
        else:
            valid_detections = confidences > conf_threshold
        
        if np.any(valid_detections):
            boxes = boxes[valid_detections]
            confidences = confidences[valid_detections]
            class_ids = class_ids[valid_detections]
            
            # 1. Convert model output (xywh) to model input XYXY
            x_centers, y_centers, widths, heights = boxes.T
            x1_model = x_centers - widths / 2
            y1_model = y_centers - heights / 2
            x2_model = x_centers + widths / 2
            y2_model = y_centers + heights / 2

            # 2. Transform to Original Image XYXY space
            x1_orig = (x1_model - x_offset) / scale
            y1_orig = (y1_model - y_offset) / scale
            x2_orig = (x2_model - x_offset) / scale
            y2_orig = (y2_model - y_offset) / scale
            
            # Ensure coordinates are valid
            x1_orig = np.maximum(0, x1_orig)
            y1_orig = np.maximum(0, y1_orig)
            
            # 3. Class-Agnostic NMS (使用您驗證過可行的 XYXY 格式)
            boxes_for_nms = np.column_stack([x1_orig, y1_orig, x2_orig, y2_orig]).astype(np.float32)
            confidences_f32 = confidences.astype(np.float32)
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(),
                confidences_f32.tolist(),
                0.0,
                0.2
            )
            
            if len(indices) > 0:
                # Handle NMS return types
                if isinstance(indices, tuple):
                    indices = indices[0]
                if hasattr(indices, 'flatten'):
                    indices = indices.flatten()
                
                # Process final NMS results
                for i in indices:
                    class_id = int(class_ids[i])
                    class_name = self.class_names.get(str(class_id), f"class_{class_id}")
                    
                    detection = {
                        "bbox": {
                            "x1": float(x1_orig[i]),
                            "y1": float(y1_orig[i]),
                            "x2": float(x2_orig[i]),
                            "y2": float(y2_orig[i])
                        },
                        "confidence": float(confidences[i]),
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    detections.append(detection)
                    
        return detections
    
    def _calc_iou_boxes(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _merge_multiscale_detections(self, all_detections, img_width, img_height, iou_threshold=0.6):
        """
        Merge detections from multiple scales using Weighted Boxes Fusion
        """
        if not all_detections:
            return []
        
        # Group by class
        class_groups = {}
        for det in all_detections:
            cls_id = det['class_id']
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)
        
        final_detections = []
        
        for cls_id, group in class_groups.items():
            if len(group) == 0:
                continue
            
            # Extract boxes and scores
            boxes = np.array([
                [d['bbox']['x1'], d['bbox']['y1'], d['bbox']['x2'], d['bbox']['y2']] 
                for d in group
            ])
            scores = np.array([d['confidence'] for d in group])
            
            # Sort by confidence
            order = np.argsort(scores)[::-1]
            boxes = boxes[order]
            scores = scores[order]
            
            # WBF-style merging
            keep = []
            processed = np.zeros(len(boxes), dtype=bool)
            
            for i in range(len(boxes)):
                if processed[i]:
                    continue
                
                current_box = boxes[i]
                current_score = scores[i]
                
                # Collect overlapping boxes
                overlapping_boxes = [current_box]
                overlapping_scores = [current_score]
                processed[i] = True
                
                for j in range(i + 1, len(boxes)):
                    if processed[j]:
                        continue
                    
                    iou = self._calc_iou_boxes(current_box, boxes[j])
                    
                    if iou > iou_threshold:
                        overlapping_boxes.append(boxes[j])
                        overlapping_scores.append(scores[j])
                        processed[j] = True
                
                # Fuse overlapping boxes
                if len(overlapping_boxes) > 1:
                    overlapping_boxes = np.array(overlapping_boxes)
                    overlapping_scores = np.array(overlapping_scores)
                    
                    # Weighted average by confidence
                    weights = overlapping_scores / overlapping_scores.sum()
                    fused_box = np.average(overlapping_boxes, axis=0, weights=weights)
                    fused_score = overlapping_scores.max()
                    
                    keep.append({'box': fused_box, 'score': fused_score})
                else:
                    keep.append({'box': current_box, 'score': current_score})
            
            # Add to final results
            class_name = self.class_names.get(str(cls_id), f"class_{cls_id}")
            for item in keep:
                final_detections.append({
                    "bbox": {
                        "x1": max(0, float(item['box'][0])),
                        "y1": max(0, float(item['box'][1])),
                        "x2": min(img_width, float(item['box'][2])),
                        "y2": min(img_height, float(item['box'][3]))
                    },
                    "confidence": float(item['score']),
                    "class_id": int(cls_id),
                    "class_name": class_name
                })
        
        return final_detections
    
    def detect_cracks(self, image_path, confidence=0.25, save_results=False):
        """Detect cracks in an image"""
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
            
            # print(f"\n🔍 Analyzing: {Path(display_path).name}")
            
            img_height, img_width = image.shape[:2]
            
            start_time = time.time()
            
            if self.use_tta:
                # Multi-scale TTA (matching Ultralytics: 1.0, 0.83, 0.67)
                scales = [1.0, 0.83, 0.67]
                # print(f"🔄 Using Multi-Scale TTA ({len(scales)} scales: {scales})...")
                
                all_detections = []
                
                for scale_factor in scales:
                    # Scale image
                    if scale_factor != 1.0:
                        scaled_h = int(img_height * scale_factor)
                        scaled_w = int(img_width * scale_factor)
                        scaled_image = cv2.resize(
                            image, 
                            (scaled_w, scaled_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        scaled_image = image
                    
                    # Preprocess
                    input_tensor, scale, x_offset, y_offset = self.preprocess_image(scaled_image)
                    
                    # Inference
                    outputs = self.session.run(
                        self.output_names, 
                        {self.input_name: input_tensor}
                    )
                    
                    # Postprocess
                    detections = self.postprocess_results(
                        outputs, scale, x_offset, y_offset, confidence
                    )
                    
                    # Transform coordinates back to original image size
                    if scale_factor != 1.0:
                        for det in detections:
                            det['bbox']['x1'] /= scale_factor
                            det['bbox']['y1'] /= scale_factor
                            det['bbox']['x2'] /= scale_factor
                            det['bbox']['y2'] /= scale_factor
                    
                    # print(f"   Scale {scale_factor:.2f}x: {len(detections)} detections")
                    all_detections.extend(detections)
                
                # print(f"   Total before merging: {len(all_detections)} detections")
                
                # Merge multi-scale detections
                detections = self._merge_multiscale_detections(
                    all_detections,
                    img_width,
                    img_height,
                    iou_threshold=0.5
                )
                
                # print(f"   After merging: {len(detections)} final detections")
            else:
                # Standard inference
                input_image, scale, x_offset, y_offset = self.preprocess_image(image)
                outputs = self.session.run(
                    self.output_names, 
                    {self.input_name: input_image}
                )
                detections = self.postprocess_results(
                    outputs, scale, x_offset, y_offset, confidence
                )
            
            detection_time = time.time() - start_time
            
            # print(f"⚡ Detection completed in {detection_time:.3f} seconds")
            
            if detections:
                # print(f"🚨 Found {len(detections)} crack(s):")
                for i, crack in enumerate(detections, 1):
                    bbox = crack["bbox"]
                    conf_percent = crack["confidence"] * 100
                    crack_type = crack["class_name"]
                    # print(f"   {i}. {crack_type.title()} crack - {conf_percent:.1f}% confidence")
                    # print(f"       Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
            # else:
                # print("✅ No cracks detected in this image")
            
            if save_results and isinstance(image_path, str):
                self.save_detection_results(image_path, image, detections)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_detection_results(self, image_path, image, detections):
        """Save detection results in LabelMe format"""
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True)
        
        image_name = Path(image_path).name
        
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

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_width)
            
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
        
        image_path_result = results_folder / f"{Path(image_path).stem}.jpg"
        cv2.imwrite(str(image_path_result), result_image)
        
        # print(f"💾 Results saved")
        # print(f"   • LabelMe data: {json_path}")
        # print(f"   • Marked image: {image_path_result}")

def parse_class_confidences(conf_str):
    """Parse class-specific confidence string"""
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
    start_time = time.time()

    print("🔧 Crack Detection Tool")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description="Detect cracks in images using AI")
    parser.add_argument("input", nargs='?', help="Image file or folder path")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, 
                        help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--class-confidences", type=str, default=None,
                        help="Per-class confidence thresholds. Format: '0:0.3,1:0.25,2:0.2,3:0.35'")
    parser.add_argument("--save", "-s", action="store_true", 
                        help="Save detection results")
    parser.add_argument("--enhance", "-e", action="store_true",
                        help="Apply image enhancement preprocessing")
    parser.add_argument("--model", "-m", type=str, default="yolov8_single",
                        choices=["yolov8_single", "yolov8_4classes"],
                        help="Model type to use (default: yolov8_single)")
    parser.add_argument("--tta", "-t", action="store_true",
                        help="Use Multi-Scale Test Time Augmentation (slower but more accurate)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("\n📁 Please provide an image file or folder:")
        print("   Example: python inference.py image.jpg")
        print("   Example: python inference.py photos/ --enhance --save --model yolov8_4classes --tta")
        print("   Example with per-class confidences: python inference.py image.jpg --class-confidences '0:0.3,1:0.25,2:0.2,3:0.35'")
        args.input = input("\nEnter path: ").strip().strip('"')
    
    detector = CrackDetector(model_type=args.model, enhance_images=args.enhance, use_tta=args.tta)
    
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
        print("🔄 Multi-Scale TTA: ENABLED")
        # print("   • Using 3 scales (1.0x, 0.83x, 0.67x)")
    else:
        print("🔄 Multi-Scale TTA: DISABLED")
        print("   • Use --tta to enable for better accuracy")

    if args.class_confidences:
        print("🎯 Per-Class Confidence")
        if detector.class_confidences:
            for class_id, conf in detector.class_confidences.items():
                class_name = detector.class_names.get(str(class_id), f"Class {class_id}")
                print(f"   • {class_name.title()}: {conf:.2f}")
    else:
        print(f"Global Confidence")
        print(f"   • Threshold: {args.confidence:.2f}")
    
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
            print(f"📂 Processing {len(image_files)} images...\n")
            
            total_cracks = 0
            images_with_cracks = 0
            start_batch = time.time()
            
            pbar = tqdm(sorted(image_files), desc="Processing", unit="img")
            for image_file in pbar:
                detections = detector.detect_cracks(str(image_file), args.confidence, args.save)
                if detections:
                    total_cracks += len(detections)
                    images_with_cracks += 1
            batch_time = time.time() - start_batch

            print(f"\n📊 Summary:")
            print(f"   • Images processed: {len(image_files)}")
            print(f"   • Images with cracks: {images_with_cracks}")
            print(f"   • Total cracks found: {total_cracks}")
            if len(image_files) > 0:
                print(f"   • Average cracks per image: {total_cracks/len(image_files):.1f}")
                print(f"   • Processing time: {batch_time:.2f}s")  # ← 新增
                print(f"   • Average time per image: {batch_time/len(image_files):.2f}s")
    
    else:
        print(f"❌ Path not found: {input_path}")
    
    print("\n✅ Detection completed!")
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes > 0:
        print(f"⏱️  Total execution time: {minutes}m {seconds:.1f}s")
    else:
        print(f"⏱️  Total execution time: {seconds:.1f}s")

if __name__ == "__main__":
    main()