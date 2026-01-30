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
    def __init__(self, model_type="yolov8_single", enhance_images=False, use_tta=False, use_tiling=False):
        """
        Initialize crack detector
        
        Args:
            model_type: Type of model to use
                - "yolov8_single": YOLOv8 Single Class
                - "yolov8_4classes": YOLOv8 4 Classes
            enhance_images: Whether to apply image enhancement
            use_tta: Whether to use Test Time Augmentation (multi-scale)
            use_tiling: Whether to use 4-tile inference
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
        self.use_tiling = use_tiling
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
        Modified: Standard YOLOv8 Letterbox preprocessing
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Optional enhancement
        image = enhance_image(image, self.enhance_images)
        
        # 取得原圖尺寸
        shape = image.shape[:2]  # [height, width]
        new_shape = (1024, 1024) # 目標尺寸 (YOLOv8 預設)
        
        # 1. 計算縮放比例 (取最小比例，確保整張圖都塞得進去)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 2. 計算 Padding (灰邊)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        # 把 Padding 分給左右/上下兩邊 (置中)
        dw /= 2  
        dh /= 2
        
        # 3. Resize
        if shape[::-1] != new_unpad:  
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # 4. 填補灰邊 (色碼 114)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 5. 標準化與轉置 (HWC -> CHW, 0-255 -> 0.0-1.0)
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        # 回傳: 影像, 縮放比, X偏移量(padding), Y偏移量(padding)
        return input_image, r, left, top
    
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
            
            # 3. Class-Agnostic NMS
            boxes_for_nms = np.column_stack([x1_orig, y1_orig, x2_orig, y2_orig]).astype(np.float32)
            confidences_f32 = confidences.astype(np.float32)
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(),
                confidences_f32.tolist(),
                0.0,
                0.35
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
    
    # ============= TILING METHODS =============
    
    def split_image_4tiles(self, img):
        """將影像切成4張 (2x2)"""
        h, w = img.shape[:2]
        h_half, w_half = h // 2, w // 2
        
        tiles = [
            img[0:h_half, 0:w_half],           # 左上
            img[0:h_half, w_half:w],           # 右上
            img[h_half:h, 0:w_half],           # 左下
            img[h_half:h, w_half:w]            # 右下
        ]
        
        offsets = [
            (0, 0),                            # 左上偏移
            (w_half, 0),                       # 右上偏移
            (0, h_half),                       # 左下偏移
            (w_half, h_half)                   # 右下偏移
        ]
        
        return tiles, offsets, (w, h)
    
    def check_boundary_proximity(self, box1, box2, w_half, h_half, margin=50):
        """
        檢查兩個框是否都靠近同一條切割邊界
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # === 檢查垂直切割線 (左右邊界) ===
        box1_right_near = abs(x1_max - w_half) < margin
        box1_left_near = abs(x1_min - w_half) < margin
        box2_right_near = abs(x2_max - w_half) < margin
        box2_left_near = abs(x2_min - w_half) < margin
        
        vertical_lr = box1_right_near and box2_left_near
        vertical_rl = box1_left_near and box2_right_near
        
        # === 檢查水平切割線 (上下邊界) ===
        box1_bottom_near = abs(y1_max - h_half) < margin
        box1_top_near = abs(y1_min - h_half) < margin
        box2_bottom_near = abs(y2_max - h_half) < margin
        box2_top_near = abs(y2_min - h_half) < margin
        
        horizontal_tb = box1_bottom_near and box2_top_near
        horizontal_bt = box1_top_near and box2_bottom_near
        
        return vertical_lr or vertical_rl or horizontal_tb or horizontal_bt
    
    def merge_tiled_detections(self, all_tile_detections, w_half, h_half, 
                                boundary_margin=50, iou_threshold=0.3):
        """
        合併4個tile的偵測結果
        """
        if not all_tile_detections:
            return []
        
        # 收集所有框
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for det in all_tile_detections:
            bbox = det['bbox']
            all_boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            all_scores.append(det['confidence'])
            all_classes.append(det['class_id'])
        
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        
        # 合併重疊框
        merged_detections = []
        used = np.zeros(len(all_boxes), dtype=bool)
        
        for i in range(len(all_boxes)):
            if used[i]:
                continue
            
            current_box = all_boxes[i]
            current_class = all_classes[i]
            overlapping_indices = [i]
            
            for j in range(i + 1, len(all_boxes)):
                if used[j] or all_classes[j] != current_class:
                    continue
                
                # 檢查 IoU
                iou = self._calc_iou_boxes(current_box, all_boxes[j])
                
                # 檢查邊界接近
                is_near_boundary = self.check_boundary_proximity(
                    current_box, all_boxes[j], 
                    w_half, h_half, boundary_margin
                )
                
                if iou > iou_threshold or is_near_boundary:
                    overlapping_indices.append(j)
                    used[j] = True
            
            # 合併框
            if len(overlapping_indices) > 1:
                overlapping_boxes = all_boxes[overlapping_indices]
                merged_box = np.array([
                    np.min(overlapping_boxes[:, 0]),  # x1
                    np.min(overlapping_boxes[:, 1]),  # y1
                    np.max(overlapping_boxes[:, 2]),  # x2
                    np.max(overlapping_boxes[:, 3])   # y2
                ])
                merged_score = np.max(all_scores[overlapping_indices])
            else:
                merged_box = current_box
                merged_score = all_scores[i]
            
            class_name = self.class_names.get(str(current_class), f"class_{current_class}")
            
            merged_detections.append({
                "bbox": {
                    "x1": float(merged_box[0]),
                    "y1": float(merged_box[1]),
                    "x2": float(merged_box[2]),
                    "y2": float(merged_box[3])
                },
                "confidence": float(merged_score),
                "class_id": int(current_class),
                "class_name": class_name
            })
            used[i] = True
        
        return merged_detections
    
    def detect_cracks_tiling(self, image, confidence=0.25):
        """
        使用4-tile inference偵測裂縫
        """
        img_height, img_width = image.shape[:2]
        
        # 切割影像
        tiles, offsets, original_size = self.split_image_4tiles(image)
        
        all_tile_detections = []
        
        # 對每個tile進行推論
        for tile_idx, (tile, (offset_x, offset_y)) in enumerate(zip(tiles, offsets)):
            # Preprocess tile
            input_tensor, scale, x_offset, y_offset = self.preprocess_image(tile)
            
            # Inference
            outputs = self.session.run(
                self.output_names, 
                {self.input_name: input_tensor}
            )
            
            # Postprocess
            tile_detections = self.postprocess_results(
                outputs, scale, x_offset, y_offset, confidence
            )
            
            # 轉換座標回原圖
            for det in tile_detections:
                det['bbox']['x1'] += offset_x
                det['bbox']['y1'] += offset_y
                det['bbox']['x2'] += offset_x
                det['bbox']['y2'] += offset_y
            
            all_tile_detections.extend(tile_detections)
        
        # 合併結果
        w_half = img_width // 2
        h_half = img_height // 2
        
        merged_detections = self.merge_tiled_detections(
            all_tile_detections,
            w_half, h_half,
            boundary_margin=50,
            iou_threshold=0.3
        )
        
        return merged_detections
    
    # ============= END TILING METHODS =============
    
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
            
            img_height, img_width = image.shape[:2]
            
            start_time = time.time()
            
            # === TILING MODE ===
            if self.use_tiling:
                detections = self.detect_cracks_tiling(image, confidence)
            
            # === TTA MODE ===
            elif self.use_tta:
                scales = [1.0, 0.83, 0.67]
                
                all_detections = []
                
                for scale_factor in scales:
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
                    
                    input_tensor, scale, x_offset, y_offset = self.preprocess_image(scaled_image)
                    
                    outputs = self.session.run(
                        self.output_names, 
                        {self.input_name: input_tensor}
                    )
                    
                    detections_scale = self.postprocess_results(
                        outputs, scale, x_offset, y_offset, confidence
                    )
                    
                    if scale_factor != 1.0:
                        for det in detections_scale:
                            det['bbox']['x1'] /= scale_factor
                            det['bbox']['y1'] /= scale_factor
                            det['bbox']['x2'] /= scale_factor
                            det['bbox']['y2'] /= scale_factor
                    
                    all_detections.extend(detections_scale)
                
                detections = self._merge_multiscale_detections(
                    all_detections,
                    img_width,
                    img_height,
                    iou_threshold=0.5
                )
            
            # === STANDARD MODE ===
            else:
                input_image, scale, x_offset, y_offset = self.preprocess_image(image)
                outputs = self.session.run(
                    self.output_names, 
                    {self.input_name: input_image}
                )
                detections = self.postprocess_results(
                    outputs, scale, x_offset, y_offset, confidence
                )
            
            detection_time = time.time() - start_time
            
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
    parser.add_argument("--tiling", action="store_true",
                        help="Use 4-tile inference for large images (better for dense cracks)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("\n📁 Please provide an image file or folder:")
        print("   Example: python inference.py image.jpg")
        print("   Example: python inference.py photos/ --enhance --save --model yolov8_4classes --tta")
        print("   Example with tiling: python inference.py image.jpg --tiling --save")
        print("   Example with per-class confidences: python inference.py image.jpg --class-confidences '0:0.3,1:0.25,2:0.2,3:0.35'")
        args.input = input("\nEnter path: ").strip().strip('"')
    
    detector = CrackDetector(
        model_type=args.model, 
        enhance_images=args.enhance, 
        use_tta=args.tta,
        use_tiling=args.tiling
    )
    
    if args.class_confidences:
        class_confs = parse_class_confidences(args.class_confidences)
        if class_confs:
            detector.set_class_confidences(class_confs)
    
    if args.enhance:
        print("🔧 Image enhancement: ENABLED")
    else:
        print("🔧 Image enhancement: DISABLED")
        print("   • Use --enhance to enable preprocessing")
    
    if args.tiling:
        print("🔲 4-Tile Inference: ENABLED")
        print("   • Image will be split into 2x2 tiles")
    elif args.tta:
        print("🔄 Multi-Scale TTA: ENABLED")
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
                print(f"   • Processing time: {batch_time:.2f}s")
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