# object_detector.py

import os
import json
import numpy as np
import uuid
from ultralytics import YOLO
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.ops import unary_union
import config

class YoloObbDetector:
    def __init__(self, model_path, tiles_dir, config_obj=None):
        print("🔄 Initializing YOLO Object Detector...")
        self.config = config_obj if config_obj is not None else config
        self.model = YOLO(model_path)
        self.tiles_dir = tiles_dir
        self.tile_coords = self._load_tile_coords()
        print("✅ YOLO Object Detector is ready.")

    def _load_tile_coords(self):
        # 타일 좌표 정보를 JSON 파일에서 불러옴
        coords_path = os.path.join(self.tiles_dir, "tile_coordinates.json")
        try:
            with open(coords_path, 'r') as f:
                print(f"  - Loading tile coordinates from {coords_path}")
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 좌표 파일({coords_path})을 찾을 수 없습니다. 타일 생성 스크립트를 먼저 실행하세요.")

    def detect_all(self, conf_threshold=0.25, nms_threshold=0.4, merge_threshold=0.3):
        """전체 탐지 프로세스: YOLO 탐지 → NMS → 분할 객체 병합"""
        
        # 체크포인트 디렉터리 생성
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

        # --- 1단계: YOLO 탐지 ---
        if os.path.exists(self.config.CHECKPOINT_STEP1_PATH):
            print("✅ [1/4] Found Step 1 checkpoint, loading YOLO detections...")
            with open(self.config.CHECKPOINT_STEP1_PATH, 'r', encoding='utf-8') as f:
                all_detections = json.load(f)
        else:
            all_detections = self._run_yolo_on_tiles(conf_threshold)
            print(f"💾 [1/4] Saving YOLO detections to checkpoint: {self.config.CHECKPOINT_STEP1_PATH}")
            with open(self.config.CHECKPOINT_STEP1_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_detections, f)

        # --- 2단계: NMS 적용 ---
        print(f"\n[2/4] NMS 시작: {len(all_detections)}개 객체 (IoU threshold={nms_threshold})")
        if os.path.exists(self.config.CHECKPOINT_STEP2_PATH):
            print("✅ [2/4] Found Step 2 checkpoint, loading NMS results...")
            with open(self.config.CHECKPOINT_STEP2_PATH, 'r', encoding='utf-8') as f:
                nms_detections = json.load(f)
        else:
            nms_detections = self._obb_nms(all_detections, nms_threshold)
            print(f"💾 [2/4] Saving NMS results to checkpoint: {self.config.CHECKPOINT_STEP2_PATH}")
            with open(self.config.CHECKPOINT_STEP2_PATH, 'w', encoding='utf-8') as f:
                json.dump(nms_detections, f)
        print(f"✅ [2/4] After NMS: {len(nms_detections)}개 객체 (제거: {len(all_detections) - len(nms_detections)}개)")

        # --- 3단계: 분할 객체 병합 ---
        print(f"\n[3/4] 병합 시작: {len(nms_detections)}개 객체 (merge threshold={merge_threshold})")
        if os.path.exists(self.config.CHECKPOINT_STEP3_PATH):
            print("✅ [3/4] Found Step 3 checkpoint, loading merged objects...")
            with open(self.config.CHECKPOINT_STEP3_PATH, 'r', encoding='utf-8') as f:
                merged_detections = json.load(f)
            print(f"  - Loaded {len(merged_detections)} objects from checkpoint.")
        else:
            merged_detections = self._merge_split_objects(nms_detections, merge_threshold)
            print(f"💾 [3/4] Saving merged objects to checkpoint: {self.config.CHECKPOINT_STEP3_PATH}")
            with open(self.config.CHECKPOINT_STEP3_PATH, 'w', encoding='utf-8') as f:
                json.dump(merged_detections, f)
        print(f"✅ [3/4] After Merge: {len(merged_detections)}개 객체 (병합: {len(nms_detections) - len(merged_detections)}개)")

        # 4단계 손상 분석은 ProcessManager에서 수행
        return merged_detections
    
    def _run_yolo_on_tiles(self, conf_threshold):
        """모든 타일에서 YOLO 객체 탐지 수행"""
        detections = []
        tile_image_dir = os.path.join(self.tiles_dir, 'tiles')
        if not os.path.exists(tile_image_dir):
            print(f"⚠️ Warning: Tile image directory not found at '{tile_image_dir}'")
            return []
        actual_tile_files = sorted([f for f in os.listdir(tile_image_dir) if f.lower().endswith(('.png', '.jpg'))])

        for tile_filename in tqdm(actual_tile_files, desc="[1/4] YOLO Detecting"):
            expected_json_key = os.path.join("results", "tiles", tile_filename).replace("\\", "/")
            if expected_json_key in self.tile_coords:
                x_offset, y_offset = self.tile_coords[expected_json_key]
            elif tile_filename in self.tile_coords:
                x_offset, y_offset = self.tile_coords[tile_filename]
            else:
                continue
            tile_path = os.path.join(tile_image_dir, tile_filename)
            results = self.model(tile_path, verbose=False, conf=conf_threshold)
            for result in results:
                if result.obb is None: continue
                for box in result.obb:
                    corners = box.xyxyxyxy[0].cpu().numpy().reshape((4, 2))
                    corners[:, 0] += x_offset; corners[:, 1] += y_offset
                    detections.append({
                        "object_id": str(uuid.uuid4()), "bbox_corners": corners.tolist(),
                        "confidence": float(box.conf[0]), "class_id": int(box.cls[0]),
                        "class_name": self.model.names[int(box.cls[0])], "source_tile": tile_filename
                    })
        return detections
    
    def _obb_nms(self, detections, default_iou_threshold):
        """클래스별 IoU 임계값을 적용하고 NMS 수행"""
        if not detections: return []
        
        detections_by_class = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append(det)

        final_keep = []
        
        for class_id, dets in detections_by_class.items():
            if not dets: continue
            
            class_name = dets[0]['class_name']
            
            iou_threshold = getattr(self.config, 'CLASS_SPECIFIC_NMS_IOU', {}).get(class_name, default_iou_threshold)

            polygons = [Polygon(d['bbox_corners']) for d in dets]
            for i, d in enumerate(dets):
                d['area'] = polygons[i].area

            if class_name in {'crosswalk'}:
                dets = sorted(dets, key=lambda x: x['area'], reverse=True)
            else:
                dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
            
            polygons = [Polygon(d['bbox_corners']) for d in dets] 

            suppressed = [False] * len(dets)
            suppressed_details = [] 
            
            for i in tqdm(range(len(dets)), desc=f"[2/4] Applying NMS for '{class_name}'"):
                if suppressed[i]: continue
                final_keep.append(dets[i])
                
                for j in range(i + 1, len(dets)):
                    if suppressed[j]: continue
                    try:
                        intersection = polygons[i].intersection(polygons[j]).area
                        union = dets[i]['area'] + dets[j]['area'] - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > iou_threshold:
                            suppressed[j] = True
                            suppressed_details.append({
                                'iou': iou,
                                'threshold': iou_threshold,
                                'kept_conf': dets[i]['confidence'],
                                'suppressed_conf': dets[j]['confidence']
                            })
                    except Exception:
                        continue
            
            if suppressed_details:
                print(f"  ⚠️ NMS로 {len(suppressed_details)}개 제거됨 (IoU threshold={iou_threshold})")

        return final_keep

    def _merge_split_objects(self, detections, merge_threshold=0.3):
        """분할된 같은 객체들을 병합"""
        if not detections: return []
        
        try:
            DO_NOT_MERGE_CLASSES = getattr(config, 'DO_NOT_MERGE_CLASSES', set())
        except AttributeError:
            DO_NOT_MERGE_CLASSES = set()

        merged_results = []
        used_indices = set()
        
        for i in tqdm(range(len(detections)), desc="[3/4] Merging Split Objects"):
            if i in used_indices: continue
            
            current_detection = detections[i]

            if current_detection['class_name'] in DO_NOT_MERGE_CLASSES:
                merged_results.append(current_detection)
                used_indices.add(i)
                continue

            merge_candidates = [current_detection]
            used_indices.add(i)
            
            for j in range(i + 1, len(detections)):
                if j in used_indices: continue
                other_detection = detections[j]
                
                if current_detection['class_name'] != other_detection['class_name']: continue
                
                if self._should_merge(current_detection, other_detection, merge_threshold):
                    merge_candidates.append(other_detection)
                    used_indices.add(j)
            
            merged_objs = self._merge_candidates(merge_candidates)
            merged_results.extend(merged_objs) 
        
        print(f"  - Merged detections into {len(merged_results)} final objects.")
        return merged_results

    def _should_merge(self, det1, det2, threshold):
        """두 객체가 병합되어야 하는지 판단"""
        try:
            poly1 = Polygon(det1['bbox_corners'])
            poly2 = Polygon(det2['bbox_corners'])
            
            def get_major_axis(poly):
                points = np.array(poly.exterior.coords)
                dist_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
                p1_idx, p2_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                p1, p2 = points[p1_idx], points[p2_idx]
                length = np.linalg.norm(p1 - p2)
                angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
                return length, angle

            len1, angle1 = get_major_axis(poly1)
            len2, angle2 = get_major_axis(poly2)

            # 1. 겹침 비율 확인
            try:
                intersection = poly1.intersection(poly2).area
                overlap1 = intersection / poly1.area if poly1.area > 0 else 0
                overlap2 = intersection / poly2.area if poly2.area > 0 else 0
                max_overlap = max(overlap1, overlap2)
            except:
                max_overlap = 0
            
            if max_overlap > 0.5: # 50% 이상 겹치면 무조건 병합
                return True

            # 2. 각도 차이 확인
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > 70: # 각도가 너무 다르면 병합 안 함
                return False

            # 3. 거리 확인
            avg_length = (len1 + len2) / 2
            min_threshold = 50  # 최소 50px 거리까지는 허용
            distance_threshold = max(avg_length * threshold, min_threshold)
            
            actual_distance = poly1.distance(poly2)
            
            if actual_distance >= distance_threshold:
                return False
            
            # 4. 연결성 확인 (버퍼 확장)
            buffer_size = (len1 + len2) / 2 * 0.1
            buffered1 = poly1.buffer(buffer_size)
            buffered2 = poly2.buffer(buffer_size)
            
            if not buffered1.intersects(buffered2):
                return False
            
            return True
        except Exception as e:
            print(f"⚠️ Error in merge decision: {e}")
            return False
    
    def _merge_candidates(self, candidates):
        """여러 후보 객체들을 병합 - MultiPolygon 대응"""
        if len(candidates) == 1:
            return [candidates[0]]
        
        try:
            polygons = [Polygon(cand['bbox_corners']) for cand in candidates]
            merged_polygon = unary_union(polygons)
            
            # MultiPolygon인 경우 (여러 조각으로 분리됨)
            if hasattr(merged_polygon, 'geoms'):
                results = []
                for geom in merged_polygon.geoms:
                    min_rect = geom.minimum_rotated_rectangle
                    coords = list(min_rect.exterior.coords)
                    best_conf = max(cand['confidence'] for cand in candidates)
                    results.append({
                        "object_id": str(uuid.uuid4()),
                        "bbox_corners": coords[:4],
                        "confidence": best_conf,
                        "class_id": candidates[0]['class_id'],
                        "class_name": candidates[0]['class_name'],
                        "merged_from_tiles": [],
                        "original_detections_count": 1
                    })
                return results
            
            # 단일 Polygon인 경우
            min_rect = merged_polygon.minimum_rotated_rectangle
            coords = list(min_rect.exterior.coords)
            best_confidence = max(cand['confidence'] for cand in candidates)
            source_tiles = list(set(cand['source_tile'] for cand in candidates))
            
            return [{
                "object_id": str(uuid.uuid4()),
                "bbox_corners": coords[:4],
                "confidence": best_confidence,
                "class_id": candidates[0]['class_id'],
                "class_name": candidates[0]['class_name'],
                "merged_from_tiles": source_tiles,
                "original_detections_count": len(candidates)
            }]
        except Exception as e:
            print(f"⚠️ Error merging candidates: {e}")
            return [max(candidates, key=lambda x: x['confidence'])]