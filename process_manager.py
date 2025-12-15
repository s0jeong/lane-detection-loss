# process_manager.py

import os
import json
import rasterio
import traceback
import numpy as np
from tqdm import tqdm
from object_detector import YoloObbDetector
from damage_analyzer import DamageAnalyzer
from utils.image_helpers import get_image_patch
from pyproj import CRS, Transformer

class ProcessManager:
    def __init__(self, config):
        self.config = config
        self.object_detector = YoloObbDetector(config.YOLO_MODEL_PATH, config.TILES_DIR)
        self.damage_analyzer = DamageAnalyzer(config)
        self._transformer_cache = {}  # 좌표 변환기 캐싱

    # 픽셀 좌표를 EPSG:4326(위도, 경도)로 변환하는 함수
    def _convert_pixels_to_epsg4326(self, tif_dataset, pixel_coords_list):
        """이미지 픽셀 좌표 리스트를 EPSG:4326 (위도, 경도) 좌표 리스트로 변환합니다."""
        if not pixel_coords_list or not hasattr(tif_dataset, 'crs'):
            return None

        valid_coords = [
            coord for coord in pixel_coords_list
            if isinstance(coord, (list, tuple)) and len(coord) >= 2
        ]

        if not valid_coords:
            return None

        src_crs_str = str(tif_dataset.crs)
        target_crs_str = "EPSG:4326"

        if (src_crs_str, target_crs_str) not in self._transformer_cache:
            src_crs = CRS(tif_dataset.crs)
            target_crs = CRS(target_crs_str)
            self._transformer_cache[(src_crs_str, target_crs_str)] = Transformer.from_crs(
                src_crs, target_crs, always_xy=True
            )
        transformer = self._transformer_cache[(src_crs_str, target_crs_str)]

        xs = [coord[0] for coord in valid_coords]
        ys = [coord[1] for coord in valid_coords]
        native_lons, native_lats = tif_dataset.xy(ys, xs)
        final_lons, final_lats = transformer.transform(native_lons, native_lats)

        return [[lon, lat] for lon, lat in zip(final_lons, final_lats)]

    def run(self):
        print("\n🚀 Starting the full analysis process...")
        # 1. 객체 탐지 수행
        all_objects = self.object_detector.detect_all(
            self.config.DETECTION_CONF_THRESHOLD,
            self.config.NMS_IOU_THRESHOLD,
            self.config.MERGE_THRESHOLD
        )
        print(f"✅ Object detection completed. Total detected: {len(all_objects)}")

        final_results = []
        processed_ids = set()

        # 2. 이전 중간 분석 결과가 있으면 이어서 실행
        if os.path.exists(self.config.CHECKPOINT_PARTIAL_ANALYSIS_PATH):
            print("\n🔄 Found checkpoint. Resuming from the previous partial analysis...")
            with open(self.config.CHECKPOINT_PARTIAL_ANALYSIS_PATH, 'r', encoding='utf-8') as f:
                final_results = json.load(f)
            processed_ids = {item['object_id'] for item in final_results}
            print(f"   - Loaded {len(final_results)} previously processed objects.")
        
        # 3. 새로 분석해야 할 객체만 추출
        objects_to_analyze = [obj for obj in all_objects if obj['object_id'] not in processed_ids]
        if len(objects_to_analyze) < len(all_objects):
            print(f"   - {len(objects_to_analyze)} remaining objects will be analyzed.")

        print("\n🔬 Starting per-object damage and segmentation analysis...")
        try:
            with rasterio.open(self.config.ORIGINAL_IMAGE_PATH) as tif_dataset:
                progress_bar = tqdm(objects_to_analyze, desc="✅ [4/4] Damage & segmentation analysis",
                                    initial=len(final_results), total=len(all_objects))
                for i, obj in enumerate(progress_bar):
                    try:
                        class_name = obj['class_name']
                        patch, local_corners, offset = get_image_patch(tif_dataset, obj['bbox_corners'])

                        result_item = {
                            "object_id": obj['object_id'],
                            "class_name": class_name,
                            "confidence": obj['confidence'],
                            "damage_percent": None,
                            "analysis_method": "excluded",
                            "bbox_corners_pixel": obj['bbox_corners'],  # 바운딩 박스 픽셀 좌표
                            "bbox_corners_epsg4326": None,              # 바운딩 박스 위경도 좌표
                            "segmentation_pixel": None,                 # 분할 결과 픽셀 좌표
                            "segmentation_epsg4326": None,              # 분할 결과 위경도 좌표
                            "source_tile": obj.get('source_tile', 'N/A'),
                            "merged_from_tiles": obj.get('merged_from_tiles', [obj.get('source_tile', 'N/A')]),
                            "original_detections_count": obj.get('original_detections_count', 1)
                        }

                        # 바운딩 박스 좌표를 EPSG:4326으로 변환
                        result_item['bbox_corners_epsg4326'] = self._convert_pixels_to_epsg4326(tif_dataset, obj['bbox_corners'])

                        # 손상 분석 수행 (제외 클래스가 아닌 경우)
                        if (patch is not None and local_corners is not None and
                            class_name not in self.config.EXCLUDED_CLASSES_FOR_DAMAGE):
                            
                            damage, segmentation = self.damage_analyzer.analyze(patch, local_corners, class_name)

                            # 분석 방법 기록
                            if class_name in self.config.SAM_CLASSES: 
                                result_item['analysis_method'] = "SAM"
                            elif class_name in self.config.BRIGHT_CONTOUR_CLASSES: 
                                result_item['analysis_method'] = "AdaptiveBrightness"
                            else: 
                                result_item['analysis_method'] = "unknown"

                            if damage is not None and segmentation is not None:
                                # 로컬 → 전역 좌표 변환
                                global_segmentation = self._convert_to_global_coordinates(
                                    segmentation, local_corners, obj['bbox_corners'], offset
                                )

                                result_item['damage_percent'] = damage
                                result_item['segmentation_pixel'] = global_segmentation
                                result_item['segmentation_epsg4326'] = self._convert_pixels_to_epsg4326(
                                    tif_dataset, global_segmentation
                                )

                        final_results.append(result_item)

                        # 50개마다 중간 저장
                        if (i + 1) % 50 == 0:
                            with open(self.config.CHECKPOINT_PARTIAL_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
                                json.dump(final_results, f)
                    except Exception as e:
                        print(f"\n❌ Error while processing object_id: {obj.get('object_id', 'N/A')}. Skipped.")
                        traceback.print_exc()
                        
        except rasterio.errors.RasterioIOError:
            print(f"❌ Error: Unable to open the original image file: '{self.config.ORIGINAL_IMAGE_PATH}'")
            return
        except Exception as e:
            print(f"❌ Unexpected error during analysis preparation: {e}")
            traceback.print_exc()
            return

        # 5. 최종 결과 저장
        print(f"\n💾 Saving final results to {self.config.FINAL_OUTPUT_PATH} ...")
        try:
            with open(self.config.FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)

            print(f"✅ Results saved successfully!")
            print(f"📊 Summary:")
            print(f"   - Total detected objects: {len(final_results)}")

            # 중간 저장 파일 삭제
            if os.path.exists(self.config.CHECKPOINT_PARTIAL_ANALYSIS_PATH):
                os.remove(self.config.CHECKPOINT_PARTIAL_ANALYSIS_PATH)

            # 분석 방법별 통계 출력
            analysis_stats = {}
            damage_analyzed = 0
            for result in final_results:
                method = result.get('analysis_method', 'unknown')
                analysis_stats[method] = analysis_stats.get(method, 0) + 1
                if result.get('damage_percent') is not None:
                    damage_analyzed += 1

            for method, count in analysis_stats.items():
                print(f"   - {method}: {count} objects")
            print(f"   - Successfully analyzed damage: {damage_analyzed} objects")

        except Exception as e:
            print(f"❌ Error while saving results: {e}")
            return

        print("\n🎉 All processes completed successfully!")

    # 로컬 분할 좌표를 전역 좌표로 변환
    def _convert_to_global_coordinates(self, local_segmentation, local_corners, global_corners, offset):
        try:
            if local_segmentation is None:
                return None

            if offset is not None:
                global_segmentation = (np.array(local_segmentation) + np.array(offset)).tolist()
                return global_segmentation

            return self._affine_transform_coordinates(local_segmentation, local_corners, global_corners)

        except Exception as e:
            print(f"⚠️ Coordinate transformation failed: {e}")
            return local_segmentation

    # 어핀 변환을 사용한 정밀 좌표 변환
    def _affine_transform_coordinates(self, local_segmentation, local_corners, global_corners):
        try:
            import cv2

            local_pts = np.array(local_corners[:3], dtype=np.float32)
            global_pts = np.array(global_corners[:3], dtype=np.float32)

            transform_matrix = cv2.getAffineTransform(local_pts, global_pts)
            seg_points = np.array(local_segmentation, dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.transform(seg_points, transform_matrix)

            return transformed_points.reshape(-1, 2).tolist()

        except Exception as e:
            print(f"⚠️ Affine transform failed: {e}")
            offset_x = global_corners[0][0] - local_corners[0][0]
            offset_y = global_corners[0][1] - local_corners[0][1]
            return (np.array(local_segmentation) + [offset_x, offset_y]).tolist()