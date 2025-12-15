# damage_analyzer.py

import cv2
import torch
import numpy as np
import traceback
from typing import Set, Optional, Tuple

# SAM2 로드
from omegaconf import OmegaConf
from hydra.utils import instantiate
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    SAM2ImagePredictor = None
    print("⚠️ SAM2 library not found.")

from detectors.sam_detector import SAMDetector
from detectors.brightness_detector import MultiThresholdBrightnessDetector

class DamageAnalyzer:
    """손상도 분석기"""
    
    def __init__(self, config):
        self.config = config
        print("🔄 Initializing Damage Analyzer ...")
        
        # SAM2 모델 로드
        self.sam_predictor = self._load_sam_model()

        # SAMDetector 초기화
        if self.sam_predictor:
            print(" - SAMDetector is running in AI-Assisted mode.")
            self.sam_detector = SAMDetector(self.sam_predictor, config.SAM_CLASSES)
        else:
            print(" - ⚠️ SAMDetector is running in FALLBACK mode (AI model failed to load).")
            self.sam_detector = SAMDetector(None, config.SAM_CLASSES)
            
        # 밝기 기반 탐지기 초기화
        self.brightness_detector = MultiThresholdBrightnessDetector(config.BRIGHT_CONTOUR_CLASSES)
        
        # 손실률 처리 가능 클래스 집합
        self.processable_classes = set(config.SAM_CLASSES) | set(config.BRIGHT_CONTOUR_CLASSES)
        
        print(f" - SAM classes: {config.SAM_CLASSES}")
        print(f" - Brightness classes: {config.BRIGHT_CONTOUR_CLASSES}")
        print(f" - No damage analysis: {config.NO_DAMAGE_ANALYSIS_CLASSES}")
        print("✅ Damage Analyzer is ready.")

    def _load_sam_model(self):
        """SAM2 모델 생성 및 로드"""
        if SAM2ImagePredictor is None:
            print("❌ SAM2 library not found.")
            return None
            
        try:
            print("  - Loading SAM2 model (hiera_base+)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = OmegaConf.load(self.config.SAM_MODEL_CONFIG)
            OmegaConf.resolve(cfg)

            # 모델 구조 생성
            sam2_model = instantiate(cfg.model, _recursive_=True)
            print("  - Model instantiated successfully.")

            # 체크포인트 적용
            ckpt_path = self.config.SAM_CHECKPOINT_PATH
            if ckpt_path:
                print(f"  - Loading checkpoint from: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                sam2_model.load_state_dict(checkpoint["model"], strict=True)
            
            # 디바이스 이동 및 평가 모드
            sam2_model.to(device)
            sam2_model.eval()
            
            predictor = SAM2ImagePredictor(sam2_model)
            print(f"  - SAM2 model loaded to {device}.")
            return predictor

        except Exception as e:
            print(f"❌ SAM2 model load failed: {e}")
            traceback.print_exc()
            return None

    def get_processable_classes(self) -> Set[str]:
        """손실률 분석 가능한 클래스 반환"""
        return self.processable_classes
    
    def should_analyze_damage(self, class_name: str) -> bool:
        """해당 클래스가 손실률 분석 대상인지 확인"""
        return class_name in self.processable_classes

    def analyze(self, image_patch: np.ndarray, local_corners: np.ndarray, 
                class_name: str) -> Tuple[Optional[float], Optional[list]]:
        """손상도 분석 수행"""
        if not self.should_analyze_damage(class_name):
            return None, None
        
        try:
            h, w = image_patch.shape[:2]
            obb_mask = np.zeros((h, w), dtype=np.uint8)
            local_corners_int = local_corners.astype(np.int32)
            cv2.fillPoly(obb_mask, [local_corners_int], 255)  # OBB 마스크 생성
            
            # Bounding box 계산
            x, y, w_box, h_box = cv2.boundingRect(local_corners_int)
            local_bbox = (x, y, x + w_box, y + h_box)

            # 클래스별 탐지기 선택
            detector = None
            if class_name in self.config.SAM_CLASSES:
                detector = self.sam_detector
            elif class_name in self.config.BRIGHT_CONTOUR_CLASSES:
                detector = self.brightness_detector
            
            if detector is None:
                return None, None

            # 형상 분석
            success, ideal_mask, ideal_contour_local = detector.analyze_shape(
                image_patch, local_bbox, class_name, obb_mask
            )
            
            if not success:
                return None, None

            # 손상도 계산
            damage = detector.calculate_damage_ratio(image_patch, ideal_mask, class_name)
            damage_percent = round(damage * 100, 2)
            
            # 컨투어를 리스트로 변환
            if isinstance(ideal_contour_local, np.ndarray):
                ideal_contour_local = ideal_contour_local.tolist()
                
            return damage_percent, ideal_contour_local

        except Exception as e:
            print(f"⚠️ DamageAnalyzer.analyze failed: {e}")
            traceback.print_exc()
            return None, None