# detectors/sam_detector.py

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional
from .base_detector import BaseDetector

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    SAM2ImagePredictor = None
    print("⚠️ SAM2 library not found. SAMDetector will not be functional.")

class SAMDetector(BaseDetector):
    """SAM2 AI를 이용한 이상적인 형태 추정 탐지기"""

    def __init__(self, predictor, class_names: list):
        super().__init__(class_names)
        self.predictor = predictor  # SAM2 Predictor 저장

    def analyze_shape(self, image_patch, bbox, class_name, obb_mask):
        """
        SAM2를 호출해 이상적인 마스크와 컨투어 반환
        """
        try:
            if self.predictor is None:
                print("⚠️ SAMDetector: Skipping AI analysis as SAM2 predictor is not initialized.")
                return self._fallback_infer(obb_mask)

            # ROI 기준 이미지/마스크
            x1, y1, x2, y2 = bbox
            roi_image = image_patch[y1:y2, x1:x2]
            roi_obb_mask = obb_mask[y1:y2, x1:x2]

            points = self._find_positive_points(roi_image, roi_obb_mask)

            if len(points) > 0:
                points[:, 0] += x1
                points[:, 1] += y1
                point_coords = points.astype(np.float32)
                point_labels = np.ones(len(points), dtype=int)
            else:
                # 포인트 없으면 중심점 사용
                point_coords = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                point_labels = np.array([1])

            # SAM2 AI 호출
            rgb_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb_patch)
            masks, scores, _ = self.predictor.predict(
                box=np.array(bbox),
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # 최고 점수 마스크 선택 후 OBB 마스크와 교집합
            best_mask_idx = np.argmax(scores)
            sam_mask = (masks[best_mask_idx] * 255).astype(np.uint8)
            final_mask = cv2.bitwise_and(sam_mask, obb_mask)

            # 1. 화면에 그릴 이상적인 컨투어 추정
            ideal_contour_points = self._infer_ideal_shape(final_mask)

            # 2. 손실률 계산용 안쪽 마스크 생성
            damage_calculation_mask = self._create_inner_mask(final_mask)

            if ideal_contour_points is not None:
                # 화면 표시용 마스크는 기존 방식 유지
                ideal_mask = np.zeros_like(final_mask, dtype=np.uint8)
                cv2.fillPoly(ideal_mask, [ideal_contour_points], 255)
                
                # 손실률 계산에는 안쪽 마스크 사용
                return True, damage_calculation_mask, ideal_contour_points

            return False, None, None

        except Exception as e:
            print(f"⚠️ SAMDetector (AI) failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None

    def _fallback_infer(self, obb_mask):
        """AI 없을 때 OBB 기반 간단 추정"""
        try:
            ideal_contour_points = self._infer_ideal_shape(obb_mask)
            if ideal_contour_points is not None:
                # 손실률 계산용 안쪽 마스크 생성
                damage_calculation_mask = self._create_inner_mask(obb_mask)
                return True, damage_calculation_mask, ideal_contour_points
            return False, None, None
        except Exception:
            return False, None, None

    def _find_positive_points(self, roi_image, roi_mask, max_points=5):
        """OBB 영역 내 SAM 프롬프트용 포인트 추출"""
        try:
            h, w = roi_image.shape[:2]
            if h == 0 or w == 0:
                return np.array([])

            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright_mask = cv2.bitwise_and(bright_mask, roi_mask)

            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return np.array([])

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            points = []
            for contour in contours[:max_points]:
                if cv2.contourArea(contour) > 50:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        points.append([cX, cY])

            return np.array(points, dtype=np.float32)
        except Exception:
            return np.array([])

    def _infer_ideal_shape(self, mask):
        """마스크에서 최소 면적 사각형(OBB) 추출"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < 100:  # 너무 작은 객체 무시
            return None
        rect = cv2.minAreaRect(main_contour)
        return np.int32(cv2.boxPoints(rect))

    def _create_inner_mask(self, mask: np.ndarray) -> np.ndarray:
        """손실률 계산용 안쪽 마스크 생성"""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or hierarchy is None:
            return mask  # 안쪽 테두리가 없으면 원본 마스크 반환
        
        # 안쪽 테두리만 필터링 (Parent >= 0인 것들)
        inner_contours = [contours[i] for i in range(len(contours)) 
                         if hierarchy[0][i][3] >= 0]
        
        if not inner_contours:
            return mask  # 안쪽 테두리가 없으면 원본 마스크 반환
        
        # 안쪽 테두리로 새 마스크 생성
        inner_mask = np.zeros_like(mask)
        cv2.drawContours(inner_mask, inner_contours, -1, 255, -1)
        
        return inner_mask