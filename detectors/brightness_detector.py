# detectors/brightness_detector.py

import cv2
import numpy as np
from typing import Optional, Tuple
from .base_detector import BaseDetector
from config import DAMAGE_CALCULATION_PARAMS

class BrightnessDetector(BaseDetector):
    """밝기 기반 컨투어 탐지 기본 클래스"""
    
    def __init__(self, class_names: list):
        super().__init__(class_names)
        self.threshold = DAMAGE_CALCULATION_PARAMS['brightness_threshold']
    
    def analyze_shape(self, image_patch, bbox, class_name, obb_mask):
        pass  # 서브클래스에서 구현

class MultiThresholdBrightnessDetector(BrightnessDetector):
    """여러 임계값 적용 + 모폴로지 스무딩 기반 최종 탐지기"""
    
    def __init__(self, class_names: list):
        super().__init__(class_names)
        self.thresholds = [190, 210, 225, 240]
    
    def analyze_shape(self, image_patch, bbox, class_name, obb_mask):
        """
        여러 임계값과 모폴로지 연산을 적용하여 가장 안정적인 컨투어 선택
        """
        try:
            gray_roi = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            roi_obb_mask = obb_mask
            
            best_contours = None
            best_area = 0
            
            is_diamond = (class_name == 'diamond')
            is_lettering = (class_name == 'lettering')

            for threshold in self.thresholds:
                # 밝기 기반 이진화
                _, bright_mask = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY)

                # OBB 마스크 적용
                final_mask = cv2.bitwise_and(bright_mask, roi_obb_mask)
                
                # 모폴로지 오픈(노이즈 제거)
                kernel = np.ones((3, 3), np.uint8)
                opened_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # 클래스별 클로즈 강도 조절
                if is_diamond or is_lettering:
                    close_kernel = np.ones((5, 5), np.uint8)
                    smoothed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
                else:
                    smoothed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                # diamond/lettering은 내부 컨투어 포함
                contour_mode = cv2.RETR_LIST if (is_diamond or is_lettering) else cv2.RETR_EXTERNAL
                
                contours, _ = cv2.findContours(smoothed_mask, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = contours

                # diamond는 면적 상위 2개만 사용
                if is_diamond and contours and len(contours) > 1:
                    filtered_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                
                if not filtered_contours:
                    continue

                current_area = sum(cv2.contourArea(c) for c in filtered_contours)

                # 가장 넓은 컨투어 선택
                if current_area > best_area:
                    best_area = current_area
                    best_contours = filtered_contours
            
            if best_contours is not None:
                # 최종 마스크 생성
                final_roi_mask = np.zeros_like(gray_roi)
                cv2.drawContours(final_roi_mask, best_contours, -1, 255, -1)
                
                return True, final_roi_mask, np.concatenate(best_contours)
            
            return False, None, None
            
        except Exception as e:
            print(f"⚠️ Multi-threshold detection failed: {e}")
            return False, None, None