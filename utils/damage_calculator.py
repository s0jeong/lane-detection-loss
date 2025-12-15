# utils/damage_calculator.py

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from config import DAMAGE_CALCULATION_PARAMS, SAM_CLASSES, BRIGHT_CONTOUR_CLASSES, MAX_PIXEL_SAMPLE

class BaseDamageCalculator(ABC):
    """손실률 계산 기본 클래스"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params if params else DAMAGE_CALCULATION_PARAMS

    @abstractmethod
    def calculate(self, roi_image: np.ndarray, detected_mask: np.ndarray) -> float:
        """ROI 이미지와 검출 마스크로 손실률 계산"""
        pass

    def _calculate_by_color_similarity(self, roi_image: np.ndarray, detected_mask: np.ndarray) -> float:
        """색상 유사도 기반 손실률 계산"""
        # 1. 입력 예외 처리
        if roi_image is None or roi_image.size == 0 or detected_mask is None or detected_mask.size == 0:
            return 1.0

        # --- 동적 침식 적용 ---
        total_mask_area = np.count_nonzero(detected_mask)
        kernel = np.ones((3, 3), np.uint8)
        
        # (1) 면적이 2000 미만: 1번만 깎음
        if total_mask_area < 2000:
            eroded_mask = cv2.erode(detected_mask, kernel, iterations=1)
            
        # (2) 면적이 2000 ~ 3000: 2번만 깎음
        elif total_mask_area < 3000:
            eroded_mask = cv2.erode(detected_mask, kernel, iterations=2)
            
        # (3) 면적이 2000 이상: 3번 깎음
        else:
            eroded_mask = cv2.erode(detected_mask, kernel, iterations=3)
        # ----------------------------------------------------

        # 2. 유효 픽셀 추출
        all_masked_pixels_bgr = roi_image[eroded_mask > 0]
        total_pixels = len(all_masked_pixels_bgr)

        # [DEBUG] 평균 픽셀 밝기 확인용 코드
        if total_pixels > 0:
            # BGR 평균 계산
            mean_bgr = np.mean(all_masked_pixels_bgr, axis=0).astype(int)
            # 밝기(Grayscale) 평균 계산 (이 값이 brightness_threshold보다 커야 함)
            gray_vals = cv2.cvtColor(all_masked_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_vals)
            
            # 로그 출력: 밝기가 150 미만인 어두운 객체만 경고처럼 출력해서 확인
            if mean_brightness < 150: 
                print(f"🔍 [Check] Dark Lane Detected! Area: {total_mask_area}, Mean Brightness: {mean_brightness:.1f}, Mean BGR: {mean_bgr}")
        
        # 3. 안전장치 (Fallback)
        # 깎았더니 픽셀이 너무 적어졌다면(20개 미만), 다시 1번만 깎은 걸로 시도
        if total_pixels < 20:
             if total_mask_area >= 1000:
                 eroded_mask = cv2.erode(detected_mask, kernel, iterations=2)
                 all_masked_pixels_bgr = roi_image[eroded_mask > 0]
                 total_pixels = len(all_masked_pixels_bgr)
             
             # 그래도 픽셀이 없으면 손상 없음(0.0)으로 처리
             if total_pixels < 20:
                 return 0.0 

        try:
            # 4. 픽셀 샘플링 (속도 최적화)
            if total_pixels > MAX_PIXEL_SAMPLE:
                indices = np.random.choice(total_pixels, MAX_PIXEL_SAMPLE, replace=False)
                sampled_pixels_bgr = all_masked_pixels_bgr[indices]
                total_pixels_in_sample = MAX_PIXEL_SAMPLE
            else:
                sampled_pixels_bgr = all_masked_pixels_bgr
                total_pixels_in_sample = total_pixels

            # 5. 기준 색상(Dominant Color) 결정 - Percentile 기반
            gray_pixels = cv2.cvtColor(sampled_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()

            # 상위 25% 밝기 픽셀을 기준으로 dominant color 선택
            percentile_threshold = np.percentile(gray_pixels, 75)  # 75 percentile = 상위 25%
            bright_pixels_bgr = sampled_pixels_bgr[gray_pixels > percentile_threshold]

            # 밝은 픽셀이 충분히 있으면 그 중에서 최빈 색상 선택
            if len(bright_pixels_bgr) > 10:
                unique_colors, counts = np.unique(bright_pixels_bgr, axis=0, return_counts=True)
                dominant_color_bgr = unique_colors[counts.argmax()]
            else:
                # 폴백: 전체 픽셀 중 상위 50% 밝기에서 선택
                fallback_threshold = np.percentile(gray_pixels, 50)
                fallback_pixels_bgr = sampled_pixels_bgr[gray_pixels > fallback_threshold]
                if len(fallback_pixels_bgr) > 0:
                    unique_colors, counts = np.unique(fallback_pixels_bgr, axis=0, return_counts=True)
                    dominant_color_bgr = unique_colors[counts.argmax()]
                else:
                    # 최후의 폴백: 전체 픽셀 중 최빈값
                    unique_colors, counts = np.unique(sampled_pixels_bgr, axis=0, return_counts=True)
                    dominant_color_bgr = unique_colors[counts.argmax()]
                    
            # 6. Delta E (색상 거리) 계산
            dominant_color_lab = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2Lab)[0][0]
            all_masked_pixels_lab = cv2.cvtColor(sampled_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab).reshape(-1, 3)
            
            diff = all_masked_pixels_lab.astype(np.float32) - dominant_color_lab.astype(np.float32)
            delta_e = np.linalg.norm(diff, axis=1)

            # config에 없으면 기본값 80.0 사용
            threshold = self.params.get('color_match_threshold', 80.0)
            good_pixels = np.sum(delta_e < threshold)
            
            damage_ratio = 1.0 - (good_pixels / total_pixels_in_sample)

            # 7. 크기 + 형태 기반 보정
            area_factor = min(1.0, total_mask_area / 2500.0)
            
            try:
                contours, _ = cv2.findContours(detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    width, height = rect[1]
                    
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        # 가늘고 긴 객체일수록 추가 보정
                        if aspect_ratio > 10:
                            shape_factor = 0.6
                        elif aspect_ratio > 5:
                            shape_factor = 0.8
                        else:
                            shape_factor = 1.0
                    else:
                        shape_factor = 1.0
                else:
                    shape_factor = 1.0
            except:
                shape_factor = 1.0
            
            scaling_factor = area_factor * shape_factor
            final_damage_ratio = damage_ratio * scaling_factor
            
            adjusted_ratio = self._apply_nonlinear_dampening(final_damage_ratio)
            return max(0.0, min(1.0, adjusted_ratio))

        except Exception as e:
            print(f"⚠️ Error in color calculation: {e}")
            return 1.0
    
    def _apply_nonlinear_dampening(self, raw_ratio: float, threshold: float = 0.03) -> float:
        """
        손상률 비선형 보정 함수
        - 3% 이하: 그대로 유지
        - 3% 이상: 제곱근 스케일로 완만하게 압축
        
        Args:
            raw_ratio: 원래 계산된 손상률 (0.0~1.0)
            threshold: 보정 시작 기준값 (기본 0.03 = 3%)
        
        Returns:
            보정된 손상률 (0.0~1.0)
        """
        if raw_ratio <= threshold:
            return raw_ratio  # 3% 이하는 그대로
        
        # 3% 초과분만 압축 (0.6 제곱)
        excess = raw_ratio - threshold
        dampened_excess = excess ** 0.6
        
        return threshold + dampened_excess

class ColorBasedDamageCalculator(BaseDamageCalculator):
    """일반 색상 기반 손실률 계산기"""
    def calculate(self, roi_image: np.ndarray, detected_mask: np.ndarray) -> float:
        return self._calculate_by_color_similarity(roi_image, detected_mask)

class EdgeBasedDamageCalculator(BaseDamageCalculator):
    """SAM/Edge 기반 손실률 계산기"""
    def calculate(self, roi_image: np.ndarray, detected_mask: np.ndarray) -> float:
        return self._calculate_by_color_similarity(roi_image, detected_mask)

class BrightnessDamageCalculator(BaseDamageCalculator):
    """밝기 기반 손실률 계산기"""
    def calculate(self, roi_image: np.ndarray, detected_mask: np.ndarray) -> float:
        return self._calculate_by_color_similarity(roi_image, detected_mask)

def get_damage_calculator(class_name: str) -> BaseDamageCalculator:
    """클래스별 적절한 손실률 계산기 반환"""
    if class_name in BRIGHT_CONTOUR_CLASSES:
        return BrightnessDamageCalculator()
    elif class_name in SAM_CLASSES:
        return EdgeBasedDamageCalculator()
    else:
        return ColorBasedDamageCalculator()