# detectors/base_detector.py

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from utils.damage_calculator import get_damage_calculator

class BaseDetector(ABC):
    
    """탐지기의 기본 추상 클래스"""
    def __init__(self, class_names: list):
        self.class_names = class_names  # 처리 가능한 클래스 목록
    
    @abstractmethod
    def analyze_shape(self, image_patch: np.ndarray, bbox: Tuple[int, int, int, int], 
                      class_name: str, obb_mask: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        객체 형상 분석 후: (성공 여부, 이상적 마스크, 이상적 윤곽선)
        """
        pass
    
    def calculate_damage_ratio(self, roi_image: np.ndarray, detected_mask: np.ndarray, class_name: str) -> float:
        """클래스별 손상률 계산"""
        calculator = get_damage_calculator(class_name)
        return calculator.calculate(roi_image, detected_mask)
    
    def can_handle(self, class_name: str) -> bool:
        """해당 클래스를 처리할 수 있는지 여부"""
        return class_name in self.class_names
    
    def draw_contours(self, image: np.ndarray, contours: list, bbox: Tuple[int, int, int, int], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
        """윤곽선 그리기 (너무 작은 윤곽선 제외)"""
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(image, [contour], -1, color, thickness)