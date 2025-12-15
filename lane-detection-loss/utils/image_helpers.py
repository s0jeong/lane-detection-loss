# utils/image_helpers.py

import cv2
import math
import rasterio
import numpy as np
from shapely.geometry import Polygon

def get_image_patch(tif_dataset, corners):
    # corners로부터 폴리곤 객체 생성
    polygon = Polygon(corners)
    # 폴리곤의 경계 좌표(최소/최대 x, y) 추출
    min_x_f, min_y_f, max_x_f, max_y_f = polygon.bounds
    min_x = int(math.floor(min_x_f))
    min_y = int(math.floor(min_y_f))
    max_x = int(math.ceil(max_x_f))
    max_y = int(math.ceil(max_y_f))
    
    # 이미지 경계 내로 좌표를 보정
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(tif_dataset.width, max_x)
    max_y = min(tif_dataset.height, max_y)
    
    # 유효하지 않은 영역일 경우 None 반환
    if max_x <= min_x or max_y <= min_y:
        return None, None, None

    # 지정된 영역(window)만큼 이미지 읽기
    window = rasterio.windows.Window(min_x, min_y, max_x - min_x, max_y - min_y)
    patch_data = tif_dataset.read(window=window)
    
    # 채널 수가 3개 이상일 때(RGB 또는 RGBA)
    if patch_data.shape[0] >= 3:
        patch_rgb = np.transpose(patch_data[:3, :, :], (1, 2, 0)) # RGB 채널만 추출
    else:
        return None, None, None # 채널이 부족할 경우 None 반환

    # 패치 내에서의 코너 좌표를 로컬 좌표로 변환
    local_corners = np.array(corners)
    local_corners[:, 0] -= min_x
    local_corners[:, 1] -= min_y
    
    # 패치(BGR), 로컬 코너 좌표, 패치의 좌상단 원본 좌표 반환
    return cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR), local_corners, (min_x, min_y)