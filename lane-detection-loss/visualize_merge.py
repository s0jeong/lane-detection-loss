# visualize_obb.py
# JSON 결과를 OBB 기준으로 시각화 후 JPG 저장

import json
import cv2
import numpy as np
import tifffile
import os

# 클래스 매핑
CLASS_ID_TO_NAME = {
    0: 'lane_line', 1: 'stop_line',
    2: 'arrow', 3: 'crosswalk', 4: 'diamond',
    5: 'zigzag', 6: 'center_line', 7: 'triangle',
    8: 'safety_zone', 9: 'lettering', 10: 'edge_line',
    11: 'bus_stop', 12: 'speed_bump'
}

# 손실률 분석 제외 클래스
NO_DAMAGE_ANALYSIS_CLASSES = ['center_line', 'safety_zone', 'edge_line', 'bus_stop', 'speed_bump']

# 입력/출력 경로
tif_path = "/home/dromii4/lane/road_cropped.tif"
json_path = "/home/dromii4/lane/lane-detection-loss/results/final_analysis_results_1205.json"
output_path = "/home/dromii4/lane/lane-detection-loss/results/visualize_1215.jpg"

# 이미지 로드
image_rgb = tifffile.imread(tif_path)
if image_rgb is None:
    raise ValueError(f"❌ Failed to load image from {tif_path}")

if len(image_rgb.shape) == 2:
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
elif image_rgb.shape[2] == 4:
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# JSON 로드
try:
    with open(json_path, "r") as f:
        annotations = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ JSON file not found at: {json_path}")
except json.JSONDecodeError:
    raise ValueError(f"❌ Failed to decode JSON from: {json_path}")

# OBB 시각화
for ann in annotations:
    cls_name = ann.get("class_name")
    if not cls_name:
        continue

    # 손실률 로딩 및 안전 처리
    damage_ratio = ann.get("damage_percent")
    if damage_ratio is None:
        continue
    try:
        damage_ratio = float(damage_ratio)
    except:
        continue

    # 0.5 이하인 경우 → 시각화 X
    if damage_ratio <= 0.5:
        continue

    # 분석 제외 클래스는 시각화 X
    if cls_name in NO_DAMAGE_ANALYSIS_CLASSES:
        continue

    # polygon points
    if not ann.get("segmentation_pixel"):
        continue
    pts = np.array(ann["segmentation_pixel"], np.int32).reshape((-1, 1, 2))

    # 색상 기준 변경
    if 0.5 < damage_ratio < 10:
        color = (0, 255, 0)          # 초록
    elif 10 <= damage_ratio <= 20:
        color = (0, 165, 255)        # 주황
    else:  # damage_ratio > 20
        color = (0, 0, 255)          # 빨강

    # 폴리라인 & 텍스트
    try:
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)
        text = f"{damage_ratio:.1f}%"
        cv2.putText(image_bgr, text, tuple(pts[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    except Exception as e:
        print(f"⚠️ Failed to draw annotation for {cls_name}: {e}")


# 결과 저장
try:
    cv2.imwrite(output_path, image_bgr)
    print(f"✅ Visualization saved at: {output_path}")
except Exception as e:
    print(f"❌ Failed to save JPG image: {e}")
