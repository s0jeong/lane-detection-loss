# config.py

# --- 1. 경로 설정 ---
SINGLE_TIF_PATH = 'lane-detection-loss-main/models/road_cropped.tif'
ORIGINAL_IMAGE_PATH = SINGLE_TIF_PATH
TILES_DIR = "lane-detection-loss-main/tiles"
YOLO_MODEL_PATH = "lane-detection-loss-main/models/yolov11L-best.pt"
SAM_CHECKPOINT_PATH = "lane-detection-loss-main/models/sam2.1_hiera_base_plus.pt"
SAM_MODEL_CONFIG = "lane-detection-loss-main/models/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
FINAL_OUTPUT_PATH = "lane-detection-loss-main/lane-detection-loss/results/final_analysis_results.json"

# --- 2. 탐지 및 NMS 파라미터 ---
DETECTION_CONF_THRESHOLD = 0.35 # 신뢰도 0.35 미만의 탐지는 초기 단계에서 무시
NMS_IOU_THRESHOLD = 0.4 # 40% 겹쳐야만 중복으로 판단
MERGE_THRESHOLD = 0.3

CLASS_SPECIFIC_NMS_IOU = {
    'crosswalk': 0.80
}

# --- 3. 손상도 분석 파라미터 ---
SAM_CLASSES = ['lane_line', 'stop_line', 'crosswalk']
BRIGHT_CONTOUR_CLASSES = ['arrow', 'diamond', 'zigzag', 'triangle', 'lettering']
NO_DAMAGE_ANALYSIS_CLASSES = ['center_line', 'safety_zone', 'edge_line', 'bus_stop', 'speed_bump']
EXCLUDED_CLASSES_FOR_DAMAGE = NO_DAMAGE_ANALYSIS_CLASSES

DAMAGE_CALCULATION_PARAMS = {
    'std_multiplier': 3.0,
    'min_tolerance': 10,
    'brightness_threshold': 100,
    'color_match_threshold': 80.0,
}
MAX_PIXEL_SAMPLE = 10000

# --- 4. 체크포인트 경로 설정 ---
CHECKPOINT_DIR = "lane-detection-loss-main/lane-detection-loss/results/checkpoints"
CHECKPOINT_STEP1_PATH = f"{CHECKPOINT_DIR}/step1_yolo_detections.json"
CHECKPOINT_STEP2_PATH = f"{CHECKPOINT_DIR}/step2_nms_detection.json"
CHECKPOINT_STEP3_PATH = f"{CHECKPOINT_DIR}/step3_merged_detections.json"
CHECKPOINT_PARTIAL_ANALYSIS_PATH = f"{CHECKPOINT_DIR}/partial_analysis_results.json"

# --- 5. 타일 관련 파라미터 ---
TILE_WIDTH = 1280
TILE_HEIGHT = 1280
TILE_OVERLAP_PERCENT = 30  # % 단위
