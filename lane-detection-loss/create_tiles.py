import os
import cv2
import json
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window

# ------------------ 사용자 설정 ------------------
SOURCE_IMAGE_PATH = "/home/dromii4/lane/tile_0613_crack_GSD92.tif"  # 원본 GeoTIFF
OUTPUT_DIR = "/home/dromii4/lane/tiles_GSD92"  # 타일 저장 디렉토리
TILE_WIDTH = 1280
TILE_HEIGHT = 1280
OVERLAP_PERCENT = 30  # % 단위
# --------------------------------------------------

# --- .aug.xml 템플릿 ---
XML_TEMPLATE = """
<PAMDataset>
 <SRS dataAxisToSRSAxisMapping="2,1">GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]</SRS>
 <GeoTransform> {geotransform} </GeoTransform>
 <Metadata domain="IMAGE_STRUCTURE">
    <MDI key="INTERLEAVE">PIXEL</MDI>
 </Metadata>
</PAMDataset>
"""

def create_tiles_with_xml():
        tiles_path = os.path.join(OUTPUT_DIR, "tiles")
        os.makedirs(tiles_path, exist_ok=True)
        print(f"📂 타일 저장 위치: {tiles_path}")

        overlap_w = int(TILE_WIDTH * (OVERLAP_PERCENT / 100))
        overlap_h = int(TILE_HEIGHT * (OVERLAP_PERCENT / 100))
        stride_w = TILE_WIDTH - overlap_w
        stride_h = TILE_HEIGHT - overlap_h

        if stride_w <= 0 or stride_h <= 0:
                print("❌ 오류: 중첩 비율이 100% 이상일 수 없습니다.")
                return

        print(f"📊 타일 크기: {TILE_WIDTH}x{TILE_HEIGHT}, 중첩: {OVERLAP_PERCENT}%, 이동 거리(가로/세로): {stride_w}px / {stride_h}px")

        tile_coordinates = {}  # 타일 생성 중 임시로 전체 경로를 키로 저장

        with rasterio.open(SOURCE_IMAGE_PATH) as dataset:
                width, height = dataset.width, dataset.height
                print(f"🖼️ 원본 이미지 크기: {width}x{height}")

                for y_offset in tqdm(range(0, height, stride_h), desc="타일 생성 중"):
                        for x_offset in range(0, width, stride_w):
                                win_width = min(TILE_WIDTH, width - x_offset)
                                win_height = min(TILE_HEIGHT, height - y_offset)

                                if win_width < TILE_WIDTH * 0.5 or win_height < TILE_HEIGHT * 0.5:
                                        continue

                                window = Window(x_offset, y_offset, win_width, win_height)
                                tile_data = dataset.read((1, 2, 3), window=window)
                                if not tile_data.any():
                                        continue

                                tile_image_rgb = np.transpose(tile_data, (1, 2, 0))
                                tile_image_bgr = cv2.cvtColor(tile_image_rgb, cv2.COLOR_RGB2BGR)

                                # PNG 저장
                                tile_filename = f"tile_{y_offset}_{x_offset}.png"
                                png_output_path = os.path.join(tiles_path, tile_filename)
                                cv2.imwrite(png_output_path, tile_image_bgr)

                                # .aug.xml 생성
                                tile_transform = rasterio.windows.transform(window, dataset.transform)
                                gt_str = f"{tile_transform.c}, {tile_transform.a}, {tile_transform.b}, {tile_transform.f}, {tile_transform.d}, {tile_transform.e}"
                                xml_content = XML_TEMPLATE.format(geotransform=gt_str)
                                xml_output_path = f"{png_output_path}.aug.xml"
                                with open(xml_output_path, 'w') as f:
                                        f.write(xml_content)

                                # 좌표 기록 (기존 로직 유지)
                                json_key = os.path.join(os.path.basename(OUTPUT_DIR), "tiles", tile_filename).replace("\\", "/")
                                tile_coordinates[json_key] = (x_offset, y_offset)

        # --- 1단계: 전체 경로 키를 가진 JSON 저장 ---
        temp_coords_path = os.path.join(OUTPUT_DIR, "tile_coordinates_temp.json")
        with open(temp_coords_path, 'w', encoding='utf-8') as f:
                json.dump(tile_coordinates, f, indent=4)

        print(f"\n🎉 타일 생성 완료! 총 {len(tile_coordinates)}개의 (.png + .aug.xml) 쌍 생성")

        # --- 2단계: JSON 키 정리  ---
        print("\n🔨 YOLO 탐지용 JSON 좌표 정리 중...")
        
        new_data = {}
        for k, v in tile_coordinates.items():
                filename = k.split("/")[-1] 
                new_data[filename] = v

        # 최종 YOLO용 좌표 파일 저장
        final_coords_path = os.path.join(OUTPUT_DIR, "tile_coordinates.json")
        with open(final_coords_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)

        print(f"🗺️ 최종 좌표 정보 파일: {final_coords_path}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_coords_path):
                os.remove(temp_coords_path)
        
        print("✅ JSON 좌표 정리 완료.")

if __name__ == "__main__":
        create_tiles_with_xml()
