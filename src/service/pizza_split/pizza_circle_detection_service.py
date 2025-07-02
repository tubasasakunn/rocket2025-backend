import cv2
import numpy as np
from typing import Tuple, Optional
try:
    from .pizza_segmentation_service import PizzaSegmentationService
except ImportError:
    from pizza_segmentation_service import PizzaSegmentationService


class PizzaCircleDetectionService:
    def __init__(self):
        self.segmentation_service = PizzaSegmentationService()
    
    def detect_circle_from_image(self, image_path: str) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        画像からピザの円を検出し、中心座標と半径を返す
        
        Args:
            image_path: 入力画像のパス
            
        Returns:
            ((center_x, center_y), radius) or None if no circle found
        """
        # セグメンテーションマスクを取得
        mask = self.segmentation_service.segment_pizza(image_path)
        
        # マスクから円を検出
        return self.detect_circle_from_mask(mask)
    
    def detect_circle_from_mask(self, mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        セグメンテーションマスクから円を検出
        
        Args:
            mask: バイナリマスク (255 = ピザ部分, 0 = 背景)
            
        Returns:
            ((center_x, center_y), radius) or None if no circle found
        """
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 最小外接円を計算
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # 整数に変換
        center = (int(center_x), int(center_y))
        radius = int(radius)
        
        return (center, radius)
    
    def draw_circle_on_image(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        画像に円を描画
        
        Args:
            image: 入力画像
            center: 円の中心座標 (x, y)
            radius: 円の半径
            
        Returns:
            円が描画された画像
        """
        # 画像をコピー
        result = image.copy()
        
        # 円を描画（緑色、線幅3）
        cv2.circle(result, center, radius, (0, 255, 0), 3)
        
        # 中心点を描画（赤色、半径5）
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        
        # 中心座標のテキストを追加
        text = f"Center: {center}, Radius: {radius}"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result


if __name__ == "__main__":
    import os
    import glob
    
    # サービスのインスタンスを作成
    service = PizzaCircleDetectionService()
    
    # resourceディレクトリ内のすべての画像を処理
    resource_dir = "resource"
    output_dir = "debug/preprocess"
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイルを取得
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(resource_dir, ext)))
    
    if not image_files:
        print(f"No image files found in {resource_dir}")
    else:
        print(f"Found {len(image_files)} image files:")
        for img_file in image_files:
            print(f"  {img_file}")
    
    # 各画像を処理
    for input_image_path in image_files:
        try:
            # ファイル名を取得
            filename = os.path.basename(input_image_path)
            name, ext = os.path.splitext(filename)
            output_image_path = os.path.join(output_dir, f"{name}_circle{ext}")
            
            print(f"\nProcessing: {input_image_path}")
            
            # 円を検出
            circle_info = service.detect_circle_from_image(input_image_path)
            
            if circle_info is None:
                print(f"  No circle detected in {filename}")
                continue
            
            center, radius = circle_info
            print(f"  Circle detected:")
            print(f"    Center: {center}")
            print(f"    Radius: {radius}")
            
            # 元の画像を読み込む
            original_image = cv2.imread(input_image_path)
            
            if original_image is None:
                print(f"  Error: Could not load image {input_image_path}")
                continue
            
            # 円をオーバーレイ
            result_image = service.draw_circle_on_image(original_image, center, radius)
            
            # 結果を保存
            cv2.imwrite(output_image_path, result_image)
            print(f"  Result saved to: {output_image_path}")
            
        except Exception as e:
            print(f"  Error processing {input_image_path}: {e}")
    
    print("\nProcessing completed.")