#!/usr/bin/env python3
"""
Corrected debug script that properly imports and uses existing services.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all services from the pizza_split directory
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


if __name__ == "__main__":
    # サービス初期化
    salami_service = SalamiSegmentationService()
    pizza_service = PizzaSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    # プロジェクトルートからの相対パスを設定
    project_root = Path(__file__).parent.parent.parent  # src/test -> src -> root
    resource_dir = project_root / "resource"
    output_dir = project_root / "debug" / "salami_segment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        exit(1)
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\nデバッグモードで実行します...\n")
    
    # 各画像を処理
    for image_path in image_files:
        print(f"\n{image_path.name}を処理中...")
        
        try:
            # 1. 前処理
            temp_preprocessed_path = output_dir / f"temp_preprocessed_{image_path.stem}.jpg"
            preprocessed_image, info = preprocess_service.preprocess_pizza_image(
                str(image_path), str(temp_preprocessed_path)
            )
            
            if info['is_transformed']:
                print(f"  楕円変換を適用しました")
                processed_image_path = str(temp_preprocessed_path)
            else:
                print("  変換は不要でした")
                processed_image_path = str(image_path)
            
            # 2. ピザマスク取得
            pizza_mask = pizza_service.segment_pizza(processed_image_path)
            
            # 3. サラミセグメンテーション実行
            salami_mask = salami_service.segment_salami(processed_image_path, pizza_mask)
            
            # 4. 表示用画像設定
            if info['is_transformed']:
                display_image = preprocessed_image.copy()
            else:
                display_image = cv2.imread(str(image_path))
            
            # 5. オーバーレイ可視化作成
            overlay = display_image.copy()
            overlay[salami_mask == 255] = [0, 0, 255]  # サラミ領域を赤で表示
            result_overlay = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
            
            # 6. 結果保存
            base_name = image_path.stem
            
            # 前処理済み画像
            preprocessed_save_path = output_dir / f"{base_name}_前処理済み.jpg"
            cv2.imwrite(str(preprocessed_save_path), display_image)
            
            # ピザマスク
            pizza_mask_path = output_dir / f"{base_name}_ピザマスク.png"
            cv2.imwrite(str(pizza_mask_path), pizza_mask)
            
            # サラミマスク
            salami_mask_path = output_dir / f"{base_name}_サラミマスク.png"
            cv2.imwrite(str(salami_mask_path), salami_mask)
            
            # オーバーレイ画像
            overlay_path = output_dir / f"{base_name}_オーバーレイ.jpg"
            cv2.imwrite(str(overlay_path), result_overlay)
            
            # 一時ファイル削除
            if temp_preprocessed_path.exists():
                temp_preprocessed_path.unlink()
            
            # 統計情報表示
            print(f"  統計情報:")
            print(f"    ピザマスクピクセル数: {np.sum(pizza_mask == 255):,}")
            print(f"    サラミマスクピクセル数: {np.sum(salami_mask == 255):,}")
            
            print(f"  結果をdebug/salami_segment/に保存しました:")
            print(f"    - 前処理済み画像: {preprocessed_save_path.name}")
            print(f"    - ピザマスク: {pizza_mask_path.name}")
            print(f"    - サラミマスク: {salami_mask_path.name}")
            print(f"    - オーバーレイ: {overlay_path.name}")
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n処理完了。debug/salami_segment/で結果を確認してください。")