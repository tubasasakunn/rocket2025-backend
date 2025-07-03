#!/usr/bin/env python
# coding: utf-8
"""
cutted.jpegとcutted2.jpegの面積を計算
"""

from src.service.pizza_split.score import PizzaScoreCalculator
from pathlib import Path

def main():
    # スコア計算機を初期化
    calculator = PizzaScoreCalculator()
    
    # 処理する画像
    images = [
        "resource/cutted.jpeg",
        "resource/cutted2.jpeg"
    ]
    
    print("=== Cutted Pizza Area Calculation ===\n")
    
    for image_path in images:
        if not Path(image_path).exists():
            print(f"画像が見つかりません: {image_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"処理中: {image_path}")
        print('='*60)
        
        try:
            # 前処理なしで計算（既にカットされた画像なので）
            print("\n【前処理なし】")
            result_no_preprocess = calculator.calculate_areas(
                image_path,
                use_preprocessing=False,
                debug=False
            )
            
            print(f"画像サイズ: {result_no_preprocess['image_shape']}")
            print(f"総ピクセル数: {result_no_preprocess['total_image_pixels']:,}")
            print(f"\nピザ面積: {result_no_preprocess['pizza_area_pixels']:,} pixels ({result_no_preprocess['pizza_area_ratio']:.1%})")
            print(f"サラミ面積: {result_no_preprocess['salami_area_pixels']:,} pixels ({result_no_preprocess['salami_area_ratio']:.1%})")
            print(f"ピザ内サラミ面積: {result_no_preprocess['salami_in_pizza_area_pixels']:,} pixels")
            print(f"ピザ内でのサラミ割合: {result_no_preprocess['salami_in_pizza_ratio']:.1%}")
            
            # 前処理ありでも計算（比較のため）
            print("\n【前処理あり】")
            result_with_preprocess = calculator.calculate_areas(
                image_path,
                use_preprocessing=True,
                debug=False
            )
            
            print(f"画像サイズ: {result_with_preprocess['image_shape']}")
            print(f"総ピクセル数: {result_with_preprocess['total_image_pixels']:,}")
            print(f"\nピザ面積: {result_with_preprocess['pizza_area_pixels']:,} pixels ({result_with_preprocess['pizza_area_ratio']:.1%})")
            print(f"サラミ面積: {result_with_preprocess['salami_area_pixels']:,} pixels ({result_with_preprocess['salami_area_ratio']:.1%})")
            print(f"ピザ内サラミ面積: {result_with_preprocess['salami_in_pizza_area_pixels']:,} pixels")
            print(f"ピザ内でのサラミ割合: {result_with_preprocess['salami_in_pizza_ratio']:.1%}")
            
            if result_with_preprocess['preprocessing_applied']:
                print("\n※楕円→円変換が適用されました")
            else:
                print("\n※変換は適用されませんでした")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()