#!/usr/bin/env python
# coding: utf-8
"""
スコア計算サービス
ピザとサラミをセグメンテーションし、それぞれの面積を計算する
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json

from pizza_segmentation_service import PizzaSegmentationService
from salami_segmentation_service import SalamiSegmentationService
from preprocess import PreprocessService


class PizzaScoreCalculator:
    """ピザとサラミの面積を計算するクラス"""
    
    def __init__(self):
        """サービスの初期化"""
        self.pizza_service = PizzaSegmentationService()
        self.salami_service = SalamiSegmentationService()
        self.preprocess_service = PreprocessService()
        
    def calculate_areas(self, image_path: str, use_preprocessing: bool = True, 
                       debug: bool = False, save_masks: bool = False) -> Dict:
        """
        画像からピザとサラミの面積を計算
        
        Args:
            image_path: 入力画像のパス
            use_preprocessing: 前処理（楕円→円変換）を使用するか
            debug: デバッグモード
            save_masks: マスク画像を保存するか
            
        Returns:
            {
                'pizza_area_pixels': int,           # ピザの面積（ピクセル数）
                'salami_area_pixels': int,          # サラミの面積（ピクセル数）
                'salami_in_pizza_area_pixels': int, # ピザ内のサラミ面積（ピクセル数）
                'total_image_pixels': int,          # 画像全体のピクセル数
                'pizza_area_ratio': float,          # ピザの面積比率（0-1）
                'salami_area_ratio': float,         # サラミの面積比率（0-1）
                'salami_in_pizza_ratio': float,     # ピザ内でのサラミの割合（0-1）
                'image_shape': tuple,               # 画像のshape (height, width, channels)
                'preprocessing_applied': bool,       # 前処理が適用されたか
                'masks': {                          # マスク画像（save_masks=Trueの場合）
                    'pizza_mask': np.ndarray,
                    'salami_mask': np.ndarray,
                    'salami_in_pizza_mask': np.ndarray
                }
            }
        """
        # 画像パスの検証
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        if debug:
            print(f"[DEBUG] 画像を処理中: {image_path}")
        
        # 処理する画像パスを決定
        processed_image_path = str(image_path)
        preprocess_info = None
        
        if use_preprocessing:
            # 前処理を実行
            if debug:
                print("[DEBUG] 前処理を実行中...")
            temp_path = Path("temp") / f"preprocessed_{image_path.name}"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            _, preprocess_info = self.preprocess_service.preprocess_pizza_image(
                str(image_path), 
                str(temp_path),
                is_debug=debug
            )
            
            if preprocess_info['is_transformed']:
                processed_image_path = str(temp_path)
                if debug:
                    print("[DEBUG] 楕円→円変換を適用しました")
        
        # 画像を読み込んで基本情報を取得
        image = cv2.imread(processed_image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {processed_image_path}")
        
        image_shape = image.shape
        total_pixels = image_shape[0] * image_shape[1]
        
        if debug:
            print(f"[DEBUG] 画像サイズ: {image_shape}")
            print(f"[DEBUG] 総ピクセル数: {total_pixels:,}")
        
        # ピザのセグメンテーション
        if debug:
            print("[DEBUG] ピザをセグメンテーション中...")
        pizza_mask = self.pizza_service.segment_pizza(processed_image_path, isDebug=False)
        pizza_area_pixels = np.sum(pizza_mask == 255)
        
        if debug:
            print(f"[DEBUG] ピザ面積: {pizza_area_pixels:,} pixels")
        
        # サラミのセグメンテーション
        if debug:
            print("[DEBUG] サラミをセグメンテーション中...")
        salami_result = self.salami_service.segment_salami(
            processed_image_path, 
            pizza_mask=pizza_mask,
            isDebug=False
        )
        
        # サラミセグメンテーションの戻り値を処理
        if isinstance(salami_result, tuple):
            salami_mask, _ = salami_result
        else:
            salami_mask = salami_result
        
        salami_area_pixels = np.sum(salami_mask == 255)
        
        if debug:
            print(f"[DEBUG] サラミ面積: {salami_area_pixels:,} pixels")
        
        # ピザ内のサラミ面積を計算（AND演算）
        salami_in_pizza_mask = cv2.bitwise_and(salami_mask, pizza_mask)
        salami_in_pizza_area_pixels = np.sum(salami_in_pizza_mask == 255)
        
        if debug:
            print(f"[DEBUG] ピザ内のサラミ面積: {salami_in_pizza_area_pixels:,} pixels")
        
        # 面積比率を計算
        pizza_area_ratio = pizza_area_pixels / total_pixels
        salami_area_ratio = salami_area_pixels / total_pixels
        salami_in_pizza_ratio = salami_in_pizza_area_pixels / pizza_area_pixels if pizza_area_pixels > 0 else 0
        
        # 結果をまとめる
        result = {
            'pizza_area_pixels': int(pizza_area_pixels),
            'salami_area_pixels': int(salami_area_pixels),
            'salami_in_pizza_area_pixels': int(salami_in_pizza_area_pixels),
            'total_image_pixels': int(total_pixels),
            'pizza_area_ratio': float(pizza_area_ratio),
            'salami_area_ratio': float(salami_area_ratio),
            'salami_in_pizza_ratio': float(salami_in_pizza_ratio),
            'image_shape': image_shape,
            'preprocessing_applied': use_preprocessing and preprocess_info and preprocess_info['is_transformed']
        }
        
        # マスクを保存する場合
        if save_masks:
            result['masks'] = {
                'pizza_mask': pizza_mask,
                'salami_mask': salami_mask,
                'salami_in_pizza_mask': salami_in_pizza_mask
            }
        
        # 一時ファイルを削除
        if use_preprocessing and Path(processed_image_path) != image_path:
            try:
                Path(processed_image_path).unlink()
            except:
                pass
        
        return result
    
    def calculate_areas_batch(self, image_paths: list, use_preprocessing: bool = True,
                            debug: bool = False) -> Dict[str, Dict]:
        """
        複数の画像をバッチ処理
        
        Args:
            image_paths: 画像パスのリスト
            use_preprocessing: 前処理を使用するか
            debug: デバッグモード
            
        Returns:
            各画像の結果を含む辞書
        """
        results = {}
        
        for image_path in image_paths:
            try:
                if debug:
                    print(f"\n処理中: {image_path}")
                
                result = self.calculate_areas(
                    image_path,
                    use_preprocessing=use_preprocessing,
                    debug=debug,
                    save_masks=False
                )
                results[str(image_path)] = result
                
            except Exception as e:
                print(f"エラー: {image_path} - {e}")
                results[str(image_path)] = {'error': str(e)}
        
        return results
    
    def save_visualization(self, image_path: str, output_dir: str = "result/score",
                         use_preprocessing: bool = True) -> Dict[str, str]:
        """
        セグメンテーション結果を可視化して保存
        
        Args:
            image_path: 入力画像パス
            output_dir: 出力ディレクトリ
            use_preprocessing: 前処理を使用するか
            
        Returns:
            保存されたファイルパスの辞書
        """
        # 出力ディレクトリを作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 面積計算（マスクも取得）
        result = self.calculate_areas(
            image_path,
            use_preprocessing=use_preprocessing,
            debug=False,
            save_masks=True
        )
        
        # 元画像を読み込み
        image = cv2.imread(str(image_path))
        base_name = Path(image_path).stem
        
        saved_paths = {}
        
        # 1. ピザマスクを保存
        pizza_mask_path = output_dir / f"{base_name}_pizza_mask.png"
        cv2.imwrite(str(pizza_mask_path), result['masks']['pizza_mask'])
        saved_paths['pizza_mask'] = str(pizza_mask_path)
        
        # 2. サラミマスクを保存
        salami_mask_path = output_dir / f"{base_name}_salami_mask.png"
        cv2.imwrite(str(salami_mask_path), result['masks']['salami_mask'])
        saved_paths['salami_mask'] = str(salami_mask_path)
        
        # 3. ピザ内サラミマスクを保存
        salami_in_pizza_path = output_dir / f"{base_name}_salami_in_pizza_mask.png"
        cv2.imwrite(str(salami_in_pizza_path), result['masks']['salami_in_pizza_mask'])
        saved_paths['salami_in_pizza_mask'] = str(salami_in_pizza_path)
        
        # 4. オーバーレイ画像を作成
        overlay = image.copy()
        
        # ピザ領域を黄色でハイライト
        pizza_colored = np.zeros_like(image)
        pizza_colored[result['masks']['pizza_mask'] == 255] = [0, 255, 255]  # 黄色（BGR）
        overlay = cv2.addWeighted(overlay, 0.7, pizza_colored, 0.3, 0)
        
        # サラミ領域を赤色でハイライト
        salami_colored = np.zeros_like(image)
        salami_colored[result['masks']['salami_in_pizza_mask'] == 255] = [0, 0, 255]  # 赤色（BGR）
        overlay = cv2.addWeighted(overlay, 0.7, salami_colored, 0.3, 0)
        
        overlay_path = output_dir / f"{base_name}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        saved_paths['overlay'] = str(overlay_path)
        
        # 5. 結果をテキストファイルに保存
        info_path = output_dir / f"{base_name}_areas.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"画像: {image_path}\n")
            f.write(f"画像サイズ: {result['image_shape']}\n")
            f.write(f"総ピクセル数: {result['total_image_pixels']:,}\n")
            f.write(f"\n")
            f.write(f"ピザ面積: {result['pizza_area_pixels']:,} pixels ({result['pizza_area_ratio']:.1%})\n")
            f.write(f"サラミ面積: {result['salami_area_pixels']:,} pixels ({result['salami_area_ratio']:.1%})\n")
            f.write(f"ピザ内サラミ面積: {result['salami_in_pizza_area_pixels']:,} pixels\n")
            f.write(f"ピザ内でのサラミ割合: {result['salami_in_pizza_ratio']:.1%}\n")
            f.write(f"\n")
            f.write(f"前処理適用: {'Yes' if result['preprocessing_applied'] else 'No'}\n")
        saved_paths['info'] = str(info_path)
        
        # 6. JSON形式でも保存
        json_path = output_dir / f"{base_name}_areas.json"
        json_result = {k: v for k, v in result.items() if k != 'masks'}
        json_result['image_shape'] = list(json_result['image_shape'])  # tupleをlistに変換
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        saved_paths['json'] = str(json_path)
        
        return saved_paths


def main():
    """メイン関数"""
    # スコア計算機を初期化
    calculator = PizzaScoreCalculator()
    
    # テスト画像
    test_images = [
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    print("=== Pizza Score Calculator ===\n")
    
    # 各画像を処理
    for image_path in test_images:
        if not Path(image_path).exists():
            print(f"画像が見つかりません: {image_path}")
            continue
        
        print(f"\n処理中: {image_path}")
        print("-" * 50)
        
        try:
            # 面積を計算
            result = calculator.calculate_areas(
                image_path,
                use_preprocessing=True,
                debug=True
            )
            
            # 結果を表示
            print("\n【計算結果】")
            print(f"画像サイズ: {result['image_shape']}")
            print(f"総ピクセル数: {result['total_image_pixels']:,}")
            print(f"\nピザ面積: {result['pizza_area_pixels']:,} pixels ({result['pizza_area_ratio']:.1%})")
            print(f"サラミ面積: {result['salami_area_pixels']:,} pixels ({result['salami_area_ratio']:.1%})")
            print(f"ピザ内サラミ面積: {result['salami_in_pizza_area_pixels']:,} pixels")
            print(f"ピザ内でのサラミ割合: {result['salami_in_pizza_ratio']:.1%}")
            
            # 可視化を保存
            print("\n可視化を保存中...")
            saved_paths = calculator.save_visualization(image_path)
            print("保存されたファイル:")
            for key, path in saved_paths.items():
                print(f"  - {key}: {path}")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # バッチ処理の例
    print("\n\n=== バッチ処理の例 ===")
    batch_results = calculator.calculate_areas_batch(test_images, debug=False)
    
    for image_path, result in batch_results.items():
        print(f"\n{image_path}:")
        if 'error' in result:
            print(f"  エラー: {result['error']}")
        else:
            print(f"  ピザ面積: {result['pizza_area_ratio']:.1%}")
            print(f"  サラミ割合: {result['salami_in_pizza_ratio']:.1%}")


if __name__ == "__main__":
    main()