#!/usr/bin/env python
# coding: utf-8
"""
複数画像のピザ・サラミ面積の標準偏差を計算するサービス
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from src.service.pizza_split.score import PizzaScoreCalculator


class MultiImageScoreCalculator:
    """複数画像の面積統計を計算するクラス"""
    
    def __init__(self):
        """初期化"""
        self.calculator = PizzaScoreCalculator()
    
    def calculate_area_statistics(self, image_paths: List[str], 
                                use_preprocessing: bool = True,
                                use_ratio: bool = False,
                                debug: bool = False) -> Dict:
        """
        複数画像のピザ・サラミ面積の統計を計算
        
        Args:
            image_paths: 画像パスのリスト
            use_preprocessing: 前処理（楕円→円変換）を使用するか
            use_ratio: 比率（0-1）を使用するか、Falseの場合はピクセル数を使用
            debug: デバッグモード
            
        Returns:
            {
                'pizza_areas': List[float],              # 各画像のピザ面積
                'salami_areas': List[float],             # 各画像のサラミ面積
                'salami_in_pizza_areas': List[float],    # 各画像のピザ内サラミ面積
                'pizza_area_mean': float,                # ピザ面積の平均
                'pizza_area_std': float,                 # ピザ面積の標準偏差
                'salami_area_mean': float,               # サラミ面積の平均
                'salami_area_std': float,                # サラミ面積の標準偏差
                'salami_in_pizza_mean': float,           # ピザ内サラミ面積の平均
                'salami_in_pizza_std': float,            # ピザ内サラミ面積の標準偏差
                'std_sum': float,                        # ピザとサラミの標準偏差の合計
                'std_sum_all': float,                    # 全標準偏差の合計
                'num_images': int,                       # 処理した画像数
                'failed_images': List[str],              # 処理に失敗した画像
                'use_ratio': bool,                       # 比率を使用したか
                'details': List[Dict]                    # 各画像の詳細情報
            }
        """
        pizza_areas = []
        salami_areas = []
        salami_in_pizza_areas = []
        failed_images = []
        details = []
        
        if debug:
            print(f"[DEBUG] {len(image_paths)}枚の画像を処理します")
            print(f"[DEBUG] 使用する値: {'比率' if use_ratio else 'ピクセル数'}")
        
        # 各画像を処理
        for i, image_path in enumerate(image_paths):
            try:
                if debug:
                    print(f"\n[DEBUG] 処理中 ({i+1}/{len(image_paths)}): {image_path}")
                
                # 面積を計算
                result = self.calculator.calculate_areas(
                    image_path,
                    use_preprocessing=use_preprocessing,
                    debug=False,
                    save_masks=False
                )
                
                # 使用する値を選択
                if use_ratio:
                    pizza_area = result['pizza_area_ratio']
                    salami_area = result['salami_area_ratio']
                    salami_in_pizza = result['salami_in_pizza_ratio']
                else:
                    pizza_area = result['pizza_area_pixels']
                    salami_area = result['salami_area_pixels']
                    salami_in_pizza = result['salami_in_pizza_area_pixels']
                
                pizza_areas.append(pizza_area)
                salami_areas.append(salami_area)
                salami_in_pizza_areas.append(salami_in_pizza)
                
                # 詳細情報を保存
                detail = {
                    'image_path': str(image_path),
                    'pizza_area': pizza_area,
                    'salami_area': salami_area,
                    'salami_in_pizza_area': salami_in_pizza,
                    'image_shape': result['image_shape'],
                    'preprocessing_applied': result['preprocessing_applied']
                }
                details.append(detail)
                
                if debug:
                    if use_ratio:
                        print(f"  ピザ面積: {pizza_area:.1%}")
                        print(f"  サラミ面積: {salami_area:.1%}")
                        print(f"  ピザ内サラミ: {salami_in_pizza:.1%}")
                    else:
                        print(f"  ピザ面積: {pizza_area:,} pixels")
                        print(f"  サラミ面積: {salami_area:,} pixels")
                        print(f"  ピザ内サラミ: {salami_in_pizza:,} pixels")
                
            except Exception as e:
                if debug:
                    print(f"  エラー: {e}")
                failed_images.append(str(image_path))
                continue
        
        # 統計量を計算
        if len(pizza_areas) == 0:
            raise ValueError("有効な画像が1枚もありません")
        
        # NumPy配列に変換
        pizza_areas_np = np.array(pizza_areas)
        salami_areas_np = np.array(salami_areas)
        salami_in_pizza_areas_np = np.array(salami_in_pizza_areas)
        
        # 平均と標準偏差を計算
        pizza_mean = np.mean(pizza_areas_np)
        pizza_std = np.std(pizza_areas_np, ddof=1) if len(pizza_areas) > 1 else 0.0
        
        salami_mean = np.mean(salami_areas_np)
        salami_std = np.std(salami_areas_np, ddof=1) if len(salami_areas) > 1 else 0.0
        
        salami_in_pizza_mean = np.mean(salami_in_pizza_areas_np)
        salami_in_pizza_std = np.std(salami_in_pizza_areas_np, ddof=1) if len(salami_in_pizza_areas) > 1 else 0.0
        
        # 標準偏差の合計
        std_sum = pizza_std + salami_std
        std_sum_all = pizza_std + salami_std + salami_in_pizza_std
        
        if debug:
            print(f"\n[DEBUG] 統計結果:")
            print(f"  処理成功: {len(pizza_areas)}枚")
            print(f"  処理失敗: {len(failed_images)}枚")
            unit = "%" if use_ratio else "pixels"
            print(f"  ピザ面積 - 平均: {pizza_mean:.4f}{unit}, 標準偏差: {pizza_std:.4f}{unit}")
            print(f"  サラミ面積 - 平均: {salami_mean:.4f}{unit}, 標準偏差: {salami_std:.4f}{unit}")
            print(f"  標準偏差の合計: {std_sum:.4f}{unit}")
        
        return {
            'pizza_areas': pizza_areas,
            'salami_areas': salami_areas,
            'salami_in_pizza_areas': salami_in_pizza_areas,
            'pizza_area_mean': float(pizza_mean),
            'pizza_area_std': float(pizza_std),
            'salami_area_mean': float(salami_mean),
            'salami_area_std': float(salami_std),
            'salami_in_pizza_mean': float(salami_in_pizza_mean),
            'salami_in_pizza_std': float(salami_in_pizza_std),
            'std_sum': float(std_sum),
            'std_sum_all': float(std_sum_all),
            'num_images': len(pizza_areas),
            'failed_images': failed_images,
            'use_ratio': use_ratio,
            'details': details
        }
    
    def calculate_std_sum(self, image_paths: List[str], 
                         use_preprocessing: bool = True,
                         use_ratio: bool = False) -> float:
        """
        標準偏差の合計のみを返すシンプルなメソッド
        
        Args:
            image_paths: 画像パスのリスト
            use_preprocessing: 前処理を使用するか
            use_ratio: 比率を使用するか
            
        Returns:
            ピザ面積とサラミ面積の標準偏差の合計
        """
        result = self.calculate_area_statistics(
            image_paths, 
            use_preprocessing=use_preprocessing,
            use_ratio=use_ratio,
            debug=False
        )
        return result['std_sum']
    
    def save_statistics_report(self, image_paths: List[str], 
                             output_path: str = "result/score/statistics_report.json",
                             use_preprocessing: bool = True,
                             use_ratio: bool = False,
                             debug: bool = False) -> str:
        """
        統計レポートを保存
        
        Args:
            image_paths: 画像パスのリスト
            output_path: 出力ファイルパス
            use_preprocessing: 前処理を使用するか
            use_ratio: 比率を使用するか
            debug: デバッグモード
            
        Returns:
            保存されたファイルパス
        """
        # 統計を計算
        stats = self.calculate_area_statistics(
            image_paths,
            use_preprocessing=use_preprocessing,
            use_ratio=use_ratio,
            debug=debug
        )
        
        # 出力ディレクトリを作成
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # レポートを作成
        report = {
            'summary': {
                'num_images': stats['num_images'],
                'failed_images': len(stats['failed_images']),
                'use_preprocessing': use_preprocessing,
                'use_ratio': use_ratio,
                'unit': 'ratio' if use_ratio else 'pixels'
            },
            'statistics': {
                'pizza_area': {
                    'mean': stats['pizza_area_mean'],
                    'std': stats['pizza_area_std'],
                    'values': stats['pizza_areas']
                },
                'salami_area': {
                    'mean': stats['salami_area_mean'],
                    'std': stats['salami_area_std'],
                    'values': stats['salami_areas']
                },
                'salami_in_pizza_area': {
                    'mean': stats['salami_in_pizza_mean'],
                    'std': stats['salami_in_pizza_std'],
                    'values': stats['salami_in_pizza_areas']
                }
            },
            'std_sum': {
                'pizza_salami': stats['std_sum'],
                'all': stats['std_sum_all']
            },
            'details': stats['details'],
            'failed_images': stats['failed_images']
        }
        
        # JSONとして保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # テキストレポートも作成
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== Pizza Multi-Image Statistics Report ===\n\n")
            f.write(f"処理画像数: {stats['num_images']}枚\n")
            f.write(f"失敗画像数: {len(stats['failed_images'])}枚\n")
            f.write(f"前処理: {'あり' if use_preprocessing else 'なし'}\n")
            f.write(f"使用単位: {'比率' if use_ratio else 'ピクセル数'}\n\n")
            
            f.write("【統計結果】\n")
            unit = "" if use_ratio else " pixels"
            
            f.write(f"\nピザ面積:\n")
            f.write(f"  平均: {stats['pizza_area_mean']:.4f}{unit}\n")
            f.write(f"  標準偏差: {stats['pizza_area_std']:.4f}{unit}\n")
            
            f.write(f"\nサラミ面積:\n")
            f.write(f"  平均: {stats['salami_area_mean']:.4f}{unit}\n")
            f.write(f"  標準偏差: {stats['salami_area_std']:.4f}{unit}\n")
            
            f.write(f"\nピザ内サラミ面積:\n")
            f.write(f"  平均: {stats['salami_in_pizza_mean']:.4f}{unit}\n")
            f.write(f"  標準偏差: {stats['salami_in_pizza_std']:.4f}{unit}\n")
            
            f.write(f"\n【標準偏差の合計】\n")
            f.write(f"  ピザ＋サラミ: {stats['std_sum']:.4f}{unit}\n")
            f.write(f"  全項目: {stats['std_sum_all']:.4f}{unit}\n")
            
            if stats['failed_images']:
                f.write(f"\n【処理失敗画像】\n")
                for img in stats['failed_images']:
                    f.write(f"  - {img}\n")
        
        print(f"統計レポートを保存しました:")
        print(f"  - JSON: {output_path}")
        print(f"  - Text: {txt_path}")
        
        return str(output_path)


def main():
    """テスト用メイン関数"""
    # 計算機を初期化
    calc = MultiImageScoreCalculator()
    
    # テスト画像セット
    test_sets = {
        'all_pizza': ['resource/pizza1.jpg', 'resource/pizza2.jpg'],
        'cutted': ['resource/cutted.jpeg', 'resource/cutted2.jpeg'],
        'mixed': ['resource/pizza1.jpg', 'resource/pizza2.jpg', 
                  'resource/cutted.jpeg', 'resource/cutted2.jpeg']
    }
    
    print("=== Multi-Image Score Calculator Test ===\n")
    
    for set_name, image_paths in test_sets.items():
        # 画像の存在確認
        valid_paths = [p for p in image_paths if Path(p).exists()]
        if not valid_paths:
            print(f"\n{set_name}: 画像が見つかりません")
            continue
        
        print(f"\n{'='*60}")
        print(f"セット: {set_name} ({len(valid_paths)}枚)")
        print('='*60)
        
        # ピクセル数での計算
        print("\n【ピクセル数ベース】")
        try:
            result_pixels = calc.calculate_area_statistics(
                valid_paths,
                use_preprocessing=True,
                use_ratio=False,
                debug=True
            )
            
            print(f"\n標準偏差の合計（ピザ＋サラミ）: {result_pixels['std_sum']:,.0f} pixels")
            
        except Exception as e:
            print(f"エラー: {e}")
        
        # 比率での計算
        print("\n【比率ベース】")
        try:
            result_ratio = calc.calculate_area_statistics(
                valid_paths,
                use_preprocessing=True,
                use_ratio=True,
                debug=True
            )
            
            print(f"\n標準偏差の合計（ピザ＋サラミ）: {result_ratio['std_sum']:.4f}")
            
        except Exception as e:
            print(f"エラー: {e}")
        
        # レポート保存
        if set_name == 'mixed':
            print("\n統計レポートを保存中...")
            report_path = calc.save_statistics_report(
                valid_paths,
                output_path=f"result/score/statistics_{set_name}.json",
                use_preprocessing=True,
                use_ratio=True,
                debug=False
            )


if __name__ == "__main__":
    main()