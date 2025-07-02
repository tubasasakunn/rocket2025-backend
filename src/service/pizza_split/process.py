#!/usr/bin/env python
# coding: utf-8
"""
ピザ分割処理パイプライン
1. 前処理（楕円→円形変換）
2. ピザとサラミの座標・半径取得
3. 移動ナイフ法による分割計算
4. SVGオーバーレイ出力
"""

import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import svgwrite

from preprocess import PreprocessService
from postprocess import PostprocessService
from pizza_circle_detection_service import PizzaCircleDetectionService
from salami_circle_detection_service import SalamiCircleDetectionService
from salami_devide import PizzaDivider


class PizzaProcessor:
    """ピザ分割処理メインクラス"""
    
    def __init__(self, output_dir="result/process"):
        """
        初期化
        
        Args:
            output_dir: 結果出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # サービスのインスタンス化
        self.preprocess_service = PreprocessService()
        self.postprocess_service = PostprocessService()
        self.pizza_detector = PizzaCircleDetectionService()
        self.salami_detector = SalamiCircleDetectionService()
        
    def process_image(self, image_path, n_pieces=4, debug=False, return_svg_only=False, quiet=False):
        """
        画像を処理してピザを分割
        
        Args:
            image_path: 入力画像パス
            n_pieces: 分割するピース数
            debug: デバッグ出力有効化
            return_svg_only: SVGのみを生成して早期リターンするか
            quiet: 標準出力を抑制するか
            
        Returns:
            dict: 処理結果
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        if not quiet:
            print(f"\n=== ピザ分割処理開始: {image_path.name} ===")
        
        # 1. 前処理（楕円→円形変換）
        if not quiet:
            print("\n1. 画像前処理...")
        preprocessed_path = self.output_dir / f"preprocessed_{self.timestamp}_{image_path.name}"
        preprocessed_img, preprocess_info = self.preprocess_service.preprocess_pizza_image(
            str(image_path), 
            str(preprocessed_path),
            is_debug=debug
        )
        if not quiet:
            print(f"   前処理完了: {preprocessed_path.name}")
        
        # 2. ピザの円検出
        if not quiet:
            print("\n2. ピザ円検出...")
        pizza_center, pizza_radius = self.pizza_detector.detect_circle_from_image(
            str(preprocessed_path)
        )
        if not quiet:
            print(f"   ピザ中心: ({pizza_center[0]:.1f}, {pizza_center[1]:.1f})")
            print(f"   ピザ半径: {pizza_radius:.1f}px")
        
        # 3. サラミの円検出
        if not quiet:
            print("\n3. サラミ円検出...")
        salami_circles = self.salami_detector.detect_salami_circles(
            str(preprocessed_path)
        )
        if not quiet:
            print(f"   検出されたサラミ数: {len(salami_circles)}")
        
        # 4. 座標を正規化（ピザ中心を原点、半径を1に）
        if not quiet:
            print("\n4. 座標正規化...")
        normalized_salami = []
        salami_radii = []
        
        for (cx, cy), r in salami_circles:
            # ピザ中心を基準に正規化
            norm_x = (cx - pizza_center[0]) / pizza_radius
            norm_y = (cy - pizza_center[1]) / pizza_radius
            norm_r = r / pizza_radius
            normalized_salami.append((norm_x, norm_y))
            salami_radii.append(norm_r)
        
        # サラミ半径の平均値を計算
        avg_salami_radius = np.mean(salami_radii) if salami_radii else 0.1
        if not quiet:
            print(f"   正規化サラミ半径（平均）: {avg_salami_radius:.4f}")
        
        # 5. 移動ナイフ法で分割
        if not quiet:
            print(f"\n5. 移動ナイフ法による{n_pieces}分割...")
        divider = PizzaDivider(
            R_pizza=1.0,
            R_salami=avg_salami_radius,
            m=len(salami_circles),
            n=n_pieces,
            N_Monte=50000,  # 高速化のため点数を調整
            seed=42,
            isDebug=debug
        )
        
        # モンテカルロ点生成
        divider.generate_monte_carlo_points()
        
        # 実際のサラミ位置を設定
        divider.centers = np.array(normalized_salami)
        
        # 各モンテカルロ点がサラミ上にあるかチェック
        divider.on_salami = np.zeros(divider.N_Monte, dtype=bool)
        for cx, cy in divider.centers:
            divider.on_salami |= (
                (divider.px - cx)**2 + (divider.py - cy)**2 <= avg_salami_radius**2
            )
        
        # 目標値計算と分割実行
        divider.calculate_targets()
        divider.divide_pizza()
        
        # quietモードでは結果出力をスキップ
        if quiet:
            divider.print_results = lambda: None
        
        # 6. SVGオーバーレイ生成（前処理済み画像用）
        if not quiet:
            print("\n6. SVGオーバーレイ生成...")
        svg_preprocessed_path = self.create_svg_overlay(
            preprocessed_path,
            pizza_center,
            pizza_radius,
            salami_circles,
            divider.cut_edges,
            n_pieces
        )
        
        # 7. 元の画像用のSVGオーバーレイ生成
        if not quiet:
            print("\n7. 元の画像用SVGオーバーレイ生成...")
        svg_original_path = self.output_dir / f"overlay_original_{self.timestamp}.svg"
        self.postprocess_service.create_svg_overlay_on_original(
            str(image_path),
            pizza_center,
            pizza_radius,
            salami_circles,
            divider.cut_edges,
            preprocess_info,
            str(svg_original_path),
            n_pieces
        )
        
        # SVGのみ返すモードの場合は早期リターン
        if return_svg_only:
            # SVGコンテンツを直接生成
            svg_content = self.postprocess_service.create_svg_content_on_original(
                str(image_path),
                pizza_center,
                pizza_radius,
                salami_circles,
                divider.cut_edges,
                preprocess_info,
                n_pieces
            )
            
            if not quiet:
                print(f"\n=== 処理完了（SVGのみ） ===")
                print(f"元画像用SVG: 生成済み")
            
            return {
                'svg_original': str(svg_original_path),
                'svg_content': svg_content,
                'pizza_center': pizza_center,
                'pizza_radius': pizza_radius,
                'salami_circles': salami_circles,
                'cut_edges': divider.cut_edges,
                'pieces': divider.pieces,
                'preprocess_info': preprocess_info
            }
        
        # 8. 元の画像にオーバーレイしたPNG画像生成
        if not quiet:
            print("\n8. 元の画像にオーバーレイしたPNG画像生成...")
        result_original_path = self.output_dir / f"result_original_{self.timestamp}.png"
        self.postprocess_service.create_overlay_image_on_original(
            str(image_path),
            pizza_center,
            pizza_radius,
            salami_circles,
            divider.cut_edges,
            preprocess_info,
            str(result_original_path),
            n_pieces
        )
        
        # 9. 結果画像生成
        if not quiet:
            print("\n9. 結果画像生成...")
        result_path = self.create_result_image(
            preprocessed_path,
            pizza_center,
            pizza_radius,
            salami_circles,
            divider.cut_edges,
            divider.pieces,
            divider.px,
            divider.py
        )
        
        if not quiet:
            print(f"\n=== 処理完了 ===")
            print(f"前処理画像: {preprocessed_path}")
            print(f"前処理済み画像用SVG: {svg_preprocessed_path}")
            print(f"元画像用SVG: {svg_original_path}")
            print(f"元画像にオーバーレイしたPNG: {result_original_path}")
            print(f"結果画像: {result_path}")
        
        return {
            'preprocessed_image': str(preprocessed_path),
            'svg_preprocessed': str(svg_preprocessed_path),
            'svg_original': str(svg_original_path),
            'result_image': str(result_path),
            'result_original_image': str(result_original_path),
            'pizza_center': pizza_center,
            'pizza_radius': pizza_radius,
            'salami_circles': salami_circles,
            'cut_edges': divider.cut_edges,
            'pieces': divider.pieces,
            'preprocess_info': preprocess_info
        }
    
    def create_svg_overlay(self, image_path, pizza_center, pizza_radius, 
                          salami_circles, cut_edges, n_pieces):
        """
        SVGオーバーレイファイルを作成
        
        Args:
            image_path: 背景画像パス
            pizza_center: ピザの中心座標
            pizza_radius: ピザの半径
            salami_circles: サラミの円情報
            cut_edges: カット線の端点情報
            n_pieces: ピース数
            
        Returns:
            svg_path: 生成されたSVGファイルパス
        """
        # 画像サイズ取得
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        
        # SVG作成
        svg_path = self.output_dir / f"overlay_{self.timestamp}.svg"
        dwg = svgwrite.Drawing(str(svg_path), size=(width, height))
        
        # 背景画像を埋め込み
        dwg.add(dwg.image(
            href=str(image_path),
            insert=(0, 0),
            size=(width, height)
        ))
        
        # ピザの円を描画
        dwg.add(dwg.circle(
            center=pizza_center,
            r=pizza_radius,
            stroke='brown',
            stroke_width=3,
            fill='none'
        ))
        
        # サラミを描画
        for (cx, cy), r in salami_circles:
            dwg.add(dwg.circle(
                center=(cx, cy),
                r=r,
                stroke='darkred',
                stroke_width=2,
                fill='indianred',
                fill_opacity=0.7
            ))
        
        # カット線を描画
        for (e1, e2) in cut_edges:
            # 正規化座標から画像座標に変換
            x1 = pizza_center[0] + e1[0] * pizza_radius
            y1 = pizza_center[1] + e1[1] * pizza_radius
            x2 = pizza_center[0] + e2[0] * pizza_radius
            y2 = pizza_center[1] + e2[1] * pizza_radius
            
            dwg.add(dwg.line(
                start=(x1, y1),
                end=(x2, y2),
                stroke='black',
                stroke_width=4
            ))
        
        dwg.save()
        return svg_path
    
    def create_result_image(self, image_path, pizza_center, pizza_radius,
                           salami_circles, cut_edges, pieces, px, py):
        """
        結果画像を生成（Matplotlib使用）
        
        Returns:
            result_path: 生成された結果画像パス
        """
        # 画像読み込み
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 図を作成
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        
        # ピザの円
        pizza_circle = Circle(
            pizza_center, pizza_radius,
            fill=False, edgecolor='brown', linewidth=3
        )
        ax.add_patch(pizza_circle)
        
        # サラミ
        for (cx, cy), r in salami_circles:
            salami = Circle(
                (cx, cy), r,
                fill=True, facecolor='indianred', 
                edgecolor='darkred', alpha=0.8
            )
            ax.add_patch(salami)
        
        # カット線
        for (e1, e2) in cut_edges:
            x1 = pizza_center[0] + e1[0] * pizza_radius
            y1 = pizza_center[1] + e1[1] * pizza_radius
            x2 = pizza_center[0] + e2[0] * pizza_radius
            y2 = pizza_center[1] + e2[1] * pizza_radius
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3)
        
        # タイトル
        ax.set_title(f'Pizza Division Result ({len(pieces)} pieces)', fontsize=16)
        ax.axis('off')
        
        # 保存
        result_path = self.output_dir / f"result_{self.timestamp}.png"
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result_path


def main():
    """メイン関数"""
    # 処理対象の画像
    images = [
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    # プロセッサーを初期化
    processor = PizzaProcessor()
    
    # 各画像を処理
    for image_path in images:
        if Path(image_path).exists():
            print(f"\n{'='*50}")
            print(f"処理中: {image_path}")
            print(f"{'='*50}")
            
            try:
                # 4分割で処理
                result = processor.process_image(
                    image_path,
                    n_pieces=4,
                    debug=False
                )
                
                print("\n処理結果:")
                for key, value in result.items():
                    if key not in ['pieces', 'cut_edges', 'salami_circles']:
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"画像が見つかりません: {image_path}")


if __name__ == "__main__":
    main()