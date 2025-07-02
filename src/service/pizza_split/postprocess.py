#!/usr/bin/env python
# coding: utf-8
"""
後処理サービス
前処理で変換されたピザ画像の座標を元の画像座標系に逆変換する
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import svgwrite
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class PostprocessService:
    """前処理の逆変換を行うサービス"""
    
    def __init__(self):
        pass
    
    def inverse_transform_point(self, point: Tuple[float, float], preprocess_info: Dict) -> Tuple[float, float]:
        """
        正規化された座標を元の画像座標に逆変換
        
        Args:
            point: 正規化された座標 (x, y)
            preprocess_info: 前処理情報
            
        Returns:
            元の画像での座標 (x, y)
        """
        x, y = point
        
        # Step 1: 512x512からクロップ領域の座標に逆変換
        norm_params = preprocess_info['normalization_params']
        scale_factor = norm_params['scale_factor']
        
        # 512x512での座標をクロップ領域での座標に変換
        x_in_crop = x / scale_factor
        y_in_crop = y / scale_factor
        
        # クロップ領域から変換後画像の座標に変換
        x1, y1, x2, y2 = norm_params['crop_region']
        x_in_transformed = x_in_crop + x1
        y_in_transformed = y_in_crop + y1
        
        # Step 2: 楕円→円変換の逆変換
        if preprocess_info['is_transformed']:
            # 変換パラメータを取得
            ellipse_params = preprocess_info['ellipse_params']
            center_x = ellipse_params['center'][0]
            center_y = ellipse_params['center'][1]
            angle = ellipse_params['angle']
            scale_x = preprocess_info['transformation_applied']['scale_x']
            scale_y = preprocess_info['transformation_applied']['scale_y']
            
            # 逆変換行列を構築
            # 1. 中心を原点に移動
            x_centered = x_in_transformed - center_x
            y_centered = y_in_transformed - center_y
            
            # 2. 元の角度に戻す回転（-angle）
            angle_rad = np.radians(-angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            x_rotated = cos_a * x_centered - sin_a * y_centered
            y_rotated = sin_a * x_centered + cos_a * y_centered
            
            # 3. スケーリングの逆変換
            x_scaled = x_rotated / scale_x
            y_scaled = y_rotated / scale_y
            
            # 4. 元の向きに戻す回転（+angle）
            angle_rad_back = np.radians(angle)
            cos_b = np.cos(angle_rad_back)
            sin_b = np.sin(angle_rad_back)
            x_final = cos_b * x_scaled - sin_b * y_scaled
            y_final = sin_b * x_scaled + cos_b * y_scaled
            
            # 5. 元の位置に戻す
            x_original = x_final + center_x
            y_original = y_final + center_y
            
            return (x_original, y_original)
        else:
            # 変換されていない場合はそのまま返す
            return (x_in_transformed, y_in_transformed)
    
    def inverse_transform_circle(self, center: Tuple[float, float], radius: float, 
                               preprocess_info: Dict) -> Tuple[Tuple[float, float], float]:
        """
        正規化された円を元の画像座標に逆変換
        
        Args:
            center: 正規化された円の中心 (x, y)
            radius: 正規化された円の半径
            preprocess_info: 前処理情報
            
        Returns:
            (元の画像での中心座標, 元の画像での半径)
        """
        # 中心座標の逆変換
        original_center = self.inverse_transform_point(center, preprocess_info)
        
        # 半径の逆変換
        norm_params = preprocess_info['normalization_params']
        scale_factor = norm_params['scale_factor']
        
        # 512x512からクロップ領域のスケールに戻す
        radius_in_crop = radius / scale_factor
        
        # 楕円変換がある場合、半径も調整が必要
        if preprocess_info['is_transformed']:
            # 楕円の場合、x方向とy方向でスケールが異なるため、
            # 平均的な半径を計算
            scale_x = preprocess_info['transformation_applied']['scale_x']
            scale_y = preprocess_info['transformation_applied']['scale_y']
            
            # 逆変換での平均スケール
            avg_scale = 2.0 / (1.0/scale_x + 1.0/scale_y)
            original_radius = radius_in_crop / avg_scale
        else:
            original_radius = radius_in_crop
        
        return original_center, original_radius
    
    def inverse_transform_cut_edges(self, cut_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                                   pizza_center: Tuple[float, float], pizza_radius: float,
                                   preprocess_info: Dict) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        正規化されたカット線を元の画像座標に逆変換
        
        Args:
            cut_edges: 正規化座標系でのカット線リスト（-1〜1の範囲）
            pizza_center: 512x512画像でのピザ中心
            pizza_radius: 512x512画像でのピザ半径
            preprocess_info: 前処理情報
            
        Returns:
            元の画像座標系でのカット線リスト
        """
        original_cut_edges = []
        
        for (e1, e2) in cut_edges:
            # 正規化座標（-1〜1）から512x512座標に変換
            x1 = pizza_center[0] + e1[0] * pizza_radius
            y1 = pizza_center[1] + e1[1] * pizza_radius
            x2 = pizza_center[0] + e2[0] * pizza_radius
            y2 = pizza_center[1] + e2[1] * pizza_radius
            
            # 各端点を逆変換
            p1_original = self.inverse_transform_point((x1, y1), preprocess_info)
            p2_original = self.inverse_transform_point((x2, y2), preprocess_info)
            
            original_cut_edges.append((p1_original, p2_original))
        
        return original_cut_edges
    
    def create_svg_overlay_on_original(self, original_image_path: str, 
                                     pizza_center_normalized: Tuple[float, float],
                                     pizza_radius_normalized: float,
                                     salami_circles_normalized: List[Tuple[Tuple[float, float], float]],
                                     cut_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                                     preprocess_info: Dict,
                                     output_path: str,
                                     n_pieces: int = 4) -> str:
        """
        元の画像にオーバーレイするSVGを作成
        
        Args:
            original_image_path: 元の画像パス
            pizza_center_normalized: 正規化されたピザ中心
            pizza_radius_normalized: 正規化されたピザ半径
            salami_circles_normalized: 正規化されたサラミ円情報
            cut_edges: カット線情報（正規化座標系）
            preprocess_info: 前処理情報
            output_path: 出力SVGパス
            n_pieces: ピース数
            
        Returns:
            生成されたSVGファイルパス
        """
        # 元の画像サイズを取得
        img = cv2.imread(original_image_path)
        height, width = img.shape[:2]
        
        # ピザの円を逆変換
        pizza_center_original, pizza_radius_original = self.inverse_transform_circle(
            pizza_center_normalized, pizza_radius_normalized, preprocess_info
        )
        
        # サラミを逆変換
        salami_circles_original = []
        for (cx, cy), r in salami_circles_normalized:
            center_original, radius_original = self.inverse_transform_circle(
                (cx, cy), r, preprocess_info
            )
            salami_circles_original.append((center_original, radius_original))
        
        # カット線を逆変換
        cut_edges_original = self.inverse_transform_cut_edges(
            cut_edges, pizza_center_normalized, pizza_radius_normalized, preprocess_info
        )
        
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        
        # 背景画像を埋め込み
        dwg.add(dwg.image(
            href=original_image_path,
            insert=(0, 0),
            size=(width, height)
        ))
        
        # ピザの楕円を描画（元の楕円形状を使用）
        if preprocess_info['is_transformed'] and preprocess_info['ellipse_params']:
            ellipse = preprocess_info['ellipse_params']
            cx, cy = ellipse['center']
            rx = ellipse['major_axis']
            ry = ellipse['minor_axis']
            angle = ellipse['angle']
            
            # SVGの楕円要素として追加
            ellipse_elem = dwg.ellipse(
                center=(cx, cy),
                r=(rx, ry),
                stroke='brown',
                stroke_width=3,
                fill='none'
            )
            ellipse_elem.rotate(angle, center=(cx, cy))
            dwg.add(ellipse_elem)
        else:
            # 円として描画
            dwg.add(dwg.circle(
                center=pizza_center_original,
                r=pizza_radius_original,
                stroke='brown',
                stroke_width=3,
                fill='none'
            ))
        
        # サラミを描画
        for (cx, cy), r in salami_circles_original:
            dwg.add(dwg.circle(
                center=(cx, cy),
                r=r,
                stroke='darkred',
                stroke_width=2,
                fill='indianred',
                fill_opacity=0.7
            ))
        
        # カット線を描画
        for (p1, p2) in cut_edges_original:
            dwg.add(dwg.line(
                start=p1,
                end=p2,
                stroke='black',
                stroke_width=4
            ))
        
        # ピース番号を表示
        if preprocess_info['is_transformed'] and preprocess_info['ellipse_params']:
            ellipse = preprocess_info['ellipse_params']
            cx, cy = ellipse['center']
            
            angle_step = 2 * np.pi / n_pieces
            for i in range(n_pieces):
                angle = i * angle_step + angle_step / 2
                # 楕円上の点を計算
                text_radius = min(ellipse['major_axis'], ellipse['minor_axis']) * 0.5
                text_x = cx + text_radius * np.cos(angle)
                text_y = cy + text_radius * np.sin(angle)
                
                dwg.add(dwg.text(
                    str(i + 1),
                    insert=(text_x, text_y),
                    text_anchor='middle',
                    font_size=20,
                    font_weight='bold',
                    fill='white',
                    stroke='black',
                    stroke_width=1
                ))
        
        dwg.save()
        return output_path
    
    def create_overlay_image_on_original(self, original_image_path: str,
                                       pizza_center_normalized: Tuple[float, float],
                                       pizza_radius_normalized: float,
                                       salami_circles_normalized: List[Tuple[Tuple[float, float], float]],
                                       cut_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                                       preprocess_info: Dict,
                                       output_path: str,
                                       n_pieces: int = 4) -> str:
        """
        元の画像にオーバーレイしたPNG画像を作成
        
        Args:
            original_image_path: 元の画像パス
            pizza_center_normalized: 正規化されたピザ中心
            pizza_radius_normalized: 正規化されたピザ半径
            salami_circles_normalized: 正規化されたサラミ円情報
            cut_edges: カット線情報（正規化座標系）
            preprocess_info: 前処理情報
            output_path: 出力画像パス
            n_pieces: ピース数
            
        Returns:
            生成された画像ファイルパス
        """
        # 元の画像を読み込み
        img = cv2.imread(original_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 図を作成
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        
        # ピザの円/楕円を逆変換して描画
        if preprocess_info['is_transformed'] and preprocess_info['ellipse_params']:
            # 楕円として描画
            ellipse = preprocess_info['ellipse_params']
            from matplotlib.patches import Ellipse
            ellipse_patch = Ellipse(
                (ellipse['center'][0], ellipse['center'][1]),
                width=ellipse['major_axis'] * 2,
                height=ellipse['minor_axis'] * 2,
                angle=ellipse['angle'],
                fill=False,
                edgecolor='brown',
                linewidth=3
            )
            ax.add_patch(ellipse_patch)
        else:
            # 円として描画
            pizza_center_original, pizza_radius_original = self.inverse_transform_circle(
                pizza_center_normalized, pizza_radius_normalized, preprocess_info
            )
            pizza_circle = Circle(
                pizza_center_original, pizza_radius_original,
                fill=False, edgecolor='brown', linewidth=3
            )
            ax.add_patch(pizza_circle)
        
        # サラミを逆変換して描画
        for (cx, cy), r in salami_circles_normalized:
            center_original, radius_original = self.inverse_transform_circle(
                (cx, cy), r, preprocess_info
            )
            salami = Circle(
                center_original, radius_original,
                fill=True, facecolor='indianred',
                edgecolor='darkred', alpha=0.8
            )
            ax.add_patch(salami)
        
        # カット線を逆変換して描画
        cut_edges_original = self.inverse_transform_cut_edges(
            cut_edges, pizza_center_normalized, pizza_radius_normalized, preprocess_info
        )
        for (p1, p2) in cut_edges_original:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3)
        
        # ピース番号を表示
        if preprocess_info['is_transformed'] and preprocess_info['ellipse_params']:
            ellipse = preprocess_info['ellipse_params']
            cx, cy = ellipse['center']
            
            angle_step = 2 * np.pi / n_pieces
            for i in range(n_pieces):
                angle = i * angle_step + angle_step / 2
                # 楕円上の点を計算
                text_radius = min(ellipse['major_axis'], ellipse['minor_axis']) * 0.5
                text_x = cx + text_radius * np.cos(angle)
                text_y = cy + text_radius * np.sin(angle)
                
                ax.text(text_x, text_y, str(i + 1),
                       fontsize=20, fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7),
                       ha='center', va='center')
        
        ax.set_title(f'Pizza Division Result on Original Image ({n_pieces} pieces)', fontsize=16)
        ax.axis('off')
        
        # 保存
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


def main():
    """テスト用メイン関数"""
    print("PostprocessService loaded successfully")
    print("This service provides inverse transformation from normalized coordinates to original image coordinates")


if __name__ == "__main__":
    main()