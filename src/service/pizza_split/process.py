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
        divider.calculate_piece_boundaries()
        
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
        
        # 7.5. ピザ爆発SVG生成（3種類）
        if not quiet:
            print("\n7.5. ピザ爆発SVG生成...")
        
        # 爆発前（アニメーションなし）
        svg_before_path = self.output_dir / f"pizza_before_{self.timestamp}.svg"
        divider.create_properly_divided_exploded_svg(
            str(svg_before_path),
            svg_size=600,
            explode_factor=0.0  # 爆発なし
        )
        if not quiet:
            print(f"   爆発前SVG生成: {svg_before_path.name}")
        
        # 爆発後（アニメーションなし）
        svg_after_path = self.output_dir / f"pizza_after_{self.timestamp}.svg"
        divider.create_properly_divided_exploded_svg(
            str(svg_after_path),
            svg_size=600,
            explode_factor=0.33  # 半径の1/3移動
        )
        if not quiet:
            print(f"   爆発後SVG生成: {svg_after_path.name}")
        
        # アニメーション付き爆発
        svg_animated_path = self.output_dir / f"pizza_animated_{self.timestamp}.svg"
        divider.create_animated_exploding_pizza_svg(
            str(svg_animated_path),
            svg_size=600,
            explode_distance=0.33,
            animation_duration=2.0
        )
        if not quiet:
            print(f"   アニメーション付きSVG生成: {svg_animated_path.name}")
        
        # 7.6. 元画像用の爆発SVG生成（逆変換適用）
        if not quiet:
            print("\n7.6. 元画像用爆発SVG生成（逆変換適用）...")
        
        # 爆発前（元画像用）
        svg_before_original_path = self.output_dir / f"pizza_before_original_{self.timestamp}.svg"
        self.create_exploded_svg_on_original(
            str(image_path),
            divider,
            preprocess_info,
            str(svg_before_original_path),
            pizza_center,
            pizza_radius,
            explode_factor=0.0,
            animated=False
        )
        if not quiet:
            print(f"   元画像用爆発前SVG: {svg_before_original_path.name}")
        
        # 爆発後（元画像用）
        svg_after_original_path = self.output_dir / f"pizza_after_original_{self.timestamp}.svg"
        self.create_exploded_svg_on_original(
            str(image_path),
            divider,
            preprocess_info,
            str(svg_after_original_path),
            pizza_center,
            pizza_radius,
            explode_factor=0.33,
            animated=False
        )
        if not quiet:
            print(f"   元画像用爆発後SVG: {svg_after_original_path.name}")
        
        # アニメーション付き（元画像用）
        svg_animated_original_path = self.output_dir / f"pizza_animated_original_{self.timestamp}.svg"
        self.create_exploded_svg_on_original(
            str(image_path),
            divider,
            preprocess_info,
            str(svg_animated_original_path),
            pizza_center,
            pizza_radius,
            explode_factor=0.33,
            animated=True,
            animation_duration=2.0
        )
        if not quiet:
            print(f"   元画像用アニメーション付きSVG: {svg_animated_original_path.name}")
        
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
                'svg_before': str(svg_before_path),
                'svg_after': str(svg_after_path),
                'svg_animated': str(svg_animated_path),
                'svg_before_original': str(svg_before_original_path),
                'svg_after_original': str(svg_after_original_path),
                'svg_animated_original': str(svg_animated_original_path),
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
        
        # 10. 各ピースのSVGを生成
        if not quiet:
            print("\n10. 各ピースのSVG生成...")
        piece_svg_dir = self.output_dir / f"pieces_{self.timestamp}"
        
        # dividerに各ピースのSVGを生成させる
        piece_svg_paths = divider.generate_piece_svgs_isolated(str(piece_svg_dir))
        
        if not quiet:
            print(f"   各ピースのSVGを生成: {piece_svg_dir}")
            for i, svg_path in enumerate(piece_svg_paths):
                print(f"     - Piece {i+1}: {Path(svg_path).name}")
        
        if not quiet:
            print(f"\n=== 処理完了 ===")
            print(f"前処理画像: {preprocessed_path}")
            print(f"前処理済み画像用SVG: {svg_preprocessed_path}")
            print(f"元画像用SVG: {svg_original_path}")
            print(f"元画像にオーバーレイしたPNG: {result_original_path}")
            print(f"結果画像: {result_path}")
            print(f"各ピースのSVG: {piece_svg_dir}")
        
        return {
            'preprocessed_image': str(preprocessed_path),
            'svg_preprocessed': str(svg_preprocessed_path),
            'svg_original': str(svg_original_path),
            'svg_before': str(svg_before_path),
            'svg_after': str(svg_after_path),
            'svg_animated': str(svg_animated_path),
            'svg_before_original': str(svg_before_original_path),
            'svg_after_original': str(svg_after_original_path),
            'svg_animated_original': str(svg_animated_original_path),
            'result_image': str(result_path),
            'result_original_image': str(result_original_path),
            'pizza_center': pizza_center,
            'pizza_radius': pizza_radius,
            'salami_circles': salami_circles,
            'cut_edges': divider.cut_edges,
            'pieces': divider.pieces,
            'piece_svgs': piece_svg_paths,
            'piece_svg_dir': str(piece_svg_dir),
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
    
    def create_exploded_svg_on_original(self, image_path, divider, preprocess_info, 
                                       output_path, pizza_center, pizza_radius,
                                       explode_factor=0.33, animated=False,
                                       animation_duration=2.0):
        """
        元画像用の爆発SVGを生成（逆変換適用）
        
        Args:
            image_path: 元画像パス
            divider: PizzaDividerインスタンス
            preprocess_info: 前処理情報
            output_path: 出力SVGパス
            pizza_center: ピザ中心（前処理済み座標）
            pizza_radius: ピザ半径（前処理済み）
            explode_factor: 爆発係数（0.0=爆発なし、0.33=半径の1/3）
            animated: アニメーションを含むか
            animation_duration: アニメーション時間（秒）
        """
        from scipy.spatial import ConvexHull
        
        # 画像情報取得
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        
        # SVG作成
        dwg = svgwrite.Drawing(str(output_path), size=(width, height))
        
        # 背景画像
        dwg.add(dwg.image(
            href=str(image_path),
            insert=(0, 0),
            size=(width, height)
        ))
        
        # 各ピースの角度を計算
        piece_angles = []
        for i in range(divider.n):
            idx = divider.pieces[i]
            # ピースの重心を計算
            cx = np.mean([divider.px[j] for j in idx])
            cy = np.mean([divider.py[j] for j in idx])
            angle = np.arctan2(cy, cx)
            piece_angles.append(angle)
        
        # アニメーション用のスタイル
        if animated:
            style_rules = []
            for i in range(divider.n):
                angle = piece_angles[i]
                # 正規化座標での移動量
                norm_offset_x = divider.R_pizza * explode_factor * np.cos(angle)
                norm_offset_y = divider.R_pizza * explode_factor * np.sin(angle)
                
                # 前処理済み画像座標での移動量
                offset_x = norm_offset_x * pizza_radius
                offset_y = norm_offset_y * pizza_radius
                
                # 逆変換を適用して元画像での移動量を計算
                start_point = (pizza_center[0], pizza_center[1])
                end_point = (pizza_center[0] + offset_x, pizza_center[1] + offset_y)
                
                # 逆変換
                start_original = self.postprocess_service.inverse_transform_point(start_point, preprocess_info)
                end_original = self.postprocess_service.inverse_transform_point(end_point, preprocess_info)
                
                # 元画像での移動量
                final_offset_x = end_original[0] - start_original[0]
                final_offset_y = end_original[1] - start_original[1]
                
                style_rules.append(f'''
        @keyframes explode_piece_{i+1} {{
            0% {{
                transform: translate(0, 0);
                opacity: 0.8;
            }}
            100% {{
                transform: translate({final_offset_x}px, {final_offset_y}px);
                opacity: 1;
            }}
        }}
        
        #animated_piece_{i+1} {{
            animation: explode_piece_{i+1} {animation_duration}s ease-out forwards;
            animation-delay: {i * 0.2}s;
            transform-origin: center;
        }}''')
            
            global_style = '\n'.join(style_rules) + '''
        
        /* ホバー時の効果 */
        g[id^="animated_piece_"] {
            transition: filter 0.3s ease;
        }
        
        g[id^="animated_piece_"]:hover {
            filter: brightness(1.2) drop-shadow(0 0 10px rgba(0,0,0,0.3));
            cursor: pointer;
        }
        '''
            
            style_elem = dwg.style(global_style)
            dwg.defs.add(style_elem)
        
        # カラーマップ
        cmap = plt.get_cmap('tab10', divider.n)
        
        # 各ピースを描画
        for i in range(divider.n):
            idx = divider.pieces[i]
            if len(idx) < 3:
                continue
            
            # ピースグループ
            piece_id = f"animated_piece_{i+1}" if animated else f"piece_{i+1}"
            piece_group = dwg.g(id=piece_id)
            
            # 爆発オフセット（アニメーションなしの場合）
            if not animated and explode_factor > 0:
                angle = piece_angles[i]
                norm_offset_x = divider.R_pizza * explode_factor * np.cos(angle)
                norm_offset_y = divider.R_pizza * explode_factor * np.sin(angle)
                
                # 前処理済み画像座標での移動量
                offset_x = norm_offset_x * pizza_radius
                offset_y = norm_offset_y * pizza_radius
                
                # 逆変換を適用
                start_point = (pizza_center[0], pizza_center[1])
                end_point = (pizza_center[0] + offset_x, pizza_center[1] + offset_y)
                
                start_original = self.postprocess_service.inverse_transform_point(start_point, preprocess_info)
                end_original = self.postprocess_service.inverse_transform_point(end_point, preprocess_info)
                
                final_offset_x = end_original[0] - start_original[0]
                final_offset_y = end_original[1] - start_original[1]
                
                piece_group = dwg.g(id=piece_id, transform=f"translate({final_offset_x},{final_offset_y})")
            
            # ピースの境界を計算（正規化座標）
            piece_points = [(divider.px[j], divider.py[j]) for j in idx]
            hull = ConvexHull(piece_points)
            hull_points_norm = [piece_points[i] for i in hull.vertices]
            
            # 正規化座標から前処理済み画像座標に変換
            hull_points_preprocessed = []
            for x_norm, y_norm in hull_points_norm:
                x_pre = pizza_center[0] + x_norm * pizza_radius
                y_pre = pizza_center[1] + y_norm * pizza_radius
                hull_points_preprocessed.append((x_pre, y_pre))
            
            # 前処理済み座標から元画像座標に逆変換
            hull_points_original = []
            for point in hull_points_preprocessed:
                original_point = self.postprocess_service.inverse_transform_point(point, preprocess_info)
                hull_points_original.append(original_point)
            
            # ピースの色
            color = cmap(i)
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                      int(color[1]*255), 
                                                      int(color[2]*255))
            
            # ピースポリゴン
            piece_group.add(dwg.polygon(
                points=hull_points_original,
                fill=color_hex,
                opacity=0.7,
                stroke='darkgray',
                stroke_width=2
            ))
            
            # クリッピングパス（サラミ用）
            clip_id = f'piece_clip_{i}'
            clip_path = dwg.defs.add(dwg.clipPath(id=clip_id))
            clip_path.add(dwg.polygon(points=hull_points_original))
            
            # サラミグループ
            salami_group = dwg.g(clip_path=f'url(#{clip_id})')
            
            # このピースに含まれるサラミを描画
            for cx_norm, cy_norm in divider.centers:
                # サラミとピースの重なりを計算
                overlap_points = 0
                for j in idx:
                    if (divider.px[j] - cx_norm)**2 + (divider.py[j] - cy_norm)**2 <= divider.R_salami**2:
                        overlap_points += 1
                
                if overlap_points > 0:
                    # サラミ中心を前処理済み座標に変換
                    cx_pre = pizza_center[0] + cx_norm * pizza_radius
                    cy_pre = pizza_center[1] + cy_norm * pizza_radius
                    
                    # 元画像座標に逆変換
                    cx_original, cy_original = self.postprocess_service.inverse_transform_point(
                        (cx_pre, cy_pre), preprocess_info
                    )
                    
                    # サラミ半径も変換
                    # 半径変換のためにサラミの端点を変換
                    edge_point_pre = (cx_pre + divider.R_salami * pizza_radius, cy_pre)
                    edge_point_original = self.postprocess_service.inverse_transform_point(
                        edge_point_pre, preprocess_info
                    )
                    r_original = abs(edge_point_original[0] - cx_original)
                    
                    # 重なり面積に応じて透明度を調整
                    total_salami_points = sum(1 for px, py in zip(divider.px, divider.py) 
                                            if (px - cx_norm)**2 + (py - cy_norm)**2 <= divider.R_salami**2)
                    overlap_ratio = overlap_points / max(total_salami_points, 1)
                    
                    if overlap_ratio > 0.8:
                        opacity = 0.9
                    else:
                        opacity = 0.7
                    
                    salami_group.add(dwg.circle(
                        center=(cx_original, cy_original),
                        r=r_original,
                        fill='indianred',
                        stroke='darkred',
                        stroke_width=1.5,
                        opacity=opacity
                    ))
            
            piece_group.add(salami_group)
            
            # ピース番号
            
            dwg.add(piece_group)
        
        
        dwg.save()
    
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