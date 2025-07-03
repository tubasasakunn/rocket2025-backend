#!/usr/bin/env python
# coding: utf-8
"""
移動ナイフ法によるピザ分割（SVG生成機能付き）
・ピザ全体の面積とサラミ面積が各ピースで同じになるように分割
・モンテカルロ法を使った近似
・結果を複数色に塗り分けて可視化
・各ピースの個別SVGを生成
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from datetime import datetime
import svgwrite
from pathlib import Path
from scipy.spatial import ConvexHull


class PizzaDivider:
    """ピザ分割クラス"""
    
    def __init__(self, R_pizza=1.0, R_salami=0.10, m=13, n=7, 
                 N_Monte=100_000, seed=0, isDebug=False):
        """
        初期化
        
        Args:
            R_pizza: ピザの半径
            R_salami: サラミの半径
            m: サラミの枚数
            n: 分割するピース数
            N_Monte: モンテカルロ法で使用する点の数
            seed: 乱数シード
            isDebug: デバッグモード（画像保存・ログ出力）
        """
        self.R_pizza = R_pizza
        self.R_salami = R_salami
        self.m = m
        self.n = n
        self.N_Monte = N_Monte
        self.seed = seed
        self.isDebug = isDebug
        self.dtheta_g = np.deg2rad(0.5)  # 角度グリッド間隔（粗探索）
        
        # デバッグ出力用ディレクトリ
        if self.isDebug:
            self.debug_dir = f"debug/salami_divide_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.debug_dir, exist_ok=True)
            self._log(f"デバッグディレクトリ作成: {self.debug_dir}")
    
    def _log(self, message):
        """デバッグログ出力"""
        if self.isDebug:
            print(f"[DEBUG] {message}")
    
    def generate_monte_carlo_points(self):
        """モンテカルロ点をピザ内に一様散布"""
        np.random.seed(self.seed)
        r = np.sqrt(np.random.rand(self.N_Monte)) * self.R_pizza
        ang = np.random.rand(self.N_Monte) * 2 * np.pi
        self.px = r * np.cos(ang)
        self.py = r * np.sin(ang)
        self.w = (np.pi * self.R_pizza**2) / self.N_Monte  # 1点あたりの面積
        
        self._log(f"モンテカルロ点生成完了: {self.N_Monte}点")
    
    def place_salami_random(self):
        """サラミをランダムに配置（重なりは許容）"""
        centers = []
        while len(centers) < self.m:
            rr = (self.R_pizza - self.R_salami) * np.sqrt(np.random.rand())
            tt = 2 * np.pi * np.random.rand()
            centers.append((rr * np.cos(tt), rr * np.sin(tt)))
        self.centers = np.asarray(centers)
        
        # 各モンテカルロ点がサラミ上にあるかチェック
        self.on_salami = np.zeros(self.N_Monte, dtype=bool)
        for cx, cy in self.centers:
            self.on_salami |= (self.px - cx)**2 + (self.py - cy)**2 <= self.R_salami**2
        
        self._log(f"サラミ配置完了: {self.m}枚")
        self._log(f"サラミ上の点数: {self.on_salami.sum()}")
    
    def calculate_targets(self):
        """目標値（面積・サラミ量）を計算"""
        self.A_goal = (np.pi * self.R_pizza**2) / self.n  # 1ピースあたりの面積
        self.B_goal = self.on_salami.sum() * self.w / self.n  # 1ピースあたりのサラミ面積
        
        self._log(f"目標面積/ピース: {self.A_goal:.4f}")
        self._log(f"目標サラミ面積/ピース: {self.B_goal:.4f}")
    
    def get_dist(self, theta, idx):
        """
        指定角度thetaの法線方向で面積がA_goalとなる距離を計算
        
        Args:
            theta: カット角度
            idx: 未確定点のインデックス
            
        Returns:
            d: カット位置までの距離
            g: サラミ超過量（得られた量 - 目標量）
            maskL: 左側半平面のマスク
        """
        nx, ny = np.cos(theta), np.sin(theta)  # 法線ベクトル
        proj = nx * self.px[idx] + ny * self.py[idx]  # 射影
        k = int(self.A_goal / self.w)  # A_goalに対応する順位
        d = np.partition(proj, k)[k]  # k番目の値を取得
        maskL = proj <= d  # 左側半平面
        B = self.on_salami[idx][maskL].sum() * self.w
        g = B - self.B_goal
        return d, g, maskL
    
    def divide_pizza(self):
        """移動ナイフ法でピザを分割"""
        self.remain = np.ones(self.N_Monte, dtype=bool)  # 未確定点
        self.pieces = []  # 各ピースの点インデックス
        self.cut_edges = []  # 各カットの両端点
        
        # n-1回カットを実行
        for i in range(self.n - 1):
            self._log(f"\nカット {i+1}/{self.n-1} を実行中...")
            
            idx = np.where(self.remain)[0]  # 未確定点のインデックス
            
            # 粗い角度グリッドでg(θ)を一括計算
            thetas = np.arange(0, 2 * np.pi + self.dtheta_g, self.dtheta_g)
            g_vals, d_vals, mask_cache = [], [], []
            
            for θ in thetas:
                d, g, maskL = self.get_dist(θ, idx)
                d_vals.append(d)
                g_vals.append(g)
                mask_cache.append(maskL)
            
            g_vals = np.asarray(g_vals)
            d_vals = np.asarray(d_vals)
            
            # g値の符号が反転する区間を検出
            sign = np.sign(g_vals)
            flip = np.where(sign[:-1] * sign[1:] < 0)[0]
            
            if len(flip) == 0:
                # 反転区間がなければ|g|最小角を使用
                j = np.argmin(np.abs(g_vals))
                theta_star = thetas[j]
                d_star = d_vals[j]
                maskL = mask_cache[j]
                self._log(f"反転区間なし、|g|最小角使用: θ={np.rad2deg(theta_star):.1f}°")
            else:
                # 反転区間で二分探索してg=0に収束
                j = flip[0]
                theta_lo, theta_hi = thetas[j], thetas[j + 1]
                
                for _ in range(25):  # 十分な回数で収束
                    theta_mid = 0.5 * (theta_lo + theta_hi)
                    d_mid, g_mid, mask_mid = self.get_dist(theta_mid, idx)
                    if np.sign(g_mid) == np.sign(g_vals[j]):
                        theta_lo = theta_mid
                    else:
                        theta_hi = theta_mid
                    theta_star, d_star, maskL = theta_mid, d_mid, mask_mid
                
                self._log(f"二分探索で収束: θ={np.rad2deg(theta_star):.1f}°")
            
            # ピース確定
            slice_idx = idx[maskL]
            self.pieces.append(slice_idx)
            self.remain[slice_idx] = False
            
            # カット線の両端点を保存
            nx, ny = np.cos(theta_star), np.sin(theta_star)
            t_edge = np.sqrt(max(0.0, self.R_pizza**2 - d_star**2))  # 数値誤差対策
            e1 = (d_star * nx - ny * t_edge, d_star * ny + nx * t_edge)
            e2 = (d_star * nx + ny * t_edge, d_star * ny - nx * t_edge)
            self.cut_edges.append((e1, e2))
            
            if self.isDebug:
                # 各カット後の状態
                A = len(slice_idx) * self.w
                B = self.on_salami[slice_idx].sum() * self.w
                self._log(f"ピース{i+1}: 面積={A:.4f}, サラミ面積={B:.4f}")
        
        # 残った点が最後のピース
        self.pieces.append(np.where(self.remain)[0])
        self._log("\n分割完了")
    
    def calculate_piece_boundaries(self):
        """各ピースの境界を計算（カット線と円弧で囲まれる領域）"""
        self.piece_boundaries = []
        
        # 各ピースについて境界を定義
        for i in range(self.n):
            boundaries = {
                'piece_index': i,
                'cut_lines': [],
                'arc_angles': []
            }
            
            # カット線の情報を追加
            if i < len(self.cut_edges):
                # このピースの右側のカット線
                boundaries['cut_lines'].append(self.cut_edges[i])
            
            if i > 0:
                # このピースの左側のカット線
                boundaries['cut_lines'].append(self.cut_edges[i-1])
            
            # このピースのモンテカルロ点から角度範囲を推定
            idx = self.pieces[i]
            if len(idx) > 0:
                # 各点の角度を計算
                angles = np.arctan2(self.py[idx], self.px[idx])
                # -π〜πの範囲を0〜2πに変換
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
                
                # 角度の範囲を計算（円を横切る場合の処理も含む）
                min_angle = np.min(angles)
                max_angle = np.max(angles)
                
                # 角度の差が大きすぎる場合は、0度を横切っている可能性
                if max_angle - min_angle > np.pi:
                    # 0度を横切っている場合の処理
                    angles_shifted = np.where(angles < np.pi, angles + 2*np.pi, angles)
                    min_angle = np.min(angles_shifted) % (2*np.pi)
                    max_angle = np.max(angles_shifted) % (2*np.pi)
                
                boundaries['arc_angles'] = [min_angle, max_angle]
            
            self.piece_boundaries.append(boundaries)
        
        self._log(f"ピース境界計算完了: {len(self.piece_boundaries)}個")
    
    def print_results(self):
        """分割結果をコンソールに出力"""
        print("\nピース   ピザ面積    サラミ面積")
        print("-" * 35)
        for i, idx in enumerate(self.pieces, 1):
            A = idx.size * self.w
            B = self.on_salami[idx].sum() * self.w
            print(f"{i:5d}  {A:10.4f}  {B:10.4f}")
    
    def visualize(self, show=True):
        """分割結果を可視化"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"移動ナイフ法によるピザ分割 (n={self.n}, m={self.m})")
        
        # ピザの背景
        ax.add_patch(Circle((0, 0), self.R_pizza,
                           fc='bisque', ec='saddlebrown', lw=3, zorder=0))
        
        # 各ピースを色分け（モンテカルロ点を散布）
        cmap = plt.get_cmap('tab10', self.n)
        for i, idx in enumerate(self.pieces):
            ax.scatter(self.px[idx], self.py[idx],
                      s=1.2,
                      color=cmap(i),
                      alpha=0.45,
                      marker='s',
                      linewidths=0,
                      zorder=1)
        
        # サラミを描画
        for cx, cy in self.centers:
            ax.add_patch(Circle((cx, cy), self.R_salami,
                               fc='indianred', ec='brown', alpha=.9, zorder=2))
        
        # カット線（弦）を描画
        for (e1, e2) in self.cut_edges:
            ax.plot([e1[0], e2[0]], [e1[1], e2[1]],
                   color='k', lw=2.2, zorder=3)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def generate_piece_svgs(self, output_dir, svg_size=400, dot_size=2):
        """
        各ピースの個別SVGを生成（モンテカルロ点で表現）
        
        Args:
            output_dir: 出力ディレクトリ
            svg_size: SVGのサイズ（正方形）
            dot_size: モンテカルロ点のサイズ
            
        Returns:
            生成されたSVGファイルパスのリスト
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        svg_paths = []
        
        for i in range(self.n):
            svg_path = output_dir / f"piece_{i+1}.svg"
            self._generate_single_piece_svg(
                piece_index=i,
                output_path=str(svg_path),
                svg_size=svg_size,
                dot_size=dot_size
            )
            svg_paths.append(str(svg_path))
            
        return svg_paths
    
    def _generate_single_piece_svg(self, piece_index, output_path, svg_size=400, dot_size=2):
        """
        単一ピースのSVGを生成（モンテカルロ点で表現）
        
        Args:
            piece_index: ピースのインデックス
            output_path: 出力パス
            svg_size: SVGのサイズ
            dot_size: モンテカルロ点のサイズ
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景を設定
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # 座標変換（-1.2〜1.2を0〜svg_sizeに変換）
        def transform_point(x, y):
            tx = (x + 1.2) * svg_size / 2.4
            ty = (-y + 1.2) * svg_size / 2.4  # y軸を反転
            return (tx, ty)
        
        # ピザの円を描画（背景として）
        center = transform_point(0, 0)
        radius = self.R_pizza * svg_size / 2.4
        dwg.add(dwg.circle(center=center, r=radius,
                          fill='bisque', stroke='saddlebrown', stroke_width=3))
        
        # このピースのモンテカルロ点を描画
        idx = self.pieces[piece_index]
        cmap = plt.get_cmap('tab10', self.n)
        color = cmap(piece_index)
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                  int(color[1]*255), 
                                                  int(color[2]*255))
        
        # モンテカルロ点を四角形で描画
        for j in idx:
            px, py = transform_point(self.px[j], self.py[j])
            dwg.add(dwg.rect(insert=(px - dot_size/2, py - dot_size/2),
                            size=(dot_size, dot_size),
                            fill=color_hex,
                            opacity=0.45))
        
        # サラミを描画（すべて表示）
        for cx, cy in self.centers:
            center = transform_point(cx, cy)
            radius = self.R_salami * svg_size / 2.4
            dwg.add(dwg.circle(center=center, r=radius,
                               fill='indianred', stroke='brown', 
                               stroke_width=1, opacity=0.9))
        
        # カット線を描画（すべて表示）
        for (e1, e2) in self.cut_edges:
            p1 = transform_point(e1[0], e1[1])
            p2 = transform_point(e2[0], e2[1])
            dwg.add(dwg.line(start=p1, end=p2,
                            stroke='black', stroke_width=2.2))
        
        # ピース番号と情報を表示
        dwg.add(dwg.text(
            f'Piece {piece_index + 1}',
            insert=(10, 25),
            font_size='18px',
            font_family='Arial',
            fill='saddlebrown',
            font_weight='bold'
        ))
        
        # ピース情報
        area = len(idx) * self.w
        salami_area = self.on_salami[idx].sum() * self.w
        
        # このピースに含まれるサラミを数える（中心点ベース）
        salami_count = 0
        for cx, cy in self.centers:
            # サラミの中心に最も近いモンテカルロ点を見つける
            distances = np.sqrt((self.px - cx)**2 + (self.py - cy)**2)
            closest_idx = np.argmin(distances)
            # そのモンテカルロ点がこのピースに属していればカウント
            if closest_idx in idx:
                salami_count += 1
        
        info_text = [
            f'Area: {area:.3f}',
            f'Salami area: {salami_area:.3f}',
            f'Salami count: ~{salami_count}'
        ]
        
        for i, text in enumerate(info_text):
            dwg.add(dwg.text(
                text,
                insert=(10, 50 + i * 20),
                font_size='14px',
                font_family='Arial',
                fill='black'
            ))
        
        dwg.save()
    
    def generate_piece_svgs_isolated(self, output_dir, svg_size=400):
        """
        各ピースの個別SVGを生成（そのピースのみを表示）
        
        Args:
            output_dir: 出力ディレクトリ
            svg_size: SVGのサイズ（正方形）
            
        Returns:
            生成されたSVGファイルパスのリスト
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        svg_paths = []
        
        for i in range(self.n):
            svg_path = output_dir / f"piece_{i+1}_isolated.svg"
            self._generate_single_piece_svg_isolated(
                piece_index=i,
                output_path=str(svg_path),
                svg_size=svg_size
            )
            svg_paths.append(str(svg_path))
            
        return svg_paths

    def _generate_single_piece_svg_isolated(self, piece_index, output_path, svg_size=400):
        """
        単一ピースのSVGを生成（そのピースのみを表示）
        
        Args:
            piece_index: ピースのインデックス
            output_path: 出力パス
            svg_size: SVGのサイズ
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景を設定
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # このピースのモンテカルロ点から境界を推定
        idx = self.pieces[piece_index]
        piece_points = [(self.px[j], self.py[j]) for j in idx]
        
        if len(piece_points) == 0:
            dwg.save()
            return
        
        # ピースの重心と範囲を計算
        cx = np.mean([p[0] for p in piece_points])
        cy = np.mean([p[1] for p in piece_points])
        max_dist = max(np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in piece_points)
        
        # 座標変換（ピースを中心に配置）
        scale = svg_size / (3 * max_dist) if max_dist > 0 else svg_size / 2.4
        
        def transform_point(x, y):
            tx = (x - cx) * scale + svg_size / 2
            ty = -(y - cy) * scale + svg_size / 2  # y軸を反転
            return (tx, ty)
        
        # ピースの背景（凸包を近似的に表現）とクリッピングパスの作成
        hull_points = []
        
        if len(piece_points) > 3:
            try:
                hull = ConvexHull(piece_points)
                hull_points = [transform_point(piece_points[i][0], piece_points[i][1]) 
                            for i in hull.vertices]
                
                # クリッピングパスを定義
                clip_path = dwg.defs.add(dwg.clipPath(id=f'piece_clip_{piece_index}'))
                clip_path.add(dwg.polygon(points=hull_points))
                
                # 凸包を描画（ピースの背景）
                dwg.add(dwg.polygon(points=hull_points,
                                fill='bisque', stroke='saddlebrown', stroke_width=2))
            except:
                # 凸包の計算に失敗した場合はスキップ
                pass
        
        # サラミを描画するグループを作成（クリッピングを適用）
        if hull_points:  # 凸包が計算できた場合のみ
            salami_group = dwg.g(clip_path=f'url(#piece_clip_{piece_index})')
            
            # このピースに含まれるサラミを描画
            for scx, scy in self.centers:
                # サラミがこのピースに含まれるかチェック（モンテカルロ点ベース）
                salami_points = 0
                for j in idx:
                    if (self.px[j] - scx)**2 + (self.py[j] - scy)**2 <= self.R_salami**2:
                        salami_points += 1
                
                if salami_points > 10:  # 十分な点がサラミ内にある場合
                    center = transform_point(scx, scy)
                    radius = self.R_salami * scale
                    salami_group.add(dwg.circle(center=center, r=radius,
                                            fill='indianred', stroke='darkred', 
                                            stroke_width=1.5, opacity=0.9))
            
            # グループをSVGに追加
            dwg.add(salami_group)
        else:
            # 凸包が計算できない場合は通常通り描画（クリッピングなし）
            for scx, scy in self.centers:
                salami_points = 0
                for j in idx:
                    if (self.px[j] - scx)**2 + (self.py[j] - scy)**2 <= self.R_salami**2:
                        salami_points += 1
                
                if salami_points > 10:
                    center = transform_point(scx, scy)
                    radius = self.R_salami * scale
                    dwg.add(dwg.circle(center=center, r=radius,
                                    fill='indianred', stroke='darkred', 
                                    stroke_width=1.5, opacity=0.9))
        dwg.save()
    
    def generate_piece_svgs_isolated_content(self, svg_size=400):
        """
        各ピースの個別SVGコンテンツを生成（ファイル保存なし）
        
        Args:
            svg_size: SVGのサイズ（正方形）
            
        Returns:
            SVGコンテンツ文字列のリスト
        """
        svg_contents = []
        
        for i in range(self.n):
            svg_content = self._generate_single_piece_svg_isolated_content(
                piece_index=i,
                svg_size=svg_size
            )
            svg_contents.append(svg_content)
            
        return svg_contents
    
    def _generate_single_piece_svg_isolated_content(self, piece_index, svg_size=400):
        """
        単一ピースのSVGコンテンツを生成（そのピースのみを表示、ファイル保存なし）
        
        Args:
            piece_index: ピースのインデックス
            svg_size: SVGのサイズ
            
        Returns:
            SVGコンテンツ文字列
        """
        # SVG作成（StringIOを使用）
        dwg = svgwrite.Drawing(size=(svg_size, svg_size))
        
        # 背景を設定
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # このピースのモンテカルロ点から境界を推定
        idx = self.pieces[piece_index]
        piece_points = [(self.px[j], self.py[j]) for j in idx]
        
        if len(piece_points) == 0:
            return dwg.tostring()
        
        # ピースの重心と範囲を計算
        cx = np.mean([p[0] for p in piece_points])
        cy = np.mean([p[1] for p in piece_points])
        max_dist = max(np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in piece_points)
        
        # 座標変換（ピースを中心に配置）
        scale = svg_size / (3 * max_dist) if max_dist > 0 else svg_size / 2.4
        
        def transform_point(x, y):
            tx = (x - cx) * scale + svg_size / 2
            ty = -(y - cy) * scale + svg_size / 2  # y軸を反転
            return (tx, ty)
        
        # ピースの背景（凸包を近似的に表現）とクリッピングパスの作成
        hull_points = []
        
        if len(piece_points) > 3:
            try:
                hull = ConvexHull(piece_points)
                hull_points = [transform_point(piece_points[i][0], piece_points[i][1]) 
                            for i in hull.vertices]
                
                # クリッピングパスを定義
                clip_path = dwg.defs.add(dwg.clipPath(id=f'piece_clip_{piece_index}'))
                clip_path.add(dwg.polygon(points=hull_points))
                
                # 凸包を描画（ピースの背景）
                dwg.add(dwg.polygon(points=hull_points,
                                fill='bisque', stroke='saddlebrown', stroke_width=2))
            except:
                # 凸包の計算に失敗した場合はスキップ
                pass
        
        # サラミを描画するグループを作成（クリッピングを適用）
        if hull_points:  # 凸包が計算できた場合のみ
            salami_group = dwg.g(clip_path=f'url(#piece_clip_{piece_index})')
            
            # このピースに含まれるサラミを描画
            for scx, scy in self.centers:
                # サラミがこのピースに含まれるかチェック（モンテカルロ点ベース）
                salami_points = 0
                for j in idx:
                    if (self.px[j] - scx)**2 + (self.py[j] - scy)**2 <= self.R_salami**2:
                        salami_points += 1
                
                if salami_points > 10:  # 十分な点がサラミ内にある場合
                    center = transform_point(scx, scy)
                    radius = self.R_salami * scale
                    salami_group.add(dwg.circle(center=center, r=radius,
                                            fill='indianred', stroke='darkred', 
                                            stroke_width=1.5, opacity=0.9))
            
            # グループをSVGに追加
            dwg.add(salami_group)
        else:
            # 凸包が計算できない場合は通常通り描画（クリッピングなし）
            for scx, scy in self.centers:
                salami_points = 0
                for j in idx:
                    if (self.px[j] - scx)**2 + (self.py[j] - scy)**2 <= self.R_salami**2:
                        salami_points += 1
                
                if salami_points > 10:
                    center = transform_point(scx, scy)
                    radius = self.R_salami * scale
                    dwg.add(dwg.circle(center=center, r=radius,
                                    fill='indianred', stroke='darkred', 
                                    stroke_width=1.5, opacity=0.9))
        
        return dwg.tostring()
    
    def get_piece_boundaries_for_postprocess(self):
        """
        各ピースの境界点を後処理用に取得（正規化座標）
        
        Returns:
            List[List[Tuple[float, float]]]: 各ピースの境界点のリスト
        """
        piece_boundaries_list = []
        
        for i in range(self.n):
            # このピースのモンテカルロ点から境界を推定
            idx = self.pieces[i]
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            
            if len(piece_points) < 3:
                piece_boundaries_list.append([])
                continue
            
            # 凸包を計算して境界点を取得
            try:
                hull = ConvexHull(piece_points)
                boundary_points = [(piece_points[v][0], piece_points[v][1]) 
                                 for v in hull.vertices]
                piece_boundaries_list.append(boundary_points)
            except:
                # 凸包の計算に失敗した場合は空リスト
                piece_boundaries_list.append([])
        
        return piece_boundaries_list
    
    def combine_isolated_svgs(self, input_dir, output_path, svg_size=800):
        """
        単独表示版SVGを組み合わせて全体表示を再現
        
        Args:
            input_dir: 単独表示版SVGが保存されているディレクトリ
            output_path: 出力する結合SVGのパス
            svg_size: 出力SVGのサイズ
        """
        from pathlib import Path
        
        input_dir = Path(input_dir)
        
        # 結合用のSVGを作成
        combined_dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        combined_dwg.add(combined_dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザの背景円を描画
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        combined_dwg.add(combined_dwg.circle(center=center, r=radius,
                                           fill='bisque', stroke='saddlebrown', stroke_width=3))
        
        # 各ピースのSVGを読み込んで配置
        for i in range(self.n):
            svg_path = input_dir / f"piece_{i+1}_isolated.svg"
            if not svg_path.exists():
                self._log(f"警告: {svg_path} が見つかりません")
                continue
            
            # SVGコンテンツを直接生成（元の位置に配置）
            piece_content = self._generate_piece_content_at_original_position(i, svg_size)
            
            # グループとして追加
            if piece_content:
                # XMLを直接解析して要素を追加
                try:
                    # svgwriteの要素として再構築
                    idx = self.pieces[i]
                    piece_points = [(self.px[j], self.py[j]) for j in idx]
                    
                    if len(piece_points) >= 3:
                        try:
                            hull = ConvexHull(piece_points)
                            hull_points = [self._transform_to_svg_coords(piece_points[j][0], piece_points[j][1], svg_size) 
                                          for j in hull.vertices]
                            
                            cmap = plt.get_cmap('tab10', self.n)
                            color = cmap(i)
                            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                                      int(color[1]*255), 
                                                                      int(color[2]*255))
                            
                            combined_dwg.add(combined_dwg.polygon(points=hull_points,
                                                                fill=color_hex,
                                                                opacity=0.6,
                                                                stroke=color_hex,
                                                                stroke_width=0.5))
                        except:
                            pass
                except:
                    pass
        
        # サラミを上に再描画（全体を統一的に表示）
        for cx, cy in self.centers:
            center = self._transform_to_svg_coords(cx, cy, svg_size)
            radius = self.R_salami * svg_size / 2.4
            combined_dwg.add(combined_dwg.circle(center=center, r=radius,
                                               fill='indianred', stroke='darkred', 
                                               stroke_width=1.5, opacity=0.9))
        
        # カット線を描画
        for (e1, e2) in self.cut_edges:
            p1 = self._transform_to_svg_coords(e1[0], e1[1], svg_size)
            p2 = self._transform_to_svg_coords(e2[0], e2[1], svg_size)
            combined_dwg.add(combined_dwg.line(start=p1, end=p2,
                                             stroke='black', stroke_width=2.5))
        
        combined_dwg.save()
        self._log(f"結合SVG生成完了: {output_path}")
    
    def _transform_to_svg_coords(self, x, y, svg_size):
        """座標をSVG座標系に変換"""
        tx = (x + 1.2) * svg_size / 2.4
        ty = (-y + 1.2) * svg_size / 2.4
        return (tx, ty)
    
    def _generate_piece_content_at_original_position(self, piece_index, svg_size):
        """
        ピースを元の位置に配置したSVGコンテンツを生成
        
        Args:
            piece_index: ピースのインデックス
            svg_size: SVGのサイズ
            
        Returns:
            SVGコンテンツ文字列（g要素の中身）
        """
        idx = self.pieces[piece_index]
        cmap = plt.get_cmap('tab10', self.n)
        color = cmap(piece_index)
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                  int(color[1]*255), 
                                                  int(color[2]*255))
        
        # ピースの凸包を計算
        piece_points = [(self.px[j], self.py[j]) for j in idx]
        
        if len(piece_points) < 3:
            return ""
        
        # 凸包を計算してポリゴンとして描画
        try:
            hull = ConvexHull(piece_points)
            hull_points = [self._transform_to_svg_coords(piece_points[i][0], piece_points[i][1], svg_size) 
                          for i in hull.vertices]
            
            # ポリゴンを作成
            polygon_str = f'<polygon points="{" ".join([f"{x},{y}" for x, y in hull_points])}" '
            polygon_str += f'fill="{color_hex}" opacity="0.6" stroke="{color_hex}" stroke-width="0.5"/>'
            
            return polygon_str
        except:
            # 凸包の計算に失敗した場合は空文字列
            return ""
    
    def create_layered_svg(self, output_path, svg_size=800):
        """
        レイヤー構造を持つSVGを生成（各ピースを独立したレイヤーとして）
        
        Args:
            output_path: 出力SVGファイルパス
            svg_size: SVGのサイズ
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景レイヤー
        background_layer = dwg.g(id='background_layer')
        background_layer.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザベースレイヤー
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        background_layer.add(dwg.circle(center=center, r=radius,
                                       fill='bisque', stroke='saddlebrown', stroke_width=3))
        dwg.add(background_layer)
        
        # 各ピースのレイヤー
        pieces_layer = dwg.g(id='pieces_layer')
        for i in range(self.n):
            piece_layer = dwg.g(id=f'piece_{i+1}_layer', class_='pizza-piece')
            
            # ピースの形状を追加
            idx = self.pieces[i]
            cmap = plt.get_cmap('tab10', self.n)
            color = cmap(i)
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                      int(color[1]*255), 
                                                      int(color[2]*255))
            
            # 凸包を計算
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            if len(piece_points) >= 3:
                try:
                    hull = ConvexHull(piece_points)
                    hull_points = [self._transform_to_svg_coords(piece_points[j][0], 
                                                                piece_points[j][1], 
                                                                svg_size) 
                                  for j in hull.vertices]
                    
                    piece_layer.add(dwg.polygon(points=hull_points,
                                              fill=color_hex,
                                              opacity=0.6,
                                              stroke=color_hex,
                                              stroke_width=0.5))
                except:
                    pass
            
            pieces_layer.add(piece_layer)
        
        dwg.add(pieces_layer)
        
        # サラミレイヤー
        salami_layer = dwg.g(id='salami_layer')
        for cx, cy in self.centers:
            center = self._transform_to_svg_coords(cx, cy, svg_size)
            radius = self.R_salami * svg_size / 2.4
            salami_layer.add(dwg.circle(center=center, r=radius,
                                       fill='indianred', stroke='darkred', 
                                       stroke_width=1.5, opacity=0.9))
        dwg.add(salami_layer)
        
        # カット線レイヤー
        cuts_layer = dwg.g(id='cuts_layer')
        for (e1, e2) in self.cut_edges:
            p1 = self._transform_to_svg_coords(e1[0], e1[1], svg_size)
            p2 = self._transform_to_svg_coords(e2[0], e2[1], svg_size)
            cuts_layer.add(dwg.line(start=p1, end=p2,
                                   stroke='black', stroke_width=2.5))
        dwg.add(cuts_layer)
        
        # CSSスタイルを追加（オプション）
        dwg.defs.add(dwg.style('''
            .pizza-piece {
                transition: opacity 0.3s ease;
            }
            .pizza-piece:hover {
                opacity: 0.8;
            }
        '''))
        
        dwg.save()
        self._log(f"レイヤー構造SVG生成完了: {output_path}")
    
    def create_exploded_svg(self, output_path, svg_size=800, explode_factor=0.33):
        """
        ピースを円弧方向にずらして表示するSVGを生成（エクスプローデッドビュー）
        
        Args:
            output_path: 出力SVGファイルパス
            svg_size: SVGのサイズ
            explode_factor: ずらす距離の係数（半径に対する比率、デフォルト0.33）
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザの円を薄く描画（ガイドとして）
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        dwg.add(dwg.circle(center=center, r=radius,
                          fill='none', stroke='lightgray', stroke_width=1,
                          stroke_dasharray='5,5', opacity=0.5))
        
        # 各ピースの中心角度を計算
        piece_angles = []
        for i in range(self.n):
            idx = self.pieces[i]
            if len(idx) > 0:
                # 各点の角度を計算
                angles = np.arctan2(self.py[idx], self.px[idx])
                # 平均角度を計算（円を横切る場合の処理を含む）
                mean_angle = np.angle(np.mean(np.exp(1j * angles)))
                piece_angles.append(mean_angle)
            else:
                piece_angles.append(0)
        
        # 各ピースを描画（ずらして配置）
        for i in range(self.n):
            # ずらす方向と距離を計算
            angle = piece_angles[i]
            offset_dist = self.R_pizza * explode_factor * svg_size / 2.4
            offset_x = offset_dist * np.cos(angle)
            offset_y = offset_dist * np.sin(angle)
            
            # ピースの凸包を計算
            idx = self.pieces[i]
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            
            if len(piece_points) >= 3:
                try:
                    hull = ConvexHull(piece_points)
                    # 変換してずらした座標を計算
                    hull_points = []
                    for j in hull.vertices:
                        x, y = piece_points[j]
                        # SVG座標に変換
                        tx = (x + 1.2) * svg_size / 2.4
                        ty = (-y + 1.2) * svg_size / 2.4
                        # ずらしを適用
                        tx += offset_x
                        ty -= offset_y  # Y軸は反転しているため
                        hull_points.append((tx, ty))
                    
                    # 色を取得
                    cmap = plt.get_cmap('tab10', self.n)
                    color = cmap(i)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                              int(color[1]*255), 
                                                              int(color[2]*255))
                    
                    # ピースを描画
                    dwg.add(dwg.polygon(points=hull_points,
                                      fill=color_hex,
                                      opacity=0.7,
                                      stroke='darkgray',
                                      stroke_width=1))
                    
                    # このピースに含まれるサラミを描画
                    for cx, cy in self.centers:
                        # サラミがこのピースに含まれるかチェック
                        salami_points = 0
                        for j in idx:
                            if (self.px[j] - cx)**2 + (self.py[j] - cy)**2 <= self.R_salami**2:
                                salami_points += 1
                        
                        if salami_points > 10:  # 十分な点がサラミ内にある場合
                            # サラミの座標を変換してずらす
                            sx = (cx + 1.2) * svg_size / 2.4 + offset_x
                            sy = (-cy + 1.2) * svg_size / 2.4 - offset_y
                            radius = self.R_salami * svg_size / 2.4
                            dwg.add(dwg.circle(center=(sx, sy), r=radius,
                                               fill='indianred', stroke='darkred', 
                                               stroke_width=1, opacity=0.9))
                    
                    # ピース番号を表示
                    # 重心を計算
                    cx = np.mean([p[0] for p in hull_points])
                    cy = np.mean([p[1] for p in hull_points])
                    dwg.add(dwg.text(str(i + 1),
                                   insert=(cx, cy),
                                   font_size='16px',
                                   font_family='Arial',
                                   fill='white',
                                   font_weight='bold',
                                   text_anchor='middle',
                                   dominant_baseline='middle'))
                    
                except:
                    pass
        
        # タイトルを追加
        dwg.add(dwg.text('Exploded Pizza View',
                       insert=(svg_size / 2, 30),
                       font_size='24px',
                       font_family='Arial',
                       fill='saddlebrown',
                       font_weight='bold',
                       text_anchor='middle'))
        
        dwg.save()
        self._log(f"エクスプローデッドビューSVG生成完了: {output_path}")
    
    def create_interactive_exploded_svg(self, output_path, svg_size=800, explode_factor=0.33):
        """
        インタラクティブなエクスプローデッドビューSVGを生成（静的な分離表示）
        
        Args:
            output_path: 出力SVGファイルパス
            svg_size: SVGのサイズ
            explode_factor: ずらす距離の係数
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザの円を薄く描画
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        dwg.add(dwg.circle(center=center, r=radius,
                          fill='none', stroke='lightgray', stroke_width=1,
                          stroke_dasharray='5,5', opacity=0.5))
        
        # 各ピースの中心角度を計算
        piece_angles = []
        for i in range(self.n):
            idx = self.pieces[i]
            if len(idx) > 0:
                angles = np.arctan2(self.py[idx], self.px[idx])
                mean_angle = np.angle(np.mean(np.exp(1j * angles)))
                piece_angles.append(mean_angle)
            else:
                piece_angles.append(0)
        
        # 各ピースをアニメーション付きで描画
        for i in range(self.n):
            # グループを作成
            piece_group = dwg.g(id=f'piece_{i+1}_group', class_='animated-piece')
            
            # アニメーションのための初期位置と最終位置を計算
            angle = piece_angles[i]
            offset_dist = self.R_pizza * explode_factor * svg_size / 2.4
            offset_x = offset_dist * np.cos(angle)
            offset_y = -offset_dist * np.sin(angle)  # Y軸反転
            
            # アニメーションを追加（アニメーションなしの静的なグループに変更）
            # svgwriteのアニメーション機能に問題があるため、静的な変換を使用
            piece_group['transform'] = f'translate({offset_x}, {offset_y})'
            
            # ピースの形状を追加（元の位置で描画）
            idx = self.pieces[i]
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            
            if len(piece_points) >= 3:
                try:
                    hull = ConvexHull(piece_points)
                    hull_points = [self._transform_to_svg_coords(piece_points[j][0], 
                                                                piece_points[j][1], 
                                                                svg_size) 
                                  for j in hull.vertices]
                    
                    cmap = plt.get_cmap('tab10', self.n)
                    color = cmap(i)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                              int(color[1]*255), 
                                                              int(color[2]*255))
                    
                    piece_group.add(dwg.polygon(points=hull_points,
                                              fill=color_hex,
                                              opacity=0.7,
                                              stroke='darkgray',
                                              stroke_width=1))
                    
                    # サラミを追加
                    for cx, cy in self.centers:
                        salami_points = 0
                        for j in idx:
                            if (self.px[j] - cx)**2 + (self.py[j] - cy)**2 <= self.R_salami**2:
                                salami_points += 1
                        
                        if salami_points > 10:
                            center = self._transform_to_svg_coords(cx, cy, svg_size)
                            radius = self.R_salami * svg_size / 2.4
                            piece_group.add(dwg.circle(center=center, r=radius,
                                                     fill='indianred', stroke='darkred', 
                                                     stroke_width=1, opacity=0.9))
                except:
                    pass
            
            dwg.add(piece_group)
        
        # CSSスタイルを追加
        dwg.defs.add(dwg.style('''
            .animated-piece {
                cursor: pointer;
            }
            .animated-piece:hover {
                opacity: 0.9;
            }
        '''))
        
        dwg.save()
        self._log(f"インタラクティブエクスプローデッドビューSVG生成完了: {output_path}")
    
    def create_properly_divided_exploded_svg(self, output_path, svg_size=800, explode_factor=0.33):
        """
        サラミを正しく分割して、各ピースを円弧方向にずらして表示するSVGを生成
        
        Args:
            output_path: 出力SVGファイルパス
            svg_size: SVGのサイズ
            explode_factor: ずらす距離の係数（半径に対する比率）
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザの円を薄く描画（ガイドとして）
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        dwg.add(dwg.circle(center=center, r=radius,
                          fill='none', stroke='lightgray', stroke_width=1,
                          stroke_dasharray='5,5', opacity=0.5))
        
        # 各ピースの中心角度を計算
        piece_angles = []
        for i in range(self.n):
            idx = self.pieces[i]
            if len(idx) > 0:
                # 各点の角度を計算
                angles = np.arctan2(self.py[idx], self.px[idx])
                # 平均角度を計算（円を横切る場合の処理を含む）
                mean_angle = np.angle(np.mean(np.exp(1j * angles)))
                piece_angles.append(mean_angle)
            else:
                piece_angles.append(0)
        
        # 各ピースを描画（ずらして配置）
        for i in range(self.n):
            # ずらす方向と距離を計算
            angle = piece_angles[i]
            offset_dist = self.R_pizza * explode_factor * svg_size / 2.4
            offset_x = offset_dist * np.cos(angle)
            offset_y = offset_dist * np.sin(angle)
            
            # ピースグループを作成
            piece_group = dwg.g(id=f'piece_{i+1}_group')
            
            # ピースの凸包を計算
            idx = self.pieces[i]
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            
            if len(piece_points) >= 3:
                try:
                    hull = ConvexHull(piece_points)
                    # 変換してずらした座標を計算
                    hull_points = []
                    for j in hull.vertices:
                        x, y = piece_points[j]
                        # SVG座標に変換
                        tx = (x + 1.2) * svg_size / 2.4
                        ty = (-y + 1.2) * svg_size / 2.4
                        # ずらしを適用
                        tx += offset_x
                        ty -= offset_y  # Y軸は反転しているため
                        hull_points.append((tx, ty))
                    
                    # クリッピングパスを定義（ずらした位置で）
                    clip_id = f'piece_clip_exploded_{i}'
                    clip_path = dwg.defs.add(dwg.clipPath(id=clip_id))
                    clip_path.add(dwg.polygon(points=hull_points))
                    
                    # 色を取得
                    cmap = plt.get_cmap('tab10', self.n)
                    color = cmap(i)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                              int(color[1]*255), 
                                                              int(color[2]*255))
                    
                    # ピースを描画
                    piece_group.add(dwg.polygon(points=hull_points,
                                               fill=color_hex,
                                               opacity=0.7,
                                               stroke='darkgray',
                                               stroke_width=1))
                    
                    # サラミグループ（クリッピング適用）
                    salami_group = dwg.g(clip_path=f'url(#{clip_id})')
                    
                    # このピースに含まれるサラミを描画
                    for cx, cy in self.centers:
                        # サラミの円とピースの交差を計算
                        # モンテカルロ点でサラミの重なり面積を推定
                        overlap_points = 0
                        for j in idx:
                            if (self.px[j] - cx)**2 + (self.py[j] - cy)**2 <= self.R_salami**2:
                                overlap_points += 1
                        
                        # 重なり面積が一定以上ならサラミを描画
                        if overlap_points > 0:
                            # サラミの中心をずらして描画
                            sx = (cx + 1.2) * svg_size / 2.4 + offset_x
                            sy = (-cy + 1.2) * svg_size / 2.4 - offset_y
                            salami_radius = self.R_salami * svg_size / 2.4
                            
                            # 重なり面積に応じて透明度を調整（部分的なサラミを表現）
                            total_salami_points = sum(1 for px, py in zip(self.px, self.py) 
                                                    if (px - cx)**2 + (py - cy)**2 <= self.R_salami**2)
                            overlap_ratio = overlap_points / max(total_salami_points, 1)
                            
                            # サラミを描画（完全に含まれる場合は通常の色、部分的な場合は薄く）
                            if overlap_ratio > 0.8:
                                salami_opacity = 0.9
                                salami_fill = 'indianred'
                            else:
                                salami_opacity = 0.6
                                salami_fill = 'lightcoral'
                            
                            salami_group.add(dwg.circle(center=(sx, sy), r=salami_radius,
                                                       fill=salami_fill, 
                                                       stroke='darkred', 
                                                       stroke_width=1, 
                                                       opacity=salami_opacity))
                    
                    piece_group.add(salami_group)
                    
                    
                except:
                    pass
            
            dwg.add(piece_group)
        
        
        dwg.save()
        self._log(f"正しく分割されたエクスプローデッドビューSVG生成完了: {output_path}")
    
    def generate_piece_svgs_with_proper_division(self, output_dir, svg_size=400):
        """
        各ピースの個別SVGを生成（サラミを正しく分割して表示）
        
        Args:
            output_dir: 出力ディレクトリ
            svg_size: SVGのサイズ
            
        Returns:
            生成されたSVGファイルパスのリスト
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        svg_paths = []
        
        for i in range(self.n):
            svg_path = output_dir / f"piece_{i+1}_properly_divided.svg"
            self._generate_single_piece_svg_with_proper_division(
                piece_index=i,
                output_path=str(svg_path),
                svg_size=svg_size
            )
            svg_paths.append(str(svg_path))
            
        return svg_paths
    
    def _generate_single_piece_svg_with_proper_division(self, piece_index, output_path, svg_size=400):
        """
        単一ピースのSVGを生成（サラミを正しく分割して表示）
        
        Args:
            piece_index: ピースのインデックス
            output_path: 出力パス
            svg_size: SVGのサイズ
        """
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        
        # 背景を設定
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # このピースのモンテカルロ点
        idx = self.pieces[piece_index]
        piece_points = [(self.px[j], self.py[j]) for j in idx]
        
        if len(piece_points) == 0:
            dwg.save()
            return
        
        # ピースの重心と範囲を計算
        cx = np.mean([p[0] for p in piece_points])
        cy = np.mean([p[1] for p in piece_points])
        max_dist = max(np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in piece_points)
        
        # 座標変換（ピースを中心に配置）
        scale = svg_size / (3 * max_dist) if max_dist > 0 else svg_size / 2.4
        
        def transform_point(x, y):
            tx = (x - cx) * scale + svg_size / 2
            ty = -(y - cy) * scale + svg_size / 2  # y軸を反転
            return (tx, ty)
        
        # ピースの凸包を計算
        if len(piece_points) > 3:
            try:
                hull = ConvexHull(piece_points)
                hull_points = [transform_point(piece_points[i][0], piece_points[i][1]) 
                            for i in hull.vertices]
                
                # クリッピングパスを定義
                clip_path = dwg.defs.add(dwg.clipPath(id=f'piece_clip_{piece_index}'))
                clip_path.add(dwg.polygon(points=hull_points))
                
                # 凸包を描画（ピースの背景）
                dwg.add(dwg.polygon(points=hull_points,
                                fill='bisque', stroke='saddlebrown', stroke_width=2))
                
                # サラミグループ（クリッピング適用）
                salami_group = dwg.g(clip_path=f'url(#piece_clip_{piece_index})')
                
                # サラミを描画
                for scx, scy in self.centers:
                    # このサラミとピースの重なりを計算
                    overlap_count = 0
                    for j in idx:
                        if (self.px[j] - scx)**2 + (self.py[j] - scy)**2 <= self.R_salami**2:
                            overlap_count += 1
                    
                    if overlap_count > 0:
                        # サラミ全体のモンテカルロ点数を計算
                        total_salami_points = sum(1 for px, py in zip(self.px, self.py) 
                                                if (px - scx)**2 + (py - scy)**2 <= self.R_salami**2)
                        overlap_ratio = overlap_count / max(total_salami_points, 1)
                        
                        # サラミを描画
                        center = transform_point(scx, scy)
                        radius = self.R_salami * scale
                        
                        # 重なり具合に応じて描画スタイルを変更
                        if overlap_ratio > 0.8:  # ほぼ完全に含まれる
                            salami_group.add(dwg.circle(center=center, r=radius,
                                                      fill='indianred', 
                                                      stroke='darkred', 
                                                      stroke_width=1.5, 
                                                      opacity=0.9))
                        elif overlap_ratio > 0.2:  # 部分的に含まれる
                            salami_group.add(dwg.circle(center=center, r=radius,
                                                      fill='lightcoral', 
                                                      stroke='darkred', 
                                                      stroke_width=1.5, 
                                                      opacity=0.7))
                        else:  # わずかに含まれる
                            salami_group.add(dwg.circle(center=center, r=radius,
                                                      fill='mistyrose', 
                                                      stroke='darkred', 
                                                      stroke_width=1, 
                                                      stroke_dasharray='2,2',
                                                      opacity=0.5))
                
                dwg.add(salami_group)
                
                # ピース情報を表示
                info_y = 20
                dwg.add(dwg.text(f'Piece {piece_index + 1}',
                               insert=(10, info_y),
                               font_size='18px',
                               font_family='Arial',
                               fill='saddlebrown',
                               font_weight='bold'))
                
                # 面積情報
                area = len(idx) * self.w
                salami_area = self.on_salami[idx].sum() * self.w
                
                info_y += 25
                dwg.add(dwg.text(f'Pizza area: {area:.3f}',
                               insert=(10, info_y),
                               font_size='14px',
                               font_family='Arial',
                               fill='black'))
                
                info_y += 20
                dwg.add(dwg.text(f'Salami area: {salami_area:.3f}',
                               insert=(10, info_y),
                               font_size='14px',
                               font_family='Arial',
                               fill='black'))
                
            except:
                pass
        
        dwg.save()
    
    def create_animated_exploding_pizza_svg(self, output_path, n_pieces=None, svg_size=800, 
                                           explode_distance=0.4, animation_duration=3.0):
        """
        ピースが外側に動くアニメーションを持つSVGを生成（既存の分割を使用）
        
        Args:
            output_path: 出力SVGファイルパス
            n_pieces: 分割数（Noneの場合は既存の分割を使用）
            svg_size: SVGのサイズ
            explode_distance: 爆発距離（半径に対する比率）
            animation_duration: アニメーション時間（秒）
        """
        # n_piecesが指定された場合のみ再分割
        original_n = None
        if n_pieces is not None and n_pieces != self.n:
            # 一時的に分割数を変更してピザを再分割
            original_n = self.n
            self.n = n_pieces
            
            # 再分割処理
            self.generate_monte_carlo_points()
            self.place_salami_random()
            self.calculate_targets()
            self.divide_pizza()
        
        # SVG作成
        dwg = svgwrite.Drawing(output_path, size=(svg_size, svg_size))
        dwg['viewBox'] = f'0 0 {svg_size} {svg_size}'
        
        # 背景
        dwg.add(dwg.rect((0, 0), (svg_size, svg_size), fill='white'))
        
        # ピザの円を薄く描画（ガイドとして）
        center = (svg_size / 2, svg_size / 2)
        radius = self.R_pizza * svg_size / 2.4
        dwg.add(dwg.circle(center=center, r=radius,
                          fill='none', stroke='lightgray', stroke_width=1,
                          stroke_dasharray='5,5', opacity=0.3))
        
        # 各ピースの中心角度を計算
        piece_angles = []
        for i in range(self.n):
            idx = self.pieces[i]
            if len(idx) > 0:
                angles = np.arctan2(self.py[idx], self.px[idx])
                mean_angle = np.angle(np.mean(np.exp(1j * angles)))
                piece_angles.append(mean_angle)
            else:
                piece_angles.append(i * 2 * np.pi / self.n)
        
        # 各ピースをアニメーション付きで描画
        for i in range(self.n):
            # ピースグループを作成
            piece_group = dwg.g(id=f'animated_piece_{i+1}')
            
            # アニメーションの移動距離を計算
            angle = piece_angles[i]
            offset_dist = self.R_pizza * explode_distance * svg_size / 2.4
            offset_x = offset_dist * np.cos(angle)
            offset_y = -offset_dist * np.sin(angle)  # Y軸反転
            
            # ピースの凸包を計算
            idx = self.pieces[i]
            piece_points = [(self.px[j], self.py[j]) for j in idx]
            
            if len(piece_points) >= 3:
                try:
                    hull = ConvexHull(piece_points)
                    # 元の位置での座標を計算
                    hull_points = []
                    for j in hull.vertices:
                        x, y = piece_points[j]
                        tx = (x + 1.2) * svg_size / 2.4
                        ty = (-y + 1.2) * svg_size / 2.4
                        hull_points.append((tx, ty))
                    
                    # クリッピングパスを定義
                    clip_id = f'piece_clip_anim_{i}'
                    clip_path = dwg.defs.add(dwg.clipPath(id=clip_id))
                    
                    # アニメーション付きクリッピングパス
                    clip_polygon = clip_path.add(dwg.polygon(points=hull_points))
                    
                    # 色を取得
                    cmap = plt.get_cmap('tab10', self.n)
                    color = cmap(i)
                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), 
                                                              int(color[1]*255), 
                                                              int(color[2]*255))
                    
                    # ピースの背景を描画
                    piece_polygon = piece_group.add(dwg.polygon(points=hull_points,
                                                               fill=color_hex,
                                                               opacity=0.8,
                                                               stroke='darkgray',
                                                               stroke_width=1))
                    
                    # サラミグループ（クリッピング適用）
                    salami_group = piece_group.add(dwg.g(clip_path=f'url(#{clip_id})'))
                    
                    # サラミを描画
                    for cx, cy in self.centers:
                        # このピースに含まれるサラミをチェック
                        overlap_points = 0
                        for j in idx:
                            if (self.px[j] - cx)**2 + (self.py[j] - cy)**2 <= self.R_salami**2:
                                overlap_points += 1
                        
                        if overlap_points > 0:
                            sx = (cx + 1.2) * svg_size / 2.4
                            sy = (-cy + 1.2) * svg_size / 2.4
                            salami_radius = self.R_salami * svg_size / 2.4
                            
                            # 重なり面積を計算
                            total_salami_points = sum(1 for px, py in zip(self.px, self.py) 
                                                    if (px - cx)**2 + (py - cy)**2 <= self.R_salami**2)
                            overlap_ratio = overlap_points / max(total_salami_points, 1)
                            
                            if overlap_ratio > 0.2:  # 20%以上重なる場合のみ描画
                                salami_opacity = 0.9 if overlap_ratio > 0.8 else 0.7
                                salami_group.add(dwg.circle(center=(sx, sy), r=salami_radius,
                                                          fill='indianred', 
                                                          stroke='darkred', 
                                                          stroke_width=1, 
                                                          opacity=salami_opacity))
                    
                    
                except Exception as e:
                    self._log(f"ピース{i+1}の描画でエラー: {e}")
            
            dwg.add(piece_group)
        
        # グローバルCSSスタイルを追加
        style_rules = []
        for i in range(self.n):
            angle = piece_angles[i]
            offset_x = self.R_pizza * explode_distance * svg_size / 2.4 * np.cos(angle)
            offset_y = -self.R_pizza * explode_distance * svg_size / 2.4 * np.sin(angle)
            
            # 各ピースのアニメーションを定義（離れた位置で停止）
            # IDとアニメーション名を一致させる（両方とも1から始まる）
            style_rules.append(f'''
        @keyframes explode_piece_{i+1} {{
            0% {{
                transform: translate(0, 0);
                opacity: 0.8;
            }}
            100% {{
                transform: translate({offset_x}px, {offset_y}px);
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
        
        # スタイルを追加
        style_elem = dwg.style(global_style)
        dwg.defs.add(style_elem)
        
        
        # 元の分割数に戻す（再分割した場合のみ）
        if original_n is not None:
            self.n = original_n
        
        dwg.save()
        self._log(f"アニメーション付きピザ爆発SVG生成完了: {output_path}")

    def run(self):
            """分割処理を実行"""
            self._log("ピザ分割処理を開始")
            
            # 1. モンテカルロ点を生成
            self.generate_monte_carlo_points()
            
            # 2. サラミを配置
            self.place_salami_random()
            
            # 3. 目標値を計算
            self.calculate_targets()
            
            # 4. 移動ナイフ法で分割
            self.divide_pizza()
            
            # 5. 結果を出力
            self.print_results()
            
            # 6. 可視化
            self.visualize()
            
            self._log("処理完了")


def main(isDebug=False):
    """メイン関数"""
    # パラメータ設定
    divider = PizzaDivider(
        R_pizza=1.0,      # ピザ半径
        R_salami=0.10,    # サラミ半径
        m=13,             # サラミ枚数
        n=7,              # ピース数
        N_Monte=100_000,  # モンテカルロ点数
        seed=0,           # 乱数シード
        isDebug=isDebug   # デバッグモード
    )
    
    # 実行
    divider.run()
    
    # SVG生成（全体を表示、各ピースを色分け）
    svg_paths = divider.generate_piece_svgs("pizza_pieces_svg")
    print(f"\nSVGファイル生成完了: {len(svg_paths)}個")
    for path in svg_paths:
        print(f"  - {path}")
    
    # SVG生成（各ピースを単独で表示）
    svg_paths_isolated = divider.generate_piece_svgs_isolated("pizza_pieces_svg")
    print(f"\n単独表示SVGファイル生成完了: {len(svg_paths_isolated)}個")
    for path in svg_paths_isolated:
        print(f"  - {path}")


if __name__ == "__main__":
    # コマンドライン引数でデバッグモードを切り替え可能
    import sys
    isDebug = "--debug" in sys.argv
    main(isDebug=isDebug)