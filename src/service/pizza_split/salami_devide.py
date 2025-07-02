#!/usr/bin/env python
# coding: utf-8
"""
移動ナイフ法によるピザ分割
・ピザ全体の面積とサラミ面積が各ピースで同じになるように分割
・モンテカルロ法を使った近似
・結果を複数色に塗り分けて可視化
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from datetime import datetime


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
    
    def _save_figure(self, fig, filename):
        """デバッグ時の図の保存"""
        if self.isDebug:
            filepath = os.path.join(self.debug_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self._log(f"図を保存: {filepath}")
    
    def generate_monte_carlo_points(self):
        """モンテカルロ点をピザ内に一様散布"""
        np.random.seed(self.seed)
        r = np.sqrt(np.random.rand(self.N_Monte)) * self.R_pizza
        ang = np.random.rand(self.N_Monte) * 2 * np.pi
        self.px = r * np.cos(ang)
        self.py = r * np.sin(ang)
        self.w = (np.pi * self.R_pizza**2) / self.N_Monte  # 1点あたりの面積
        
        self._log(f"モンテカルロ点生成完了: {self.N_Monte}点")
        
        if self.isDebug:
            # モンテカルロ点の分布を可視化
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(self.px, self.py, s=0.5, alpha=0.3)
            ax.add_patch(Circle((0, 0), self.R_pizza, fill=False, ec='black', lw=2))
            ax.set_aspect('equal')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"モンテカルロ点分布 (N={self.N_Monte})")
            self._save_figure(fig, "monte_carlo_points.png")
            plt.close(fig)
    
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
        
        if self.isDebug:
            # サラミ配置を可視化
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.add_patch(Circle((0, 0), self.R_pizza, fc='bisque', ec='black', lw=2))
            for cx, cy in self.centers:
                ax.add_patch(Circle((cx, cy), self.R_salami, fc='indianred', ec='brown'))
            ax.set_aspect('equal')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"サラミ配置 (m={self.m})")
            self._save_figure(fig, "salami_placement.png")
            plt.close(fig)
    
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
    
    def find_optimal_cut(self, remain):
        """
        最適なカット角度を見つける
        
        Args:
            remain: 未確定点のマスク
            
        Returns:
            theta_star: 最適角度
            d_star: 最適距離
            maskL: 最適カットでの左側半平面マスク
            slice_idx: このピースに含まれる点のインデックス
        """
        idx = np.where(remain)[0]  # 未確定点のインデックス
        
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
        
        slice_idx = idx[maskL]
        return theta_star, d_star, maskL, slice_idx
    
    def calculate_cut_edges(self, theta, d):
        """
        カット線の両端点を計算
        
        Args:
            theta: カット角度
            d: カット位置までの距離
            
        Returns:
            e1, e2: カット線の両端点
        """
        nx, ny = np.cos(theta), np.sin(theta)
        t_edge = np.sqrt(max(0.0, self.R_pizza**2 - d**2))  # 数値誤差対策
        e1 = (d * nx - ny * t_edge, d * ny + nx * t_edge)
        e2 = (d * nx + ny * t_edge, d * ny - nx * t_edge)
        return e1, e2
    
    def divide_pizza(self):
        """移動ナイフ法でピザを分割"""
        self.remain = np.ones(self.N_Monte, dtype=bool)  # 未確定点
        self.pieces = []  # 各ピースの点インデックス
        self.cut_edges = []  # 各カットの両端点
        
        # n-1回カットを実行
        for i in range(self.n - 1):
            self._log(f"\nカット {i+1}/{self.n-1} を実行中...")
            
            # 最適なカットを見つける
            theta_star, d_star, maskL, slice_idx = self.find_optimal_cut(self.remain)
            
            # ピース確定
            self.pieces.append(slice_idx)
            self.remain[slice_idx] = False
            
            # カット線の両端点を保存
            e1, e2 = self.calculate_cut_edges(theta_star, d_star)
            self.cut_edges.append((e1, e2))
            
            if self.isDebug:
                # 各カット後の状態を可視化
                A = len(slice_idx) * self.w
                B = self.on_salami[slice_idx].sum() * self.w
                self._log(f"ピース{i+1}: 面積={A:.4f}, サラミ面積={B:.4f}")
        
        # 残った点が最後のピース
        self.pieces.append(np.where(self.remain)[0])
        self._log("\n分割完了")
    
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
        
        if self.isDebug:
            self._save_figure(fig, "final_result.png")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
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


if __name__ == "__main__":
    # コマンドライン引数でデバッグモードを切り替え可能
    import sys
    isDebug = "--debug" in sys.argv
    main(isDebug=isDebug)