#!/usr/bin/env python
# coding: utf-8
"""
修正されたアニメーション付きピザ爆発SVGのテストスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.pizza_split.salami_devide import PizzaDivider


def test_fixed_animation():
    """修正されたアニメーションをテスト"""
    print("=== 修正されたアニメーション付きピザ爆発SVGテスト ===\n")
    
    # 出力ディレクトリを作成
    output_dir = "fixed_animated_pizza"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5分割でテスト
    print("5分割のピザでアニメーションをテスト")
    print("-" * 40)
    
    # ピザ分割器を初期化
    divider = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.10,
        m=12,
        n=5,  # 初期値（後で変更される）
        N_Monte=80_000,
        seed=42,
        isDebug=True
    )
    
    # アニメーション付きSVGを生成
    output_path = os.path.join(output_dir, "fixed_animated_pizza_5pieces.svg")
    
    print(f"生成中: {output_path}")
    divider.create_animated_exploding_pizza_svg(
        output_path=output_path,
        n_pieces=5,
        svg_size=700,
        explode_distance=0.5,  # 50%の距離で爆発
        animation_duration=2.0  # 2秒で完了
    )
    
    print("完了!")
    print("\n特徴:")
    print("- すべてのピース（1番を含む）がアニメーション")
    print("- ピースは離れた位置で停止")
    print("- ホバー時に明るくなり影が付く")
    print("- 各ピースは0.2秒ずつ遅れて動き始める")
    
    # 3分割のシンプルなバージョンも生成
    print("\n\n3分割のシンプルバージョンも生成")
    print("-" * 40)
    
    simple_output_path = os.path.join(output_dir, "fixed_animated_pizza_3pieces.svg")
    print(f"生成中: {simple_output_path}")
    
    divider.create_animated_exploding_pizza_svg(
        output_path=simple_output_path,
        n_pieces=3,
        svg_size=600,
        explode_distance=0.4,
        animation_duration=1.5
    )
    
    print("完了!")
    
    print("\n=== テスト完了 ===")
    print(f"\n生成されたファイルは '{output_dir}' ディレクトリに保存されています:")
    for file in os.listdir(output_dir):
        if file.endswith('.svg'):
            print(f"  - {file}")
    print("\nWebブラウザでファイルを開いて、すべてのピースが正しく動くことを確認してください。")


if __name__ == "__main__":
    test_fixed_animation()