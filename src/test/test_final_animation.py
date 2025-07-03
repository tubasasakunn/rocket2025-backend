#!/usr/bin/env python
# coding: utf-8
"""
最終的に修正されたアニメーション付きピザ爆発SVGのテストスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.pizza_split.salami_devide import PizzaDivider


def test_final_animation():
    """最終的に修正されたアニメーションをテスト"""
    print("=== 最終修正版アニメーション付きピザ爆発SVGテスト ===\n")
    
    # 出力ディレクトリを作成
    output_dir = "final_animated_pizza"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5分割でテスト
    print("5分割のピザでアニメーションをテスト（番号1も正しく動くはず）")
    print("-" * 60)
    
    # ピザ分割器を初期化
    divider = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.10,
        m=15,
        n=5,  # 初期値（後で変更される）
        N_Monte=80_000,
        seed=12345,
        isDebug=True
    )
    
    # アニメーション付きSVGを生成
    output_path = os.path.join(output_dir, "final_animated_pizza_5pieces.svg")
    
    print(f"生成中: {output_path}")
    divider.create_animated_exploding_pizza_svg(
        output_path=output_path,
        n_pieces=5,
        svg_size=700,
        explode_distance=0.5,  # 50%の距離で爆発
        animation_duration=2.0  # 2秒で完了
    )
    
    print("完了!")
    print("\n確認ポイント:")
    print("✓ 番号1のピースが正しくアニメーション")
    print("✓ 各ピースが順番に爆発（0.2秒ずつ遅延）")
    print("✓ ピースは離れた位置で停止")
    print("✓ ホバー時に明るくなり影が付く")
    
    # 異なる分割数でも生成
    test_cases = [
        {"n": 3, "distance": 0.4, "desc": "シンプルな3分割"},
        {"n": 4, "distance": 0.45, "desc": "バランスの良い4分割"},
        {"n": 6, "distance": 0.35, "desc": "細かい6分割"},
    ]
    
    print("\n\n他の分割パターンも生成")
    print("-" * 60)
    
    for case in test_cases:
        output_path = os.path.join(output_dir, f"final_animated_pizza_{case['n']}pieces.svg")
        print(f"\n{case['desc']}: {output_path}")
        
        divider.create_animated_exploding_pizza_svg(
            output_path=output_path,
            n_pieces=case['n'],
            svg_size=600,
            explode_distance=case['distance'],
            animation_duration=1.5
        )
        print("  生成完了!")
    
    print("\n=== テスト完了 ===")
    print(f"\n生成されたファイルは '{output_dir}' ディレクトリに保存されています:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.svg'):
            print(f"  - {file}")
    
    print("\n重要: Webブラウザでファイルを開いて、番号1のピースが正しく動くことを確認してください！")
    print("すべてのピースが中心から外側へ爆発し、離れた位置で停止するはずです。")


if __name__ == "__main__":
    test_final_animation()