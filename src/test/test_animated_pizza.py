#!/usr/bin/env python
# coding: utf-8
"""
アニメーション付きピザ爆発SVGのテストスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.pizza_split.salami_devide import PizzaDivider


def test_animated_pizza():
    """アニメーション付きピザ爆発SVGをテスト"""
    print("=== アニメーション付きピザ爆発SVGテスト ===\n")
    
    # 出力ディレクトリを作成
    output_dir = "animated_pizza_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 異なる分割数でテスト
    test_cases = [
        {"n_pieces": 3, "explode_distance": 0.3, "duration": 2.5},
        {"n_pieces": 5, "explode_distance": 0.4, "duration": 3.0},
        {"n_pieces": 6, "explode_distance": 0.35, "duration": 3.5},
        {"n_pieces": 8, "explode_distance": 0.45, "duration": 4.0},
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nテストケース {i+1}: {test_case['n_pieces']}分割")
        print("-" * 40)
        
        # ピザ分割器を初期化
        divider = PizzaDivider(
            R_pizza=1.0,
            R_salami=0.08,  # 少し小さめのサラミ
            m=15,           # サラミの数
            n=7,            # 初期分割数（後で変更される）
            N_Monte=50_000, # 計算速度のため少なめ
            seed=42 + i,    # 各テストで異なるシード
            isDebug=True
        )
        
        # アニメーション付きSVGを生成
        output_path = os.path.join(output_dir, f"animated_pizza_{test_case['n_pieces']}pieces.svg")
        
        print(f"  生成中: {output_path}")
        divider.create_animated_exploding_pizza_svg(
            output_path=output_path,
            n_pieces=test_case['n_pieces'],
            svg_size=600,
            explode_distance=test_case['explode_distance'],
            animation_duration=test_case['duration']
        )
        print(f"  完了!")
        print(f"  - 爆発距離: {test_case['explode_distance'] * 100}%")
        print(f"  - アニメーション時間: {test_case['duration']}秒")
    
    # 大きめのアニメーションも生成
    print("\n\n大きめのアニメーションを生成...")
    print("-" * 40)
    
    divider_large = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.10,
        m=20,
        n=7,
        N_Monte=100_000,
        seed=123,
        isDebug=True
    )
    
    large_output_path = os.path.join(output_dir, "animated_pizza_large_5pieces.svg")
    print(f"生成中: {large_output_path}")
    
    divider_large.create_animated_exploding_pizza_svg(
        output_path=large_output_path,
        n_pieces=5,
        svg_size=800,
        explode_distance=0.5,
        animation_duration=4.0
    )
    
    print("完了!")
    
    print("\n=== テスト完了 ===")
    print(f"\n生成されたファイルは '{output_dir}' ディレクトリに保存されています:")
    print("以下のファイルをWebブラウザで開いてアニメーションを確認してください:")
    for file in os.listdir(output_dir):
        if file.endswith('.svg'):
            print(f"  - {file}")
    print("\nヒント: ピースにマウスをホバーするとアニメーションが一時停止します")


def test_animated_with_proper_division():
    """正しく分割されたサラミでアニメーションをテスト"""
    print("\n\n=== 正しく分割されたサラミでのアニメーションテスト ===\n")
    
    output_dir = "animated_pizza_output"
    
    # 特別なケース：サラミが正しく分割されていることを強調
    divider = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.12,  # 大きめのサラミ
        m=10,           # サラミの数を少なめに
        n=5,
        N_Monte=100_000,
        seed=999,
        isDebug=True
    )
    
    output_path = os.path.join(output_dir, "animated_pizza_properly_divided_demo.svg")
    print(f"生成中: {output_path}")
    
    divider.create_animated_exploding_pizza_svg(
        output_path=output_path,
        n_pieces=5,
        svg_size=800,
        explode_distance=0.6,  # 大きく爆発
        animation_duration=5.0  # ゆっくりアニメーション
    )
    
    print("完了!")
    print("このデモでは、各ピースに含まれるサラミの部分のみが表示されます")


if __name__ == "__main__":
    test_animated_pizza()
    test_animated_with_proper_division()