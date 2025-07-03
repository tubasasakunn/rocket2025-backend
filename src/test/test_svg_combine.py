#!/usr/bin/env python
# coding: utf-8
"""
SVG組み合わせ機能のテストスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.pizza_split.salami_devide import PizzaDivider


def test_svg_combination():
    """SVG組み合わせ機能をテスト"""
    print("=== SVG組み合わせテスト開始 ===\n")
    
    # ピザ分割器を初期化
    divider = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.10,
        m=13,
        n=7,
        N_Monte=100_000,
        seed=42,
        isDebug=True
    )
    
    print("1. ピザ分割処理を実行...")
    # モンテカルロ点を生成
    divider.generate_monte_carlo_points()
    
    # サラミを配置
    divider.place_salami_random()
    
    # 目標値を計算
    divider.calculate_targets()
    
    # 移動ナイフ法で分割
    divider.divide_pizza()
    
    # 結果を出力
    divider.print_results()
    
    print("\n2. 各種SVGファイルを生成...")
    
    # 出力ディレクトリを作成
    output_dir = "test_svg_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 単独表示版SVGを生成
    print("   - 単独表示版SVGを生成中...")
    isolated_paths = divider.generate_piece_svgs_isolated(output_dir)
    print(f"   生成完了: {len(isolated_paths)}個のファイル")
    
    # 組み合わせSVGを生成
    print("\n3. 単独表示版を組み合わせて全体表示を再現...")
    combined_path = os.path.join(output_dir, "combined_from_isolated.svg")
    divider.combine_isolated_svgs(output_dir, combined_path, svg_size=800)
    print(f"   組み合わせSVG生成: {combined_path}")
    
    # レイヤー構造SVGを生成
    print("\n4. レイヤー構造SVGを生成...")
    layered_path = os.path.join(output_dir, "layered_pizza.svg")
    divider.create_layered_svg(layered_path, svg_size=800)
    print(f"   レイヤー構造SVG生成: {layered_path}")
    
    # エクスプローデッドビューSVGを生成
    print("\n5. エクスプローデッドビューSVGを生成...")
    exploded_path = os.path.join(output_dir, "exploded_pizza.svg")
    divider.create_exploded_svg(exploded_path, svg_size=800, explode_factor=0.33)
    print(f"   エクスプローデッドビューSVG生成: {exploded_path}")
    
    # インタラクティブエクスプローデッドビューSVGを生成
    print("\n6. インタラクティブエクスプローデッドビューSVGを生成...")
    interactive_path = os.path.join(output_dir, "interactive_exploded_pizza.svg")
    divider.create_interactive_exploded_svg(interactive_path, svg_size=800, explode_factor=0.33)
    print(f"   インタラクティブSVG生成: {interactive_path}")
    
    # 正しく分割されたピースのSVGを生成
    print("\n7. 正しく分割されたピースのSVGを生成...")
    properly_divided_paths = divider.generate_piece_svgs_with_proper_division(output_dir)
    print(f"   正しく分割されたピース生成完了: {len(properly_divided_paths)}個のファイル")
    
    # 正しく分割されたエクスプローデッドビューを生成
    print("\n8. 正しく分割されたエクスプローデッドビューSVGを生成...")
    properly_exploded_path = os.path.join(output_dir, "properly_divided_exploded_pizza.svg")
    divider.create_properly_divided_exploded_svg(properly_exploded_path, svg_size=800, explode_factor=0.33)
    print(f"   正しく分割されたエクスプローデッドビューSVG生成: {properly_exploded_path}")
    
    # 通常の全体表示SVGも生成（比較用）
    print("\n9. 通常の全体表示SVGを生成（比較用）...")
    normal_paths = divider.generate_piece_svgs(output_dir)
    print(f"   通常版生成完了: {len(normal_paths)}個のファイル")
    
    print("\n=== テスト完了 ===")
    print(f"\n生成されたファイルは '{output_dir}' ディレクトリに保存されています:")
    print("  - piece_*_isolated.svg: 各ピースの単独表示（従来版）")
    print("  - piece_*_properly_divided.svg: 各ピースの単独表示（サラミ正しく分割）")
    print("  - combined_from_isolated.svg: 単独表示を組み合わせた全体表示")
    print("  - layered_pizza.svg: レイヤー構造を持つ全体表示")
    print("  - exploded_pizza.svg: ピースをずらして表示（エクスプローデッドビュー）")
    print("  - interactive_exploded_pizza.svg: インタラクティブエクスプローデッドビュー")
    print("  - properly_divided_exploded_pizza.svg: 正しく分割されたエクスプローデッドビュー")
    print("  - piece_*.svg: 通常の全体表示（比較用）")


def test_svg_content_generation():
    """SVGコンテンツ生成機能をテスト（ファイル保存なし）"""
    print("\n=== SVGコンテンツ生成テスト ===")
    
    divider = PizzaDivider(
        R_pizza=1.0,
        R_salami=0.08,
        m=10,
        n=5,
        N_Monte=50_000,
        seed=123
    )
    
    # 分割処理
    divider.generate_monte_carlo_points()
    divider.place_salami_random()
    divider.calculate_targets()
    divider.divide_pizza()
    
    # SVGコンテンツを生成
    print("SVGコンテンツを生成中...")
    svg_contents = divider.generate_piece_svgs_isolated_content(svg_size=400)
    
    print(f"生成されたSVGコンテンツ数: {len(svg_contents)}")
    
    # 最初のピースのSVGコンテンツの一部を表示
    if svg_contents:
        print(f"\n最初のピースのSVGコンテンツ（先頭200文字）:")
        print(svg_contents[0][:200] + "...")


if __name__ == "__main__":
    test_svg_combination()
    test_svg_content_generation()