#!/usr/bin/env python
# coding: utf-8
"""
爆発SVG統合版のprocess.pyをテスト
"""

import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# pizza_splitディレクトリもパスに追加
sys.path.append(str(Path(__file__).parent.parent / "service" / "pizza_split"))

from service.pizza_split.process import PizzaProcessor

def main():
    """テスト実行"""
    print("=== ピザ爆発SVG統合版テスト ===\n")
    
    # プロセッサー初期化
    processor = PizzaProcessor(output_dir="result/exploded_test")
    
    # テスト画像
    test_images = [
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\n{'='*60}")
            print(f"処理中: {image_path}")
            print(f"{'='*60}")
            
            try:
                # 4分割で処理
                result = processor.process_image(
                    image_path,
                    n_pieces=4,
                    debug=False,
                    return_svg_only=False,  # フル処理を実行
                    quiet=False  # 進捗を表示
                )
                
                print("\n生成されたSVGファイル:")
                print("-" * 40)
                
                # 前処理済み画像用
                print("\n[前処理済み画像用SVG]")
                print(f"  通常版: {Path(result['svg_preprocessed']).name}")
                print(f"  爆発前: {Path(result['svg_before']).name}")
                print(f"  爆発後: {Path(result['svg_after']).name}")
                print(f"  アニメ: {Path(result['svg_animated']).name}")
                
                # 元画像用（逆変換適用）
                print("\n[元画像用SVG（逆変換適用）]")
                print(f"  通常版: {Path(result['svg_original']).name}")
                print(f"  爆発前: {Path(result['svg_before_original']).name}")
                print(f"  爆発後: {Path(result['svg_after_original']).name}")
                print(f"  アニメ: {Path(result['svg_animated_original']).name}")
                
                # その他の結果
                print("\n[その他の出力ファイル]")
                print(f"  前処理画像: {Path(result['preprocessed_image']).name}")
                print(f"  結果画像: {Path(result['result_image']).name}")
                print(f"  元画像オーバーレイPNG: {Path(result['result_original_image']).name}")
                print(f"  各ピースSVG: {result['piece_svg_dir']}")
                
                # ピザとサラミの情報
                print(f"\n[検出結果]")
                print(f"  ピザ中心: ({result['pizza_center'][0]:.1f}, {result['pizza_center'][1]:.1f})")
                print(f"  ピザ半径: {result['pizza_radius']:.1f}px")
                print(f"  サラミ数: {len(result['salami_circles'])}")
                print(f"  ピース数: {len(result['pieces'])}")
                
            except Exception as e:
                print(f"\nエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n画像が見つかりません: {image_path}")
    
    print("\n" + "="*60)
    print("テスト完了！")
    print("生成されたSVGファイルはWebブラウザで開いて確認してください。")
    print("特に以下のファイルに注目:")
    print("  - pizza_animated_original_*.svg : 元画像用のアニメーション付きSVG")
    print("  - pizza_after_original_*.svg : 元画像用の爆発後の状態")
    print("="*60)

if __name__ == "__main__":
    main()