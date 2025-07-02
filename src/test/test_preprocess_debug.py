#!/usr/bin/env python3
"""
preprocess.pyのisDebugフラグのテスト
"""
import sys
import os
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from service.pizza_split.preprocess import PreprocessService


def test_debug_modes():
    """デバッグモードのオン/オフをテスト"""
    service = PreprocessService()
    test_image = "resource/pizza1.jpg"
    
    if not Path(test_image).exists():
        print(f"テスト画像が見つかりません: {test_image}")
        return
    
    print("="*60)
    print("テスト1: デバッグモードOFF (is_debug=False)")
    print("="*60)
    
    # デバッグモードOFFで実行
    result, info = service.preprocess_pizza_image(
        test_image,
        output_path="debug/test_normal_mode.jpg",
        is_debug=False
    )
    print(f"✓ 変換完了: is_transformed={info['is_transformed']}")
    print("  → デバッグ出力なし")
    
    print("\n" + "="*60)
    print("テスト2: デバッグモードON (is_debug=True)")
    print("="*60)
    
    # デバッグモードONで実行
    result, info = service.preprocess_pizza_image(
        test_image,
        output_path="debug/test_debug_mode.jpg",
        is_debug=True
    )
    print(f"✓ 変換完了: is_transformed={info['is_transformed']}")
    print("  → デバッグ出力あり")
    
    print("\n✅ テスト完了")


if __name__ == "__main__":
    test_debug_modes()