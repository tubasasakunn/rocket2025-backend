#!/usr/bin/env python
# coding: utf-8
"""
画像と分割人数を受け取ってSVGを返すAPI
"""

import sys
import argparse
from pathlib import Path
from typing import Union, Optional

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))

from process import PizzaProcessor


def process_pizza_image(image_path: Union[str, Path], n_people: int, output_dir: Optional[str] = None) -> str:
    """
    ピザ画像を指定人数で分割し、SVGコンテンツを返す
    
    Args:
        image_path: 入力画像のパス
        n_people: 分割する人数
        output_dir: 出力ディレクトリ（省略時は"result/process"）
        
    Returns:
        str: 生成されたSVGの内容（XML文字列）
        
    Raises:
        FileNotFoundError: 画像ファイルが見つからない場合
        ValueError: 分割人数が不正な場合
    """
    # 入力検証
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    
    if n_people < 2:
        raise ValueError(f"分割人数は2人以上である必要があります: {n_people}")
    
    if n_people > 99:
        raise ValueError(f"分割人数が多すぎます（最大99人）: {n_people}")
    
    # プロセッサーを初期化
    if output_dir:
        processor = PizzaProcessor(output_dir=output_dir)
    else:
        processor = PizzaProcessor()
    
    # 画像を処理
    print(f"ピザ画像を{n_people}人で分割します...", file=sys.stderr)
    result = processor.process_image(
        str(image_path),
        n_pieces=n_people,
        debug=False,
        return_svg_only=True,  # SVGのみを返すモード
        quiet=True  # 静かなモード
    )
    
    # SVGコンテンツを取得
    svg_content = result.get('svg_content')
    if not svg_content:
        raise RuntimeError("SVGコンテンツの生成に失敗しました")
    
    print(f"SVGコンテンツを生成しました（{len(svg_content)}文字）", file=sys.stderr)
    return svg_content


def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(
        description='ピザ画像を指定人数で分割してSVGを生成します'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='入力画像のパス'
    )
    parser.add_argument(
        'n_people',
        type=int,
        help='分割する人数（2〜99）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力ディレクトリ（省略時は"result/process"）'
    )
    
    args = parser.parse_args()
    
    try:
        svg_content = process_pizza_image(
            args.image_path,
            args.n_people,
            args.output_dir
        )
        # SVGコンテンツを標準出力に出力
        print(svg_content)
        return 0
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())