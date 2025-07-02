#!/usr/bin/env python
# coding: utf-8
"""
Pizza Score APIのテストスクリプト
"""

import requests
import json
from pathlib import Path


def test_score_endpoint():
    """スコアエンドポイントのテスト"""
    # APIのURL（ローカルサーバー）
    base_url = "http://localhost:8000"
    
    # テスト画像（プロジェクトルートからの相対パス）
    image_files = [
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    # 存在するファイルのみ使用
    valid_files = [f for f in image_files if Path(f).exists()]
    
    if len(valid_files) < 2:
        print("テスト画像が不足しています")
        return
    
    # ファイルを準備
    files = []
    for file_path in valid_files:
        files.append(
            ('files', (Path(file_path).name, open(file_path, 'rb'), 'image/jpeg'))
        )
    
    try:
        # APIを呼び出し
        print("=== Pizza Score API テスト ===\n")
        print(f"送信画像: {[Path(f).name for f in valid_files]}")
        print("\nAPIを呼び出し中...")
        
        response = requests.post(f"{base_url}/api/pizza/score", files=files)
        
        # ファイルを閉じる
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n【結果】")
            print(f"総合スコア: {result['score']:.1f}点 / 100点")
            print(f"\n【スコア詳細】")
            print(f"  ピザスコア: {result['score_details']['pizza_score']:.1f}点")
            print(f"  サラミスコア: {result['score_details']['salami_score']:.1f}点")
            print(f"  重み付け: ピザ={result['score_details']['weights']['pizza_weight']}, "
                  f"サラミ={result['score_details']['weights']['salami_weight']}")
            
            print(f"\n【統計情報】")
            print(f"  処理画像数: {result['statistics']['num_images']}枚")
            print(f"  ピザ面積:")
            print(f"    平均: {result['statistics']['pizza_mean']:.3f}")
            print(f"    標準偏差: {result['statistics']['pizza_std']:.4f}")
            print(f"  サラミ面積:")
            print(f"    平均: {result['statistics']['salami_mean']:.3f}")
            print(f"    標準偏差: {result['statistics']['salami_std']:.4f}")
            
            if result['failed_images']:
                print(f"\n【処理失敗画像】")
                for img in result['failed_images']:
                    print(f"  - {img}")
            
            # JSON形式でも保存
            with open('test_api_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n結果をtest_api_result.jsonに保存しました")
            
        else:
            print(f"\nエラー: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nエラー: APIサーバーに接続できません")
        print("以下のコマンドでサーバーを起動してください:")
        print("cd src/service/pizza_split && uvicorn api_score:app --reload")
    except Exception as e:
        print(f"\nエラー: {e}")


def test_info_endpoint():
    """情報エンドポイントのテスト"""
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/api/pizza/score/info")
        
        if response.status_code == 200:
            info = response.json()
            
            print("\n=== スコア計算仕様 ===")
            print(f"\n【概要】")
            print(f"  {info['scoring_method']['description']}")
            print(f"  スコア範囲: {info['scoring_method']['score_range']}")
            print(f"  重み付け: {info['scoring_method']['weight']}")
            
            print(f"\n【定数】")
            for key, value in info['constants'].items():
                print(f"  {key}: {value}")
            
            print(f"\n【計算式】")
            for key, value in info['formula'].items():
                if key != 'note':
                    print(f"  {key}: {value}")
            print(f"  ※ {info['formula']['note']}")
            
    except Exception as e:
        print(f"\nエラー: {e}")


def main():
    """メイン関数"""
    print("Pizza Score APIのテストを開始します\n")
    
    # 情報エンドポイントをテスト
    test_info_endpoint()
    
    # スコアエンドポイントをテスト
    print("\n" + "="*50 + "\n")
    test_score_endpoint()


if __name__ == "__main__":
    main()