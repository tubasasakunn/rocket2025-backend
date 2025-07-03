#!/usr/bin/env python
# coding: utf-8
"""
顔検出・感情認識統合APIのテスト用スクリプト
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path

# リソースディレクトリ
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resource"

def test_face_emotion_api():
    """
    顔検出・感情認識統合APIをテスト
    """
    # テスト用の顔画像ファイルパス
    image_path = RESOURCE_DIR / "face.jpg"
    
    # ファイルが存在することを確認
    if not image_path.exists():
        print(f"テスト用画像が見つかりません: {image_path}")
        return
    
    # 画像をBase64エンコード
    with open(image_path, "rb") as f:
        image_data = f.read()
        image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
    
    # リクエストデータ
    request_data = {
        "count": 1,  # face.jpgには1つの顔があると想定
        "image": image_base64
    }
    
    # APIリクエスト
    try:
        response = requests.post(
            "http://localhost:8000/api/face/emotion",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # レスポンスの表示
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        # 検出された顔の数と感情認識結果を表示
        if response.status_code == 200:
            data = response.json()
            print(f"\n検出された顔の数: {data['detected']}")
            print(f"返された結果の数: {len(data['results'])}")
            
            # 各顔の感情認識結果を表示
            for i, result in enumerate(data['results']):
                print(f"\n顔 {i+1}:")
                if result['dominant'] is None:
                    print(f"  検出なし（プレースホルダー）")
                else:
                    print(f"  最も強い感情: {result['dominant']}")
                    print(f"  感情スコア: {json.dumps(result['scores'], indent=2, ensure_ascii=False)}")
                print(f"  支払い確率: {result.get('pay', 'N/A')}")
                print(f"  顔画像のプレビュー: {result['image'][:50]}...")
            
            # 支払い確率の合計が1になることを確認
            total_pay = sum(result.get('pay', 0) for result in data['results'])
            print(f"\n支払い確率の合計: {total_pay}")
            if abs(total_pay - 1.0) < 0.001:  # 浮動小数点の誤差を考慮
                print("支払い確率の合計は1.0です（正常）")
            else:
                print(f"警告: 支払い確率の合計が1.0ではありません: {total_pay}")
    
    except Exception as e:
        print(f"エラー: {str(e)}")

if __name__ == "__main__":
    test_face_emotion_api()
