#!/usr/bin/env python
# coding: utf-8
"""
顔検出APIのテスト用スクリプト
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path

# リソースディレクトリ
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resource"

def test_face_find_api():
    """
    顔検出APIをテスト
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
            "http://localhost:8000/api/face/find",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # レスポンスの表示
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        # 検出された顔の数を表示
        if response.status_code == 200:
            data = response.json()
            print(f"\n検出された顔の数: {data['detected']}")
            print(f"返された顔画像の数: {len(data['faces'])}")
            print(f"最初の顔画像のプレビュー: {data['faces'][0][:50]}...")
    
    except Exception as e:
        print(f"エラー: {str(e)}")

if __name__ == "__main__":
    test_face_find_api()
