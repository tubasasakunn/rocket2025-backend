#!/usr/bin/env python
# coding: utf-8
"""
エラー診断用スクリプト

UnicodeDecodeErrorが発生しているAPIエンドポイントを特定するためのスクリプト
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path

# 親ディレクトリをパスに追加（相対インポート用）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# APIのベースURL
API_BASE_URL = "http://localhost:9000/api"

def load_test_image(filename):
    """テスト用の画像をロードしてBase64エンコード"""
    image_path = Path(parent_dir) / "resource" / filename
    
    if not image_path.exists():
        print(f"エラー: ファイル {image_path} が見つかりません")
        return None
        
    with open(image_path, "rb") as f:
        image_data = f.read()
        
    # Base64エンコード
    base64_data = base64.b64encode(image_data).decode('ascii')
    return base64_data

def test_face_find_api():
    """顔検出APIをテスト"""
    print("\n=== 顔検出API (/api/face/find) のテスト ===")
    
    # テスト画像をロード
    base64_image = load_test_image("face.jpg")
    if not base64_image:
        return False
        
    # リクエストデータ
    payload = {
        "count": 1,
        "image": f"data:image/jpeg;base64,{base64_image}"
    }
    
    try:
        # APIリクエスト
        response = requests.post(f"{API_BASE_URL}/face/find", json=payload)
        
        # レスポンスの確認
        if response.status_code == 200:
            print("✅ 成功: 顔検出APIは正常に動作しています")
            return True
        else:
            print(f"❌ エラー: ステータスコード {response.status_code}")
            print(f"エラーメッセージ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 例外発生: {str(e)}")
        return False

def test_emotion_recognition_api():
    """感情認識APIをテスト"""
    print("\n=== 感情認識API (/api/emotion/recognition) のテスト ===")
    
    # テスト画像をロード
    image_path = Path(parent_dir) / "resource" / "face.jpg"
    
    if not image_path.exists():
        print(f"エラー: ファイル {image_path} が見つかりません")
        return False
    
    try:
        # マルチパートフォームデータとしてファイルを送信
        with open(image_path, "rb") as f:
            files = {"file": ("face.jpg", f, "image/jpeg")}
            response = requests.post(f"{API_BASE_URL}/emotion/recognition", files=files)
        
        # レスポンスの確認
        if response.status_code == 200:
            print("✅ 成功: 感情認識APIは正常に動作しています")
            return True
        else:
            print(f"❌ エラー: ステータスコード {response.status_code}")
            print(f"エラーメッセージ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 例外発生: {str(e)}")
        return False

def test_face_emotion_api():
    """顔検出・感情認識統合APIをテスト"""
    print("\n=== 顔検出・感情認識統合API (/api/face/emotion) のテスト ===")
    
    # テスト画像をロード
    base64_image = load_test_image("face.jpg")
    if not base64_image:
        return False
        
    # リクエストデータ
    payload = {
        "count": 1,
        "image": f"data:image/jpeg;base64,{base64_image}"
    }
    
    try:
        # APIリクエスト
        response = requests.post(f"{API_BASE_URL}/face/emotion", json=payload)
        
        # レスポンスの確認
        if response.status_code == 200:
            print("✅ 成功: 顔検出・感情認識統合APIは正常に動作しています")
            return True
        else:
            print(f"❌ エラー: ステータスコード {response.status_code}")
            print(f"エラーメッセージ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 例外発生: {str(e)}")
        return False

def test_pizza_cutter_api():
    """ピザカッターAPIをテスト"""
    print("\n=== ピザカッターAPI (/api/pizza-cutter/divide) のテスト ===")
    
    # テスト画像をロード
    image_path = Path(parent_dir) / "resource" / "pizza1.jpg"
    
    if not image_path.exists():
        print(f"エラー: ファイル {image_path} が見つかりません")
        return False
    
    try:
        # マルチパートフォームデータとしてファイルを送信
        with open(image_path, "rb") as f:
            files = {"file": ("pizza1.jpg", f, "image/jpeg")}
            data = {"n_pieces": 4}
            response = requests.post(f"{API_BASE_URL}/pizza-cutter/divide", files=files, data=data)
        
        # レスポンスの確認
        if response.status_code == 200:
            print("✅ 成功: ピザカッターAPIは正常に動作しています")
            return True
        else:
            print(f"❌ エラー: ステータスコード {response.status_code}")
            print(f"エラーメッセージ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 例外発生: {str(e)}")
        return False

def main():
    """メイン関数"""
    print("=== APIエンドポイント診断ツール ===")
    print("UnicodeDecodeErrorの原因となっているAPIエンドポイントを特定します")
    
    results = {}
    
    # 各APIをテスト
    results["face_find"] = test_face_find_api()
    results["emotion_recognition"] = test_emotion_recognition_api()
    results["face_emotion"] = test_face_emotion_api()
    results["pizza_cutter"] = test_pizza_cutter_api()
    
    # 結果サマリー
    print("\n=== 診断結果 ===")
    for api_name, success in results.items():
        status = "✅ 正常" if success else "❌ エラー"
        print(f"{api_name}: {status}")
    
    # 失敗したAPIがあるか
    failed_apis = [api for api, success in results.items() if not success]
    if failed_apis:
        print("\n問題のあるAPI:")
        for api in failed_apis:
            print(f"- {api}")
        
        print("\n対策:")
        print("1. エンドポイントごとのデータ処理方法を確認してください")
        print("2. バイナリデータが適切にBase64エンコードされているか確認してください")
        print("3. JSONレスポンスにバイナリデータが直接含まれていないか確認してください")
    else:
        print("\nすべてのAPIは正常に動作しています")

if __name__ == "__main__":
    main()
