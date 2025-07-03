#!/usr/bin/env python
# coding: utf-8
"""
顔検出APIのテスト
"""

import os
import sys
import pytest
import base64
import json
from fastapi.testclient import TestClient
from pathlib import Path

# テスト対象のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app

# テストクライアント
client = TestClient(app)

# リソースディレクトリ
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resource"


def _load_image_as_base64(image_path):
    """画像ファイルをBase64エンコードして返す"""
    with open(image_path, "rb") as f:
        image_data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"


def test_face_find_success_match_count():
    """
    顔検出APIの正常系テスト - 期待する顔の数と一致する場合
    """
    # テスト用の顔画像ファイルパス
    image_path = RESOURCE_DIR / "face.jpg"
    
    # ファイルが存在することを確認
    assert image_path.exists(), f"テスト用画像が見つかりません: {image_path}"
    
    # 画像をBase64エンコード
    image_base64 = _load_image_as_base64(image_path)
    
    # リクエストデータ
    request_data = {
        "count": 1,  # face.jpgには1つの顔があると想定
        "image": image_base64
    }
    
    # APIリクエスト
    response = client.post("/api/face/find", json=request_data)
    
    # レスポンスの検証
    assert response.status_code == 200, f"APIエラー: {response.text}"
    
    # レスポンスの内容を検証
    data = response.json()
    assert "detected" in data, "レスポンスに 'detected' キーがありません"
    assert "faces" in data, "レスポンスに 'faces' キーがありません"
    assert "file" in data, "レスポンスに 'file' キーがありません"
    
    # 検出された顔の数が期待通りであることを確認
    assert data["detected"] == 1, f"検出された顔の数が期待と一致しません: {data['detected']}"
    
    # 顔画像が返されていることを確認
    assert len(data["faces"]) == 1, f"返された顔画像の数が期待と一致しません: {len(data['faces'])}"
    
    # 顔画像がBase64形式であることを確認
    assert data["faces"][0].startswith("data:image/jpeg;base64,"), "顔画像がBase64形式ではありません"


def test_face_find_mismatch_count():
    """
    顔検出APIの異常系テスト - 期待する顔の数と一致しない場合
    """
    # テスト用の顔画像ファイルパス
    image_path = RESOURCE_DIR / "face.jpg"
    
    # ファイルが存在することを確認
    assert image_path.exists(), f"テスト用画像が見つかりません: {image_path}"
    
    # 画像をBase64エンコード
    image_base64 = _load_image_as_base64(image_path)
    
    # リクエストデータ（期待する顔の数を実際より多く設定）
    request_data = {
        "count": 2,  # face.jpgには1つの顔しかないと想定
        "image": image_base64
    }
    
    # APIリクエスト
    response = client.post("/api/face/find", json=request_data)
    
    # 400 Bad Request が返ることを確認
    assert response.status_code == 400, f"期待する顔の数と一致しないエラーが発生しませんでした: {response.status_code}"
    assert "期待される顔の数" in response.json()["detail"], "エラーメッセージが正しくありません"


def test_face_find_empty_image():
    """
    顔検出APIの異常系テスト - 空の画像データを送信した場合
    """
    # リクエストデータ（空の画像データ）
    request_data = {
        "count": 1,
        "image": ""
    }
    
    # APIリクエスト
    response = client.post("/api/face/find", json=request_data)
    
    # 400 Bad Request が返ることを確認
    assert response.status_code == 400, f"空画像エラーが発生しませんでした: {response.status_code}"
    assert "画像データが空です" in response.json()["detail"], "エラーメッセージが正しくありません"


def test_face_find_no_face():
    """
    顔検出APIの異常系テスト - 顔が含まれない画像を送信した場合
    
    注: このテストはピザの画像など、顔が含まれない画像を使用します。
    """
    # テスト用のピザ画像ファイルパス
    image_path = RESOURCE_DIR / "pizza1.jpg"
    
    # ファイルが存在することを確認
    assert image_path.exists(), f"テスト用画像が見つかりません: {image_path}"
    
    # 画像をBase64エンコード
    image_base64 = _load_image_as_base64(image_path)
    
    # リクエストデータ
    request_data = {
        "count": 1,
        "image": image_base64
    }
    
    # APIリクエスト
    response = client.post("/api/face/find", json=request_data)
    
    # 400 Bad Request が返ることを確認
    assert response.status_code == 400, f"顔なしエラーが発生しませんでした: {response.status_code}"
    assert "顔を検出できませんでした" in response.json()["detail"], "エラーメッセージが正しくありません"
