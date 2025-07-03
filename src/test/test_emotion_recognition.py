#!/usr/bin/env python
# coding: utf-8
"""
感情認識APIのテスト
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# テスト対象のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app

# テストクライアント
client = TestClient(app)

# リソースディレクトリ
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resource"


def test_emotion_recognition_success():
    """
    感情認識APIの正常系テスト
    """
    # テスト用の顔画像ファイルパス
    image_path = RESOURCE_DIR / "face.jpg"
    
    # ファイルが存在することを確認
    assert image_path.exists(), f"テスト用画像が見つかりません: {image_path}"
    
    # マルチパートフォームデータでファイルをアップロード
    with open(image_path, "rb") as f:
        files = {"file": ("face.jpg", f, "image/jpeg")}
        response = client.post("/api/emotion/recognition", files=files)
    
    # レスポンスの検証
    assert response.status_code == 200, f"APIエラー: {response.text}"
    
    # レスポンスの内容を検証
    data = response.json()
    assert "dominant" in data, "レスポンスに 'dominant' キーがありません"
    assert "scores" in data, "レスポンスに 'scores' キーがありません"
    assert "file" in data, "レスポンスに 'file' キーがありません"
    
    # ファイル名が正しいことを確認
    assert data["file"] == "face.jpg", f"ファイル名が一致しません: {data['file']}"
    
    # スコアが辞書型であることを確認
    assert isinstance(data["scores"], dict), "scores が辞書型ではありません"
    
    # 感情スコアのキーが存在することを確認
    expected_emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for emotion in expected_emotions:
        assert emotion in data["scores"], f"感情スコア '{emotion}' がありません"


def test_emotion_recognition_empty_file():
    """
    空ファイルを送信した場合のテスト
    """
    # 空のファイルを作成
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/emotion/recognition", files=files)
    
    # 400 Bad Request が返ることを確認
    assert response.status_code == 400, f"空ファイルエラーが発生しませんでした: {response.status_code}"
    assert "空のファイル" in response.json()["detail"], "エラーメッセージが正しくありません"


def test_emotion_recognition_no_face():
    """
    顔が含まれない画像を送信した場合のテスト
    
    注: このテストはモックを使用するか、顔のない画像を用意する必要があります。
    現在は実装されていませんが、将来的に追加することを推奨します。
    """
    pass
