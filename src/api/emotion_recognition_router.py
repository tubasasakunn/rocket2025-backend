#!/usr/bin/env python
# coding: utf-8
"""
感情認識APIのルーター
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys
import os
from typing import Dict, Any, Union, Optional

# サービスのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from service.emotion_recognition.service import EmotionRecognitionService

# ルーターの作成
router = APIRouter(tags=["emotion-recognition"])

# サービスインスタンス
emotion_service = EmotionRecognitionService()


@router.post("/emotion/recognition")
async def recognize_emotion(file: UploadFile = File(...)):
    """
    顔画像から感情を認識
    
    Parameters:
    - file: 顔画像ファイル
    
    Returns:
    - dominant: 最も強い感情（検出できない場合はNone）
    - scores: 各感情のスコア（0-1の範囲、合計1）
    - file: アップロードされたファイル名
    
    Example Response:
    ```json
    {
        "dominant": "happy",
        "scores": {
            "angry": 0.01,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.92,
            "sad": 0.03,
            "surprise": 0.02,
            "neutral": 0.02
        },
        "file": "face.jpg"
    }
    ```
    """
    try:
        # ファイルの内容を読み込み
        contents = await file.read()
        
        # ファイルが空でないか確認
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="空のファイルが送信されました"
            )
        
        # 感情分析を実行
        result = emotion_service.analyze(contents)
        
        # 顔が検出されなかった場合
        if result["dominant"] is None:
            raise HTTPException(
                status_code=400,
                detail="画像から顔を検出できませんでした"
            )
        
        # レスポンスを作成
        response = {
            "dominant": result["dominant"],
            "scores": result["scores"],
            "file": file.filename
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")
