#!/usr/bin/env python
# coding: utf-8
"""
顔検出・感情認識統合APIのルーター
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import sys
import os
import logging
from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field

# サービスのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from service.face_find.service import FaceFindService
from service.emotion_recognition.service import EmotionRecognitionService, EMOTION_PAY_WEIGHTS
from utils.binary import safe_b64decode, safe_b64encode, format_image_data_uri

# ロガーの設定
logger = logging.getLogger("app.api.face_emotion_router")

# ルーターの作成
router = APIRouter(tags=["face-emotion"])

# サービスインスタンス
face_find_service = FaceFindService()
emotion_service = EmotionRecognitionService()


# リクエストモデル
class FaceFindRequest(BaseModel):
    count: int = Field(..., description="期待される顔の数", ge=1)
    image: str = Field(..., description="Base64エンコードされた画像データ")


@router.post("/face/emotion")
async def face_emotion(request: FaceFindRequest):
    """
    画像から顔を検出し、各顔の感情を認識して返す（JSON形式）
    
    Parameters:
    - count: 期待される顔の数
    - image: Base64エンコードされた画像データ
    
    Returns:
    - detected: 検出された顔の数
    - results: 検出された顔ごとの結果
      - image: 顔の画像（Base64エンコード、200x200にリサイズ済み）
      - dominant: 最も強い感情
      - scores: 各感情のスコア
      - pay: 支払い確率（0-1の範囲、全顔の合計が1になる）
    - file: "upload.jpg"（固定値）
    
    Example Response:
    ```json
    {
        "detected": 2,
        "results": [
            {
                "image": "data:image/jpeg;base64,...",
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
                "pay": 0.65
            },
            {
                "image": "data:image/jpeg;base64,...",
                "dominant": "neutral",
                "scores": {
                    "angry": 0.05,
                    "disgust": 0.01,
                    "fear": 0.02,
                    "happy": 0.1,
                    "sad": 0.15,
                    "surprise": 0.05,
                    "neutral": 0.62
                },
                "pay": 0.35
            }
        ],
        "file": "upload.jpg"
    }
    ```
    """
    try:
        # 顔検出を実行
        face_result = face_find_service.analyze(request.image, request.count)
        
        # 各顔に対して感情認識を実行
        results = []
        pay_raws = []  # 支払い確率の原値を保存するリスト
        missing_count = 0  # 感情認識に失敗した顔の数
        
        for face_b64 in face_result["faces"]:
            # プレースホルダー画像かどうかを確認
            if face_b64.startswith("placeholder://"):
                # プレースホルダーの場合、プレフィックスを削除して画像部分だけを取得
                image_part = face_b64.replace("placeholder://", "")
                
                # 結果を追加（感情認識はNull）
                results.append({
                    "image": image_part,
                    "dominant": None,
                    "scores": {}
                })
                pay_raws.append(None)
                missing_count += 1
            else:
                # 通常の顔画像の場合
                # 安全なBase64デコード処理を使用
                face_bytes, success = safe_b64decode(face_b64)
                
                if not success or face_bytes is None:
                    # デコードに失敗した場合のエラーハンドリング
                    logger.warning(f"顔画像のBase64デコードに失敗しました")
                    results.append({
                        "image": face_b64,
                        "dominant": None,
                        "scores": {}
                    })
                    pay_raws.append(None)
                    missing_count += 1
                    continue
                
                # 感情認識を実行
                try:
                    emotion_result = emotion_service.analyze(face_bytes)
                    
                    # 支払い確率の原値を計算
                    pay_raw = EmotionRecognitionService.calc_pay_probability(emotion_result["scores"])
                    
                    # 結果を追加
                    results.append({
                        "image": face_b64,
                        "dominant": emotion_result["dominant"],
                        "scores": emotion_result["scores"]
                    })
                    pay_raws.append(pay_raw)
                except Exception as e:
                    logger.error(f"感情認識エラー: {str(e)}")
                    # 感情認識に失敗した場合は null を設定
                    results.append({
                        "image": face_b64,
                        "dominant": None,
                        "scores": {}
                    })
                    pay_raws.append(None)
                    missing_count += 1
        
        # 支払い確率を正規化
        sum_raw = sum(raw for raw in pay_raws if raw is not None)
        
        # 各顔の支払い確率を計算
        for i, pay_raw in enumerate(pay_raws):
            if pay_raw is not None:
                # 感情認識に成功した顔
                if sum_raw > 0:
                    results[i]["pay"] = float(pay_raw / (sum_raw + missing_count))
                else:
                    # すべての顔の支払い確率の原値が0の場合は均等に分配
                    results[i]["pay"] = 1.0 / len(pay_raws)
            else:
                # 感情認識に失敗した顔
                if sum_raw > 0:
                    results[i]["pay"] = 1.0 / (sum_raw + missing_count)
                else:
                    # すべての顔の支払い確率の原値が0の場合は均等に分配
                    results[i]["pay"] = 1.0 / len(pay_raws)
        
        # レスポンスを作成
        response = {
            "detected": face_result["detected"],
            "results": results,
            "file": "upload.jpg"
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"入力値エラー: {str(e)}")
        # 顔検出に関するエラー
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"予期しないエラー: {str(e)}")
        # その他のエラー
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")


@router.post("/face/emotion/form")
async def face_emotion_form(
    file: UploadFile = File(...),
    count: int = Form(..., description="期待される顔の数", ge=1)
):
    """
    画像から顔を検出し、各顔の感情を認識して返す（マルチパートフォームデータ形式）
    
    Parameters:
    - file: 画像ファイル
    - count: 期待される顔の数
    
    Returns:
    - detected: 検出された顔の数
    - results: 検出された顔ごとの結果
      - image: 顔の画像（Base64エンコード、200x200にリサイズ済み）
      - dominant: 最も強い感情
      - scores: 各感情のスコア
      - pay: 支払い確率（0-1の範囲、全顔の合計が1になる）
    - file: アップロードされたファイル名
    
    Example Response:
    ```json
    {
        "detected": 2,
        "results": [
            {
                "image": "data:image/jpeg;base64,...",
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
                "pay": 0.65
            },
            {
                "image": "data:image/jpeg;base64,...",
                "dominant": "neutral",
                "scores": {
                    "angry": 0.05,
                    "disgust": 0.01,
                    "fear": 0.02,
                    "happy": 0.1,
                    "sad": 0.15,
                    "surprise": 0.05,
                    "neutral": 0.62
                },
                "pay": 0.35
            }
        ],
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
        
        # Base64エンコード（安全な関数を使用）
        encoded_data, success = safe_b64encode(contents)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="画像データのBase64エンコードに失敗しました"
            )
        
        # データURIフォーマットに変換
        image_b64 = format_image_data_uri(contents)
        if image_b64 is None:
            raise HTTPException(
                status_code=400,
                detail="画像データのフォーマットに失敗しました"
            )
        
        # 顔検出を実行
        face_result = face_find_service.analyze(image_b64, count)
        
        # 各顔に対して感情認識を実行
        results = []
        pay_raws = []  # 支払い確率の原値を保存するリスト
        missing_count = 0  # 感情認識に失敗した顔の数
        
        for face_b64 in face_result["faces"]:
            # プレースホルダー画像かどうかを確認
            if face_b64.startswith("placeholder://"):
                # プレースホルダーの場合、プレフィックスを削除して画像部分だけを取得
                image_part = face_b64.replace("placeholder://", "")
                
                # 結果を追加（感情認識はNull）
                results.append({
                    "image": image_part,
                    "dominant": None,
                    "scores": {}
                })
                pay_raws.append(None)
                missing_count += 1
            else:
                # 通常の顔画像の場合
                # 安全なBase64デコード処理を使用
                face_bytes, success = safe_b64decode(face_b64)
                
                if not success or face_bytes is None:
                    # デコードに失敗した場合のエラーハンドリング
                    logger.warning(f"顔画像のBase64デコードに失敗しました")
                    results.append({
                        "image": face_b64,
                        "dominant": None,
                        "scores": {}
                    })
                    pay_raws.append(None)
                    missing_count += 1
                    continue
                
                # 感情認識を実行
                try:
                    emotion_result = emotion_service.analyze(face_bytes)
                    
                    # 支払い確率の原値を計算
                    pay_raw = EmotionRecognitionService.calc_pay_probability(emotion_result["scores"])
                    
                    # 結果を追加
                    results.append({
                        "image": face_b64,
                        "dominant": emotion_result["dominant"],
                        "scores": emotion_result["scores"]
                    })
                    pay_raws.append(pay_raw)
                except Exception as e:
                    logger.error(f"感情認識エラー: {str(e)}")
                    # 感情認識に失敗した場合は null を設定
                    results.append({
                        "image": face_b64,
                        "dominant": None,
                        "scores": {}
                    })
                    pay_raws.append(None)
                    missing_count += 1
        
        # 支払い確率を正規化
        sum_raw = sum(raw for raw in pay_raws if raw is not None)
        
        # 各顔の支払い確率を計算
        for i, pay_raw in enumerate(pay_raws):
            if pay_raw is not None:
                # 感情認識に成功した顔
                if sum_raw > 0:
                    results[i]["pay"] = float(pay_raw / (sum_raw + missing_count))
                else:
                    # すべての顔の支払い確率の原値が0の場合は均等に分配
                    results[i]["pay"] = 1.0 / len(pay_raws)
            else:
                # 感情認識に失敗した顔
                if sum_raw > 0:
                    results[i]["pay"] = 1.0 / (sum_raw + missing_count)
                else:
                    # すべての顔の支払い確率の原値が0の場合は均等に分配
                    results[i]["pay"] = 1.0 / len(pay_raws)
        
        # レスポンスを作成
        response = {
            "detected": face_result["detected"],
            "results": results,
            "file": file.filename
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"入力値エラー: {str(e)}")
        # 顔検出に関するエラー
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"予期しないエラー: {str(e)}")
        # その他のエラー
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")
