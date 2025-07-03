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

# サービスのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from service.face_find.service import FaceFindService
from service.emotion_recognition.service import EmotionRecognitionService, EMOTION_PAY_WEIGHTS

# ロガーの設定
logger = logging.getLogger("app.api.face_emotion_router")

# ルーターの作成
router = APIRouter(tags=["face-emotion"])

# サービスインスタンス
face_find_service = FaceFindService()
emotion_service = EmotionRecognitionService()


@router.post("/face/emotion")
async def face_emotion(
    file: UploadFile = File(...),
    count: int = Form(..., description="期待される顔の数", ge=1)
):
    """
    画像から顔を検出し、各顔の感情を認識して返す
    
    Parameters:
    - file: 画像ファイル
    - count: 期待される顔の数
    
    Returns:
    - detected: 検出された顔の数
    - results: 検出された顔ごとの結果
      - dominant: 最も強い感情
      - scores: 各感情のスコア
      - pay: 支払い確率（0-1の範囲、全顔の合計が1になる）
    - file: アップロードされたファイル名
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
        
        # 顔検出を実行
        face_result = face_find_service.analyze(contents, count)
        
        # 各顔に対して感情認識を実行
        results = []
        pay_raws = []  # 支払い確率の原値を保存するリスト
        missing_count = 0  # 感情認識に失敗した顔の数
        
        print(f"検出された顔の数: {face_result['detected']}")
        print(f"期待される顔の数: {count}")
        print(face_result)
        
        for face_bytes in face_result["faces"]:
            # プレースホルダー画像かどうかを確認
            if face_bytes is None:
                # プレースホルダーの場合
                results.append({
                    "dominant": None,
                    "scores": {}
                })
                pay_raws.append(None)
                missing_count += 1
            else:
                # 通常の顔画像の場合
                # 感情認識を実行
                try:
                    emotion_result = emotion_service.analyze(face_bytes)
                    
                    # 支払い確率の原値を計算
                    pay_raw = EmotionRecognitionService.calc_pay_probability(emotion_result["scores"])
                    
                    # 結果を追加
                    results.append({
                        "dominant": emotion_result["dominant"],
                        "scores": emotion_result["scores"]
                    })
                    pay_raws.append(pay_raw)
                except Exception as e:
                    logger.error(f"感情認識エラー: {str(e)}")
                    # 感情認識に失敗した場合は null を設定
                    results.append({
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
