#!/usr/bin/env python
# coding: utf-8
"""
顔検出APIのルーター
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import sys
import os
import base64
from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field

# サービスのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from service.face_find.service import FaceFindService

# ルーターの作成
router = APIRouter(tags=["face-find"])

# サービスインスタンス
face_find_service = FaceFindService()


# リクエストモデル
class FaceFindRequest(BaseModel):
    count: int = Field(..., description="期待される顔の数", ge=1)
    image: str = Field(..., description="Base64エンコードされた画像データ")


@router.post("/face/find")
async def find_face(request: FaceFindRequest):
    """
    画像から顔を検出し、リサイズして返す（JSON形式）
    
    Parameters:
    - count: 期待される顔の数
    - image: Base64エンコードされた画像データ
    
    Returns:
    - detected: 検出された顔の数
    - faces: 検出された顔の画像（Base64エンコード、200x200にリサイズ済み）
    - file: "upload.jpg"（固定値）
    
    Example Response:
    ```json
    {
        "detected": 2,
        "faces": [
            "data:image/jpeg;base64,...",
            "data:image/jpeg;base64,..."
        ],
        "file": "upload.jpg"
    }
    ```
    """
    try:
        # 顔検出を実行
        result = face_find_service.analyze(request.image, request.count)
        
        # レスポンスを作成
        response = {
            "detected": result["detected"],
            "faces": result["faces"],
            "file": "upload.jpg"
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        # 顔検出に関するエラー
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # その他のエラー
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")


@router.post("/face/find/form")
async def find_face_form(
    file: UploadFile = File(...),
    count: int = Form(..., description="期待される顔の数", ge=1)
):
    """
    画像から顔を検出し、リサイズして返す（マルチパートフォームデータ形式）
    
    Parameters:
    - file: 画像ファイル
    - count: 期待される顔の数
    
    Returns:
    - detected: 検出された顔の数
    - faces: 検出された顔の画像（Base64エンコード、200x200にリサイズ済み）
    - file: アップロードされたファイル名
    
    Example Response:
    ```json
    {
        "detected": 2,
        "faces": [
            "data:image/jpeg;base64,...",
            "data:image/jpeg;base64,..."
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
        
        # Base64エンコード
        image_b64 = f"data:image/jpeg;base64,{base64.b64encode(contents).decode('utf-8')}"
        
        # 顔検出を実行
        result = face_find_service.analyze(image_b64, count)
        
        # レスポンスを作成
        response = {
            "detected": result["detected"],
            "faces": result["faces"],
            "file": file.filename
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        # 顔検出に関するエラー
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # その他のエラー
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")
