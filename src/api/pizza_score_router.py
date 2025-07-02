#!/usr/bin/env python
# coding: utf-8
"""
ピザスコア計算APIのルーター
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
from pathlib import Path

# サービスのインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from service.pizza_split.score_multi import MultiImageScoreCalculator
from service.pizza_split.api_score import PizzaScoreAPI, ScoreConstants

# ルーターの作成
router = APIRouter(tags=["pizza-score"])

# APIインスタンス
score_api = PizzaScoreAPI()


@router.post("/pizza/score")
async def calculate_pizza_score(files: List[UploadFile] = File(...)):
    """
    複数のピザ画像から分割均等性スコアを計算
    
    Parameters:
    - files: 複数のピザ画像ファイル（最低2枚）
    
    Returns:
    - score: 総合スコア（0-100点、高いほど均等）
    - score_details: スコアの詳細（ピザ・サラミ個別スコア等）
    - statistics: 統計情報（標準偏差、平均値等）
    - images_processed: 処理された画像数
    - failed_images: 処理に失敗した画像
    - files: アップロードされたファイル名
    
    Example Response:
    ```json
    {
        "score": 79.8,
        "score_details": {
            "total_score": 79.8,
            "pizza_score": 36.8,
            "salami_score": 87.9,
            "weights": {
                "pizza_weight": 1.0,
                "salami_weight": 5.0
            }
        },
        "statistics": {
            "num_images": 2,
            "pizza_std": 0.0100,
            "salami_std": 0.0124,
            "pizza_mean": 0.7788,
            "salami_mean": 0.1707
        },
        "images_processed": 2,
        "failed_images": [],
        "files": ["pizza1.jpg", "pizza2.jpg"]
    }
    ```
    """
    try:
        # ファイル数チェック
        if len(files) < ScoreConstants.MIN_IMAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"最低{ScoreConstants.MIN_IMAGES}枚以上の画像が必要です"
            )
        
        # スコアを計算
        result = await score_api.process_uploaded_files(files)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")


@router.get("/pizza/score/info")
async def get_score_info():
    """
    スコア計算の仕様を取得
    
    Returns:
    - scoring_method: スコア計算方法の説明
    - constants: 使用している定数
    - formula: 計算式の説明
    
    Example Response:
    ```json
    {
        "scoring_method": {
            "description": "複数のピザ画像から、ピザとサラミの面積の標準偏差を計算し、均等性をスコア化します",
            "score_range": "0-100点（高いほど均等）",
            "weight": "サラミの標準偏差はピザの5倍重要視されます"
        },
        "constants": {
            "pizza_weight": 1.0,
            "salami_weight": 5.0,
            "pizza_std_baseline": 0.10,
            "salami_std_baseline": 0.05,
            "decay_rate": 2.0
        },
        "formula": {
            "individual_score": "score = 100 * exp(-decay_rate * std / baseline)",
            "total_score": "(pizza_weight * pizza_score + salami_weight * salami_score) / (pizza_weight + salami_weight)",
            "note": "標準偏差が基準値の場合、スコアは約37点となります"
        }
    }
    ```
    """
    return {
        "scoring_method": {
            "description": "複数のピザ画像から、ピザとサラミの面積の標準偏差を計算し、均等性をスコア化します",
            "score_range": "0-100点（高いほど均等）",
            "weight": "サラミの標準偏差はピザの5倍重要視されます"
        },
        "constants": {
            "pizza_weight": ScoreConstants.PIZZA_WEIGHT,
            "salami_weight": ScoreConstants.SALAMI_WEIGHT,
            "pizza_std_baseline": ScoreConstants.PIZZA_STD_BASELINE,
            "salami_std_baseline": ScoreConstants.SALAMI_STD_BASELINE,
            "decay_rate": ScoreConstants.DECAY_RATE,
            "min_images": ScoreConstants.MIN_IMAGES
        },
        "formula": {
            "individual_score": "score = 100 * exp(-decay_rate * std / baseline)",
            "total_score": "(pizza_weight * pizza_score + salami_weight * salami_score) / (pizza_weight + salami_weight)",
            "note": "標準偏差が基準値の場合、スコアは約37点となります"
        }
    }