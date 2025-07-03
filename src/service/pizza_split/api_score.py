#!/usr/bin/env python
# coding: utf-8
"""
複数画像のピザ分割均等性スコアを計算するAPI
標準偏差が小さいほど高スコア（100点満点）
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import numpy as np
import tempfile
import os
from pathlib import Path
import shutil

from src.service.pizza_split.score_multi import MultiImageScoreCalculator


# 定数定義
class ScoreConstants:
    """スコア計算用の定数"""
    # スコア計算の重み付け
    PIZZA_WEIGHT = 1.0    # ピザの重み
    SALAMI_WEIGHT = 5.0   # サラミの重み（5倍重要）
    
    # 標準偏差の基準値（比率ベース）
    # これらの値で50点となるように設定
    PIZZA_STD_BASELINE = 0.10    # ピザ面積の標準偏差基準値（10%）
    SALAMI_STD_BASELINE = 0.05   # サラミ面積の標準偏差基準値（5%）
    
    # スコア計算パラメータ
    MAX_SCORE = 100.0             # 最高点
    MIN_SCORE = 0.0               # 最低点
    DECAY_RATE = 2.0              # 減衰率（標準偏差が基準値のn倍でスコアが1/eになる）
    
    # 処理設定
    USE_PREPROCESSING = True      # 前処理（楕円→円変換）を使用
    USE_RATIO = True             # 比率ベースで計算（Falseの場合はピクセル数）
    MIN_IMAGES = 2               # 最小画像数


class PizzaScoreAPI:
    """ピザ分割スコアを計算するAPIクラス"""
    
    def __init__(self):
        """初期化"""
        self.calculator = MultiImageScoreCalculator()
        self.constants = ScoreConstants()
    
    def calculate_score_from_std(self, pizza_std: float, salami_std: float) -> Dict[str, float]:
        """
        標準偏差からスコアを計算
        
        Args:
            pizza_std: ピザ面積の標準偏差
            salami_std: サラミ面積の標準偏差
            
        Returns:
            {
                'total_score': float,      # 総合スコア（0-100）
                'pizza_score': float,      # ピザのスコア（0-100）
                'salami_score': float,     # サラミのスコア（0-100）
                'weighted_score': float    # 重み付け前のスコア
            }
        """
        # 各要素のスコアを計算（指数関数的減衰）
        # std = 0で100点、std = baselineで約37点となる
        pizza_score = self.constants.MAX_SCORE * np.exp(
            -self.constants.DECAY_RATE * pizza_std / self.constants.PIZZA_STD_BASELINE
        )
        salami_score = self.constants.MAX_SCORE * np.exp(
            -self.constants.DECAY_RATE * salami_std / self.constants.SALAMI_STD_BASELINE
        )
        
        # 重み付け平均でスコアを計算
        total_weight = self.constants.PIZZA_WEIGHT + self.constants.SALAMI_WEIGHT
        weighted_score = (
            self.constants.PIZZA_WEIGHT * pizza_score + 
            self.constants.SALAMI_WEIGHT * salami_score
        ) / total_weight
        
        # スコアを0-100の範囲に制限
        total_score = np.clip(weighted_score, self.constants.MIN_SCORE, self.constants.MAX_SCORE)
        
        return {
            'total_score': float(total_score),
            'pizza_score': float(np.clip(pizza_score, 0, 100)),
            'salami_score': float(np.clip(salami_score, 0, 100)),
            'weighted_score': float(weighted_score)
        }
    
    def process_images(self, image_paths: List[str]) -> Dict:
        """
        画像リストを処理してスコアを計算
        
        Args:
            image_paths: 画像パスのリスト
            
        Returns:
            完全な結果辞書
        """
        # 画像数チェック
        if len(image_paths) < self.constants.MIN_IMAGES:
            raise ValueError(f"最低{self.constants.MIN_IMAGES}枚以上の画像が必要です")
        
        # 統計を計算
        stats = self.calculator.calculate_area_statistics(
            image_paths,
            use_preprocessing=self.constants.USE_PREPROCESSING,
            use_ratio=self.constants.USE_RATIO,
            debug=False
        )
        
        # スコアを計算
        scores = self.calculate_score_from_std(
            stats['pizza_area_std'],
            stats['salami_area_std']
        )
        
        # 結果をまとめる
        result = {
            'score': scores['total_score'],
            'score_details': {
                'total_score': scores['total_score'],
                'pizza_score': scores['pizza_score'],
                'salami_score': scores['salami_score'],
                'weights': {
                    'pizza_weight': self.constants.PIZZA_WEIGHT,
                    'salami_weight': self.constants.SALAMI_WEIGHT
                }
            },
            'statistics': {
                'num_images': stats['num_images'],
                'pizza_std': stats['pizza_area_std'],
                'salami_std': stats['salami_area_std'],
                'pizza_mean': stats['pizza_area_mean'],
                'salami_mean': stats['salami_area_mean']
            },
            'images_processed': len(image_paths),
            'failed_images': stats['failed_images']
        }
        
        return result
    
    async def process_uploaded_files(self, files: List[UploadFile]) -> Dict:
        """
        アップロードされたファイルを処理
        
        Args:
            files: アップロードファイルのリスト
            
        Returns:
            処理結果
        """
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_paths = []
            
            try:
                # ファイルを一時保存
                for i, file in enumerate(files):
                    # ファイル拡張子を取得
                    ext = Path(file.filename).suffix or '.jpg'
                    temp_path = Path(temp_dir) / f"image_{i}{ext}"
                    
                    # ファイルを保存
                    with open(temp_path, 'wb') as f:
                        content = await file.read()
                        f.write(content)
                    
                    temp_paths.append(str(temp_path))
                
                # 画像を処理
                result = self.process_images(temp_paths)
                
                # ファイル名を追加
                result['files'] = [f.filename for f in files]
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))


# FastAPI アプリケーション
app = FastAPI(title="Pizza Score API", version="1.0.0")
score_api = PizzaScoreAPI()


@app.post("/api/pizza/score")
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


@app.get("/api/pizza/score/info")
async def get_score_info():
    """
    スコア計算の仕様を取得
    
    Returns:
    - scoring_method: スコア計算方法の説明
    - constants: 使用している定数
    - formula: 計算式の説明
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
            "decay_rate": ScoreConstants.DECAY_RATE
        },
        "formula": {
            "individual_score": "score = 100 * exp(-decay_rate * std / baseline)",
            "total_score": "(pizza_weight * pizza_score + salami_weight * salami_score) / (pizza_weight + salami_weight)",
            "note": "標準偏差が基準値の場合、スコアは約37点となります"
        }
    }


# テスト用の関数
def test_score_calculation():
    """スコア計算のテスト"""
    api = PizzaScoreAPI()
    
    # テストケース
    test_cases = [
        {"pizza_std": 0.0, "salami_std": 0.0, "name": "完璧な均等分割"},
        {"pizza_std": 0.05, "salami_std": 0.025, "name": "良好な分割"},
        {"pizza_std": 0.10, "salami_std": 0.05, "name": "基準値での分割"},
        {"pizza_std": 0.15, "salami_std": 0.075, "name": "やや不均等な分割"},
        {"pizza_std": 0.20, "salami_std": 0.10, "name": "不均等な分割"},
    ]
    
    print("=== スコア計算テスト ===\n")
    print("ケース名                  | Pizza STD | Salami STD | Pizza Score | Salami Score | Total Score")
    print("-" * 95)
    
    for case in test_cases:
        scores = api.calculate_score_from_std(case["pizza_std"], case["salami_std"])
        print(f"{case['name']:<24} | {case['pizza_std']:>9.3f} | {case['salami_std']:>10.3f} | "
              f"{scores['pizza_score']:>11.1f} | {scores['salami_score']:>12.1f} | {scores['total_score']:>11.1f}")


# ローカルテスト用
def test_with_local_images():
    """ローカル画像でのテスト"""
    api = PizzaScoreAPI()
    
    # テスト画像
    test_images = [
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    # 存在する画像のみ使用
    valid_images = [img for img in test_images if Path(img).exists()]
    
    if len(valid_images) >= 2:
        print("\n=== ローカル画像でのテスト ===")
        result = api.process_images(valid_images)
        
        print(f"\n処理画像数: {result['images_processed']}")
        print(f"総合スコア: {result['score']:.1f}点")
        print(f"  - ピザスコア: {result['score_details']['pizza_score']:.1f}点")
        print(f"  - サラミスコア: {result['score_details']['salami_score']:.1f}点")
        print(f"\n標準偏差:")
        print(f"  - ピザ: {result['statistics']['pizza_std']:.4f}")
        print(f"  - サラミ: {result['statistics']['salami_std']:.4f}")


if __name__ == "__main__":
    # テストを実行
    test_score_calculation()
    test_with_local_images()
    
    # サーバー起動の案内
    print("\n\nAPIサーバーを起動するには以下のコマンドを実行してください:")
    print("uvicorn api_score:app --reload --host 0.0.0.0 --port 8000")