#!/usr/bin/env python
# coding: utf-8
"""
感情認識サービス

顔画像から感情を分析するサービス
"""

from fer import FER
import cv2
import numpy as np
import io
from typing import Dict, Optional, Union, List
import logging

# utils からバイナリユーティリティをインポート
from utils.binary import safe_b64decode, safe_b64encode, format_image_data_uri

logger = logging.getLogger("app.service.emotion_recognition")

# 感情ごとの支払い意欲の重み付け
EMOTION_PAY_WEIGHTS = {
    "angry": 0.15,
    "disgust": 0.15,
    "fear": 0.20,
    "sad": 0.20,
    "surprise": 0.80,
    "happy": 0.90,
    "neutral": 0.05
}


class EmotionRecognitionService:
    """
    感情認識サービスクラス
    
    FERライブラリを使用して顔画像から感情を分析します。
    """
    
    def __init__(self):
        """
        感情認識サービスの初期化
        
        MTCNNモデルを使用して顔検出の精度を向上させます。
        """
        self.detector = FER(mtcnn=True)
    
    def analyze(self, image_bytes: bytes) -> Dict[str, Union[Optional[str], Dict[str, float]]]:
        """
        画像から感情を分析
        
        Parameters:
        - image_bytes: バイナリ形式の画像データ
        
        Returns:
        - dominant: 最も強い感情（検出できない場合はNone）
        - scores: 各感情のスコア（0-1の範囲、合計1）
          - angry: 怒り
          - disgust: 嫌悪
          - fear: 恐怖
          - happy: 幸福
          - sad: 悲しみ
          - surprise: 驚き
          - neutral: 無表情
        """
        try:
            # バイナリデータからOpenCV形式の画像に変換
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("画像のデコードに失敗しました")
                raise ValueError("画像のデコードに失敗しました")
            
            # 感情分析を実行
            emotions = self.detector.detect_emotions(img)
            
            # 顔が検出されなかった場合
            if not emotions:
                logger.info("顔が検出されませんでした")
                return {"dominant": None, "scores": {}}
            
            # 最初に検出された顔の感情スコアを取得
            scores = emotions[0]["emotions"]
            
            # 最も強い感情を特定
            dominant = max(scores, key=scores.get)
            
            # スコアを通常の辞書に変換し、float型に明示的に変換（JSONシリアライズを確実にするため）
            scores_dict = {k: float(v) for k, v in scores.items()}
            
        except UnicodeDecodeError as e:
            logger.error(f"Unicode デコードエラー: {str(e)}")
            # Unicode関連のエラーをキャッチしてより分かりやすいメッセージに変換
            raise ValueError(f"画像データのエンコードに問題があります: {str(e)}")
        except ValueError as e:
            logger.error(f"値エラー: {str(e)}")
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"感情認識エラー: {str(e)}")
            raise ValueError(f"感情認識処理中にエラーが発生しました: {str(e)}")
        
        return {
            "dominant": dominant,
            "scores": scores_dict
        }
    
    @staticmethod
    def calc_pay_probability(scores: Optional[Dict[str, float]]) -> Optional[float]:
        """
        感情スコアから支払い確率の原値を計算
        
        Parameters:
        - scores: 各感情のスコア（0-1の範囲、合計1）
        
        Returns:
        - pay_raw: 支払い確率の原値（正規化前）
          Noneの場合は感情認識に失敗したことを示す
        """
        if scores is None or not scores:
            return None
            
        pay_raw = 0.0
        for emotion, score in scores.items():
            if emotion in EMOTION_PAY_WEIGHTS:
                pay_raw += score * EMOTION_PAY_WEIGHTS[emotion]
                
        return pay_raw
