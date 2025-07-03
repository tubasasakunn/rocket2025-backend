#!/usr/bin/env python
# coding: utf-8
"""
顔検出サービス

画像から顔を検出し、200x200にリサイズして返すサービス
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple

# ロガーの設定
logger = logging.getLogger("app.service.face_find")


class FaceFindService:
    """
    顔検出サービスクラス
    
    OpenCVのHaar Cascadeを使用して顔を検出し、リサイズします。
    """
    
    def __init__(self):
        """
        顔検出サービスの初期化
        
        Haar Cascadeの顔検出器を読み込みます。
        """
        # 顔検出器の読み込み
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def _bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        """
        バイト列をOpenCV画像に変換
        
        Parameters:
        - image_bytes: 画像のバイト列
        
        Returns:
        - np.ndarray: OpenCV形式の画像
        """
        try:
            # バイナリデータからOpenCV形式の画像に変換
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("画像形式が不正です")
                raise ValueError("画像のデコードに失敗しました")
                
            return img
            
        except Exception as e:
            logger.error(f"画像変換エラー: {str(e)}")
            raise ValueError(f"画像変換中にエラーが発生しました: {str(e)}")
    
    def _image_to_bytes(self, img: np.ndarray) -> bytes:
        """
        OpenCV画像をバイト列に変換
        
        Parameters:
        - img: OpenCV形式の画像
        
        Returns:
        - bytes: JPEG形式のバイト列
        """
        try:
            # JPEG形式に変換
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"画像のバイト変換中にエラー: {str(e)}")
            raise ValueError(f"画像のバイト変換に失敗しました: {str(e)}")
    
    def _create_placeholder_image(self) -> None:
        """
        プレースホルダー用の空白画像を生成
        
        Returns:
        - None: プレースホルダーを示すNone値
        """
        return None
    
    def analyze(self, image_bytes: bytes, expected_count: int) -> Dict[str, Union[int, List[Optional[bytes]]]]:
        """
        画像から顔を検出し、リサイズして返す
        
        Parameters:
        - image_bytes: 画像のバイト列
        - expected_count: 期待される顔の数
        
        Returns:
        - detected: 検出された顔の数
        - faces: 検出された顔の画像（バイト列、200x200にリサイズ済み）
                 検出数が期待値に満たない場合はプレースホルダー（None）で補完
        
        Raises:
        - ValueError: 画像が無効な場合
        """
        # 画像が空の場合
        if not image_bytes:
            logger.error("空の画像データが指定されました")
            raise ValueError("画像データが空です")
        
        # バイト列をOpenCV画像に変換
        try:
            img = self._bytes_to_image(image_bytes)
        except Exception as e:
            logger.error(f"画像のデコードに失敗: {str(e)}")
            raise ValueError(f"画像のデコードに失敗しました: {str(e)}")
        
        # 画像が無効な場合
        if img is None or img.size == 0:
            logger.error("無効な画像データ")
            raise ValueError("無効な画像データです")
        
        # グレースケールに変換（顔検出の精度向上）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 顔検出（パラメータを調整して偽陽性を減らす）
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # 値を大きくして偽陽性を減らす
            minSize=(50, 50)  # 最小サイズを大きくして小さな誤検出を防ぐ
        )
        
        # 顔が検出されなかった場合
        if len(faces) == 0:
            logger.info(f"顔が検出されませんでした。期待値: {expected_count}")
            # プレースホルダー画像で補完
            face_images = [self._create_placeholder_image() for _ in range(expected_count)]
            return {
                "detected": 0,
                "faces": face_images
            }
        
        # 検出された顔をリサイズしてバイト列に変換
        face_images = []
        for (x, y, w, h) in faces:
            # 顔の領域を切り出し
            face_roi = img[y:y+h, x:x+w]
            
            # 200x200にリサイズ
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # バイト列に変換
            face_bytes = self._image_to_bytes(face_resized)
            
            # リストに追加
            face_images.append(face_bytes)
        
        # 検出数が期待値に満たない場合、プレースホルダーで補完
        detected_count = len(faces)
        if detected_count < expected_count:
            logger.info(f"検出された顔の数 ({detected_count}) が期待値 ({expected_count}) より少ないため、プレースホルダーを追加します")
            for _ in range(expected_count - detected_count):
                face_images.append(self._create_placeholder_image())
        
        # 検出数が期待値を超える場合は、期待値分だけ返す
        if detected_count > expected_count:
            logger.info(f"検出された顔の数 ({detected_count}) が期待値 ({expected_count}) を超えています")
            face_images = face_images[:expected_count]
            detected_count = expected_count
        
        return {
            "detected": detected_count,
            "faces": face_images
        }
