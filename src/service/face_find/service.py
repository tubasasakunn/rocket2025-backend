#!/usr/bin/env python
# coding: utf-8
"""
顔検出サービス

画像から顔を検出し、200x200にリサイズして返すサービス
"""

import cv2
import numpy as np
import base64
import re
from typing import Dict, List, Union, Optional


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
        
    def _base64_to_image(self, base64_str: str) -> np.ndarray:
        """
        Base64文字列をOpenCV画像に変換
        
        Parameters:
        - base64_str: Base64エンコードされた画像データ
        
        Returns:
        - np.ndarray: OpenCV形式の画像
        """
        # data:image/jpeg;base64, プレフィックスを削除
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
            
        # Base64をデコード
        img_data = base64.b64decode(base64_str)
        
        # バイナリデータからOpenCV形式の画像に変換
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    def _image_to_base64(self, img: np.ndarray) -> str:
        """
        OpenCV画像をBase64文字列に変換
        
        Parameters:
        - img: OpenCV形式の画像
        
        Returns:
        - str: Base64エンコードされた画像データ (data:image/jpeg;base64, プレフィックス付き)
        """
        # JPEG形式に変換
        _, buffer = cv2.imencode('.jpg', img)
        
        # Base64にエンコード
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        # data:image/jpeg;base64, プレフィックスを追加
        return f"data:image/jpeg;base64,{base64_str}"
    
    def analyze(self, image_b64: str, expected_count: int) -> Dict[str, Union[int, List[str]]]:
        """
        画像から顔を検出し、リサイズして返す
        
        Parameters:
        - image_b64: Base64エンコードされた画像データ
        - expected_count: 期待される顔の数
        
        Returns:
        - detected: 検出された顔の数
        - faces: 検出された顔の画像（Base64エンコード、200x200にリサイズ済み）
        
        Raises:
        - ValueError: 画像が無効、または期待される顔の数と検出された顔の数が一致しない場合
        """
        # 画像が空の場合
        if not image_b64:
            raise ValueError("画像データが空です")
        
        # Base64をOpenCV画像に変換
        try:
            img = self._base64_to_image(image_b64)
        except Exception as e:
            raise ValueError(f"画像のデコードに失敗しました: {str(e)}")
        
        # 画像が無効な場合
        if img is None or img.size == 0:
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
            raise ValueError("画像から顔を検出できませんでした")
        
        # 期待される顔の数と検出された顔の数が一致しない場合
        if len(faces) != expected_count:
            raise ValueError(f"期待される顔の数({expected_count})と検出された顔の数({len(faces)})が一致しません")
        
        # 検出された顔をリサイズしてBase64に変換
        face_images = []
        for (x, y, w, h) in faces:
            # 顔の領域を切り出し
            face_roi = img[y:y+h, x:x+w]
            
            # 200x200にリサイズ
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Base64に変換
            face_b64 = self._image_to_base64(face_resized)
            
            # リストに追加
            face_images.append(face_b64)
        
        return {
            "detected": len(faces),
            "faces": face_images
        }
