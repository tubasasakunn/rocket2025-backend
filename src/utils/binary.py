#!/usr/bin/env python
# coding: utf-8
"""
バイナリデータ処理用のユーティリティ関数

Base64エンコード/デコードや画像処理に関する共通関数
"""

import base64
import re
import logging
from typing import Union, Tuple, Optional

logger = logging.getLogger("app.utils.binary")

def safe_b64decode(base64_str: str) -> Tuple[Optional[bytes], bool]:
    """
    安全にBase64文字列をデコードする関数
    
    エラーが発生した場合でも、アプリケーションがクラッシュしないように
    例外を内部で処理します。
    
    Parameters:
    - base64_str: Base64エンコードされた文字列
    
    Returns:
    - (bytes, True): 成功した場合、デコードされたバイナリデータとTrueのタプル
    - (None, False): 失敗した場合、Noneとfalseのタプル
    """
    try:
        # データURIスキームの処理
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # 空の場合はエラー
        if not base64_str:
            logger.warning("空のBase64文字列が指定されました")
            return None, False
        
        # Base64のバリデーション
        if not re.match(r'^[A-Za-z0-9+/=]+$', base64_str.strip()):
            logger.warning("無効なBase64形式です")
            return None, False
        
        # デコード実行
        decoded_data = base64.b64decode(base64_str)
        return decoded_data, True
        
    except Exception as e:
        logger.error(f"Base64デコード中にエラーが発生: {str(e)}")
        return None, False


def safe_b64encode(binary_data: bytes) -> Tuple[Optional[str], bool]:
    """
    安全にバイナリデータをBase64エンコードする関数
    
    Parameters:
    - binary_data: エンコードするバイナリデータ
    
    Returns:
    - (str, True): 成功した場合、Base64エンコード文字列とTrueのタプル
    - (None, False): 失敗した場合、Noneとfalseのタプル
    """
    try:
        # 空の場合はエラー
        if not binary_data:
            logger.warning("空のバイナリデータが指定されました")
            return None, False
            
        # エンコード実行
        encoded_data = base64.b64encode(binary_data).decode('ascii')
        return encoded_data, True
        
    except Exception as e:
        logger.error(f"Base64エンコード中にエラーが発生: {str(e)}")
        return None, False


def format_image_data_uri(binary_data: bytes, mime_type: str = "image/jpeg") -> Optional[str]:
    """
    バイナリデータをデータURI形式に変換
    
    Parameters:
    - binary_data: 画像バイナリデータ
    - mime_type: MIMEタイプ（デフォルトはimage/jpeg）
    
    Returns:
    - str: データURI形式の文字列（data:image/jpeg;base64,...）
    - None: 変換に失敗した場合
    """
    encoded_str, success = safe_b64encode(binary_data)
    if not success:
        return None
        
    return f"data:{mime_type};base64,{encoded_str}"
