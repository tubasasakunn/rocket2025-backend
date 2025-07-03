#!/usr/bin/env python
# coding: utf-8
"""
FastAPI ミドルウェア

APIリクエストとレスポンスを適切に処理するためのミドルウェア
"""

import base64
from typing import Any, Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from fastapi import Request
import logging
import traceback

logger = logging.getLogger("app.middleware")

class BinaryDataHandlingMiddleware(BaseHTTPMiddleware):
    """
    バイナリデータ処理用ミドルウェア
    
    JSONレスポンス内のバイナリデータを適切に処理します
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # エラーログ記録
            logger.error(f"バイナリデータ処理中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
            
            # バイナリデコードエラーの場合
            if isinstance(e, UnicodeDecodeError):
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "バイナリデータのエンコード/デコード中にエラーが発生しました。データがBase64形式でエンコードされているか確認してください。"
                    }
                )
            
            # その他のエラー
            return JSONResponse(
                status_code=500,
                content={"detail": f"サーバーエラーが発生しました: {str(e)}"}
            )


def encode_bytes_in_dict(data: Any) -> Any:
    """
    辞書内のbytesオブジェクトを再帰的にBase64エンコードする関数
    
    Parameters:
    - data: 変換対象のデータ（任意の型）
    
    Returns:
    - 変換後のデータ（バイナリデータはBase64エンコード済み）
    """
    if isinstance(data, bytes):
        # バイナリデータをBase64エンコード
        return base64.b64encode(data).decode("utf-8")
    elif isinstance(data, dict):
        # 辞書内の各要素を再帰的に処理
        return {key: encode_bytes_in_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        # リスト内の各要素を再帰的に処理
        return [encode_bytes_in_dict(item) for item in data]
    else:
        # その他の型はそのまま返す
        return data
