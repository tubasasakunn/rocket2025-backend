import os
import time
import hmac
import hashlib
import base64
import json
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel


class PayPayClient:
    """
    PayPay API クライアント
    PayPay API v2 を使用して支払いリクエストを作成・管理するクライアント
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        merchant_id: str,
        is_production: bool = False
    ):
        """
        PayPay API クライアントの初期化
        
        Args:
            api_key: PayPay API キー
            api_secret: PayPay API シークレット
            merchant_id: マーチャントID (X-ASSUME-MERCHANT)
            is_production: 本番環境の場合 True、Sandbox の場合 False
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.merchant_id = merchant_id
        
        # 環境に応じたベースURLを設定
        if is_production:
            self.base_url = "https://api.paypay.ne.jp"
        else:
            self.base_url = "https://stg-api.paypay.ne.jp"
    
    def generate_signature(
        self,
        method: str,
        path: str,
        timestamp: int,
        body: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        HMAC-SHA256 署名を生成
        
        Args:
            method: HTTP メソッド (GET, POST, etc.)
            path: API パス (/v2/...)
            timestamp: UNIX タイムスタンプ (秒)
            body: リクエストボディ (JSON シリアライズ可能なオブジェクト)
            
        Returns:
            Base64 エンコードされた署名
        """
        # リクエストボディがある場合は JSON 文字列に変換
        content_type = "application/json"
        body_string = ""
        if body:
            body_string = json.dumps(body)
        
        # 署名対象文字列を作成
        signature_string = f"{method}\n{path}\n{timestamp}\n{content_type}\n{body_string}"
        
        # HMAC-SHA256 で署名
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        # Base64 エンコード
        return base64.b64encode(signature).decode('utf-8')
    
    async def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        PayPay API にリクエストを送信
        
        Args:
            method: HTTP メソッド (GET, POST, etc.)
            path: API パス (/v2/...)
            body: リクエストボディ (JSON シリアライズ可能なオブジェクト)
            params: クエリパラメータ
            
        Returns:
            API レスポンス (JSON)
        """
        # タイムスタンプを生成 (UNIX 秒)
        timestamp = int(time.time())
        
        # 署名を生成
        signature = self.generate_signature(method, path, timestamp, body)
        
        # ヘッダーを作成
        headers = {
            "Content-Type": "application/json",
            "X-ASSUME-MERCHANT": self.merchant_id,
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": str(timestamp),
            "X-SIGNATURE": signature
        }
        
        # リクエストURLを作成
        url = f"{self.base_url}{path}"
        
        # リクエストを送信
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None,
                params=params
            )
            
            # レスポンスをJSONとしてパース
            try:
                response_data = response.json()
            except Exception:
                response_data = {"error": "Failed to parse response as JSON"}
            
            # エラーチェック
            if response.status_code >= 400:
                error_message = response_data.get("message", "Unknown error")
                raise Exception(f"PayPay API error: {response.status_code} - {error_message}")
            
            return response_data
    
    async def create_qr_code(
        self,
        amount: int,
        merchant_payment_id: str,
        order_description: str,
        redirect_url: Optional[str] = None,
        redirect_type: str = "WEB_LINK"
    ) -> Dict[str, Any]:
        """
        QRコード支払いリクエストを作成
        
        Args:
            amount: 支払い金額 (整数)
            merchant_payment_id: マーチャント側で管理する支払いID (一意)
            order_description: 支払い内容の説明
            redirect_url: 支払い完了後のリダイレクトURL
            redirect_type: リダイレクトタイプ (APP_DEEP_LINK or WEB_LINK)
            
        Returns:
            API レスポンス (QRコード情報を含む)
        """
        path = "/v2/codes"
        
        # リクエストボディを作成
        body = {
            "merchantPaymentId": merchant_payment_id,
            "amount": {
                "amount": amount,
                "currency": "JPY"
            },
            "codeType": "ORDER_QR",
            "orderDescription": order_description,
            "isAuthorization": False,
            "redirectUrl": redirect_url,
            "redirectType": redirect_type
        }
        
        # リダイレクトURLが指定されていない場合は削除
        if not redirect_url:
            body.pop("redirectUrl", None)
            body.pop("redirectType", None)
        
        # APIリクエストを送信
        return await self.request("POST", path, body)
    
    async def get_payment_details(self, merchant_payment_id: str) -> Dict[str, Any]:
        """
        支払い詳細を取得
        
        Args:
            merchant_payment_id: マーチャント側で管理する支払いID
            
        Returns:
            支払い詳細情報
        """
        path = f"/v2/payments/{merchant_payment_id}"
        
        # APIリクエストを送信
        return await self.request("GET", path)
    
    async def cancel_payment(
        self,
        merchant_payment_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        支払いをキャンセル
        
        Args:
            merchant_payment_id: マーチャント側で管理する支払いID
            reason: キャンセル理由
            
        Returns:
            キャンセル結果
        """
        path = f"/v2/payments/{merchant_payment_id}/cancel"
        
        # リクエストボディを作成
        body = {}
        if reason:
            body["reason"] = reason
        
        # APIリクエストを送信
        return await self.request("POST", path, body)
    
    async def refund_payment(
        self,
        merchant_payment_id: str,
        amount: int,
        merchant_refund_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        支払いを返金
        
        Args:
            merchant_payment_id: マーチャント側で管理する支払いID
            amount: 返金金額
            merchant_refund_id: マーチャント側で管理する返金ID (一意)
            reason: 返金理由
            
        Returns:
            返金結果
        """
        path = f"/v2/refunds"
        
        # リクエストボディを作成
        body = {
            "merchantRefundId": merchant_refund_id,
            "paymentId": merchant_payment_id,
            "amount": {
                "amount": amount,
                "currency": "JPY"
            },
            "requestedAt": int(time.time())
        }
        
        if reason:
            body["reason"] = reason
        
        # APIリクエストを送信
        return await self.request("POST", path, body)
