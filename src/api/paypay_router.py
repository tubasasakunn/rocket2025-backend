from fastapi import APIRouter, HTTPException, Depends, Request, Header
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import hmac
import hashlib
import base64
import time
import json

from service.paypay.split_payment_service import SplitPaymentService
from service.paypay.settings import get_paypay_settings


# モデル定義
class ParticipantRequest(BaseModel):
    name: str
    share: float = Field(..., description="割合 (0.0 ~ 1.0) または具体的な金額")


class CreateSplitRequest(BaseModel):
    total_amount: int = Field(..., description="合計金額 (円)")
    created_by: str = Field(..., description="作成者名")
    participants: List[ParticipantRequest] = Field(..., description="参加者リスト")
    description: str = Field("ピザの割り勘", description="支払い説明")
    currency: str = Field("JPY", description="通貨")


class WebhookRequest(BaseModel):
    merchant_payment_id: str
    status: str


# ルーター作成
router = APIRouter(prefix="/paypay", tags=["paypay"])


# 依存関係
def get_split_payment_service() -> SplitPaymentService:
    return SplitPaymentService()


# Webhook 署名検証
async def verify_webhook_signature(
    request: Request,
    x_paypay_signature: Optional[str] = Header(None),
    x_paypay_timestamp: Optional[str] = Header(None)
) -> bool:
    """
    PayPay Webhook の署名を検証
    
    Args:
        request: リクエスト
        x_paypay_signature: PayPay 署名ヘッダー
        x_paypay_timestamp: PayPay タイムスタンプヘッダー
        
    Returns:
        署名が有効な場合は True
    """
    if not x_paypay_signature or not x_paypay_timestamp:
        return False
    
    # 設定を取得
    settings = get_paypay_settings()
    
    # リクエストボディを取得
    body = await request.body()
    body_str = body.decode("utf-8")
    
    # 署名対象文字列を作成
    signature_string = f"{x_paypay_timestamp}\n{body_str}"
    
    # HMAC-SHA256 で署名
    signature = hmac.new(
        settings.paypay_api_secret.encode("utf-8"),
        signature_string.encode("utf-8"),
        hashlib.sha256
    ).digest()
    
    # Base64 エンコード
    expected_signature = base64.b64encode(signature).decode("utf-8")
    
    # 署名を比較
    return hmac.compare_digest(expected_signature, x_paypay_signature)


# エンドポイント定義
@router.post("/split", response_model=Dict[str, Any])
async def create_split_payment(
    request: CreateSplitRequest,
    service: SplitPaymentService = Depends(get_split_payment_service)
):
    """
    割り勘支払いを作成
    
    各参加者に対して PayPay QR コードを生成し、支払いリクエストを作成します。
    """
    try:
        # 参加者情報を辞書に変換
        participants = [{"name": p.name, "share": p.share} for p in request.participants]
        
        # 割り勘支払いを作成
        result = await service.create_split_payment(
            total_amount=request.total_amount,
            created_by=request.created_by,
            participants=participants,
            description=request.description,
            currency=request.currency
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/split/{split_id}", response_model=Dict[str, Any])
async def get_split_payment_status(
    split_id: str,
    service: SplitPaymentService = Depends(get_split_payment_service)
):
    """
    割り勘支払いの状態を取得
    
    各参加者の支払い状況を含む割り勘情報を返します。
    """
    try:
        result = await service.get_split_payment_status(split_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook", response_model=Dict[str, Any])
async def handle_webhook(
    request: Request,
    webhook_data: WebhookRequest,
    is_valid: bool = Depends(verify_webhook_signature),
    service: SplitPaymentService = Depends(get_split_payment_service)
):
    """
    PayPay Webhook を処理
    
    PayPay からの支払い状態更新通知を処理します。
    """
    # 署名検証
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        # Webhook を処理
        result = await service.handle_payment_webhook(
            merchant_payment_id=webhook_data.merchant_payment_id,
            status=webhook_data.status
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/callback")
async def payment_callback():
    """
    PayPay 支払い完了後のコールバック
    
    PayPay アプリでの支払い完了後にリダイレクトされるエンドポイント。
    フロントエンドにリダイレクトするなどの処理を行います。
    """
    # 実際のアプリケーションでは、フロントエンドにリダイレクトするなどの処理を行う
    return {"message": "Payment completed"}
