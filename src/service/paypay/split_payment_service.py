import uuid
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .client import PayPayClient
from .repository import PayPayRepository, Participant, SplitBill, PaymentRequest
from .settings import get_paypay_settings


class SplitPaymentService:
    """
    割り勘支払いサービス
    PayPay APIを使用して割り勘の支払いリクエストを作成・管理する
    """
    
    def __init__(self, client: Optional[PayPayClient] = None, repository: Optional[PayPayRepository] = None):
        """
        サービスの初期化
        
        Args:
            client: PayPay API クライアント (None の場合は自動作成)
            repository: データリポジトリ (None の場合は自動作成)
        """
        # 設定を読み込み
        settings = get_paypay_settings()
        
        # クライアントが指定されていない場合は作成
        self.client = client or PayPayClient(
            api_key=settings.paypay_api_key,
            api_secret=settings.paypay_api_secret,
            merchant_id=settings.paypay_merchant_id,
            is_production=settings.paypay_is_production
        )
        
        # リポジトリが指定されていない場合は作成
        self.repository = repository or PayPayRepository()
        
        # アプリケーションのホスト
        self.app_host = settings.app_host
    
    async def create_split_payment(
        self,
        total_amount: int,
        created_by: str,
        participants: List[Dict[str, Any]],
        description: str = "ピザの割り勘",
        currency: str = "JPY"
    ) -> Dict[str, Any]:
        """
        割り勘支払いを作成
        
        Args:
            total_amount: 合計金額
            created_by: 作成者
            participants: 参加者リスト [{"name": "名前", "share": 金額または割合}]
            description: 支払い説明
            currency: 通貨
            
        Returns:
            作成された割り勘情報と支払いリクエスト
        """
        # 参加者情報を Participant オブジェクトに変換
        participant_objects = [Participant(**p) for p in participants]
        
        # 割り勘を作成
        split_bill = self.repository.create_split_bill(
            total_amount=total_amount,
            created_by=created_by,
            participants=participant_objects,
            currency=currency
        )
        
        # 各参加者の支払い金額を計算
        payment_amounts = self._calculate_payment_amounts(total_amount, participant_objects)
        
        # 各参加者の支払いリクエストを作成
        payment_requests = []
        for participant, amount in zip(participant_objects, payment_amounts):
            # 支払いリクエストを作成
            payment_request = await self._create_payment_request(
                split_bill.id,
                participant.name,
                amount,
                f"{description} - {participant.name}の支払い"
            )
            payment_requests.append(payment_request)
        
        # レスポンスを作成
        response = {
            "split_id": split_bill.id,
            "total_amount": total_amount,
            "currency": currency,
            "created_by": created_by,
            "payment_requests": [
                {
                    "id": req.id,
                    "participant_name": req.participant_name,
                    "amount": req.amount,
                    "pay_url": req.pay_url,
                    "qr_code_base64": req.qr_code_base64,
                    "status": req.status
                }
                for req in payment_requests
            ]
        }
        
        return response
    
    def _calculate_payment_amounts(self, total_amount: int, participants: List[Participant]) -> List[int]:
        """
        各参加者の支払い金額を計算
        
        Args:
            total_amount: 合計金額
            participants: 参加者リスト
            
        Returns:
            各参加者の支払い金額リスト
        """
        # 参加者の share が割合 (0.0 ~ 1.0) か具体的な金額かを判断
        is_ratio = all(0.0 <= p.share <= 1.0 for p in participants)
        
        if is_ratio:
            # 割合の場合
            # 合計が1.0になるように正規化
            total_ratio = sum(p.share for p in participants)
            normalized_ratios = [p.share / total_ratio for p in participants]
            
            # 各参加者の支払い金額を計算 (切り捨て)
            amounts = [int(total_amount * ratio) for ratio in normalized_ratios]
            
            # 端数を調整 (最後の参加者に割り当て)
            remainder = total_amount - sum(amounts)
            if remainder > 0:
                amounts[-1] += remainder
            
            return amounts
        else:
            # 具体的な金額の場合
            # 合計金額と一致するか確認
            total_share = sum(p.share for p in participants)
            if total_share != total_amount:
                # 一致しない場合は比例配分
                ratio = total_amount / total_share
                amounts = [int(p.share * ratio) for p in participants]
                
                # 端数を調整
                remainder = total_amount - sum(amounts)
                if remainder > 0:
                    amounts[-1] += remainder
                
                return amounts
            else:
                # 一致する場合はそのまま返す
                return [int(p.share) for p in participants]
    
    async def _create_payment_request(
        self,
        split_bill_id: str,
        participant_name: str,
        amount: int,
        description: str
    ) -> PaymentRequest:
        """
        支払いリクエストを作成
        
        Args:
            split_bill_id: 割り勘ID
            participant_name: 参加者名
            amount: 金額
            description: 支払い説明
            
        Returns:
            作成された PaymentRequest
        """
        # マーチャント支払いIDを生成
        merchant_payment_id = f"split_{split_bill_id}_{uuid.uuid4().hex}"
        
        # コールバックURL
        callback_url = f"{self.app_host}/api/paypay/callback"
        
        # PayPay QRコードを作成
        qr_response = await self.client.create_qr_code(
            amount=amount,
            merchant_payment_id=merchant_payment_id,
            order_description=description,
            redirect_url=callback_url
        )
        
        # QRコードとURLを取得
        qr_code_base64 = qr_response.get("data", {}).get("codeId")
        pay_url = qr_response.get("data", {}).get("url")
        
        # 支払いリクエストを作成
        payment_request = self.repository.create_payment_request(
            split_bill_id=split_bill_id,
            participant_name=participant_name,
            amount=amount,
            merchant_payment_id=merchant_payment_id,
            pay_url=pay_url,
            qr_code_base64=qr_code_base64
        )
        
        return payment_request
    
    async def get_split_payment_status(self, split_id: str) -> Dict[str, Any]:
        """
        割り勘の支払い状況を取得
        
        Args:
            split_id: 割り勘ID
            
        Returns:
            割り勘情報と支払い状況
        """
        # 割り勘情報を取得
        split_bill = self.repository.get_split_bill(split_id)
        if not split_bill:
            raise ValueError(f"Split bill not found: {split_id}")
        
        # 支払いリクエストを取得
        payment_requests = self.repository.get_payment_requests_by_split_id(split_id)
        
        # 各支払いリクエストの状態を更新
        updated_requests = []
        for request in payment_requests:
            try:
                # PayPay APIから支払い状態を取得
                payment_details = await self.client.get_payment_details(request.merchant_payment_id)
                
                # 支払い状態を更新
                status = payment_details.get("data", {}).get("status")
                if status == "COMPLETED":
                    request.status = "PAID"
                elif status in ["AUTHORIZED", "REAUTHORIZING"]:
                    request.status = "AUTHORIZED"
                elif status in ["FAILED", "CANCELED"]:
                    request.status = "CANCELLED"
                
                # リポジトリを更新
                self.repository.update_payment_request(request)
                
            except Exception as e:
                print(f"Error updating payment status: {e}")
            
            updated_requests.append(request)
        
        # 全ての支払いが完了しているか確認
        all_paid = all(req.status == "PAID" for req in updated_requests)
        if all_paid and split_bill.status == "OPEN":
            split_bill.status = "COMPLETE"
            self.repository.update_split_bill(split_bill)
        
        # レスポンスを作成
        response = {
            "split_id": split_bill.id,
            "total_amount": split_bill.total_amount,
            "currency": split_bill.currency,
            "created_by": split_bill.created_by,
            "status": split_bill.status,
            "created_at": split_bill.created_at,
            "payment_requests": [
                {
                    "id": req.id,
                    "participant_name": req.participant_name,
                    "amount": req.amount,
                    "pay_url": req.pay_url,
                    "qr_code_base64": req.qr_code_base64,
                    "status": req.status
                }
                for req in updated_requests
            ]
        }
        
        return response
    
    async def handle_payment_webhook(self, merchant_payment_id: str, status: str) -> Dict[str, Any]:
        """
        PayPay Webhookを処理
        
        Args:
            merchant_payment_id: マーチャント支払いID
            status: 支払い状態
            
        Returns:
            更新された支払いリクエスト情報
        """
        # 支払いリクエストを取得
        payment_request = self.repository.get_payment_request_by_merchant_id(merchant_payment_id)
        if not payment_request:
            raise ValueError(f"Payment request not found: {merchant_payment_id}")
        
        # 支払い状態を更新
        if status == "COMPLETED":
            payment_request.status = "PAID"
        elif status in ["AUTHORIZED", "REAUTHORIZING"]:
            payment_request.status = "AUTHORIZED"
        elif status in ["FAILED", "CANCELED"]:
            payment_request.status = "CANCELLED"
        
        # リポジトリを更新
        self.repository.update_payment_request(payment_request)
        
        # 割り勘情報を取得
        split_bill = self.repository.get_split_bill(payment_request.split_bill_id)
        if not split_bill:
            raise ValueError(f"Split bill not found: {payment_request.split_bill_id}")
        
        # 全ての支払いリクエストを取得
        all_requests = self.repository.get_payment_requests_by_split_id(split_bill.id)
        
        # 全ての支払いが完了しているか確認
        all_paid = all(req.status == "PAID" for req in all_requests)
        if all_paid and split_bill.status == "OPEN":
            split_bill.status = "COMPLETE"
            self.repository.update_split_bill(split_bill)
        
        # レスポンスを作成
        response = {
            "payment_request_id": payment_request.id,
            "merchant_payment_id": merchant_payment_id,
            "status": payment_request.status,
            "split_bill_id": split_bill.id,
            "split_bill_status": split_bill.status
        }
        
        return response
