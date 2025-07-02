import json
import uuid
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel


class Participant(BaseModel):
    """参加者情報"""
    name: str
    share: float  # 割合 (0.0 ~ 1.0) または具体的な金額


class PaymentRequest(BaseModel):
    """支払いリクエスト"""
    id: str
    split_bill_id: str
    participant_name: str
    amount: int
    merchant_payment_id: str
    pay_url: Optional[str] = None
    qr_code_base64: Optional[str] = None
    status: str = "UNPAID"  # UNPAID, PAID, CANCELLED
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now()
        data["updated_at"] = datetime.now()
        super().__init__(**data)


class SplitBill(BaseModel):
    """割り勘情報"""
    id: str
    total_amount: int
    currency: str = "JPY"
    created_by: str
    participants: List[Participant]
    payment_requests: List[str] = []  # PaymentRequest の ID リスト
    status: str = "OPEN"  # OPEN, COMPLETE, CANCELLED
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now()
        data["updated_at"] = datetime.now()
        super().__init__(**data)


class PayPayRepository:
    """
    PayPay 関連データのリポジトリ
    現在はインメモリ実装だが、将来的にはデータベースに置き換え可能
    """
    
    def __init__(self, storage_dir: str = "result"):
        """
        リポジトリの初期化
        
        Args:
            storage_dir: データを保存するディレクトリ
        """
        self.storage_dir = storage_dir
        self.split_bills: Dict[str, SplitBill] = {}
        self.payment_requests: Dict[str, PaymentRequest] = {}
        
        # ストレージディレクトリが存在しない場合は作成
        os.makedirs(storage_dir, exist_ok=True)
        
        # 保存されたデータがあれば読み込む
        self._load_data()
    
    def _load_data(self):
        """保存されたデータを読み込む"""
        split_bills_path = os.path.join(self.storage_dir, "split_bills.json")
        payment_requests_path = os.path.join(self.storage_dir, "payment_requests.json")
        
        try:
            if os.path.exists(split_bills_path):
                with open(split_bills_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        split_bill = SplitBill(**item)
                        self.split_bills[split_bill.id] = split_bill
        except Exception as e:
            print(f"Error loading split bills: {e}")
        
        try:
            if os.path.exists(payment_requests_path):
                with open(payment_requests_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        payment_request = PaymentRequest(**item)
                        self.payment_requests[payment_request.id] = payment_request
        except Exception as e:
            print(f"Error loading payment requests: {e}")
    
    def _save_data(self):
        """データを保存する"""
        split_bills_path = os.path.join(self.storage_dir, "split_bills.json")
        payment_requests_path = os.path.join(self.storage_dir, "payment_requests.json")
        
        # SplitBill を保存
        with open(split_bills_path, "w") as f:
            json.dump([bill.dict() for bill in self.split_bills.values()], f, default=str)
        
        # PaymentRequest を保存
        with open(payment_requests_path, "w") as f:
            json.dump([req.dict() for req in self.payment_requests.values()], f, default=str)
    
    def create_split_bill(
        self,
        total_amount: int,
        created_by: str,
        participants: List[Participant],
        currency: str = "JPY"
    ) -> SplitBill:
        """
        新しい割り勘を作成
        
        Args:
            total_amount: 合計金額
            created_by: 作成者
            participants: 参加者リスト
            currency: 通貨
            
        Returns:
            作成された SplitBill
        """
        split_id = str(uuid.uuid4())
        
        split_bill = SplitBill(
            id=split_id,
            total_amount=total_amount,
            currency=currency,
            created_by=created_by,
            participants=participants
        )
        
        self.split_bills[split_id] = split_bill
        self._save_data()
        
        return split_bill
    
    def get_split_bill(self, split_id: str) -> Optional[SplitBill]:
        """
        割り勘情報を取得
        
        Args:
            split_id: 割り勘ID
            
        Returns:
            SplitBill または None (存在しない場合)
        """
        return self.split_bills.get(split_id)
    
    def update_split_bill(self, split_bill: SplitBill) -> SplitBill:
        """
        割り勘情報を更新
        
        Args:
            split_bill: 更新する SplitBill
            
        Returns:
            更新された SplitBill
        """
        split_bill.updated_at = datetime.now()
        self.split_bills[split_bill.id] = split_bill
        self._save_data()
        
        return split_bill
    
    def create_payment_request(
        self,
        split_bill_id: str,
        participant_name: str,
        amount: int,
        merchant_payment_id: str,
        pay_url: Optional[str] = None,
        qr_code_base64: Optional[str] = None
    ) -> PaymentRequest:
        """
        支払いリクエストを作成
        
        Args:
            split_bill_id: 割り勘ID
            participant_name: 参加者名
            amount: 金額
            merchant_payment_id: PayPay マーチャント支払いID
            pay_url: 支払いURL
            qr_code_base64: QRコード (Base64)
            
        Returns:
            作成された PaymentRequest
        """
        request_id = str(uuid.uuid4())
        
        payment_request = PaymentRequest(
            id=request_id,
            split_bill_id=split_bill_id,
            participant_name=participant_name,
            amount=amount,
            merchant_payment_id=merchant_payment_id,
            pay_url=pay_url,
            qr_code_base64=qr_code_base64
        )
        
        self.payment_requests[request_id] = payment_request
        
        # SplitBill に PaymentRequest を関連付け
        split_bill = self.split_bills.get(split_bill_id)
        if split_bill:
            split_bill.payment_requests.append(request_id)
            self.update_split_bill(split_bill)
        
        self._save_data()
        
        return payment_request
    
    def get_payment_request(self, request_id: str) -> Optional[PaymentRequest]:
        """
        支払いリクエストを取得
        
        Args:
            request_id: 支払いリクエストID
            
        Returns:
            PaymentRequest または None (存在しない場合)
        """
        return self.payment_requests.get(request_id)
    
    def get_payment_requests_by_split_id(self, split_id: str) -> List[PaymentRequest]:
        """
        割り勘IDに紐づく支払いリクエストを取得
        
        Args:
            split_id: 割り勘ID
            
        Returns:
            PaymentRequest のリスト
        """
        split_bill = self.split_bills.get(split_id)
        if not split_bill:
            return []
        
        return [
            self.payment_requests.get(request_id)
            for request_id in split_bill.payment_requests
            if request_id in self.payment_requests
        ]
    
    def update_payment_request(self, payment_request: PaymentRequest) -> PaymentRequest:
        """
        支払いリクエストを更新
        
        Args:
            payment_request: 更新する PaymentRequest
            
        Returns:
            更新された PaymentRequest
        """
        payment_request.updated_at = datetime.now()
        self.payment_requests[payment_request.id] = payment_request
        self._save_data()
        
        return payment_request
    
    def get_payment_request_by_merchant_id(self, merchant_payment_id: str) -> Optional[PaymentRequest]:
        """
        マーチャント支払いIDから支払いリクエストを取得
        
        Args:
            merchant_payment_id: マーチャント支払いID
            
        Returns:
            PaymentRequest または None (存在しない場合)
        """
        for request in self.payment_requests.values():
            if request.merchant_payment_id == merchant_payment_id:
                return request
        
        return None
