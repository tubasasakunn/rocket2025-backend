import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.service.paypay.client import PayPayClient
from src.service.paypay.repository import PayPayRepository, Participant, SplitBill, PaymentRequest
from src.service.paypay.split_payment_service import SplitPaymentService


# テスト用のモックデータ
def create_mock_qr_response():
    return {
        "data": {
            "codeId": "data:image/png;base64,mockQRCodeBase64",
            "url": "https://stg-link.paypay.ne.jp/mock_url",
            "status": "CREATED"
        }
    }


def create_mock_payment_details(status="COMPLETED"):
    return {
        "data": {
            "status": status,
            "paymentId": "mock_payment_id",
            "merchantPaymentId": "mock_merchant_payment_id",
            "amount": {
                "amount": 1000,
                "currency": "JPY"
            },
            "requestedAt": int(datetime.now().timestamp()),
            "completedAt": int(datetime.now().timestamp())
        }
    }


# テスト用のフィクスチャ
@pytest.fixture
def mock_client():
    client = AsyncMock(spec=PayPayClient)
    client.create_qr_code.return_value = create_mock_qr_response()
    client.get_payment_details.return_value = create_mock_payment_details()
    return client


@pytest.fixture
def mock_repository():
    repository = MagicMock(spec=PayPayRepository)
    
    # create_split_bill のモック
    def mock_create_split_bill(total_amount, created_by, participants, currency="JPY"):
        split_id = str(uuid.uuid4())
        split_bill = SplitBill(
            id=split_id,
            total_amount=total_amount,
            currency=currency,
            created_by=created_by,
            participants=participants
        )
        return split_bill
    
    repository.create_split_bill.side_effect = mock_create_split_bill
    
    # create_payment_request のモック
    def mock_create_payment_request(split_bill_id, participant_name, amount, merchant_payment_id, pay_url=None, qr_code_base64=None):
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
        return payment_request
    
    repository.create_payment_request.side_effect = mock_create_payment_request
    
    return repository


@pytest.mark.asyncio
async def test_create_split_payment(mock_client, mock_repository):
    """割り勘支払い作成のテスト"""
    # テスト用のサービスを作成
    service = SplitPaymentService(client=mock_client, repository=mock_repository)
    service.app_host = "http://localhost:8000"
    
    # テスト用のパラメータ
    total_amount = 3000
    created_by = "テスト太郎"
    participants = [
        {"name": "テスト太郎", "share": 0.4},
        {"name": "テスト花子", "share": 0.3},
        {"name": "テスト次郎", "share": 0.3}
    ]
    
    # 割り勘支払いを作成
    result = await service.create_split_payment(
        total_amount=total_amount,
        created_by=created_by,
        participants=participants
    )
    
    # 結果の検証
    assert result["total_amount"] == total_amount
    assert result["created_by"] == created_by
    assert len(result["payment_requests"]) == 3
    
    # 各参加者の支払い金額の合計が total_amount と一致することを確認
    total_payment = sum(req["amount"] for req in result["payment_requests"])
    assert total_payment == total_amount
    
    # PayPay API が正しく呼び出されたことを確認
    assert mock_client.create_qr_code.call_count == 3
    
    # リポジトリが正しく呼び出されたことを確認
    mock_repository.create_split_bill.assert_called_once()
    assert mock_repository.create_payment_request.call_count == 3


@pytest.mark.asyncio
async def test_get_split_payment_status(mock_client, mock_repository):
    """割り勘支払い状態取得のテスト"""
    # テスト用のデータを準備
    split_id = str(uuid.uuid4())
    split_bill = SplitBill(
        id=split_id,
        total_amount=3000,
        currency="JPY",
        created_by="テスト太郎",
        participants=[
            Participant(name="テスト太郎", share=0.4),
            Participant(name="テスト花子", share=0.3),
            Participant(name="テスト次郎", share=0.3)
        ],
        payment_requests=["req1", "req2", "req3"]
    )
    
    payment_requests = [
        PaymentRequest(
            id="req1",
            split_bill_id=split_id,
            participant_name="テスト太郎",
            amount=1200,
            merchant_payment_id="merchant_id_1",
            pay_url="https://example.com/1",
            qr_code_base64="base64_1",
            status="UNPAID"
        ),
        PaymentRequest(
            id="req2",
            split_bill_id=split_id,
            participant_name="テスト花子",
            amount=900,
            merchant_payment_id="merchant_id_2",
            pay_url="https://example.com/2",
            qr_code_base64="base64_2",
            status="UNPAID"
        ),
        PaymentRequest(
            id="req3",
            split_bill_id=split_id,
            participant_name="テスト次郎",
            amount=900,
            merchant_payment_id="merchant_id_3",
            pay_url="https://example.com/3",
            qr_code_base64="base64_3",
            status="UNPAID"
        )
    ]
    
    # モックの設定
    mock_repository.get_split_bill.return_value = split_bill
    mock_repository.get_payment_requests_by_split_id.return_value = payment_requests
    
    # テスト用のサービスを作成
    service = SplitPaymentService(client=mock_client, repository=mock_repository)
    
    # 割り勘支払い状態を取得
    result = await service.get_split_payment_status(split_id)
    
    # 結果の検証
    assert result["split_id"] == split_id
    assert result["total_amount"] == 3000
    assert result["created_by"] == "テスト太郎"
    assert len(result["payment_requests"]) == 3
    
    # PayPay API が正しく呼び出されたことを確認
    assert mock_client.get_payment_details.call_count == 3
    
    # リポジトリが正しく呼び出されたことを確認
    mock_repository.get_split_bill.assert_called_once_with(split_id)
    mock_repository.get_payment_requests_by_split_id.assert_called_once_with(split_id)
    assert mock_repository.update_payment_request.call_count == 3


@pytest.mark.asyncio
async def test_handle_payment_webhook(mock_client, mock_repository):
    """PayPay Webhook 処理のテスト"""
    # テスト用のデータを準備
    split_id = str(uuid.uuid4())
    merchant_payment_id = "merchant_id_1"
    
    payment_request = PaymentRequest(
        id="req1",
        split_bill_id=split_id,
        participant_name="テスト太郎",
        amount=1200,
        merchant_payment_id=merchant_payment_id,
        pay_url="https://example.com/1",
        qr_code_base64="base64_1",
        status="UNPAID"
    )
    
    split_bill = SplitBill(
        id=split_id,
        total_amount=3000,
        currency="JPY",
        created_by="テスト太郎",
        participants=[
            Participant(name="テスト太郎", share=0.4),
            Participant(name="テスト花子", share=0.3),
            Participant(name="テスト次郎", share=0.3)
        ],
        payment_requests=["req1", "req2", "req3"]
    )
    
    # モックの設定
    mock_repository.get_payment_request_by_merchant_id.return_value = payment_request
    mock_repository.get_split_bill.return_value = split_bill
    mock_repository.get_payment_requests_by_split_id.return_value = [
        payment_request,
        PaymentRequest(
            id="req2",
            split_bill_id=split_id,
            participant_name="テスト花子",
            amount=900,
            merchant_payment_id="merchant_id_2",
            pay_url="https://example.com/2",
            qr_code_base64="base64_2",
            status="PAID"
        ),
        PaymentRequest(
            id="req3",
            split_bill_id=split_id,
            participant_name="テスト次郎",
            amount=900,
            merchant_payment_id="merchant_id_3",
            pay_url="https://example.com/3",
            qr_code_base64="base64_3",
            status="PAID"
        )
    ]
    
    # テスト用のサービスを作成
    service = SplitPaymentService(client=mock_client, repository=mock_repository)
    
    # Webhook を処理
    result = await service.handle_payment_webhook(
        merchant_payment_id=merchant_payment_id,
        status="COMPLETED"
    )
    
    # 結果の検証
    assert result["merchant_payment_id"] == merchant_payment_id
    assert result["status"] == "PAID"
    assert result["split_bill_id"] == split_id
    
    # リポジトリが正しく呼び出されたことを確認
    mock_repository.get_payment_request_by_merchant_id.assert_called_once_with(merchant_payment_id)
    mock_repository.update_payment_request.assert_called_once()
    mock_repository.get_split_bill.assert_called_once_with(split_id)
    mock_repository.get_payment_requests_by_split_id.assert_called_once_with(split_id)
    mock_repository.update_split_bill.assert_called_once()
