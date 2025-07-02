# PayPay 割り勘機能

このドキュメントでは、ピザ割り勘アプリの PayPay 割り勘機能の使い方について説明します。

## 目次

1. [概要](#概要)
2. [セットアップ](#セットアップ)
3. [API エンドポイント](#api-エンドポイント)
4. [使用例](#使用例)
5. [開発ガイド](#開発ガイド)

## 概要

PayPay 割り勘機能は、ピザを注文した代表者がまとめて支払い、参加者に PayPay 送金で後から精算するための機能です。
各参加者に対して PayPay の支払いリクエスト（QR コード / URL）を生成し、参加者はそれを使って代表者に送金することができます。

### 主な機能

- 参加者ごとの支払い金額を計算（均等割り or 比率指定）
- 各参加者向けの PayPay 支払いリクエスト（QR コード / URL）を生成
- 支払い状況の追跡
- PayPay からの Webhook による支払い状態の自動更新

## セットアップ

### 1. PayPay Developer アカウントの作成

1. [PayPay Developer Portal](https://developer.paypay.ne.jp/) にアクセスし、アカウントを作成
2. アプリケーションを作成し、API キー、API シークレット、マーチャント ID を取得

### 2. 環境変数の設定

`.env.example` ファイルを `.env` としてコピーし、PayPay の API 認証情報を設定します。

```bash
cp .env.example .env
```

`.env` ファイルを編集し、以下の項目を設定します。

```
PAYPAY_API_KEY=your_api_key_here
PAYPAY_API_SECRET=your_api_secret_here
PAYPAY_MERCHANT_ID=your_merchant_id_here
PAYPAY_IS_PRODUCTION=false  # 開発中は false (Sandbox)
APP_HOST=http://localhost:8000
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. アプリケーションの起動

```bash
cd src
uvicorn main:app --reload
```

アプリケーションが http://localhost:8000 で起動します。
API ドキュメントは http://localhost:8000/docs で確認できます。

## API エンドポイント

### 割り勘作成

```
POST /api/paypay/split
```

#### リクエスト例

```json
{
  "total_amount": 3000,
  "created_by": "田中太郎",
  "participants": [
    {
      "name": "田中太郎",
      "share": 0.4
    },
    {
      "name": "鈴木花子",
      "share": 0.3
    },
    {
      "name": "佐藤次郎",
      "share": 0.3
    }
  ],
  "description": "ピザの割り勘"
}
```

#### レスポンス例

```json
{
  "split_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_amount": 3000,
  "currency": "JPY",
  "created_by": "田中太郎",
  "payment_requests": [
    {
      "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "participant_name": "田中太郎",
      "amount": 1200,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "UNPAID"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "participant_name": "鈴木花子",
      "amount": 900,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "UNPAID"
    },
    {
      "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "participant_name": "佐藤次郎",
      "amount": 900,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "UNPAID"
    }
  ]
}
```

### 割り勘状態取得

```
GET /api/paypay/split/{split_id}
```

#### レスポンス例

```json
{
  "split_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_amount": 3000,
  "currency": "JPY",
  "created_by": "田中太郎",
  "status": "OPEN",
  "created_at": "2025-07-02T12:00:00",
  "payment_requests": [
    {
      "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "participant_name": "田中太郎",
      "amount": 1200,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "PAID"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "participant_name": "鈴木花子",
      "amount": 900,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "UNPAID"
    },
    {
      "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "participant_name": "佐藤次郎",
      "amount": 900,
      "pay_url": "https://stg-link.paypay.ne.jp/...",
      "qr_code_base64": "data:image/png;base64,...",
      "status": "UNPAID"
    }
  ]
}
```

### PayPay Webhook

```
POST /api/paypay/webhook
```

PayPay からの支払い状態更新通知を受け取るエンドポイントです。
PayPay Developer Portal で Webhook URL を設定する必要があります。

## 使用例

### フロントエンド実装例

```javascript
// 割り勘作成
async function createSplitPayment() {
  const response = await fetch('/api/paypay/split', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      total_amount: 3000,
      created_by: "田中太郎",
      participants: [
        { name: "田中太郎", share: 0.4 },
        { name: "鈴木花子", share: 0.3 },
        { name: "佐藤次郎", share: 0.3 }
      ],
      description: "ピザの割り勘"
    })
  });
  
  const data = await response.json();
  
  // QRコードを表示
  data.payment_requests.forEach(request => {
    // QRコードを表示
    const qrImage = document.createElement('img');
    qrImage.src = request.qr_code_base64;
    document.getElementById('qr-container').appendChild(qrImage);
    
    // 支払いURLを表示
    const payLink = document.createElement('a');
    payLink.href = request.pay_url;
    payLink.textContent = `${request.participant_name}の支払いリンク`;
    document.getElementById('links-container').appendChild(payLink);
  });
  
  // 割り勘IDを保存
  localStorage.setItem('splitId', data.split_id);
}

// 支払い状況の確認
async function checkPaymentStatus() {
  const splitId = localStorage.getItem('splitId');
  const response = await fetch(`/api/paypay/split/${splitId}`);
  const data = await response.json();
  
  // 支払い状況を表示
  data.payment_requests.forEach(request => {
    const statusElement = document.getElementById(`status-${request.participant_name}`);
    statusElement.textContent = request.status === 'PAID' ? '支払い済み' : '未払い';
  });
  
  // 全員支払い完了したかチェック
  if (data.status === 'COMPLETE') {
    alert('全員の支払いが完了しました！');
  }
}
```

## 開発ガイド

### ディレクトリ構造

```
src/
├── api/
│   └── paypay_router.py   # API エンドポイント
├── service/
│   └── paypay/
│       ├── client.py      # PayPay API クライアント
│       ├── repository.py  # データ永続化
│       ├── settings.py    # 設定
│       └── split_payment_service.py  # 割り勘ロジック
└── main.py                # アプリケーションエントリーポイント
```

### PayPay API ドキュメント

詳細な API 仕様については、[PayPay API ドキュメント](https://developer.paypay.ne.jp/products/docs/webpayment) を参照してください。

### テスト

テストを実行するには以下のコマンドを使用します。

```bash
pytest
```

### 本番環境への移行

本番環境に移行する際は、以下の点に注意してください。

1. `.env` ファイルの `PAYPAY_IS_PRODUCTION` を `true` に設定
2. 本番用の API キー、API シークレット、マーチャント ID を設定
3. `APP_HOST` を本番環境の URL に設定
4. PayPay Developer Portal で Webhook URL を本番環境の URL に設定
5. CORS 設定を適切に制限 (`main.py` の `allow_origins` を修正)
