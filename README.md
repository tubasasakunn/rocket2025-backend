# ピザ分配API

## 概要

ピザとサラミの画像から理想的な切り方を提案し、実際の切り方との差分をスコアリングするAPIサービスです。

## API仕様

### 1. 理想的な切り方を計算
```
POST /api/calculate-ideal-cut
```

**リクエスト**
```json
{
  "image": "base64_encoded_image",
  "num_pieces": 4
}
```

**レスポンス**
```json
{
  "svg": "<svg>...</svg>"
}
```

### 2. スコア計算
```
POST /api/calculate-score
```

**リクエスト**
```json
{
  "actual_image": "base64_encoded_image",
  "ideal_image": "base64_encoded_image"
}
```

**レスポンス**
```json
{
  "score": 85.5
}
```

### 3. スコア保存
```
POST /api/save-score
```

**リクエスト**
```json
{
  "account_name": "ピザ太郎",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "score": 85.5
}
```

**レスポンス**
```json
{
  "success": true
}
```

### 4. ランキング取得
```
GET /api/ranking
```

**レスポンス**
```json
{
  "ranking": [
    {
      "rank": 1,
      "account_name": "ピザマスター",
      "score": 98.5
    },
    {
      "rank": 2,
      "account_name": "サラミ王",
      "score": 92.0
    }
  ]
}
```

## ディレクトリ構成

```
pizza-sharing-api/
├── api/
│   ├── index.py                    # Vercelエントリーポイント / FastAPIアプリケーション定義
│   ├── ideal_cut.py               # 理想的な切り方計算エンドポイント
│   ├── scoring.py                 # スコア計算エンドポイント
│   └── ranking.py                 # ランキングエンドポイント
├── services/
│   ├── __init__.py
│   ├── image_processor.py         # 画像解析処理（ピザ・サラミ検出）
│   ├── cut_calculator.py          # 最適な切り方の計算ロジック
│   ├── svg_generator.py           # SVG形式での切り方可視化
│   ├── score_calculator.py        # スコア算出ロジック
│   └── spreadsheet_service.py     # Google Apps Script連携
├── utils/
│   ├── __init__.py
│   └── geometry.py                # 幾何学計算ユーティリティ
├── pyproject.toml                 # uv設定ファイル
├── requirements.txt               # 依存関係
├── vercel.json                    # Vercel設定
└── README.md
```

## 技術スタック

- Python 3.12
- uv (パッケージ管理)
- FastAPI
- OpenCV (画像処理)
- NumPy (数値計算)
- Google Apps Script + スプレッドシート (データ保存)
- Vercel (デプロイ)
