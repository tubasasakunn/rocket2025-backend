# ユーザーレコードAPI ドキュメント

## 概要

このAPIは、Google Sheetsに保存されたユーザーデータを操作するためのエンドポイントを提供します。
データはGoogle Apps Script Webアプリを通じて操作されます。

## ベースURL

```
https://your-vercel-deployment-url/api/user-records
```

## エンドポイント

### 1. すべてのユーザーを取得

**リクエスト**:
```
GET /api/user-records
```

**レスポンス例**:
```json
[
  {
    "id": 1,
    "account": "user1",
    "score": 150,
    "update_at": "2025-07-02T10:30:45.000Z"
  },
  {
    "id": 2,
    "account": "user2",
    "score": 75,
    "update_at": "2025-07-01T15:20:10.000Z"
  }
]
```

### 2. ユーザーをIDで検索

**リクエスト**:
```
GET /api/user-records/{user_id}
```

**パラメータ**:
- `user_id`: ユーザーID (数値)

**レスポンス例**:
```json
{
  "id": 1,
  "account": "user1",
  "score": 150,
  "update_at": "2025-07-02T10:30:45.000Z"
}
```

### 3. ユーザーをアカウント名で検索

**リクエスト**:
```
GET /api/user-records/by-account/{account}
```

**パラメータ**:
- `account`: ユーザーアカウント名 (文字列)

**レスポンス例**:
```json
{
  "id": 1,
  "account": "user1",
  "score": 150,
  "update_at": "2025-07-02T10:30:45.000Z"
}
```

### 4. 新規ユーザー作成

**リクエスト**:
```
POST /api/user-records
```

**リクエストボディ**:
```json
{
  "account": "new_user",
  "score": 100
}
```

**レスポンス例**:
```json
{
  "id": 3,
  "account": "new_user",
  "score": 100,
  "update_at": "2025-07-02T14:45:30.000Z"
}
```

### 5. ユーザースコア更新

**リクエスト**:
```
PUT /api/user-records/{identifier}/score
```

**パラメータ**:
- `identifier`: ユーザーIDまたはアカウント名

**リクエストボディ**:
```json
{
  "score": 200,
  "create_if_not_exist": true
}
```

**レスポンス例**:
```json
{
  "id": 1,
  "account": "user1",
  "score": 200,
  "update_at": "2025-07-02T15:00:00.000Z"
}
```

### 6. ヘルスチェック

**リクエスト**:
```
GET /api/user-records/health
```

**レスポンス例**:
```json
{
  "status": "ok",
  "message": "User Record API is running"
}
```

## エラーレスポンス

### ユーザーが見つからない場合 (404 Not Found)
```json
{
  "detail": "User with ID 999 not found"
}
```

### 重複するアカウント名で作成した場合 (409 Conflict)
```json
{
  "detail": "User with account 'existing_user' already exists"
}
```

### サーバーエラー (500 Internal Server Error)
```json
{
  "detail": "Failed to get users: GAS API error: Error message"
}
```

## cURLコマンド例

### 全ユーザー取得
```bash
curl -X GET "https://your-api-url/api/user-records"
```

### ユーザーIDで検索
```bash
curl -X GET "https://your-api-url/api/user-records/1"
```

### アカウント名で検索
```bash
curl -X GET "https://your-api-url/api/user-records/by-account/user1"
```

### 新規ユーザー作成
```bash
curl -X POST "https://your-api-url/api/user-records" \
  -H "Content-Type: application/json" \
  -d '{"account": "new_user", "score": 100}'
```

### スコア更新
```bash
curl -X PUT "https://your-api-url/api/user-records/user1/score" \
  -H "Content-Type: application/json" \
  -d '{"score": 200, "create_if_not_exist": true}'
```
