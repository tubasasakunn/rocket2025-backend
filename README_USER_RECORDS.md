# ユーザーレコード管理機能

このAPIは、Google Apps Scriptでスプレッドシートに保存されたユーザーデータを操作するためのインターフェースを提供します。

## セットアップ手順

1. `.env.example` ファイルを `.env` にコピーします:
   ```bash
   cp .env.example .env
   ```

2. Google Apps Scriptを公開します:
   - GASプロジェクト (gas/Code.gs, SpreadsheetService.gs) をウェブアプリとして公開
   - デプロイ時の設定:
     - 新しいデプロイを作成
     - アクセスできるユーザー: 「全員 (匿名を含む)」を選択
     - デプロイボタンをクリック

3. 公開されたWebアプリのURLを `.env` ファイルに設定します:
   ```
   GAS_WEBAPP_URL=https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID_HERE/exec
   ```

## API エンドポイント

### ユーザーレコードAPI

基本URL: `/api/user-records`

#### 全ユーザー取得
- **GET** `/api/user-records/`
- すべてのユーザーレコードを取得します

#### ヘルスチェック
- **GET** `/api/user-records/health`
- APIが正常に動作していることを確認します

#### ユーザー取得 (ID)
- **GET** `/api/user-records/{user_id}`
- IDによるユーザー検索

#### ユーザー取得 (アカウント名)
- **GET** `/api/user-records/by-account/{account}`
- アカウント名によるユーザー検索

#### ユーザー作成
- **POST** `/api/user-records/`
- リクエスト例:
  ```json
  {
    "account": "test_user",
    "score": 100
  }
  ```

#### スコア更新
- **PUT** `/api/user-records/{identifier}/score`
- identifierにはユーザーIDまたはアカウント名を指定できます
- リクエスト例:
  ```json
  {
    "score": 200,
    "create_if_not_exist": true
  }
  ```

## 使用例 (Python)

```python
import requests
import json

# APIのベースURL
BASE_URL = "https://your-api-domain.com/api/user-records"

# すべてのユーザーを取得
response = requests.get(f"{BASE_URL}/")
users = response.json()
print("All users:", json.dumps(users, indent=2))

# 特定のユーザーを取得 (ID)
user_id = 1
response = requests.get(f"{BASE_URL}/{user_id}")
user = response.json()
print(f"User {user_id}:", json.dumps(user, indent=2))

# 新しいユーザーを作成
new_user_data = {
    "account": "new_user",
    "score": 50
}
response = requests.post(f"{BASE_URL}/", json=new_user_data)
created_user = response.json()
print("Created user:", json.dumps(created_user, indent=2))

# ユーザーのスコアを更新
update_data = {
    "score": 150,
    "create_if_not_exist": True
}
response = requests.put(f"{BASE_URL}/new_user/score", json=update_data)
updated_user = response.json()
print("Updated user:", json.dumps(updated_user, indent=2))
```

## トラブルシューティング

1. **GAS WebアプリURLが正しく設定されていない**
   - GASプロジェクトが正しく公開されているか確認
   - 環境変数 `GAS_WEBAPP_URL` が正しいURLを指しているか確認

2. **GASのスプレッドシートアクセス権限エラー**
   - スプレッドシートのIDが正しく設定されているか確認
   - スクリプトが実行されるGoogleアカウントがスプレッドシートにアクセスできるか確認

3. **CORS関連のエラー**
   - GASスクリプトの `appsscript.json` ファイルで、webapp.accessが "ANYONE" に設定されているか確認
