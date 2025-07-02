# Rocket2025 Spreadsheet Manager

Google Apps Script for managing user accounts, scores, and timestamps in a Google Spreadsheet.

## 設定方法

1. Googleスプレッドシートを作成し、そのIDをコピーする
   - スプレッドシートのURLは通常 `https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit` の形式
   - この `{SPREADSHEET_ID}` 部分をコピーします

2. スクリプトプロパティーにスプレッドシートIDを設定する
   - スクリプトエディタで「ファイル」→「プロジェクトのプロパティ」→「スクリプトのプロパティ」を開く
   - 「行を追加」をクリック
   - プロパティに「SPREADSHEET_ID」、値に上でコピーしたスプレッドシートIDを入力
   - 「保存」をクリック

3. 以下のコマンドでコードを変更した場合はプッシュする
   ```bash
   clasp push
   ```

4. 変更をデプロイする
   ```bash
   clasp deploy --description "説明"
   ```

## API仕様

### レコード作成 (POST)

```
URL: https://script.google.com/macros/s/{DEPLOYMENT_ID}/exec
Method: POST
Content-Type: application/json
```

リクエスト本文:
```json
{
  "action": "create",
  "account": "ユーザー名",
  "score": 100  // オプション、デフォルト: 0
}
```

### スコア更新 (POST)

```
URL: https://script.google.com/macros/s/{DEPLOYMENT_ID}/exec
Method: POST
Content-Type: application/json
```

リクエスト本文:
```json
{
  "action": "update",
  "identifier": "ユーザー名またはID",
  "score": 150,
  "createIfNotExist": true // オプション、デフォルト: true（存在しないユーザーを自動作成）
}
```

**注意**: `createIfNotExist` を `false` に設定すると、存在しないユーザーの場合はエラーを返します。デフォルトでは `true` で、存在しないユーザー名が指定された場合は新規アカウントを作成します。

### 全レコード取得 (GET)

```
URL: https://script.google.com/macros/s/{DEPLOYMENT_ID}/exec
Method: GET
```

### アカウント名でレコード検索 (GET)

```
URL: https://script.google.com/macros/s/{DEPLOYMENT_ID}/exec?action=findByAccount&account=ユーザー名
Method: GET
```

### IDでレコード検索 (GET)

```
URL: https://script.google.com/macros/s/{DEPLOYMENT_ID}/exec?action=findById&id=1
Method: GET
```

## レスポンス形式

### 成功時
```json
{
  "status": "success",
  "data": {...} // 操作に応じたデータ
}
```

### エラー時
```json
{
  "status": "error",
  "message": "エラー内容"
}
```

## 注意事項

- このスクリプトを使用するには、スクリプトプロパティーに `SPREADSHEET_ID` を設定する必要があります
  - 設定方法は上記の「設定方法」セクションを参照してください
  - スプレッドシートIDを設定しない場合、初回実行時に新しいスプレッドシートが自動的に作成されます
- WebアプリのURLは `clasp deployments` コマンドで確認できます
- 初めてWebアプリにアクセスする際に権限の承認が必要な場合があります
