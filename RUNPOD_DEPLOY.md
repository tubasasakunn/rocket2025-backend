# RunPodへのデプロイ手順

このドキュメントでは、ピザ割り勘アプリバックエンドをRunPodにデプロイする方法について説明します。

## 前提条件

- RunPodアカウントが必要です
- DockerHubアカウントが必要です（コンテナイメージの保存用）
- Dockerがローカル環境にインストールされていること

## デプロイ手順

### 1. Dockerイメージのビルド

```sh
docker build -t your-dockerhub-username/pizza-app-backend:latest .
```

### 2. DockerHubへのプッシュ

```sh
docker login
docker push your-dockerhub-username/pizza-app-backend:latest
```

### 3. RunPodでのサーバーレスデプロイ

1. RunPod管理コンソール（https://www.runpod.io/console/serverless）にアクセス
2. 「New Endpoint」をクリック
3. 以下の情報を入力：
   - Name: pizza-app-backend
   - Docker Image: your-dockerhub-username/pizza-app-backend:latest
   - Docker Container Start Command: `python -m runpod_handler`
   - Min Memory (GB): 8
   - vCPU: 2
   - GPU Type: CPU (GPUが不要の場合)
   - Min Instances: 1
   - Max Instances: 3

4. 「Deploy」をクリックして完了

### 4. 環境変数の設定

RunPod管理コンソールから、作成したエンドポイントの「Settings」を開き、必要な環境変数を設定します：

- PAYPAY_API_KEY: PayPayのAPIキー
- PAYPAY_API_SECRET: PayPayのAPIシークレット
- PAYPAY_MERCHANT_ID: PayPay加盟店ID
- その他、必要に応じて追加の環境変数

### 5. エンドポイントの確認

デプロイが完了すると、RunPodから固有のエンドポイントURLが発行されます。このURLにAPIリクエストを送信することで、バックエンドサービスにアクセスできます。

```
https://api.runpod.ai/v2/{endpoint-id}/run
```

## 注意事項

- 画像処理を行うアプリケーションのため、メモリ使用量が大きくなる可能性があります。必要に応じてスペックを調整してください。
- サーバーレス関数は使用されていない時間はコストがかかりません。
- 初回リクエスト時はコールドスタートがあるため、レスポンスに時間がかかることがあります。

## ローカルでのテスト

RunPodハンドラーをローカルでテストするには、以下のコマンドを実行します：

```sh
python runpod_handler.py
```

これによりUvicornサーバーが起動し、ポート8000でAPIが利用可能になります。

## トラブルシューティング

- ログはRunPod管理コンソールの「Logs」タブで確認できます。
- コンテナ内でのデバッグが必要な場合は、RunPodの「Connect to SSH」機能を使用してコンテナ内に接続できます。
