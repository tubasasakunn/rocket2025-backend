import os
import runpod
import uvicorn
from src.main import app

# RunPodのエンドポイントハンドラー
def handler(event):
    """
    RunPod APIリクエストをFastAPIアプリケーションに転送するハンドラー
    """
    try:
        method = event.get("method", "GET")
        path = event.get("path", "/")
        query_params = event.get("query", {})
        headers = event.get("headers", {})
        body = event.get("body", None)

        # FastAPIに転送するためのリクエストを作成
        # 注意: 実際の実装では、より複雑なリクエスト処理が必要になる場合があります
        return {
            "statusCode": 200,
            "body": f"RunPod request: {method} {path}"
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }

# RunPod Serverless モードで動作
runpod.serverless.start({"handler": handler})

# ローカル開発用のUvicorn起動部分
if __name__ == "__main__":
    # 環境変数からポート番号を取得（RunPodの場合は8000）
    port = int(os.environ.get("PORT", 8000))
    
    # FastAPIアプリケーションを起動
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
