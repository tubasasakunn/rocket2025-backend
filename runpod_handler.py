import os
import sys
import runpod
import uvicorn

# デバッグ: Pythonパスとインストール済みパッケージを表示
print("PYTHONPATH:", sys.path)
print("Current directory:", os.getcwd())

try:
    # svgwriteを明示的にインポート
    import svgwrite
    print("Successfully imported svgwrite")
except ImportError as e:
    print(f"Failed to import svgwrite: {e}")
    # svgwriteを自動インストールして再試行
    import subprocess
    print("Attempting to install svgwrite...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "svgwrite"])
    import svgwrite
    print("Successfully installed and imported svgwrite")

# アプリケーションをインポート
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

# メイン処理
def main():
    # 環境変数からモードを取得
    mode = os.environ.get("RUNPOD_MODE", "serverless")
    
    if mode.lower() == "serverless":
        # RunPod Serverless モードで動作
        print("Starting in RunPod serverless mode...")
        runpod.serverless.start({"handler": handler})
    else:
        # ローカル開発用のUvicorn起動
        print("Starting in local development mode...")
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=port,
            reload=False
        )

if __name__ == "__main__":
    main()
