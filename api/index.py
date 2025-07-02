from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sys
import os

# src ディレクトリをインポートパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.main import app
    
    # エラーハンドラを追加
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {str(exc)}"}
        )
        
    # VercelのサーバーレスFunctionとして動作させるための設定
    handler = app
except Exception as e:
    # インポートエラーなどのデバッグのために、最小限のアプリを作成
    app = FastAPI()
    
    @app.get("/debug")
    def debug_info():
        return {
            "error": str(e),
            "sys_path": sys.path,
            "cwd": os.getcwd(),
            "files": os.listdir(os.getcwd())
        }
    
    handler = app
