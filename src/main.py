from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.paypay_router import router as paypay_router


# FastAPI アプリケーションの作成
app = FastAPI(
    title="ピザ割り勘アプリ API",
    description="ピザの具を平等に切り分け、PayPayで割り勘できるAPIサーバー",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(paypay_router, prefix="/api")


# ルートエンドポイント
@app.get("/")
async def root():
    return {
        "message": "ピザ割り勘アプリ API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


# アプリケーション起動
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
