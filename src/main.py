from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ログ設定
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app")
logger.info(f"Starting application with log level: {log_level}")

# 正しいインポートパスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 自作ミドルウェアのインポート
from middleware import BinaryDataHandlingMiddleware

try:
    from src.api.pizza_cutter_router import router as pizza_cutter_router
    from src.api.calculate_ideal_cut import router as calculate_ideal_cut_router
    from src.api.user_record_router import router as user_record_router
    from src.api.pizza_score_router import router as pizza_score_router
    from src.api.emotion_recognition_router import router as emotion_recognition_router
    from src.api.face_find_router import router as face_find_router
    from src.api.face_emotion_router import router as face_emotion_router
except ImportError:
    # ローカル環境での相対インポート
    from api.pizza_cutter_router import router as pizza_cutter_router
    from api.calculate_ideal_cut import router as calculate_ideal_cut_router
    from api.user_record_router import router as user_record_router
    from api.pizza_score_router import router as pizza_score_router
    from api.emotion_recognition_router import router as emotion_recognition_router
    from api.face_find_router import router as face_find_router
    from api.face_emotion_router import router as face_emotion_router


# FastAPI アプリケーションの作成
app = FastAPI(
    title="ピザ割り勘アプリ API",
    description="ピザの具を平等に切り分けるアプリのAPIサーバー",
    version="1.0.0",
    debug=os.getenv("FASTAPI_DEBUG", "false").lower() == "true"
)

# カスタムミドルウェア追加 - バイナリデータ処理
app.add_middleware(BinaryDataHandlingMiddleware)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(pizza_cutter_router, prefix="/api")
app.include_router(calculate_ideal_cut_router, prefix="/api")
app.include_router(user_record_router, prefix="/api")
app.include_router(pizza_score_router, prefix="/api")
app.include_router(emotion_recognition_router, prefix="/api")
app.include_router(face_find_router, prefix="/api")
app.include_router(face_emotion_router, prefix="/api")

# エラーハンドリング
@app.exception_handler(UnicodeDecodeError)
async def unicode_decode_exception_handler(request, exc):
    logger.error(f"UnicodeDecodeError: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "バイナリデータのデコード中にエラーが発生しました。データがBase64形式でエンコードされているか確認してください。",
            "error_type": "UnicodeDecodeError"
        }
    )


# ルートエンドポイント
@app.get("/")
async def root():
    return {
        "message": "ピザ割り勘アプリ API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# グローバル例外ハンドラーを追加
@app.exception_handler(UnicodeDecodeError)
async def unicode_decode_exception_handler(request, exc):
    logger.error(f"UnicodeDecodeError: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "バイナリデータのデコード中にエラーが発生しました。Base64エンコーディングを確認してください。",
            "error": str(exc)
        }
    )


# アプリケーション起動
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
