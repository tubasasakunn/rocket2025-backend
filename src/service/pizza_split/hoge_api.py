#!/usr/bin/env python
# coding: utf-8
"""
FastAPI版のピザ分割API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
import tempfile
import os
from pathlib import Path
import sys

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))

from hoge import process_pizza_image

app = FastAPI(title="Pizza Division API", version="1.0.0")


@app.get("/")
async def root():
    """APIの情報を返す"""
    return {
        "message": "Pizza Division API",
        "endpoints": {
            "/divide": "POST - Upload pizza image and get SVG with division lines",
            "/docs": "GET - API documentation"
        }
    }


@app.post("/divide")
async def divide_pizza(
    image: UploadFile = File(..., description="ピザ画像ファイル"),
    n_people: int = Form(..., ge=2, le=99, description="分割人数（2〜99）")
):
    """
    ピザ画像を指定人数で分割してSVGを返す
    
    Args:
        image: アップロードされたピザ画像
        n_people: 分割する人数（2〜99）
    
    Returns:
        SVGコンテンツ
    """
    # 画像の検証
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
        content = await image.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # ピザ分割処理を実行
        svg_content = process_pizza_image(tmp_path, n_people)
        
        # SVGレスポンスを返す
        return Response(
            content=svg_content,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f'inline; filename="pizza_divided_{n_people}.svg"'
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理中にエラーが発生しました: {str(e)}")
    
    finally:
        # 一時ファイルを削除
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/divide/json")
async def divide_pizza_json(
    image: UploadFile = File(..., description="ピザ画像ファイル"),
    n_people: int = Form(..., ge=2, le=99, description="分割人数（2〜99）")
):
    """
    ピザ画像を指定人数で分割してSVGをJSON形式で返す
    
    Args:
        image: アップロードされたピザ画像
        n_people: 分割する人数（2〜99）
    
    Returns:
        JSONレスポンス（SVGコンテンツを含む）
    """
    # 画像の検証
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
        content = await image.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # ピザ分割処理を実行
        svg_content = process_pizza_image(tmp_path, n_people)
        
        return {
            "success": True,
            "n_people": n_people,
            "filename": image.filename,
            "svg_content": svg_content
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理中にエラーが発生しました: {str(e)}")
    
    finally:
        # 一時ファイルを削除
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    # APIサーバーを起動
    uvicorn.run(app, host="0.0.0.0", port=8000)