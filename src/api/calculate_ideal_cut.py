from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import sys
import uuid
from pathlib import Path
import shutil
import tempfile

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from service.pizza_split.process import PizzaProcessor

# レスポンスモデル
class CalculateIdealCutResponse(BaseModel):
    svg: str

# ルーター作成
router = APIRouter()

@router.post("/calculate-ideal-cut", response_model=CalculateIdealCutResponse)
async def calculate_ideal_cut(
    file: UploadFile = File(..., description="ピザの画像ファイル (JPG, PNG)")
):
    """
    ピザ画像を解析して理想的なカット線のSVGを生成
    """
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一時ディレクトリと一意のファイル名生成
    temp_dir = Path(tempfile.mkdtemp())
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    upload_path = temp_dir / f"{file_id}{file_extension}"
    
    try:
        # ファイル保存
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # PizzaProcessorを使用して処理
        processor = PizzaProcessor(output_dir=str(temp_dir / "output"))
        
        # SVGのみを生成（高速化のため）
        result = processor.process_image(
            str(upload_path),
            n_pieces=4,  # デフォルトで4分割
            debug=False,
            return_svg_only=True,
            quiet=True  # ログ出力を抑制
        )
        
        # SVGコンテンツを取得
        svg_content = result.get('svg_content', '')
        
        if not svg_content:
            raise HTTPException(status_code=500, detail="SVG生成に失敗しました")
        
        return CalculateIdealCutResponse(svg=svg_content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理中にエラーが発生しました: {str(e)}")
    
    finally:
        # 一時ファイルをクリーンアップ
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass