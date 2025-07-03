from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import uuid
from pathlib import Path
import shutil
import numpy as np

from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.pizza_circle_detection_service import PizzaCircleDetectionService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.salami_circle_detection_service import SalamiCircleDetectionService
from service.pizza_split.preprocess import PreprocessService
from service.pizza_split.process import PizzaProcessor


# モデル定義
class PizzaAnalysisResponse(BaseModel):
    """ピザ解析結果のレスポンス"""
    success: bool
    pizza_circle: Optional[Dict[str, Any]] = None
    salami_circles: List[Dict[str, Any]] = []
    preprocessing_applied: bool = False
    error_message: Optional[str] = None


class PizzaDivisionResponse(BaseModel):
    """ピザ分割結果のレスポンス"""
    success: bool
    svg_before_explosion: Optional[str] = None  # 爆発前のSVG
    svg_after_explosion: Optional[str] = None   # 爆発後のSVG
    svg_animated: Optional[str] = None          # アニメーション付きSVG
    error_message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    message: str
    services_available: List[str]


# ルーター作成
router = APIRouter(prefix="/pizza-cutter", tags=["pizza-cutter"])

# アップロード用ディレクトリ
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# サービス依存関係
def get_pizza_segmentation_service() -> PizzaSegmentationService:
    return PizzaSegmentationService()


def get_pizza_circle_detection_service() -> PizzaCircleDetectionService:
    return PizzaCircleDetectionService()


def get_salami_segmentation_service() -> SalamiSegmentationService:
    return SalamiSegmentationService()


def get_salami_circle_detection_service() -> SalamiCircleDetectionService:
    return SalamiCircleDetectionService()


def get_preprocess_service() -> PreprocessService:
    return PreprocessService()


# エンドポイント定義
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    ピザカッター API のヘルスチェック
    
    各サービスの動作確認を行います。
    """
    try:
        # 各サービスのインスタンス化テスト
        services = []
        
        # Pizza Segmentation Service
        try:
            get_pizza_segmentation_service()
            services.append("pizza_segmentation")
        except Exception:
            pass
            
        # Pizza Circle Detection Service
        try:
            get_pizza_circle_detection_service()
            services.append("pizza_circle_detection")
        except Exception:
            pass
            
        # Salami Segmentation Service
        try:
            get_salami_segmentation_service()
            services.append("salami_segmentation")
        except Exception:
            pass
            
        # Salami Circle Detection Service
        try:
            get_salami_circle_detection_service()
            services.append("salami_circle_detection")
        except Exception:
            pass
        
        return HealthCheckResponse(
            status="healthy",
            message="Pizza Cutter API is running",
            services_available=services
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/divide", response_model=PizzaDivisionResponse)
async def divide_pizza(
    file: UploadFile = File(..., description="ピザの画像ファイル (JPG, PNG)"),
    n_pieces: int = Form(4, description="分割するピース数（デフォルト: 4）")
):
    """
    ピザ画像を指定されたピース数に分割
    
    アップロードされた画像に対して以下の処理を実行:
    1. 前処理（楕円→円形変換）
    2. ピザ領域の検出と円近似
    3. サラミの検出と個別円検出
    4. 移動ナイフ法による分割
    5. 全体SVGと各ピースのSVGを生成
    """
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一意のファイル名生成
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    upload_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    try:
        # ファイル保存
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # PizzaProcessorを使用して処理
        processor = PizzaProcessor(output_dir=str(UPLOAD_DIR / "process" / file_id))
        result = processor.process_image(
            str(upload_path),
            n_pieces=n_pieces,
            debug=False,
            return_svg_only=True,
            quiet=True
        )
        
        # 爆発SVGを読み込む
        svg_before = None
        svg_after = None
        svg_animated = None
        
        # return_svg_onlyの場合、SVGパスが返される
        if 'svg_before_original' in result and result['svg_before_original']:
            before_path = Path(result['svg_before_original'])
            if before_path.exists():
                svg_before = before_path.read_text()
        
        if 'svg_after_original' in result and result['svg_after_original']:
            after_path = Path(result['svg_after_original'])
            if after_path.exists():
                svg_after = after_path.read_text()
        
        if 'svg_animated_original' in result and result['svg_animated_original']:
            animated_path = Path(result['svg_animated_original'])
            if animated_path.exists():
                svg_animated = animated_path.read_text()
        
        return PizzaDivisionResponse(
            success=True,
            svg_before_explosion=svg_before,
            svg_after_explosion=svg_after,
            svg_animated=svg_animated
        )
    
    except Exception as e:
        return PizzaDivisionResponse(
            success=False,
            error_message=str(e)
        )
    
    finally:
        # アップロードファイルをクリーンアップ
        try:
            if upload_path.exists():
                upload_path.unlink()
            # process_dirもクリーンアップ
            process_dir = UPLOAD_DIR / "process" / file_id
            if process_dir.exists():
                shutil.rmtree(process_dir)
        except Exception:
            pass


@router.post("/analyze", response_model=PizzaAnalysisResponse)
async def analyze_pizza(
    file: UploadFile = File(..., description="ピザの画像ファイル (JPG, PNG)")
):
    """
    ピザ画像を解析してピザの形状とサラミの位置を検出
    
    アップロードされた画像に対して以下の処理を実行:
    1. 前処理（楕円→円形変換）
    2. ピザ領域の検出と円近似
    3. サラミの検出と個別円検出
    """
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一意のファイル名生成
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    upload_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    try:
        # ファイル保存
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. 前処理
        preprocessing_applied = False
        processed_image_path = upload_path
        
        try:
            preprocess_service = get_preprocess_service()
            preprocessed_image, preprocessing_info = preprocess_service.preprocess_pizza_image(
                str(upload_path), 
                str(UPLOAD_DIR / f"preprocessed_{file_id}{file_extension}")
            )
            if preprocessing_info.get("transformation_applied", False):
                preprocessing_applied = True
                processed_image_path = UPLOAD_DIR / f"preprocessed_{file_id}{file_extension}"
        except Exception as e:
            # 前処理が失敗しても元画像で続行
            print(f"Preprocessing failed, using original image: {e}")
        
        # 2. ピザ円検出
        pizza_circle = None
        try:
            circle_detection_service = get_pizza_circle_detection_service()
            center, radius = circle_detection_service.detect_circle_from_image(str(processed_image_path))
            pizza_circle = {
                "center": {"x": float(center[0]), "y": float(center[1])},
                "radius": float(radius)
            }
        except Exception as e:
            print(f"Pizza circle detection failed: {e}")
        
        # 3. サラミ円検出
        salami_circles = []
        try:
            salami_circle_service = get_salami_circle_detection_service()
            detected_circles = salami_circle_service.detect_salami_circles(str(processed_image_path))
            
            for i, (center, radius) in enumerate(detected_circles):
                salami_circles.append({
                    "id": i + 1,
                    "center": {"x": float(center[0]), "y": float(center[1])},
                    "radius": float(radius)
                })
        except Exception as e:
            print(f"Salami circle detection failed: {e}")
        
        return PizzaAnalysisResponse(
            success=True,
            pizza_circle=pizza_circle,
            salami_circles=salami_circles,
            preprocessing_applied=preprocessing_applied
        )
    
    except Exception as e:
        return PizzaAnalysisResponse(
            success=False,
            error_message=str(e)
        )
    
    finally:
        # アップロードファイルをクリーンアップ
        try:
            if upload_path.exists():
                upload_path.unlink()
            preprocessed_path = UPLOAD_DIR / f"preprocessed_{file_id}{file_extension}"
            if preprocessed_path.exists():
                preprocessed_path.unlink()
        except Exception:
            pass


@router.get("/test", response_model=Dict[str, Any])
async def test_with_sample_images():
    """
    サンプル画像を使用した疎通テスト
    
    resource/ フォルダのサンプル画像を使用してAPIの動作確認を行います。
    """
    try:
        results = {}
        resource_dir = Path("resource")
        
        if not resource_dir.exists():
            raise HTTPException(status_code=404, detail="resource directory not found")
        
        # サンプル画像を検索
        sample_images = list(resource_dir.glob("pizza*.jpg")) + list(resource_dir.glob("pizza*.png"))
        
        if not sample_images:
            raise HTTPException(status_code=404, detail="No sample pizza images found in resource directory")
        
        for image_path in sample_images[:2]:  # 最大2つの画像をテスト
            try:
                # ピザ円検出
                circle_detection_service = get_pizza_circle_detection_service()
                center, radius = circle_detection_service.detect_circle_from_image(str(image_path))
                
                # サラミ円検出
                salami_circle_service = get_salami_circle_detection_service()
                detected_circles = salami_circle_service.detect_salami_circles(str(image_path))
                
                results[image_path.name] = {
                    "pizza_circle": {
                        "center": {"x": float(center[0]), "y": float(center[1])},
                        "radius": float(radius)
                    },
                    "salami_count": len(detected_circles),
                    "salami_circles": [
                        {
                            "center": {"x": float(c[0]), "y": float(c[1])},
                            "radius": float(r)
                        } for c, r in detected_circles
                    ]
                }
            except Exception as e:
                results[image_path.name] = {
                    "error": str(e)
                }
        
        return {
            "success": True,
            "message": "Sample image analysis completed",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))