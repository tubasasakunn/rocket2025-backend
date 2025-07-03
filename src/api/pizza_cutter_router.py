from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import uuid
from pathlib import Path
import shutil
import numpy as np
import cv2
import base64
from PIL import Image, ImageDraw
import io
import xml.etree.ElementTree as ET
import re

from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.pizza_circle_detection_service import PizzaCircleDetectionService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.salami_circle_detection_service import SalamiCircleDetectionService
from service.pizza_split.preprocess import PreprocessService
from service.pizza_split.process import PizzaProcessor
# from service.pizza_split.score import process_pizza_image, analyze_pizza_regions, calculate_scores, calculate_fairness_score


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
    piece_svgs: List[str] = []                  # 各ピースの個別SVG（色付き）
    overlay_image: Optional[str] = None         # 元画像にオーバーレイしたPNG画像（base64）
    error_message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    message: str
    services_available: List[str]


class PizzaScoreResponse(BaseModel):
    """ピザスコア計算結果のレスポンス"""
    success: bool
    fairness_score: float = Field(description="公平性スコア（0-100）")
    error_message: Optional[str] = None


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


def parse_svg_elements(svg_content: str) -> dict:
    """
    SVGコンテンツから描画要素を抽出
    
    Args:
        svg_content: SVGコンテンツ文字列
        
    Returns:
        dict: 描画要素の辞書
    """
    elements = {
        'circles': [],
        'lines': [],
        'polygons': []
    }
    
    try:
        # SVGのXMLを解析
        root = ET.fromstring(svg_content)
        
        # 名前空間を除去してタグ名のみ取得
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag == 'circle':
                try:
                    cx = float(elem.get('cx', 0))
                    cy = float(elem.get('cy', 0))
                    r = float(elem.get('r', 0))
                    stroke = elem.get('stroke', 'black')
                    stroke_width = float(elem.get('stroke-width', 1))
                    fill = elem.get('fill', 'none')
                    elements['circles'].append({
                        'cx': cx, 'cy': cy, 'r': r, 
                        'stroke': stroke, 'stroke_width': stroke_width, 'fill': fill
                    })
                except (ValueError, TypeError):
                    continue
                    
            elif tag == 'line':
                try:
                    x1 = float(elem.get('x1', 0))
                    y1 = float(elem.get('y1', 0))
                    x2 = float(elem.get('x2', 0))
                    y2 = float(elem.get('y2', 0))
                    stroke = elem.get('stroke', 'black')
                    stroke_width = float(elem.get('stroke-width', 1))
                    elements['lines'].append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'stroke': stroke, 'stroke_width': stroke_width
                    })
                except (ValueError, TypeError):
                    continue
                    
            elif tag == 'polygon':
                try:
                    points_str = elem.get('points', '')
                    if points_str:
                        points = []
                        coords = points_str.replace(',', ' ').split()
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                points.append((float(coords[i]), float(coords[i+1])))
                        
                        stroke = elem.get('stroke', 'black')
                        stroke_width = float(elem.get('stroke-width', 1))
                        fill = elem.get('fill', 'none')
                        elements['polygons'].append({
                            'points': points, 'stroke': stroke, 
                            'stroke_width': stroke_width, 'fill': fill
                        })
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"SVG解析エラー: {e}")
    
    return elements


def color_name_to_rgb(color_name: str) -> tuple:
    """色名をRGB値に変換"""
    color_map = {
        'black': (0, 0, 0),
        'brown': (165, 42, 42),
        'darkred': (139, 0, 0),
        'indianred': (205, 92, 92),
        'darkgray': (169, 169, 169),
        'gray': (128, 128, 128),
        'bisque': (255, 228, 196),
        'saddlebrown': (139, 69, 19),
        'none': None
    }
    return color_map.get(color_name.lower(), (0, 0, 0))


def create_svg_overlay_base64(original_image_path: str, svg_content: str) -> str:
    """
    元画像にSVGをMatplotlibでレンダリングして重ねてbase64エンコードされたPNG画像を生成
    
    Args:
        original_image_path: 元画像のパス
        svg_content: オーバーレイするSVGコンテンツ
        
    Returns:
        base64エンコードされたPNG画像文字列
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.patches import Polygon, Circle
        import xml.etree.ElementTree as ET
        
        # 元画像を読み込み
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            raise ValueError(f"画像を読み込めません: {original_image_path}")
        
        height, width = original_img.shape[:2]
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Matplotlibで図を作成
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Y軸を反転（画像座標系に合わせる）
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 背景画像を表示
        ax.imshow(original_img_rgb, extent=[0, width, height, 0])
        
        # SVGをパースしてMatplotlibで描画
        try:
            root = ET.fromstring(svg_content)
            
            # 全ての要素を処理
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if tag == 'polygon':
                    points_str = elem.get('points', '')
                    if points_str:
                        # 座標をパース
                        coords = points_str.replace(',', ' ').split()
                        points = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                points.append([float(coords[i]), float(coords[i+1])])
                        
                        if len(points) >= 3:
                            fill = elem.get('fill', 'none')
                            stroke = elem.get('stroke', 'black')
                            stroke_width = float(elem.get('stroke-width', 1))
                            opacity = float(elem.get('opacity', 1.0))
                            
                            # 透明度を上げる（より薄く）
                            alpha = opacity * 0.8  # 元の透明度の30%
                            
                            # Matplotlibのポリゴンを追加
                            polygon = Polygon(points, 
                                            facecolor=fill if fill != 'none' else 'none',
                                            edgecolor=stroke,
                                            linewidth=stroke_width,
                                            alpha=alpha)
                            ax.add_patch(polygon)
                
                elif tag == 'circle':
                    # サラミ（円）は表示しない
                    pass
        
        except Exception as e:
            print(f"SVG解析エラー: {e}")
        
        # 図をメモリに保存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                   pad_inches=0, dpi=100, facecolor='white')
        plt.close(fig)  # メモリリークを防ぐ
        
        img_buffer.seek(0)
        
        # base64エンコード
        base64_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
        
    except Exception as e:
        print(f"Matplotlibオーバーレイ画像生成エラー: {e}")
        # エラー時は元画像をそのままbase64エンコードして返す
        try:
            with open(original_image_path, 'rb') as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{base64_str}"
        except:
            return ""


def create_overlay_png_base64(original_image_path: str, svg_content: str) -> str:
    """
    元画像にSVGをオーバーレイしてbase64エンコードされたPNG画像を生成
    
    Args:
        original_image_path: 元画像のパス
        svg_content: オーバーレイするSVGコンテンツ
        
    Returns:
        base64エンコードされたPNG画像文字列
    """
    try:
        # 元画像を読み込み
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            raise ValueError(f"画像を読み込めません: {original_image_path}")
        
        height, width = original_img.shape[:2]
        
        # OpenCVからPILに変換（BGR→RGB）
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 元画像をHTMLに埋め込むためのbase64エンコード
        _, img_buffer = cv2.imencode('.png', original_img)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        
        # SVGに元画像を背景として埋め込む
        svg_with_background = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- 背景画像 -->
  <image x="0" y="0" width="{width}" height="{height}" 
         xlink:href="data:image/png;base64,{img_base64}"/>
  
  <!-- オーバーレイ要素 -->
  {extract_svg_body(svg_content)}
</svg>'''
        
        # wkhtmltoimageやpuppeteerの代わりに、SVGをHTMLとしてレンダリング
        try:
            # SVGを直接PIL/Pillowで処理できる形に変換
            from PIL import ImageDraw
            import re
            
            pil_img = Image.fromarray(original_img_rgb).convert('RGBA')
            overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # 簡単なSVG要素の抽出と描画
            draw_simple_svg_elements(draw, svg_content)
            
            # オーバーレイ合成
            combined = Image.alpha_composite(pil_img, overlay)
            
        except Exception as e:
            print(f"SVG描画エラー、元画像のみ返却: {e}")
            combined = Image.fromarray(original_img_rgb)
        
        # PNGとして保存してbase64エンコード
        img_buffer = io.BytesIO()
        combined.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # base64エンコード
        base64_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
        
    except Exception as e:
        print(f"オーバーレイ画像生成エラー: {e}")
        # エラー時は元画像をそのままbase64エンコードして返す
        try:
            with open(original_image_path, 'rb') as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{base64_str}"
        except:
            return ""


def extract_svg_body(svg_content: str) -> str:
    """SVGコンテンツから<svg>タグの中身を抽出"""
    try:
        # <svg>タグの開始と終了を見つける
        start_match = re.search(r'<svg[^>]*>', svg_content)
        end_match = re.search(r'</svg>', svg_content)
        
        if start_match and end_match:
            start = start_match.end()
            end = end_match.start()
            return svg_content[start:end]
        else:
            return svg_content
    except:
        return svg_content


def draw_simple_svg_elements(draw, svg_content: str):
    """シンプルなSVG要素を描画"""
    import re
    
    # ポリゴンの描画（ピザピース）
    polygon_pattern = r'<polygon[^>]*points="([^"]*)"[^>]*(?:stroke="([^"]*)")?[^>]*(?:stroke-width="([^"]*)")?[^>]*(?:fill="([^"]*)")?[^>]*/?>'
    for match in re.finditer(polygon_pattern, svg_content):
        try:
            points_str = match.group(1)
            stroke = match.group(2) or 'black'
            stroke_width = int(float(match.group(3) or 2))
            fill = match.group(4) or 'none'
            
            # 座標をパース
            coords = points_str.replace(',', ' ').split()
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((float(coords[i]), float(coords[i+1])))
            
            if len(points) >= 3:
                # 塗りつぶし
                if fill and fill != 'none':
                    fill_color = color_name_to_rgb(fill)
                    if fill_color:
                        fill_rgba = fill_color + (100,)  # より薄い半透明
                        draw.polygon(points, fill=fill_rgba)
                
                # 輪郭
                stroke_color = color_name_to_rgb(stroke)
                if stroke_color and stroke_width > 0:
                    stroke_rgba = stroke_color + (255,)
                    draw.polygon(points, outline=stroke_rgba, width=stroke_width)
        except Exception as e:
            print(f"Polygon描画エラー: {e}")
            continue
    
    # 円の描画（サラミ）
    circle_pattern = r'<circle[^>]*cx="([^"]*)"[^>]*cy="([^"]*)"[^>]*r="([^"]*)"[^>]*(?:stroke="([^"]*)")?[^>]*(?:stroke-width="([^"]*)")?[^>]*(?:fill="([^"]*)")?[^>]*/?>'
    for match in re.finditer(circle_pattern, svg_content):
        try:
            cx, cy, r = float(match.group(1)), float(match.group(2)), float(match.group(3))
            stroke = match.group(4) or 'darkred'
            stroke_width = int(float(match.group(5) or 2))
            fill = match.group(6) or 'indianred'
            
            bbox = [cx - r, cy - r, cx + r, cy + r]
            
            # 塗りつぶし
            if fill and fill != 'none':
                fill_color = color_name_to_rgb(fill)
                if fill_color:
                    fill_rgba = fill_color + (180,)  # サラミは少し濃く
                    draw.ellipse(bbox, fill=fill_rgba)
            
            # 輪郭
            stroke_color = color_name_to_rgb(stroke)
            if stroke_color and stroke_width > 0:
                stroke_rgba = stroke_color + (255,)
                draw.ellipse(bbox, outline=stroke_rgba, width=stroke_width)
        except Exception as e:
            print(f"Circle描画エラー: {e}")
            continue
    
    # 線の描画
    line_pattern = r'<line[^>]*x1="([^"]*)"[^>]*y1="([^"]*)"[^>]*x2="([^"]*)"[^>]*y2="([^"]*)"[^>]*(?:stroke="([^"]*)")?[^>]*(?:stroke-width="([^"]*)")?[^>]*/?>'
    for match in re.finditer(line_pattern, svg_content):
        try:
            x1, y1, x2, y2 = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
            stroke = match.group(5) or 'black'
            stroke_width = int(float(match.group(6) or 2))
            
            stroke_color = color_name_to_rgb(stroke)
            if stroke_color and stroke_width > 0:
                stroke_rgba = stroke_color + (255,)
                draw.line([(x1, y1), (x2, y2)], fill=stroke_rgba, width=stroke_width)
        except Exception as e:
            print(f"Line描画エラー: {e}")
            continue


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
    6. 元画像にオーバーレイしたPNG画像を生成（base64エンコード）
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
        
        # 各ピースの色付きSVGとオーバーレイ画像を生成
        piece_svgs = []
        overlay_image = None
        
        if 'cut_edges' in result and 'pieces' in result:
            from service.pizza_split.salami_devide import PizzaDivider
            
            # サラミ半径の平均値を計算
            salami_radii = [r for _, r in result['salami_circles']]
            avg_salami_radius = np.mean(salami_radii) / result['pizza_radius'] if salami_radii else 0.1
            
            # dividerインスタンスを作成
            divider = PizzaDivider(
                R_pizza=1.0,
                R_salami=avg_salami_radius,
                m=len(result['salami_circles']),
                n=n_pieces,
                N_Monte=50000,
                seed=42,
                isDebug=False
            )
            
            # 既に計算された結果を設定
            divider.pieces = result['pieces']
            divider.cut_edges = result['cut_edges']
            divider.n = n_pieces
            
            # 正規化されたサラミ位置を設定
            normalized_salami = []
            for (cx, cy), r in result['salami_circles']:
                norm_x = (cx - result['pizza_center'][0]) / result['pizza_radius']
                norm_y = (cy - result['pizza_center'][1]) / result['pizza_radius']
                normalized_salami.append((norm_x, norm_y))
            divider.centers = np.array(normalized_salami)
            
            # モンテカルロ点を生成（必要なため）
            divider.generate_monte_carlo_points()
            divider.px = result.get('px', divider.px) if 'px' in result else divider.px
            divider.py = result.get('py', divider.py) if 'py' in result else divider.py
            
            # 色付きピースSVGを生成
            piece_svgs = divider.generate_colored_piece_svgs(svg_size=400)
            
            # オーバーレイ画像を生成（爆発前のSVGを使用）
            if svg_before:
                try:
                    overlay_image = create_svg_overlay_base64(str(upload_path), svg_before)
                except Exception as e:
                    print(f"オーバーレイ画像生成エラー: {e}")
                    overlay_image = None
        
        return PizzaDivisionResponse(
            success=True,
            svg_before_explosion=svg_before,
            svg_after_explosion=svg_after,
            svg_animated=svg_animated,
            piece_svgs=piece_svgs,
            overlay_image=overlay_image
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


@router.post("/score", response_model=PizzaScoreResponse)
async def calculate_pizza_score(
    file: UploadFile = File(..., description="ピザの画像ファイル (JPG, PNG)")
):
    """
    ピザ画像の公平性スコアを計算
    
    アップロードされた画像に対して以下の処理を実行:
    1. ピザ領域の検出（形態学的分離処理付き）
    2. サラミの検出
    3. 各領域ごとのピザ・サラミ面積計算
    4. 標準偏差に基づく公平性スコアの算出
    
    Returns:
        - 全体のピザ・サラミ面積
        - 各領域の詳細情報
        - 公平性スコア（0-100: 100が完全に公平）
    """
    
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    
    # 一意のファイル名生成
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    upload_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    output_dir = UPLOAD_DIR / "score_output" / file_id
    
    try:
        # ファイル保存
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        import cv2
        
        # サービスを初期化
        pizza_service = get_pizza_segmentation_service()
        salami_service = get_salami_segmentation_service()
        
        # 画像を読み込み
        original_image = cv2.imread(str(upload_path))
        
        # 1. ピザセグメンテーション
        pizza_mask = pizza_service.segment_pizza(str(upload_path), isDebug=False)
        
        # 2. サラミセグメンテーション
        salami_result = salami_service.segment_salami(str(upload_path))
        
        if isinstance(salami_result, tuple):
            salami_mask, _ = salami_result
        else:
            salami_mask = salami_result
        
        # 3. 各領域の分析
        num_labels, labels = cv2.connectedComponents(pizza_mask.astype(np.uint8))
        
        region_stats = []
        pizza_pixels_list = []
        salami_pixels_list = []
        
        for label in range(1, num_labels):
            region_mask = (labels == label).astype(np.uint8) * 255
            pizza_pixels = int(np.sum(region_mask > 0))
            
            salami_in_region = cv2.bitwise_and(region_mask, salami_mask)
            salami_pixels = int(np.sum(salami_in_region > 0))
            
            coverage = salami_pixels / pizza_pixels if pizza_pixels > 0 else 0
            
            region_stats.append({
                'region_id': label,
                'pizza_pixels': pizza_pixels,
                'salami_pixels': salami_pixels,
                'coverage_ratio': coverage,
                'coverage_percent': coverage * 100
            })
            
            pizza_pixels_list.append(pizza_pixels)
            salami_pixels_list.append(salami_pixels)
        
        # 4. 標準偏差とfairness scoreの計算
        std_pizza = float(np.std(pizza_pixels_list)) if len(pizza_pixels_list) > 1 else 0.0
        std_salami = float(np.std(salami_pixels_list)) if len(salami_pixels_list) > 1 else 0.0
        
        # Fairness score計算
        if len(pizza_pixels_list) > 1:
            mean_pizza = np.mean(pizza_pixels_list)
            mean_salami = np.mean(salami_pixels_list)
            
            cv_pizza = std_pizza / mean_pizza if mean_pizza > 0 else 0
            cv_salami = std_salami / mean_salami if mean_salami > 0 else 0
            
            k = 3.0
            pizza_fairness = 100 * np.exp(-k * cv_pizza)
            salami_fairness = 100 * np.exp(-k * cv_salami)
            
            fairness_score = 0.3 * pizza_fairness + 0.7 * salami_fairness
        else:
            fairness_score = 100.0
        
        # 5. 全体の統計
        total_pizza_area = int(np.sum(pizza_mask > 0))
        total_salami_area = int(np.sum(salami_mask > 0))
        salami_coverage_percent = (total_salami_area / total_pizza_area * 100) if total_pizza_area > 0 else 0.0
        
        return PizzaScoreResponse(
            success=True,
            fairness_score=fairness_score
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PizzaScoreResponse(
            success=False,
            fairness_score=0.0,
            error_message=str(e)
        )
    
    finally:
        # クリーンアップ
        try:
            if upload_path.exists():
                upload_path.unlink()
            if output_dir.exists():
                shutil.rmtree(output_dir)
        except Exception:
            pass