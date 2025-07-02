# Pizza Split Service - 詳細技術仕様書

## 概要

Pizza Split Serviceは、ピザ画像から以下の処理を行うコンピュータビジョンサービス群です：
1. ピザの形状を検出（楕円・円形）し、円形に正規化
2. ピザ領域とサラミ（トッピング）領域のセグメンテーション
3. 移動ナイフ法によるピザの公平な分割計算
4. 分割結果のSVG生成と元画像への逆変換

## サービス一覧

### 1. pizza_segmentation_service.py
**目的**: YOLOv8を使用してピザ領域をセグメンテーション

**主要クラス**: `PizzaSegmentationService`

**メソッド**:
- `segment_pizza(image_path: str, isDebug: bool = False) -> np.ndarray`
  - 入力: 画像パス
  - 出力: バイナリマスク（255=ピザ, 0=背景）
  - YOLOv8nモデルを使用（クラスID: 53）

**技術詳細**:
- PyTorch 2.6+のweights_only問題を回避するパッチ実装
- モデルファイル: `yolov8n-seg.pt`
- 複数のピザが検出された場合、すべてをマスクに統合

### 2. pizza_circle_detection_service.py
**目的**: セグメンテーションマスクからピザの円形近似

**主要クラス**: `PizzaCircleDetectionService`

**メソッド**:
- `detect_circle_from_image(image_path: str, isDebug: bool = False) -> Optional[Tuple[Tuple[int, int], int]]`
  - 入力: 画像パス
  - 出力: ((中心x, 中心y), 半径) または None
  
- `draw_circle_on_image(image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray`
  - 検出した円を画像上に描画

**技術詳細**:
- 最大輪郭を選択して最小外接円を計算
- 円形度（circularity）を参考値として計算
- 前処理サービスと連携して楕円→円変換後の画像を処理

### 3. salami_segmentation_service.py
**目的**: 色ベースでサラミ領域を検出

**主要クラス**: `SalamiSegmentationService`

**定数（マジックナンバー）**:
```python
# 前処理パラメータ
BILATERAL_D = 9                    # バイラテラルフィルタ直径
BILATERAL_SIGMA_COLOR = 50         # カラーシグマ値
BILATERAL_SIGMA_SPACE = 50         # スペースシグマ値
CLAHE_CLIP_LIMIT = 2.0            # CLAHE制限値
CLAHE_TILE_GRID_SIZE = (8, 8)    # CLAHEタイルサイズ

# HSV色範囲（サラミ検出用）
H_MIN = 165   # 色相最小値（330°÷2）
H_MAX = 10    # 色相最大値（20°÷2、0をまたぐ）
S_MIN = 30    # 彩度最小値
S_MAX = 255   # 彩度最大値
V_MIN = 50    # 明度最小値
V_MAX = 140   # 明度最大値

# フィルタリング
MIN_CONTOUR_AREA = 1700  # 最小面積（ピクセル）
```

**処理パイプライン**:
1. 画像前処理（ガウシアンブラー → バイラテラルフィルタ → CLAHE）
2. HSV色空間変換
3. 色範囲でマスク作成（0度をまたぐ赤色に対応）
4. 穴埋め処理（`_fill_holes_only`）
5. モルフォロジー処理（クロージング → オープニング）
6. 面積フィルタリング

### 4. salami_circle_detection_service.py
**目的**: サラミ領域から個別の円形サラミを検出

**主要クラス**: `SalamiCircleDetectionService`

**定数**:
```python
MIN_AREA_THRESHOLD = 200        # 最小面積閾値
MIN_RADIUS = 12                 # 最小半径
MAX_RADIUS_RATIO = 0.15         # 画像サイズに対する最大半径比
CIRCULARITY_THRESHOLD = 0.3     # 円形度閾値
DISTANCE_THRESHOLD_RATIO = 0.5  # 距離変換ピーク検出閾値比
```

**アルゴリズム**:
1. ピザとサラミマスクをAND演算で結合
2. 距離変換（`cv2.distanceTransform`）を実行
3. 縮小処理による分割判定（`_detect_circles_by_erosion`）
   - 80%閾値で領域を分割
   - 分割された場合は各領域を個別処理
   - 分割されない場合は全体を1つの円として処理
4. 領域成長法で元のマスクまで拡張（`_expand_region_to_original`）
5. 各領域から円を生成（中心は距離変換の最大値位置）

### 5. preprocess.py
**目的**: 楕円形のピザを円形に変換し、512x512に正規化

**主要クラス**: `PreprocessService`

**定数**:
```python
ellipse_threshold = 0.1  # 楕円判定閾値（長軸と短軸の差が10%以上）
```

**変換プロセス**:
1. セグメンテーションマスクから楕円フィッティング（`cv2.fitEllipse`）
2. 楕円判定（長軸/短軸の比率）
3. アフィン変換行列の構築：
   - 中心を原点に移動
   - 長軸がx軸に合うよう回転（-angle）
   - x軸をy軸に合わせてスケーリング（scale_x = minor/major）
   - 元の向きに回転（+angle）
   - 元の位置に移動
4. 512x512に正規化（ピザ半径でクロップ＆リサイズ）

**出力情報**:
```python
{
    'original_shape': tuple,        # 元画像のshape
    'is_transformed': bool,         # 変換を適用したか
    'ellipse_params': {            # 検出された楕円パラメータ
        'center': (x, y),
        'major_axis': int,
        'minor_axis': int,
        'angle': float
    },
    'transformation_applied': {     # 適用された変換
        'scale_x': float,
        'scale_y': float,
        'angle': float
    },
    'normalization_params': {      # 正規化パラメータ
        'crop_region': tuple,
        'crop_size': tuple,
        'target_size': (512, 512),
        'scale_factor': float,
        'center_in_transformed': tuple,
        'radius_in_transformed': int
    }
}
```

### 6. postprocess.py
**目的**: 前処理の逆変換を行い、結果を元の画像座標系に戻す

**主要クラス**: `PostprocessService`

**メソッド**:
- `inverse_transform_point(point, preprocess_info)`: 点の逆変換
- `inverse_transform_circle(center, radius, preprocess_info)`: 円の逆変換
- `inverse_transform_cut_edges(cut_edges, pizza_center, pizza_radius, preprocess_info)`: カット線の逆変換
- `create_svg_overlay_on_original(...)`: 元画像用SVG生成
- `create_overlay_image_on_original(...)`: 元画像にオーバーレイしたPNG生成

**逆変換アルゴリズム**:
1. 512x512座標 → クロップ領域座標
2. クロップ領域座標 → 変換後画像座標
3. 変換後画像座標 → 元画像座標（楕円変換の逆適用）

### 7. salami_devide.py
**目的**: 移動ナイフ法によるピザの公平な分割

**主要クラス**: `PizzaDivider`

**パラメータ**:
```python
R_pizza = 1.0      # ピザ半径（正規化）
R_salami = 0.10    # サラミ半径（正規化）
m = 13             # サラミ枚数
n = 7              # 分割ピース数
N_Monte = 100_000  # モンテカルロ点数
```

**アルゴリズム（移動ナイフ法）**:
1. モンテカルロ点をピザ内に一様散布
2. 各点がサラミ上にあるかチェック
3. 目標値計算：
   - A_goal = πR²/n（1ピースあたりの面積）
   - B_goal = 総サラミ面積/n（1ピースあたりのサラミ面積）
4. n-1回のカット実行：
   - 角度θを0〜2πで探索
   - 各θで面積がA_goalとなる距離dを計算
   - サラミ量の誤差g(θ) = B - B_goalを評価
   - g(θ) = 0となる角度を二分探索で見つける
5. カット線の端点を計算して保存

**SVG生成機能**:
- `generate_piece_svgs()`: 全体表示（各ピースを色分け）
- `generate_piece_svgs_isolated()`: 各ピースを個別に表示
- 凸包計算によるピース境界の近似

### 8. process.py
**目的**: 全体のパイプラインを統合したメインプロセッサ

**主要クラス**: `PizzaProcessor`

**処理フロー**:
1. 画像前処理（楕円→円形変換、512x512正規化）
2. ピザ円検出
3. サラミ円検出
4. 座標正規化（ピザ中心を原点、半径を1に）
5. 移動ナイフ法で分割計算
6. SVGオーバーレイ生成（前処理済み画像用）
7. 元画像用SVGオーバーレイ生成（逆変換適用）
8. 結果画像生成（Matplotlib使用）
9. 各ピースの個別SVG生成

**メソッド**:
- `process_image(image_path, n_pieces=4, debug=False, return_svg_only=False, quiet=False)`
  - 完全な処理パイプラインを実行
  - 戻り値：処理結果の辞書

## 使用方法

### 基本的な使用例
```python
from process import PizzaProcessor

# プロセッサーを初期化
processor = PizzaProcessor(output_dir="result/process")

# 画像を4分割で処理
result = processor.process_image(
    "resource/pizza1.jpg",
    n_pieces=4,
    debug=True
)

# 結果の確認
print(f"前処理画像: {result['preprocessed_image']}")
print(f"元画像用SVG: {result['svg_original']}")
print(f"各ピースSVG: {result['piece_svgs']}")
```

### 個別サービスの使用例
```python
# ピザセグメンテーション
from pizza_segmentation_service import PizzaSegmentationService
service = PizzaSegmentationService()
mask = service.segment_pizza("pizza.jpg", isDebug=True)

# サラミ円検出
from salami_circle_detection_service import SalamiCircleDetectionService
service = SalamiCircleDetectionService()
circles = service.detect_salami_circles("pizza.jpg", isDebug=True)
```

## 依存関係
- OpenCV (cv2)
- NumPy
- Matplotlib
- svgwrite
- ultralytics (YOLO)
- scipy (凸包計算用)
- PyTorch (YOLOモデル用)

## ディレクトリ構造
```
src/service/pizza_split/
├── pizza_segmentation_service.py      # ピザ領域検出
├── pizza_circle_detection_service.py  # ピザ円近似
├── salami_segmentation_service.py     # サラミ色検出
├── salami_circle_detection_service.py # サラミ円検出
├── preprocess.py                      # 前処理（楕円→円）
├── postprocess.py                     # 後処理（逆変換）
├── salami_devide.py                   # 移動ナイフ法分割
├── process.py                         # メインパイプライン
└── yolov8n-seg.pt                    # YOLOモデル
```

## 注意事項
- 入力画像はJPEG/PNG形式をサポート
- YOLOモデルファイル（yolov8n-seg.pt）が必要
- 楕円形のピザは自動的に円形に変換される
- サラミ検出は赤色系の色相（HSV）に基づく
- 移動ナイフ法はモンテカルロ法による近似計算