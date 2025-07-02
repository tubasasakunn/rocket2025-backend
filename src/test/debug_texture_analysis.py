import cv2
import numpy as np
from pathlib import Path
import sys
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
from matplotlib import cm

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


def compute_entropy(image, radius=7):
    """
    エントロピー特徴量を計算
    
    Args:
        image: 入力画像（BGR）
        radius: 計算用の円形カーネルの半径
        
    Returns:
        エントロピーマップ
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return entropy(gray, disk(radius))


def local_std(image, window_size=15):
    """
    局所標準偏差を計算
    
    Args:
        image: 入力画像（BGR）
        window_size: ウィンドウサイズ
        
    Returns:
        局所標準偏差マップ
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 局所平均
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    mean = cv2.filter2D(gray, -1, kernel)
    
    # 局所分散
    sqr_mean = cv2.filter2D(gray**2, -1, kernel)
    variance = sqr_mean - mean**2
    
    return np.sqrt(np.maximum(variance, 0))


def create_heatmap(feature_map, title="Feature Map"):
    """
    特徴量マップをヒートマップとして可視化
    
    Args:
        feature_map: 特徴量マップ
        title: タイトル
        
    Returns:
        カラーマップ画像（BGR）
    """
    # 正規化（0-255）
    normalized = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # カラーマップ適用（jet colormap）
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    
    return colored


def analyze_texture_features(image_path, pizza_mask, salami_mask, output_dir):
    """
    テクスチャ特徴量を分析し、可視化結果を保存
    
    Args:
        image_path: 入力画像のパス
        pizza_mask: ピザ領域のマスク
        salami_mask: サラミ領域のマスク（色ベース）
        output_dir: 出力ディレクトリ
    """
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像が読み込めません: {image_path}")
    
    base_name = Path(image_path).stem
    
    # ピザ領域のみを抽出
    masked_image = cv2.bitwise_and(image, image, mask=pizza_mask)
    
    print(f"\n{base_name}のテクスチャ特徴量を計算中...")
    
    # エントロピー計算
    print("  - エントロピー計算中...")
    entropy_map = compute_entropy(masked_image, radius=7)
    
    # 局所標準偏差計算
    print("  - 局所標準偏差計算中...")
    std_map = local_std(masked_image, window_size=15)
    
    # ピザ領域外をマスク
    entropy_map_masked = np.where(pizza_mask > 0, entropy_map, 0)
    std_map_masked = np.where(pizza_mask > 0, std_map, 0)
    
    # ヒートマップ作成
    entropy_heatmap = create_heatmap(entropy_map_masked, "Entropy")
    std_heatmap = create_heatmap(std_map_masked, "Local STD")
    
    # トマトソース領域とサラミ領域の特徴量統計を計算
    # トマトソース領域 = ピザ領域 - サラミ領域
    tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
    
    # サラミ領域の特徴量
    salami_entropy = entropy_map_masked[salami_mask > 0]
    salami_std = std_map_masked[salami_mask > 0]
    
    # トマトソース領域の特徴量
    tomato_entropy = entropy_map_masked[tomato_mask > 0]
    tomato_std = std_map_masked[tomato_mask > 0]
    
    # 統計情報を出力
    print(f"\n  統計情報:")
    print(f"  サラミ領域:")
    if len(salami_entropy) > 0:
        print(f"    - エントロピー: 平均={salami_entropy.mean():.2f}, 標準偏差={salami_entropy.std():.2f}")
        print(f"    - 局所標準偏差: 平均={salami_std.mean():.2f}, 標準偏差={salami_std.std():.2f}")
    else:
        print(f"    - サラミが検出されませんでした")
    
    print(f"  トマトソース領域:")
    if len(tomato_entropy) > 0:
        print(f"    - エントロピー: 平均={tomato_entropy.mean():.2f}, 標準偏差={tomato_entropy.std():.2f}")
        print(f"    - 局所標準偏差: 平均={tomato_std.mean():.2f}, 標準偏差={tomato_std.std():.2f}")
    else:
        print(f"    - トマトソース領域が検出されませんでした")
    
    # 結果を保存
    # 元画像
    cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), image)
    
    # マスク画像
    cv2.imwrite(str(output_dir / f"{base_name}_pizza_mask.png"), pizza_mask)
    cv2.imwrite(str(output_dir / f"{base_name}_salami_mask.png"), salami_mask)
    cv2.imwrite(str(output_dir / f"{base_name}_tomato_mask.png"), tomato_mask)
    
    # 特徴量ヒートマップ
    cv2.imwrite(str(output_dir / f"{base_name}_entropy_heatmap.jpg"), entropy_heatmap)
    cv2.imwrite(str(output_dir / f"{base_name}_std_heatmap.jpg"), std_heatmap)
    
    # オーバーレイ画像（元画像とヒートマップの合成）
    entropy_overlay = cv2.addWeighted(image, 0.5, entropy_heatmap, 0.5, 0)
    std_overlay = cv2.addWeighted(image, 0.5, std_heatmap, 0.5, 0)
    cv2.imwrite(str(output_dir / f"{base_name}_entropy_overlay.jpg"), entropy_overlay)
    cv2.imwrite(str(output_dir / f"{base_name}_std_overlay.jpg"), std_overlay)
    
    # 特徴量の分布をヒストグラムとして保存
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # エントロピーヒストグラム
    axes[0, 0].hist(salami_entropy, bins=50, alpha=0.7, label='Salami', color='red')
    axes[0, 0].hist(tomato_entropy, bins=50, alpha=0.7, label='Tomato', color='orange')
    axes[0, 0].set_title('Entropy Distribution')
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 局所標準偏差ヒストグラム
    axes[0, 1].hist(salami_std, bins=50, alpha=0.7, label='Salami', color='red')
    axes[0, 1].hist(tomato_std, bins=50, alpha=0.7, label='Tomato', color='orange')
    axes[0, 1].set_title('Local STD Distribution')
    axes[0, 1].set_xlabel('Local STD')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # エントロピーマップの画像表示
    axes[1, 0].imshow(entropy_map_masked, cmap='jet')
    axes[1, 0].set_title('Entropy Map')
    axes[1, 0].axis('off')
    
    # 局所標準偏差マップの画像表示
    axes[1, 1].imshow(std_map_masked, cmap='jet')
    axes[1, 1].set_title('Local STD Map')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{base_name}_texture_analysis.png"))
    plt.close()
    
    print(f"  結果を保存しました: {output_dir}")
    
    return {
        'entropy_map': entropy_map_masked,
        'std_map': std_map_masked,
        'salami_entropy_mean': salami_entropy.mean() if len(salami_entropy) > 0 else 0,
        'salami_std_mean': salami_std.mean() if len(salami_std) > 0 else 0,
        'tomato_entropy_mean': tomato_entropy.mean() if len(tomato_entropy) > 0 else 0,
        'tomato_std_mean': tomato_std.mean() if len(tomato_std) > 0 else 0
    }


def main():
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\nテクスチャ分析を開始します...\n")
    
    # 各画像を処理
    for image_path in image_files:
        print(f"\n{'='*60}")
        print(f"{image_path.name}を処理中...")
        
        try:
            # 1. 前処理
            temp_preprocessed_path = output_dir / f"temp_preprocessed_{image_path.stem}.jpg"
            preprocessed_image, info = preprocess_service.preprocess_pizza_image(
                str(image_path), str(temp_preprocessed_path)
            )
            
            if info['is_transformed']:
                print(f"  楕円変換を適用しました")
                processed_image_path = str(temp_preprocessed_path)
            else:
                print(f"  変換は不要でした")
                processed_image_path = str(image_path)
            
            # 2. ピザマスク取得
            pizza_mask = pizza_service.segment_pizza(processed_image_path)
            
            # 3. サラミマスク取得（色ベース）
            salami_mask = salami_service.segment_salami(processed_image_path, pizza_mask)
            
            # 4. テクスチャ特徴量分析
            results = analyze_texture_features(
                processed_image_path, pizza_mask, salami_mask, output_dir
            )
            
            # 一時ファイル削除
            if temp_preprocessed_path.exists():
                temp_preprocessed_path.unlink()
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"処理完了。{output_dir}で結果を確認してください。")


if __name__ == "__main__":
    main()