import cv2
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


def compute_texture_features(image, mask):
    """
    様々なテクスチャ特徴量を計算
    
    Args:
        image: 入力画像（BGR）
        mask: 対象領域のマスク
        
    Returns:
        特徴量の辞書
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # マスク領域のピクセルを抽出
    masked_pixels = gray[mask > 0]
    
    features = {}
    
    # 1. 基本統計量
    if len(masked_pixels) > 0:
        features['mean'] = np.mean(masked_pixels)
        features['std'] = np.std(masked_pixels)
        features['min'] = np.min(masked_pixels)
        features['max'] = np.max(masked_pixels)
        features['range'] = features['max'] - features['min']
    else:
        features['mean'] = 0
        features['std'] = 0
        features['min'] = 0
        features['max'] = 0
        features['range'] = 0
    
    # 2. エントロピー（複数の半径で計算）
    for radius in [3, 5, 7, 9]:
        entropy_map = entropy(gray, disk(radius))
        masked_entropy = entropy_map[mask > 0]
        if len(masked_entropy) > 0:
            features[f'entropy_r{radius}_mean'] = np.mean(masked_entropy)
            features[f'entropy_r{radius}_std'] = np.std(masked_entropy)
        else:
            features[f'entropy_r{radius}_mean'] = 0
            features[f'entropy_r{radius}_std'] = 0
    
    # 3. 局所標準偏差（複数のウィンドウサイズ）
    for window_size in [5, 9, 15, 21]:
        std_map = local_std(image, window_size)
        masked_std = std_map[mask > 0]
        if len(masked_std) > 0:
            features[f'local_std_w{window_size}_mean'] = np.mean(masked_std)
            features[f'local_std_w{window_size}_std'] = np.std(masked_std)
        else:
            features[f'local_std_w{window_size}_mean'] = 0
            features[f'local_std_w{window_size}_std'] = 0
    
    # 4. Laplacian（エッジ検出）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    masked_laplacian = laplacian_abs[mask > 0]
    if len(masked_laplacian) > 0:
        features['laplacian_mean'] = np.mean(masked_laplacian)
        features['laplacian_std'] = np.std(masked_laplacian)
    else:
        features['laplacian_mean'] = 0
        features['laplacian_std'] = 0
    
    # 5. Sobel勾配
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    masked_sobel = sobel_magnitude[mask > 0]
    if len(masked_sobel) > 0:
        features['sobel_mean'] = np.mean(masked_sobel)
        features['sobel_std'] = np.std(masked_sobel)
    else:
        features['sobel_mean'] = 0
        features['sobel_std'] = 0
    
    return features


def local_std(image, window_size=15):
    """局所標準偏差を計算"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 局所平均
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    mean = cv2.filter2D(gray, -1, kernel)
    
    # 局所分散
    sqr_mean = cv2.filter2D(gray**2, -1, kernel)
    variance = sqr_mean - mean**2
    
    return np.sqrt(np.maximum(variance, 0))


def create_feature_maps(image, pizza_mask):
    """
    様々な特徴量マップを生成
    
    Args:
        image: 入力画像（BGR）
        pizza_mask: ピザ領域のマスク
        
    Returns:
        特徴量マップの辞書
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    maps = {}
    
    # エントロピーマップ（複数の半径）
    for radius in [3, 5, 7, 9]:
        entropy_map = entropy(gray, disk(radius))
        maps[f'entropy_r{radius}'] = np.where(pizza_mask > 0, entropy_map, 0)
    
    # 局所標準偏差マップ（複数のウィンドウサイズ）
    for window_size in [5, 9, 15, 21]:
        std_map = local_std(image, window_size)
        maps[f'local_std_w{window_size}'] = np.where(pizza_mask > 0, std_map, 0)
    
    # Laplacianマップ
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    maps['laplacian'] = np.where(pizza_mask > 0, np.abs(laplacian), 0)
    
    # Sobelマップ
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    maps['sobel'] = np.where(pizza_mask > 0, sobel_magnitude, 0)
    
    return maps


def plot_feature_distributions(salami_features, tomato_features, output_path):
    """
    特徴量の分布を比較するプロット
    
    Args:
        salami_features: サラミ領域の特徴量
        tomato_features: トマトソース領域の特徴量
        output_path: 出力パス
    """
    # 特徴量名のリスト
    feature_names = list(salami_features.keys())
    
    # プロット作成
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names[:20]):  # 最大20個の特徴量を表示
        ax = axes[i]
        
        # バープロット
        x = ['Salami', 'Tomato']
        y = [salami_features[feature_name], tomato_features[feature_name]]
        
        bars = ax.bar(x, y, color=['red', 'orange'])
        ax.set_title(feature_name)
        ax.set_ylabel('Value')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom')
        
        # 差分をパーセンテージで表示
        if tomato_features[feature_name] != 0:
            diff_percent = ((salami_features[feature_name] - tomato_features[feature_name]) 
                          / tomato_features[feature_name] * 100)
            ax.text(0.5, max(y) * 1.1, f'Diff: {diff_percent:+.1f}%', 
                   transform=ax.transData, ha='center')
    
    # 使用していないサブプロットを非表示
    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_all_feature_maps(maps, base_name, output_dir):
    """
    全ての特徴量マップを保存
    
    Args:
        maps: 特徴量マップの辞書
        base_name: ファイル名のベース
        output_dir: 出力ディレクトリ
    """
    # 特徴量マップを3x4のグリッドで表示
    n_maps = len(maps)
    n_cols = 4
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, (name, feature_map) in enumerate(maps.items()):
        if i < len(axes):
            ax = axes[i]
            im = ax.imshow(feature_map, cmap='jet')
            ax.set_title(name)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 使用していないサブプロットを非表示
    for i in range(len(maps), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_all_feature_maps.png", dpi=150)
    plt.close()
    
    # 個別の特徴量マップも保存
    for name, feature_map in maps.items():
        # ヒートマップとして保存
        normalized = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{base_name}_{name}.jpg"), colored)


def main():
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\nテクスチャ調査を開始します...\n")
    
    # 全画像の特徴量を収集
    all_salami_features = []
    all_tomato_features = []
    
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
            
            # 4. トマトソースマスク作成
            tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
            
            # 画像読み込み
            image = cv2.imread(processed_image_path)
            
            # 5. 特徴量計算
            print("  特徴量を計算中...")
            salami_features = compute_texture_features(image, salami_mask)
            tomato_features = compute_texture_features(image, tomato_mask)
            
            all_salami_features.append(salami_features)
            all_tomato_features.append(tomato_features)
            
            # 6. 特徴量マップ生成
            print("  特徴量マップを生成中...")
            feature_maps = create_feature_maps(image, pizza_mask)
            
            # 7. 結果保存
            base_name = image_path.stem
            
            # 特徴量分布の比較プロット
            plot_feature_distributions(
                salami_features, tomato_features, 
                output_dir / f"{base_name}_feature_comparison.png"
            )
            
            # 全特徴量マップの保存
            save_all_feature_maps(feature_maps, base_name, output_dir)
            
            # マスク画像の保存
            cv2.imwrite(str(output_dir / f"{base_name}_pizza_mask.png"), pizza_mask)
            cv2.imwrite(str(output_dir / f"{base_name}_salami_mask.png"), salami_mask)
            cv2.imwrite(str(output_dir / f"{base_name}_tomato_mask.png"), tomato_mask)
            
            # 特徴量の詳細を出力
            print("\n  特徴量の比較:")
            print("  " + "-"*50)
            print(f"  {'Feature':<30} {'Salami':>10} {'Tomato':>10} {'Diff %':>10}")
            print("  " + "-"*50)
            
            for feature_name in sorted(salami_features.keys()):
                salami_val = salami_features[feature_name]
                tomato_val = tomato_features[feature_name]
                if tomato_val != 0:
                    diff_percent = ((salami_val - tomato_val) / tomato_val * 100)
                else:
                    diff_percent = 0
                print(f"  {feature_name:<30} {salami_val:>10.2f} {tomato_val:>10.2f} {diff_percent:>10.1f}")
            
            # 一時ファイル削除
            if temp_preprocessed_path.exists():
                temp_preprocessed_path.unlink()
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全画像の平均特徴量を計算
    if all_salami_features and all_tomato_features:
        print(f"\n{'='*60}")
        print("全画像の平均特徴量:")
        print("=" * 60)
        
        # 平均を計算
        avg_salami_features = {}
        avg_tomato_features = {}
        
        for feature_name in all_salami_features[0].keys():
            avg_salami_features[feature_name] = np.mean([f[feature_name] for f in all_salami_features])
            avg_tomato_features[feature_name] = np.mean([f[feature_name] for f in all_tomato_features])
        
        # 平均特徴量の比較プロット
        plot_feature_distributions(
            avg_salami_features, avg_tomato_features, 
            output_dir / "average_feature_comparison.png"
        )
        
        print(f"  {'Feature':<30} {'Salami':>10} {'Tomato':>10} {'Diff %':>10}")
        print("  " + "-"*50)
        
        for feature_name in sorted(avg_salami_features.keys()):
            salami_val = avg_salami_features[feature_name]
            tomato_val = avg_tomato_features[feature_name]
            if tomato_val != 0:
                diff_percent = ((salami_val - tomato_val) / tomato_val * 100)
            else:
                diff_percent = 0
            print(f"  {feature_name:<30} {salami_val:>10.2f} {tomato_val:>10.2f} {diff_percent:>10.1f}")
    
    print(f"\n{'='*60}")
    print(f"処理完了。{output_dir}で結果を確認してください。")


if __name__ == "__main__":
    main()