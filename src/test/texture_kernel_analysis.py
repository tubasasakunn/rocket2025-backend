import cv2
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


def compute_entropy_kernel(gray_patch):
    """
    9x9パッチのエントロピーを計算
    
    Args:
        gray_patch: グレースケールパッチ（9x9）
        
    Returns:
        エントロピー値
    """
    # ヒストグラムを計算
    hist, _ = np.histogram(gray_patch.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    
    # 確率に変換
    hist = hist / hist.sum()
    
    # エントロピー計算
    hist = hist[hist > 0]  # ゼロ要素を除外
    entropy_value = -np.sum(hist * np.log2(hist))
    
    return entropy_value


def compute_std_kernel(gray_patch):
    """
    9x9パッチの標準偏差を計算
    
    Args:
        gray_patch: グレースケールパッチ（9x9）
        
    Returns:
        標準偏差
    """
    return np.std(gray_patch)


def compute_texture_maps_kernel(image, kernel_size=9):
    """
    9x9カーネルでスライディングウィンドウを使用してテクスチャマップを計算
    
    Args:
        image: 入力画像（BGR）
        kernel_size: カーネルサイズ（デフォルト9）
        
    Returns:
        (エントロピーマップ, 標準偏差マップ)
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # パディング
    pad_size = kernel_size // 2
    padded = cv2.copyMakeBorder(gray, pad_size, pad_size, pad_size, pad_size, 
                                cv2.BORDER_REFLECT)
    
    # 出力マップの初期化
    h, w = gray.shape
    entropy_map = np.zeros((h, w), dtype=np.float32)
    std_map = np.zeros((h, w), dtype=np.float32)
    
    # スライディングウィンドウで計算
    for i in range(h):
        for j in range(w):
            # 9x9パッチを抽出
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            
            # エントロピーと標準偏差を計算
            entropy_map[i, j] = compute_entropy_kernel(patch)
            std_map[i, j] = compute_std_kernel(patch)
            
        # 進捗表示
        if i % 50 == 0:
            print(f"    処理中: {i}/{h} 行完了", end='\r')
    
    print(f"    処理完了: {h}/{h} 行              ")
    
    return entropy_map, std_map


def create_feature_visualization(image, entropy_map, std_map, pizza_mask, salami_mask, 
                               base_name, output_dir):
    """
    特徴量マップの詳細な可視化
    
    Args:
        image: 元画像（BGR）
        entropy_map: エントロピーマップ
        std_map: 標準偏差マップ
        pizza_mask: ピザマスク
        salami_mask: サラミマスク
        base_name: ファイル名ベース
        output_dir: 出力ディレクトリ
    """
    # グレースケール画像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # トマトソースマスク
    tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
    
    # マスク適用
    entropy_pizza = np.where(pizza_mask > 0, entropy_map, 0)
    std_pizza = np.where(pizza_mask > 0, std_map, 0)
    
    # 統計情報計算
    salami_entropy = entropy_map[salami_mask > 0]
    tomato_entropy = entropy_map[tomato_mask > 0]
    salami_std = std_map[salami_mask > 0]
    tomato_std = std_map[tomato_mask > 0]
    
    # 大きな図を作成（3x4グリッド）
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 元画像（グレースケール）
    plt.subplot(3, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original (Grayscale)')
    plt.axis('off')
    
    # 2. ピザマスク
    plt.subplot(3, 4, 2)
    plt.imshow(pizza_mask, cmap='gray')
    plt.title('Pizza Mask')
    plt.axis('off')
    
    # 3. サラミマスク
    plt.subplot(3, 4, 3)
    plt.imshow(salami_mask, cmap='gray')
    plt.title('Salami Mask')
    plt.axis('off')
    
    # 4. トマトソースマスク
    plt.subplot(3, 4, 4)
    plt.imshow(tomato_mask, cmap='gray')
    plt.title('Tomato Sauce Mask')
    plt.axis('off')
    
    # 5. エントロピーマップ（フル）
    plt.subplot(3, 4, 5)
    im1 = plt.imshow(entropy_map, cmap='jet')
    plt.title('Entropy Map (Full)')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 6. エントロピーマップ（ピザ領域のみ）
    plt.subplot(3, 4, 6)
    im2 = plt.imshow(entropy_pizza, cmap='jet')
    plt.title('Entropy Map (Pizza Only)')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 7. 標準偏差マップ（フル）
    plt.subplot(3, 4, 7)
    im3 = plt.imshow(std_map, cmap='jet')
    plt.title('Local STD Map (Full)')
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 8. 標準偏差マップ（ピザ領域のみ）
    plt.subplot(3, 4, 8)
    im4 = plt.imshow(std_pizza, cmap='jet')
    plt.title('Local STD Map (Pizza Only)')
    plt.colorbar(im4, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 9. エントロピーヒストグラム
    plt.subplot(3, 4, 9)
    plt.hist(salami_entropy, bins=50, alpha=0.7, label='Salami', color='red', density=True)
    plt.hist(tomato_entropy, bins=50, alpha=0.7, label='Tomato', color='orange', density=True)
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Entropy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. 標準偏差ヒストグラム
    plt.subplot(3, 4, 10)
    plt.hist(salami_std, bins=50, alpha=0.7, label='Salami', color='red', density=True)
    plt.hist(tomato_std, bins=50, alpha=0.7, label='Tomato', color='orange', density=True)
    plt.xlabel('Local STD')
    plt.ylabel('Density')
    plt.title('Local STD Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. 統計情報テーブル
    plt.subplot(3, 4, 11)
    plt.axis('off')
    stats_text = f"""Statistics (9x9 kernel):
    
Entropy:
  Salami: μ={np.mean(salami_entropy):.3f}, σ={np.std(salami_entropy):.3f}
  Tomato: μ={np.mean(tomato_entropy):.3f}, σ={np.std(tomato_entropy):.3f}
  Difference: {(np.mean(salami_entropy) - np.mean(tomato_entropy)):.3f}
  
Local STD:
  Salami: μ={np.mean(salami_std):.2f}, σ={np.std(salami_std):.2f}
  Tomato: μ={np.mean(tomato_std):.2f}, σ={np.std(tomato_std):.2f}
  Difference: {(np.mean(salami_std) - np.mean(tomato_std)):.2f}"""
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             transform=plt.gca().transAxes, verticalalignment='center')
    
    # 12. 2D散布図（エントロピー vs 標準偏差）
    plt.subplot(3, 4, 12)
    # サンプリング（表示用）
    n_samples = min(5000, len(salami_entropy), len(tomato_entropy))
    if len(salami_entropy) > 0:
        salami_idx = np.random.choice(len(salami_entropy), n_samples, replace=True)
        plt.scatter(salami_entropy[salami_idx], salami_std[salami_idx], 
                   alpha=0.3, s=1, c='red', label='Salami')
    if len(tomato_entropy) > 0:
        tomato_idx = np.random.choice(len(tomato_entropy), n_samples, replace=True)
        plt.scatter(tomato_entropy[tomato_idx], tomato_std[tomato_idx], 
                   alpha=0.3, s=1, c='orange', label='Tomato')
    plt.xlabel('Entropy')
    plt.ylabel('Local STD')
    plt.title('Feature Space (2D)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_kernel_analysis.png", dpi=150)
    plt.close()
    
    # 個別のヒートマップも保存
    # エントロピーマップ
    entropy_normalized = cv2.normalize(entropy_pizza, None, 0, 255, cv2.NORM_MINMAX)
    entropy_colored = cv2.applyColorMap(entropy_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{base_name}_entropy_9x9.jpg"), entropy_colored)
    
    # 標準偏差マップ
    std_normalized = cv2.normalize(std_pizza, None, 0, 255, cv2.NORM_MINMAX)
    std_colored = cv2.applyColorMap(std_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{base_name}_std_9x9.jpg"), std_colored)
    
    # グレースケール画像も保存
    cv2.imwrite(str(output_dir / f"{base_name}_grayscale.jpg"), gray)


def main():
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_kernel_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\n9x9カーネルによるテクスチャ分析を開始します...\n")
    
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
            print("  ピザ領域を検出中...")
            pizza_mask = pizza_service.segment_pizza(processed_image_path)
            
            # 3. サラミマスク取得（色ベース）
            print("  サラミ領域を検出中...")
            salami_mask = salami_service.segment_salami(processed_image_path, pizza_mask)
            
            # 画像読み込み
            image = cv2.imread(processed_image_path)
            
            # 4. 9x9カーネルでテクスチャマップ計算
            print("  9x9カーネルでテクスチャ特徴量を計算中...")
            entropy_map, std_map = compute_texture_maps_kernel(image, kernel_size=9)
            
            # 5. 可視化と保存
            print("  結果を可視化中...")
            base_name = image_path.stem
            create_feature_visualization(
                image, entropy_map, std_map, pizza_mask, salami_mask, 
                base_name, output_dir
            )
            
            print(f"  完了: {output_dir}/{base_name}_kernel_analysis.png")
            
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
    print("\n生成されたファイル:")
    print("  - *_kernel_analysis.png : 総合分析結果")
    print("  - *_grayscale.jpg      : グレースケール画像")
    print("  - *_entropy_9x9.jpg    : エントロピーマップ（ヒートマップ）")
    print("  - *_std_9x9.jpg        : 標準偏差マップ（ヒートマップ）")


if __name__ == "__main__":
    main()