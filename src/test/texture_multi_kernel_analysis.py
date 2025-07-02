import cv2
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


def compute_entropy_kernel(gray_patch):
    """
    パッチのエントロピーを計算
    
    Args:
        gray_patch: グレースケールパッチ
        
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
    パッチの標準偏差を計算
    
    Args:
        gray_patch: グレースケールパッチ
        
    Returns:
        標準偏差
    """
    return np.std(gray_patch)


def compute_texture_maps_kernel(image, kernel_size):
    """
    指定サイズのカーネルでスライディングウィンドウを使用してテクスチャマップを計算
    
    Args:
        image: 入力画像（BGR）
        kernel_size: カーネルサイズ
        
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
    start_time = time.time()
    total_pixels = h * w
    processed = 0
    
    for i in range(h):
        for j in range(w):
            # パッチを抽出
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            
            # エントロピーと標準偏差を計算
            entropy_map[i, j] = compute_entropy_kernel(patch)
            std_map[i, j] = compute_std_kernel(patch)
            
            processed += 1
            
        # 進捗表示
        if i % 20 == 0:
            elapsed = time.time() - start_time
            eta = elapsed * (h - i) / (i + 1) if i > 0 else 0
            print(f"    カーネル{kernel_size}x{kernel_size}: {i}/{h} 行完了 (ETA: {eta:.1f}秒)", end='\r')
    
    elapsed = time.time() - start_time
    print(f"    カーネル{kernel_size}x{kernel_size}: 完了 ({elapsed:.1f}秒)              ")
    
    return entropy_map, std_map


def create_multi_kernel_visualization(image, results, pizza_mask, salami_mask, 
                                    base_name, output_dir):
    """
    複数のカーネルサイズの結果を比較可視化
    
    Args:
        image: 元画像（BGR）
        results: {kernel_size: (entropy_map, std_map)}の辞書
        pizza_mask: ピザマスク
        salami_mask: サラミマスク
        base_name: ファイル名ベース
        output_dir: 出力ディレクトリ
    """
    # グレースケール画像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # トマトソースマスク
    tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
    
    # カーネルサイズのリスト
    kernel_sizes = sorted(results.keys())
    
    # 統計情報を収集
    stats = {}
    for kernel_size in kernel_sizes:
        entropy_map, std_map = results[kernel_size]
        
        # マスク適用
        entropy_pizza = np.where(pizza_mask > 0, entropy_map, 0)
        std_pizza = np.where(pizza_mask > 0, std_map, 0)
        
        # 統計情報計算
        salami_entropy = entropy_map[salami_mask > 0]
        tomato_entropy = entropy_map[tomato_mask > 0]
        salami_std = std_map[salami_mask > 0]
        tomato_std = std_map[tomato_mask > 0]
        
        stats[kernel_size] = {
            'salami_entropy_mean': np.mean(salami_entropy) if len(salami_entropy) > 0 else 0,
            'salami_entropy_std': np.std(salami_entropy) if len(salami_entropy) > 0 else 0,
            'tomato_entropy_mean': np.mean(tomato_entropy) if len(tomato_entropy) > 0 else 0,
            'tomato_entropy_std': np.std(tomato_entropy) if len(tomato_entropy) > 0 else 0,
            'salami_std_mean': np.mean(salami_std) if len(salami_std) > 0 else 0,
            'salami_std_std': np.std(salami_std) if len(salami_std) > 0 else 0,
            'tomato_std_mean': np.mean(tomato_std) if len(tomato_std) > 0 else 0,
            'tomato_std_std': np.std(tomato_std) if len(tomato_std) > 0 else 0,
        }
    
    # 大きな比較図を作成
    n_kernels = len(kernel_sizes)
    fig = plt.figure(figsize=(20, 5 * n_kernels + 3))
    
    # 最初の行：元画像とマスク
    plt.subplot(n_kernels + 1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original (Grayscale)')
    plt.axis('off')
    
    plt.subplot(n_kernels + 1, 4, 2)
    plt.imshow(pizza_mask, cmap='gray')
    plt.title('Pizza Mask')
    plt.axis('off')
    
    plt.subplot(n_kernels + 1, 4, 3)
    plt.imshow(salami_mask, cmap='gray')
    plt.title('Salami Mask')
    plt.axis('off')
    
    plt.subplot(n_kernels + 1, 4, 4)
    plt.imshow(tomato_mask, cmap='gray')
    plt.title('Tomato Sauce Mask')
    plt.axis('off')
    
    # 各カーネルサイズの結果を表示
    for idx, kernel_size in enumerate(kernel_sizes):
        entropy_map, std_map = results[kernel_size]
        entropy_pizza = np.where(pizza_mask > 0, entropy_map, 0)
        std_pizza = np.where(pizza_mask > 0, std_map, 0)
        
        row = idx + 2  # 2行目から開始
        
        # エントロピーマップ
        plt.subplot(n_kernels + 1, 4, (row-1)*4 + 1)
        im1 = plt.imshow(entropy_pizza, cmap='jet')
        plt.title(f'Entropy {kernel_size}x{kernel_size}')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 標準偏差マップ
        plt.subplot(n_kernels + 1, 4, (row-1)*4 + 2)
        im2 = plt.imshow(std_pizza, cmap='jet')
        plt.title(f'Local STD {kernel_size}x{kernel_size}')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # エントロピーヒストグラム
        plt.subplot(n_kernels + 1, 4, (row-1)*4 + 3)
        salami_entropy = entropy_map[salami_mask > 0]
        tomato_entropy = entropy_map[tomato_mask > 0]
        plt.hist(salami_entropy, bins=30, alpha=0.7, label='Salami', color='red', density=True)
        plt.hist(tomato_entropy, bins=30, alpha=0.7, label='Tomato', color='orange', density=True)
        plt.xlabel('Entropy')
        plt.ylabel('Density')
        plt.title(f'Entropy Dist. (k={kernel_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 統計情報
        plt.subplot(n_kernels + 1, 4, (row-1)*4 + 4)
        plt.axis('off')
        s = stats[kernel_size]
        stats_text = f"""Kernel {kernel_size}x{kernel_size} Statistics:

Entropy:
  Salami: μ={s['salami_entropy_mean']:.3f}, σ={s['salami_entropy_std']:.3f}
  Tomato: μ={s['tomato_entropy_mean']:.3f}, σ={s['tomato_entropy_std']:.3f}
  Diff: {(s['salami_entropy_mean'] - s['tomato_entropy_mean']):.3f}

Local STD:
  Salami: μ={s['salami_std_mean']:.2f}, σ={s['salami_std_std']:.2f}
  Tomato: μ={s['tomato_std_mean']:.2f}, σ={s['tomato_std_std']:.2f}
  Diff: {(s['salami_std_mean'] - s['tomato_std_mean']):.2f}"""
        
        plt.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                transform=plt.gca().transAxes, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_multi_kernel_comparison.png", dpi=150)
    plt.close()
    
    # 個別のヒートマップも保存
    for kernel_size in kernel_sizes:
        entropy_map, std_map = results[kernel_size]
        entropy_pizza = np.where(pizza_mask > 0, entropy_map, 0)
        std_pizza = np.where(pizza_mask > 0, std_map, 0)
        
        # エントロピーマップ
        entropy_normalized = cv2.normalize(entropy_pizza, None, 0, 255, cv2.NORM_MINMAX)
        entropy_colored = cv2.applyColorMap(entropy_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{base_name}_entropy_{kernel_size}x{kernel_size}.jpg"), entropy_colored)
        
        # 標準偏差マップ
        std_normalized = cv2.normalize(std_pizza, None, 0, 255, cv2.NORM_MINMAX)
        std_colored = cv2.applyColorMap(std_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{base_name}_std_{kernel_size}x{kernel_size}.jpg"), std_colored)
    
    # 統計サマリーを作成
    summary_fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # エントロピー平均値の比較
    kernel_labels = [f'{k}x{k}' for k in kernel_sizes]
    salami_entropy_means = [stats[k]['salami_entropy_mean'] for k in kernel_sizes]
    tomato_entropy_means = [stats[k]['tomato_entropy_mean'] for k in kernel_sizes]
    
    x = np.arange(len(kernel_labels))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, salami_entropy_means, width, label='Salami', color='red')
    axes[0, 0].bar(x + width/2, tomato_entropy_means, width, label='Tomato', color='orange')
    axes[0, 0].set_xlabel('Kernel Size')
    axes[0, 0].set_ylabel('Mean Entropy')
    axes[0, 0].set_title('Entropy Mean by Kernel Size')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(kernel_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 標準偏差平均値の比較
    salami_std_means = [stats[k]['salami_std_mean'] for k in kernel_sizes]
    tomato_std_means = [stats[k]['tomato_std_mean'] for k in kernel_sizes]
    
    axes[0, 1].bar(x - width/2, salami_std_means, width, label='Salami', color='red')
    axes[0, 1].bar(x + width/2, tomato_std_means, width, label='Tomato', color='orange')
    axes[0, 1].set_xlabel('Kernel Size')
    axes[0, 1].set_ylabel('Mean Local STD')
    axes[0, 1].set_title('Local STD Mean by Kernel Size')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(kernel_labels)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 差分のプロット（エントロピー）
    entropy_diffs = [s-t for s, t in zip(salami_entropy_means, tomato_entropy_means)]
    axes[1, 0].plot(kernel_labels, entropy_diffs, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Kernel Size')
    axes[1, 0].set_ylabel('Difference (Salami - Tomato)')
    axes[1, 0].set_title('Entropy Difference by Kernel Size')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 差分のプロット（標準偏差）
    std_diffs = [s-t for s, t in zip(salami_std_means, tomato_std_means)]
    axes[1, 1].plot(kernel_labels, std_diffs, 'ro-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Kernel Size')
    axes[1, 1].set_ylabel('Difference (Salami - Tomato)')
    axes[1, 1].set_title('Local STD Difference by Kernel Size')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_kernel_statistics_summary.png", dpi=150)
    plt.close()


def main():
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_multi_kernel")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\n複数カーネルサイズ（16, 32, 64）でテクスチャ分析を開始します...\n")
    
    # テストするカーネルサイズ
    kernel_sizes = [16, 32, 64]
    
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
            
            # 4. 各カーネルサイズでテクスチャマップ計算
            print(f"  複数のカーネルサイズでテクスチャ特徴量を計算中...")
            results = {}
            for kernel_size in kernel_sizes:
                entropy_map, std_map = compute_texture_maps_kernel(image, kernel_size)
                results[kernel_size] = (entropy_map, std_map)
            
            # 5. 可視化と保存
            print("  結果を可視化中...")
            base_name = image_path.stem
            create_multi_kernel_visualization(
                image, results, pizza_mask, salami_mask, 
                base_name, output_dir
            )
            
            print(f"  完了:")
            print(f"    - 比較図: {output_dir}/{base_name}_multi_kernel_comparison.png")
            print(f"    - 統計サマリー: {output_dir}/{base_name}_kernel_statistics_summary.png")
            
            # グレースケール画像も保存
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(output_dir / f"{base_name}_grayscale.jpg"), gray)
            
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
    print("  - *_multi_kernel_comparison.png    : カーネルサイズ別の比較")
    print("  - *_kernel_statistics_summary.png  : 統計情報のサマリー")
    print("  - *_entropy_NxN.jpg               : 各カーネルサイズのエントロピーマップ")
    print("  - *_std_NxN.jpg                   : 各カーネルサイズの標準偏差マップ")


if __name__ == "__main__":
    main()