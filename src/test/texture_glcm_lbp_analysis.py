import cv2
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


def compute_glcm_features(image, mask, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    GLCM（Gray Level Co-occurrence Matrix）特徴量を計算
    
    Args:
        image: 入力画像（BGR）
        mask: 対象領域のマスク
        distances: 距離のリスト
        angles: 角度のリスト
        
    Returns:
        特徴量の辞書
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # マスク領域を抽出
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0
    
    # 8ビットに量子化（GLCMの計算を高速化）
    levels = 64  # グレーレベル数を減らして計算を高速化
    masked_gray = (masked_gray // (256 // levels)).astype(np.uint8)
    
    features = {}
    
    # 各距離と角度でGLCMを計算
    for distance in distances:
        # GLCMを計算
        glcm = graycomatrix(masked_gray, distances=[distance], angles=angles, 
                          levels=levels, symmetric=True, normed=True)
        
        # 各特徴量を計算
        # コントラスト：隣接ピクセル間の強度差
        contrast = graycoprops(glcm, 'contrast')
        features[f'glcm_contrast_d{distance}_mean'] = np.mean(contrast)
        features[f'glcm_contrast_d{distance}_std'] = np.std(contrast)
        
        # 相関：ピクセル値の線形依存性
        correlation = graycoprops(glcm, 'correlation')
        features[f'glcm_correlation_d{distance}_mean'] = np.mean(correlation)
        features[f'glcm_correlation_d{distance}_std'] = np.std(correlation)
        
        # エネルギー（ASM）：テクスチャの均一性
        energy = graycoprops(glcm, 'energy')
        features[f'glcm_energy_d{distance}_mean'] = np.mean(energy)
        features[f'glcm_energy_d{distance}_std'] = np.std(energy)
        
        # ホモジェニティ：局所的な均一性
        homogeneity = graycoprops(glcm, 'homogeneity')
        features[f'glcm_homogeneity_d{distance}_mean'] = np.mean(homogeneity)
        features[f'glcm_homogeneity_d{distance}_std'] = np.std(homogeneity)
    
    return features


def compute_lbp_features(image, mask, radius_list=[1, 3, 5], n_points_list=[8, 16, 24]):
    """
    LBP（Local Binary Pattern）特徴量を計算
    
    Args:
        image: 入力画像（BGR）
        mask: 対象領域のマスク
        radius_list: 半径のリスト
        n_points_list: サンプリング点数のリスト
        
    Returns:
        特徴量の辞書
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = {}
    
    for radius in radius_list:
        for n_points in n_points_list:
            # n_points が radius に対して妥当かチェック
            if n_points > 8 * radius:
                continue
                
            # LBPを計算
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # マスク領域のLBP値を抽出
            masked_lbp = lbp[mask > 0]
            
            if len(masked_lbp) > 0:
                # ヒストグラムを計算
                n_bins = n_points + 2  # uniform LBPのビン数
                hist, _ = np.histogram(masked_lbp, bins=n_bins, range=(0, n_bins), density=True)
                
                # 統計量
                features[f'lbp_r{radius}_p{n_points}_mean'] = np.mean(masked_lbp)
                features[f'lbp_r{radius}_p{n_points}_std'] = np.std(masked_lbp)
                features[f'lbp_r{radius}_p{n_points}_entropy'] = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + 1e-10))
            else:
                features[f'lbp_r{radius}_p{n_points}_mean'] = 0
                features[f'lbp_r{radius}_p{n_points}_std'] = 0
                features[f'lbp_r{radius}_p{n_points}_entropy'] = 0
    
    return features


def compute_lbp_map(gray, radius=3, n_points=24):
    """
    LBPマップを計算（可視化用）
    
    Args:
        gray: グレースケール画像
        radius: LBPの半径
        n_points: サンプリング点数
        
    Returns:
        LBPマップ
    """
    return local_binary_pattern(gray, n_points, radius, method='uniform')


def compute_glcm_contrast_map(gray, window_size=15, distance=1):
    """
    GLCMコントラストマップを計算（可視化用）
    
    Args:
        gray: グレースケール画像
        window_size: ウィンドウサイズ
        distance: GLCM距離
        
    Returns:
        コントラストマップ
    """
    h, w = gray.shape
    contrast_map = np.zeros((h, w), dtype=np.float32)
    
    pad = window_size // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # 8ビットに量子化
    levels = 32
    padded_quantized = (padded // (256 // levels)).astype(np.uint8)
    
    for i in range(h):
        for j in range(w):
            # ウィンドウを抽出
            window = padded_quantized[i:i+window_size, j:j+window_size]
            
            # GLCMを計算
            glcm = graycomatrix(window, distances=[distance], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                              levels=levels, symmetric=True, normed=True)
            
            # コントラストを計算
            contrast = graycoprops(glcm, 'contrast')
            contrast_map[i, j] = np.mean(contrast)
        
        if i % 50 == 0:
            print(f"    GLCMコントラストマップ計算中: {i}/{h} 行完了", end='\r')
    
    print(f"    GLCMコントラストマップ計算完了: {h}/{h} 行              ")
    
    return contrast_map


def create_glcm_lbp_visualization(image, pizza_mask, salami_mask, base_name, output_dir):
    """
    GLCMとLBP特徴量の可視化
    
    Args:
        image: 元画像（BGR）
        pizza_mask: ピザマスク
        salami_mask: サラミマスク
        base_name: ファイル名ベース
        output_dir: 出力ディレクトリ
    """
    # グレースケール画像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # トマトソースマスク
    tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
    
    print("  GLCM特徴量を計算中...")
    salami_glcm = compute_glcm_features(image, salami_mask)
    tomato_glcm = compute_glcm_features(image, tomato_mask)
    
    print("  LBP特徴量を計算中...")
    salami_lbp = compute_lbp_features(image, salami_mask)
    tomato_lbp = compute_lbp_features(image, tomato_mask)
    
    print("  特徴量マップを生成中...")
    # LBPマップ（複数の設定）
    lbp_r1_p8 = compute_lbp_map(gray, radius=1, n_points=8)
    lbp_r3_p24 = compute_lbp_map(gray, radius=3, n_points=24)
    
    # GLCMコントラストマップ
    glcm_contrast = compute_glcm_contrast_map(gray, window_size=15, distance=1)
    
    # マスク適用
    lbp_r1_p8_masked = np.where(pizza_mask > 0, lbp_r1_p8, 0)
    lbp_r3_p24_masked = np.where(pizza_mask > 0, lbp_r3_p24, 0)
    glcm_contrast_masked = np.where(pizza_mask > 0, glcm_contrast, 0)
    
    # 大きな可視化図を作成
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 元画像（グレースケール）
    plt.subplot(4, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original (Grayscale)')
    plt.axis('off')
    
    # 2. マスク
    plt.subplot(4, 4, 2)
    plt.imshow(salami_mask, cmap='gray')
    plt.title('Salami Mask')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(tomato_mask, cmap='gray')
    plt.title('Tomato Mask')
    plt.axis('off')
    
    # 4. LBPマップ (r=1, p=8)
    plt.subplot(4, 4, 5)
    im1 = plt.imshow(lbp_r1_p8_masked, cmap='jet')
    plt.title('LBP Map (r=1, p=8)')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 5. LBPマップ (r=3, p=24)
    plt.subplot(4, 4, 6)
    im2 = plt.imshow(lbp_r3_p24_masked, cmap='jet')
    plt.title('LBP Map (r=3, p=24)')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 6. GLCMコントラストマップ
    plt.subplot(4, 4, 7)
    im3 = plt.imshow(glcm_contrast_masked, cmap='jet')
    plt.title('GLCM Contrast Map')
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 7-8. LBPヒストグラム
    plt.subplot(4, 4, 9)
    salami_lbp_vals = lbp_r1_p8[salami_mask > 0]
    tomato_lbp_vals = lbp_r1_p8[tomato_mask > 0]
    plt.hist(salami_lbp_vals, bins=10, alpha=0.7, label='Salami', color='red', density=True)
    plt.hist(tomato_lbp_vals, bins=10, alpha=0.7, label='Tomato', color='orange', density=True)
    plt.xlabel('LBP Value')
    plt.ylabel('Density')
    plt.title('LBP Distribution (r=1, p=8)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. GLCM特徴量比較（バープロット）
    plt.subplot(4, 4, 10)
    glcm_features = ['contrast', 'correlation', 'energy', 'homogeneity']
    salami_vals = [salami_glcm[f'glcm_{feat}_d1_mean'] for feat in glcm_features]
    tomato_vals = [tomato_glcm[f'glcm_{feat}_d1_mean'] for feat in glcm_features]
    
    x = np.arange(len(glcm_features))
    width = 0.35
    
    plt.bar(x - width/2, salami_vals, width, label='Salami', color='red')
    plt.bar(x + width/2, tomato_vals, width, label='Tomato', color='orange')
    plt.xlabel('GLCM Features')
    plt.ylabel('Value')
    plt.title('GLCM Features Comparison (d=1)')
    plt.xticks(x, glcm_features, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. 統計情報テーブル
    plt.subplot(4, 4, 11)
    plt.axis('off')
    stats_text = """GLCM Statistics (distance=1):
    
Contrast:
  Salami: μ={:.3f}, σ={:.3f}
  Tomato: μ={:.3f}, σ={:.3f}
  
Energy:
  Salami: μ={:.3f}, σ={:.3f}
  Tomato: μ={:.3f}, σ={:.3f}
  
LBP Statistics (r=3, p=24):
  Salami: μ={:.2f}, σ={:.2f}
  Tomato: μ={:.2f}, σ={:.2f}""".format(
        salami_glcm['glcm_contrast_d1_mean'], salami_glcm['glcm_contrast_d1_std'],
        tomato_glcm['glcm_contrast_d1_mean'], tomato_glcm['glcm_contrast_d1_std'],
        salami_glcm['glcm_energy_d1_mean'], salami_glcm['glcm_energy_d1_std'],
        tomato_glcm['glcm_energy_d1_mean'], tomato_glcm['glcm_energy_d1_std'],
        salami_lbp['lbp_r3_p24_mean'], salami_lbp['lbp_r3_p24_std'],
        tomato_lbp['lbp_r3_p24_mean'], tomato_lbp['lbp_r3_p24_std']
    )
    
    plt.text(0.05, 0.5, stats_text, fontsize=11, family='monospace',
             transform=plt.gca().transAxes, verticalalignment='center')
    
    # 12. 特徴量の詳細比較表
    plt.subplot(4, 2, 7)
    plt.axis('off')
    
    # GLCM特徴量の表
    glcm_table_data = []
    for feat in ['contrast', 'correlation', 'energy', 'homogeneity']:
        for d in [1, 3, 5]:
            key = f'glcm_{feat}_d{d}_mean'
            if key in salami_glcm and key in tomato_glcm:
                diff = ((salami_glcm[key] - tomato_glcm[key]) / (tomato_glcm[key] + 1e-10)) * 100
                glcm_table_data.append([f'{feat} (d={d})', 
                                       f'{salami_glcm[key]:.3f}',
                                       f'{tomato_glcm[key]:.3f}',
                                       f'{diff:+.1f}%'])
    
    # 表を描画
    table = plt.table(cellText=glcm_table_data,
                     colLabels=['GLCM Feature', 'Salami', 'Tomato', 'Diff'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('GLCM Features Detailed Comparison', pad=20)
    
    # 13. LBP特徴量の表
    plt.subplot(4, 2, 8)
    plt.axis('off')
    
    lbp_table_data = []
    for r in [1, 3, 5]:
        for p in [8, 16, 24]:
            key_mean = f'lbp_r{r}_p{p}_mean'
            key_entropy = f'lbp_r{r}_p{p}_entropy'
            if key_mean in salami_lbp and key_mean in tomato_lbp:
                diff_mean = ((salami_lbp[key_mean] - tomato_lbp[key_mean]) / (tomato_lbp[key_mean] + 1e-10)) * 100
                lbp_table_data.append([f'r={r}, p={p}',
                                      f'{salami_lbp[key_mean]:.2f}',
                                      f'{tomato_lbp[key_mean]:.2f}',
                                      f'{diff_mean:+.1f}%'])
    
    table2 = plt.table(cellText=lbp_table_data,
                      colLabels=['LBP Config', 'Salami μ', 'Tomato μ', 'Diff'],
                      cellLoc='center',
                      loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    plt.title('LBP Features Detailed Comparison', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_glcm_lbp_analysis.png", dpi=150)
    plt.close()
    
    # 個別のマップも保存
    # LBPマップ
    lbp_r1_p8_norm = cv2.normalize(lbp_r1_p8_masked, None, 0, 255, cv2.NORM_MINMAX)
    lbp_r1_p8_colored = cv2.applyColorMap(lbp_r1_p8_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{base_name}_lbp_r1_p8.jpg"), lbp_r1_p8_colored)
    
    lbp_r3_p24_norm = cv2.normalize(lbp_r3_p24_masked, None, 0, 255, cv2.NORM_MINMAX)
    lbp_r3_p24_colored = cv2.applyColorMap(lbp_r3_p24_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{base_name}_lbp_r3_p24.jpg"), lbp_r3_p24_colored)
    
    # GLCMコントラストマップ
    glcm_contrast_norm = cv2.normalize(glcm_contrast_masked, None, 0, 255, cv2.NORM_MINMAX)
    glcm_contrast_colored = cv2.applyColorMap(glcm_contrast_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{base_name}_glcm_contrast.jpg"), glcm_contrast_colored)
    
    return {
        'salami_glcm': salami_glcm,
        'tomato_glcm': tomato_glcm,
        'salami_lbp': salami_lbp,
        'tomato_lbp': tomato_lbp
    }


def main():
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_glcm_lbp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\nGLCMとLBPによるテクスチャ分析を開始します...\n")
    
    # 全画像の結果を収集
    all_results = []
    
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
            
            # 4. GLCMとLBP分析
            base_name = image_path.stem
            results = create_glcm_lbp_visualization(
                image, pizza_mask, salami_mask, base_name, output_dir
            )
            all_results.append(results)
            
            print(f"  完了: {output_dir}/{base_name}_glcm_lbp_analysis.png")
            
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
    
    # 全画像の平均結果をまとめる
    if all_results:
        print(f"\n{'='*60}")
        print("全画像の平均結果:")
        print("="*60)
        
        # 主要な特徴量の平均を計算
        avg_features = {}
        feature_names = ['glcm_contrast_d1_mean', 'glcm_energy_d1_mean', 
                        'glcm_homogeneity_d1_mean', 'lbp_r3_p24_mean']
        
        for feat_name in feature_names:
            salami_vals = [r['salami_glcm'].get(feat_name, r['salami_lbp'].get(feat_name, 0)) 
                          for r in all_results]
            tomato_vals = [r['tomato_glcm'].get(feat_name, r['tomato_lbp'].get(feat_name, 0)) 
                          for r in all_results]
            
            avg_salami = np.mean(salami_vals)
            avg_tomato = np.mean(tomato_vals)
            diff = ((avg_salami - avg_tomato) / (avg_tomato + 1e-10)) * 100
            
            print(f"{feat_name:30} Salami: {avg_salami:8.3f}, Tomato: {avg_tomato:8.3f}, Diff: {diff:+6.1f}%")
    
    print(f"\n{'='*60}")
    print(f"処理完了。{output_dir}で結果を確認してください。")
    print("\n生成されたファイル:")
    print("  - *_glcm_lbp_analysis.png : GLCMとLBPの総合分析結果")
    print("  - *_lbp_r*_p*.jpg        : LBPマップ（ヒートマップ）")
    print("  - *_glcm_contrast.jpg    : GLCMコントラストマップ")


if __name__ == "__main__":
    main()