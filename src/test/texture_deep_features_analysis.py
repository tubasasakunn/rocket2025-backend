import cv2
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.preprocess import PreprocessService


class FeatureExtractor:
    """VGG16とResNetから中間層特徴を抽出するクラス"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # VGG16モデル
        self.vgg16 = models.vgg16(pretrained=True).to(device)
        self.vgg16.eval()
        
        # ResNet50モデル
        self.resnet50 = models.resnet50(pretrained=True).to(device)
        self.resnet50.eval()
        
        # 画像の前処理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # VGG16の中間層フック
        self.vgg_features = {}
        def get_vgg_activation(name):
            def hook(model, input, output):
                self.vgg_features[name] = output.detach()
            return hook
        
        # 複数の層から特徴を取得
        self.vgg16.features[4].register_forward_hook(get_vgg_activation('conv1_2'))   # 64ch
        self.vgg16.features[9].register_forward_hook(get_vgg_activation('conv2_2'))   # 128ch
        self.vgg16.features[16].register_forward_hook(get_vgg_activation('conv3_3'))  # 256ch
        self.vgg16.features[23].register_forward_hook(get_vgg_activation('conv4_3'))  # 512ch
        
        # ResNetの中間層フック
        self.resnet_features = {}
        def get_resnet_activation(name):
            def hook(model, input, output):
                self.resnet_features[name] = output.detach()
            return hook
        
        self.resnet50.layer1.register_forward_hook(get_resnet_activation('layer1'))  # 256ch
        self.resnet50.layer2.register_forward_hook(get_resnet_activation('layer2'))  # 512ch
        self.resnet50.layer3.register_forward_hook(get_resnet_activation('layer3'))  # 1024ch
        self.resnet50.layer4.register_forward_hook(get_resnet_activation('layer4'))  # 2048ch
    
    def extract_patch_features(self, image_patch):
        """
        画像パッチから特徴を抽出
        
        Args:
            image_patch: 画像パッチ（BGR）
            
        Returns:
            VGGとResNetの特徴辞書
        """
        # BGRからRGBに変換
        rgb_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        
        # テンソルに変換
        input_tensor = self.transform(rgb_patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # VGG16の推論
            _ = self.vgg16(input_tensor)
            
            # ResNet50の推論
            _ = self.resnet50(input_tensor)
        
        # 特徴をコピー
        vgg_feats = self.vgg_features.copy()
        resnet_feats = self.resnet_features.copy()
        
        return vgg_feats, resnet_feats


def extract_region_features(image, mask, feature_extractor, patch_size=64, stride=32):
    """
    マスク領域から深層特徴を抽出
    
    Args:
        image: 入力画像（BGR）
        mask: 領域マスク
        feature_extractor: 特徴抽出器
        patch_size: パッチサイズ
        stride: ストライド
        
    Returns:
        特徴統計量の辞書
    """
    h, w = mask.shape
    features_dict = {
        'vgg_conv1_2': [], 'vgg_conv2_2': [], 'vgg_conv3_3': [], 'vgg_conv4_3': [],
        'resnet_layer1': [], 'resnet_layer2': [], 'resnet_layer3': [], 'resnet_layer4': []
    }
    
    # スライディングウィンドウでパッチを抽出
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # マスクの中心がマスク内にあるかチェック
            center_y = y + patch_size // 2
            center_x = x + patch_size // 2
            
            if mask[center_y, center_x] > 0:
                # パッチを抽出
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # 特徴抽出
                vgg_feats, resnet_feats = feature_extractor.extract_patch_features(patch)
                
                # VGG特徴を保存（Global Average Pooling）
                for layer_name, feat in vgg_feats.items():
                    gap_feat = feat.mean(dim=[2, 3]).cpu().numpy().flatten()
                    features_dict[f'vgg_{layer_name}'].append(gap_feat)
                
                # ResNet特徴を保存（Global Average Pooling）
                for layer_name, feat in resnet_feats.items():
                    gap_feat = feat.mean(dim=[2, 3]).cpu().numpy().flatten()
                    features_dict[f'resnet_{layer_name}'].append(gap_feat)
    
    # 統計量を計算
    stats = {}
    for feat_name, feat_list in features_dict.items():
        if feat_list:
            feat_array = np.array(feat_list)
            stats[f'{feat_name}_mean'] = np.mean(feat_array, axis=0)
            stats[f'{feat_name}_std'] = np.std(feat_array, axis=0)
            stats[f'{feat_name}_samples'] = len(feat_list)
        else:
            stats[f'{feat_name}_mean'] = None
            stats[f'{feat_name}_std'] = None
            stats[f'{feat_name}_samples'] = 0
    
    return stats, features_dict


def visualize_feature_maps(image, pizza_mask, salami_mask, feature_extractor, 
                          base_name, output_dir):
    """
    特徴マップの可視化
    
    Args:
        image: 入力画像（BGR）
        pizza_mask: ピザマスク
        salami_mask: サラミマスク
        feature_extractor: 特徴抽出器
        base_name: ファイル名ベース
        output_dir: 出力ディレクトリ
    """
    # トマトソースマスク
    tomato_mask = cv2.bitwise_and(pizza_mask, cv2.bitwise_not(salami_mask))
    
    print("  深層特徴を抽出中...")
    # 各領域から特徴を抽出
    salami_stats, salami_features = extract_region_features(
        image, salami_mask, feature_extractor, patch_size=64, stride=32
    )
    tomato_stats, tomato_features = extract_region_features(
        image, tomato_mask, feature_extractor, patch_size=64, stride=32
    )
    
    # 可視化用の図を作成
    fig = plt.figure(figsize=(20, 24))
    
    # 1. 元画像とマスク
    plt.subplot(6, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(6, 3, 2)
    plt.imshow(salami_mask, cmap='gray')
    plt.title('Salami Mask')
    plt.axis('off')
    
    plt.subplot(6, 3, 3)
    plt.imshow(tomato_mask, cmap='gray')
    plt.title('Tomato Mask')
    plt.axis('off')
    
    # 2. 特徴量の比較（各層）
    layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
    
    for idx, layer in enumerate(layer_names):
        plt.subplot(6, 3, 4 + idx)
        
        # VGG特徴の一部を比較（最初の10次元）
        if salami_stats[f'vgg_{layer}_mean'] is not None and tomato_stats[f'vgg_{layer}_mean'] is not None:
            salami_feat = salami_stats[f'vgg_{layer}_mean'][:10]
            tomato_feat = tomato_stats[f'vgg_{layer}_mean'][:10]
            
            x = np.arange(len(salami_feat))
            width = 0.35
            
            plt.bar(x - width/2, salami_feat, width, label='Salami', color='red', alpha=0.7)
            plt.bar(x + width/2, tomato_feat, width, label='Tomato', color='orange', alpha=0.7)
            plt.xlabel('Feature Dimension')
            plt.ylabel('Mean Activation')
            plt.title(f'VGG16 {layer} Features')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 3. ResNet特徴の比較
    resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    for idx, layer in enumerate(resnet_layers):
        plt.subplot(6, 3, 8 + idx)
        
        if salami_stats[f'resnet_{layer}_mean'] is not None and tomato_stats[f'resnet_{layer}_mean'] is not None:
            salami_feat = salami_stats[f'resnet_{layer}_mean'][:10]
            tomato_feat = tomato_stats[f'resnet_{layer}_mean'][:10]
            
            x = np.arange(len(salami_feat))
            width = 0.35
            
            plt.bar(x - width/2, salami_feat, width, label='Salami', color='red', alpha=0.7)
            plt.bar(x + width/2, tomato_feat, width, label='Tomato', color='orange', alpha=0.7)
            plt.xlabel('Feature Dimension')
            plt.ylabel('Mean Activation')
            plt.title(f'ResNet50 {layer} Features')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 4. t-SNE/PCA可視化（VGG conv4_3）
    plt.subplot(6, 3, 13)
    if (salami_features['vgg_conv4_3'] and tomato_features['vgg_conv4_3'] and 
        len(salami_features['vgg_conv4_3']) > 10 and len(tomato_features['vgg_conv4_3']) > 10):
        
        # 特徴を結合
        salami_feats = np.array(salami_features['vgg_conv4_3'])
        tomato_feats = np.array(tomato_features['vgg_conv4_3'])
        
        # サンプル数を制限
        n_samples = min(100, len(salami_feats), len(tomato_feats))
        salami_sample = salami_feats[np.random.choice(len(salami_feats), n_samples, replace=False)]
        tomato_sample = tomato_feats[np.random.choice(len(tomato_feats), n_samples, replace=False)]
        
        all_feats = np.vstack([salami_sample, tomato_sample])
        
        # PCA
        pca = PCA(n_components=2)
        pca_feats = pca.fit_transform(all_feats)
        
        # プロット
        plt.scatter(pca_feats[:n_samples, 0], pca_feats[:n_samples, 1], 
                   c='red', alpha=0.5, label='Salami')
        plt.scatter(pca_feats[n_samples:, 0], pca_feats[n_samples:, 1], 
                   c='orange', alpha=0.5, label='Tomato')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of VGG conv4_3 Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. 統計情報
    plt.subplot(6, 3, 14)
    plt.axis('off')
    
    stats_text = f"""Deep Features Statistics:
    
VGG16 conv4_3:
  Salami samples: {salami_stats['vgg_conv4_3_samples']}
  Tomato samples: {tomato_stats['vgg_conv4_3_samples']}
  
ResNet50 layer4:
  Salami samples: {salami_stats['resnet_layer4_samples']}
  Tomato samples: {tomato_stats['resnet_layer4_samples']}
  
Feature Extraction:
  Patch size: 64x64
  Stride: 32
  Model: Pretrained on ImageNet"""
    
    plt.text(0.05, 0.5, stats_text, fontsize=12, family='monospace',
             transform=plt.gca().transAxes, verticalalignment='center')
    
    # 6. 特徴距離の計算
    plt.subplot(6, 3, 15)
    
    distances = []
    layer_labels = []
    
    # VGG層
    for layer in layer_names:
        if salami_stats[f'vgg_{layer}_mean'] is not None and tomato_stats[f'vgg_{layer}_mean'] is not None:
            s_feat = salami_stats[f'vgg_{layer}_mean']
            t_feat = tomato_stats[f'vgg_{layer}_mean']
            # コサイン距離
            cos_sim = np.dot(s_feat, t_feat) / (np.linalg.norm(s_feat) * np.linalg.norm(t_feat))
            distances.append(1 - cos_sim)
            layer_labels.append(f'VGG\n{layer}')
    
    # ResNet層
    for layer in resnet_layers:
        if salami_stats[f'resnet_{layer}_mean'] is not None and tomato_stats[f'resnet_{layer}_mean'] is not None:
            s_feat = salami_stats[f'resnet_{layer}_mean']
            t_feat = tomato_stats[f'resnet_{layer}_mean']
            cos_sim = np.dot(s_feat, t_feat) / (np.linalg.norm(s_feat) * np.linalg.norm(t_feat))
            distances.append(1 - cos_sim)
            layer_labels.append(f'ResNet\n{layer}')
    
    plt.bar(range(len(distances)), distances)
    plt.xlabel('Layer')
    plt.ylabel('Cosine Distance')
    plt.title('Feature Distance between Salami and Tomato')
    plt.xticks(range(len(distances)), layer_labels, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. 活性化マップの可視化（1枚の画像で）
    plt.subplot(6, 3, 16)
    
    # 画像の中心付近からパッチを抽出
    h, w = image.shape[:2]
    center_patch = image[h//2-32:h//2+32, w//2-32:w//2+32]
    vgg_feats, _ = feature_extractor.extract_patch_features(center_patch)
    
    # conv1_2の最初の16チャンネルを可視化
    if 'conv1_2' in vgg_feats:
        feat_map = vgg_feats['conv1_2'][0, :16].cpu().numpy()
        # 4x4グリッドで表示
        grid = np.zeros((4*14, 4*14))
        for i in range(4):
            for j in range(4):
                if i*4+j < 16:
                    # 特徴マップをリサイズ
                    fm = cv2.resize(feat_map[i*4+j], (14, 14))
                    grid[i*14:(i+1)*14, j*14:(j+1)*14] = fm
        
        plt.imshow(grid, cmap='viridis')
        plt.title('VGG conv1_2 Activation Maps (16ch)')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_deep_features_analysis.png", dpi=150)
    plt.close()
    
    # 特徴距離の詳細を保存
    results_summary = {
        'salami_samples': salami_stats['vgg_conv4_3_samples'],
        'tomato_samples': tomato_stats['vgg_conv4_3_samples'],
        'feature_distances': dict(zip(layer_labels, distances)) if distances else {}
    }
    
    return results_summary


def create_feature_heatmaps(image, feature_extractor, base_name, output_dir, 
                           patch_size=64, stride=16):
    """
    特徴量ヒートマップを作成
    
    Args:
        image: 入力画像（BGR）
        feature_extractor: 特徴抽出器
        base_name: ファイル名ベース
        output_dir: 出力ディレクトリ
        patch_size: パッチサイズ
        stride: ストライド
    """
    h, w = image.shape[:2]
    
    # 出力ヒートマップのサイズ
    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1
    
    # VGG conv4_3の平均活性化でヒートマップを作成
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    
    print(f"    特徴ヒートマップを生成中...", end='', flush=True)
    
    for i, y in enumerate(range(0, h - patch_size + 1, stride)):
        for j, x in enumerate(range(0, w - patch_size + 1, stride)):
            # パッチを抽出
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # 特徴抽出
            vgg_feats, _ = feature_extractor.extract_patch_features(patch)
            
            # conv4_3の平均活性化
            if 'conv4_3' in vgg_feats:
                activation = vgg_feats['conv4_3'].mean().item()
                heatmap[i, j] = activation
    
    print(" 完了")
    
    # ヒートマップを元画像サイズにリサイズ
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # 正規化
    heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    # オーバーレイ
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    # 保存
    cv2.imwrite(str(output_dir / f"{base_name}_vgg_conv4_3_heatmap.jpg"), heatmap_colored)
    cv2.imwrite(str(output_dir / f"{base_name}_vgg_conv4_3_overlay.jpg"), overlay)
    
    return heatmap_resized


def main():
    # GPUが利用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    
    # 特徴抽出器を初期化
    print("深層学習モデルを読み込み中...")
    feature_extractor = FeatureExtractor(device=device)
    
    # サービス初期化
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/texture_deep_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("\nVGG16/ResNetの中間層特徴によるテクスチャ分析を開始します...\n")
    
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
            
            # 4. 深層特徴分析
            base_name = image_path.stem
            results = visualize_feature_maps(
                image, pizza_mask, salami_mask, feature_extractor, 
                base_name, output_dir
            )
            all_results.append(results)
            
            # 5. 特徴ヒートマップ生成
            create_feature_heatmaps(image, feature_extractor, base_name, output_dir)
            
            print(f"  完了: {output_dir}/{base_name}_deep_features_analysis.png")
            
            # 一時ファイル削除
            if temp_preprocessed_path.exists():
                temp_preprocessed_path.unlink()
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 結果のサマリー
    if all_results:
        print(f"\n{'='*60}")
        print("深層特徴分析のサマリー:")
        print("="*60)
        
        # 平均的な特徴距離を表示
        if all_results[0]['feature_distances']:
            avg_distances = {}
            for layer in all_results[0]['feature_distances'].keys():
                distances = [r['feature_distances'].get(layer, 0) for r in all_results]
                avg_distances[layer] = np.mean(distances)
            
            print("\n平均コサイン距離（サラミ vs トマトソース）:")
            for layer, dist in sorted(avg_distances.items(), key=lambda x: x[1], reverse=True):
                print(f"  {layer.replace(chr(10), ' '):<20}: {dist:.4f}")
    
    print(f"\n{'='*60}")
    print(f"処理完了。{output_dir}で結果を確認してください。")
    print("\n生成されたファイル:")
    print("  - *_deep_features_analysis.png : VGG16/ResNet特徴の総合分析")
    print("  - *_vgg_conv4_3_heatmap.jpg   : VGG conv4_3層の活性化ヒートマップ")
    print("  - *_vgg_conv4_3_overlay.jpg   : ヒートマップのオーバーレイ")


if __name__ == "__main__":
    main()