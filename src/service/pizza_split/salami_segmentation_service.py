import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import os

class SalamiSegmentationService:
    def __init__(self, model_path: str = None):
        """
        YOLOv11を使用したサラミセグメンテーションサービス
        
        Args:
            model_path: YOLOv11モデルのパス
        """
        if model_path is None:
            # プロジェクトルートのweights.ptを使用
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = str(project_root / "weights.pt")
        self.model = YOLO(model_path)
    
    def segment_salami(self, image_path: str) -> np.ndarray:
        """
        画像からサラミをセグメンテーション
        
        Args:
            image_path: 入力画像のパス
            
        Returns:
            サラミ領域のマスク（255: サラミ, 0: その他）
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像が読み込めません: {image_path}")
        
        # YOLO推論実行
        results = self.model(image_path)
        
        # マスクを初期化
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 結果からマスクを作成
        for result in results:
            if result.masks is not None:
                # セグメンテーションマスクを取得
                for mask_data in result.masks.data:
                    # マスクをリサイズして元画像サイズに合わせる
                    mask_resized = cv2.resize(
                        mask_data.cpu().numpy(), 
                        (image.shape[1], image.shape[0])
                    )
                    # 閾値処理でバイナリマスクに変換
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    # 複数のマスクを結合
                    mask = cv2.bitwise_or(mask, binary_mask)
        
        return mask
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255)) -> np.ndarray:
        """
        画像にマスクをオーバーレイ
        
        Args:
            image: 元画像
            mask: セグメンテーションマスク
            color: オーバーレイ色 (B, G, R)
            
        Returns:
            オーバーレイ画像
        """
        overlay = image.copy()
        overlay[mask == 255] = color
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    def save_results(self, image_path: str, mask: np.ndarray, overlay: np.ndarray, output_dir: Path):
        """
        結果を保存
        
        Args:
            image_path: 元画像のパス
            mask: セグメンテーションマスク
            overlay: オーバーレイ画像
            output_dir: 出力ディレクトリ
        """
        base_name = Path(image_path).stem
        
        # マスクを保存
        mask_path = output_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # オーバーレイを保存
        overlay_path = output_dir / f"{base_name}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        
        print(f"結果を保存しました: {base_name}")
        print(f"  - マスク: {mask_path}")
        print(f"  - オーバーレイ: {overlay_path}")


if __name__ == "__main__":
    # サービス初期化
    salami_service = SalamiSegmentationService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(resource_dir.glob(ext))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        exit(1)
    
    print(f"\n{len(image_files)}個の画像ファイルが見つかりました")
    print("サラミセグメンテーションを実行します...\n")
    
    # 各画像を処理
    for image_path in image_files:
        print(f"{image_path.name}を処理中...")
        
        try:
            # 1. サラミセグメンテーション実行
            mask = salami_service.segment_salami(str(image_path))
            
            # 2. 元画像を読み込み
            image = cv2.imread(str(image_path))
            
            # 3. オーバーレイ作成
            overlay = salami_service.create_overlay(image, mask)
            
            # 4. 結果保存
            salami_service.save_results(str(image_path), mask, overlay, result_dir)
            
            # 5. 統計情報表示
            salami_pixels = np.sum(mask == 255)
            total_pixels = mask.shape[0] * mask.shape[1]
            ratio = (salami_pixels / total_pixels) * 100
            
            print(f"  統計情報:")
            print(f"    サラミピクセル数: {salami_pixels:,}")
            print(f"    全体に占める割合: {ratio:.2f}%")
            print()
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            continue
    
    print(f"処理完了。{result_dir}で結果を確認してください。")