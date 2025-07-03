import cv2
import numpy as np
from pathlib import Path
import os

# PyTorch 2.6+ の weights_only 問題を解決
import torch
_original_load = torch.load

def patched_load(f, map_location=None, pickle_module=None, **kwargs):
    """torch.load をパッチして weights_only=False を強制"""
    kwargs['weights_only'] = False
    return _original_load(f, map_location, pickle_module, **kwargs)

torch.load = patched_load

from ultralytics import YOLO


class PizzaSegmentationService:
    """ピザ領域をセグメンテーションするサービスクラス"""
    
    def __init__(self):
        """YOLOモデルを初期化"""
        # モデルファイルのパスを解決
        current_dir = Path(__file__).parent
        model_path = current_dir / 'yolov8n-seg.pt'
        
        # モデルが存在しない場合は、プロジェクトルートを確認
        if not model_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / 'yolov8n-seg.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"YOLOモデルファイルが見つかりません: {model_path}")
        
        # YOLO モデルをロード
        self.model = YOLO(str(model_path))
        self.pizza_class_id = 53  # YOLOにおけるピザのクラスID
    
    def segment_pizza(self, image_path: str, isDebug: bool = False) -> np.ndarray:
        """
        画像からピザ領域をセグメンテーション
        
        Args:
            image_path: 入力画像のパス
            isDebug: デバッグモード（Trueの場合、途中経過を保存）
            
        Returns:
            ピザ領域のバイナリマスク（255: ピザ, 0: 背景）
        """
        # 画像を読み込み
        image = self._load_image(image_path)
        
        if isDebug:
            print(f"[DEBUG] 画像を読み込みました: {image_path}")
            print(f"[DEBUG] 画像サイズ: {image.shape}")
        
        # YOLOでセグメンテーション実行
        results = self._run_yolo_inference(image, isDebug)
        
        # マスクを生成
        mask = self._create_pizza_mask(image, results, isDebug)
        
        if isDebug:
            debug_path = os.path.join('debug', 'pizza_segmentation_raw.png')
            os.makedirs('debug', exist_ok=True)
            cv2.imwrite(debug_path, mask)
            print(f"[DEBUG] セグメンテーション結果を保存: {debug_path}")
        
        return mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """画像を読み込み"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        return image
    
    def _run_yolo_inference(self, image: np.ndarray, isDebug: bool) -> list:
        """YOLOモデルで推論実行"""
        if isDebug:
            print("[DEBUG] YOLOモデルで推論を実行中...")
        
        results = self.model(image, verbose=False)
        
        if isDebug:
            print(f"[DEBUG] 検出結果数: {len(results)}")
        
        return results
    
    def _create_pizza_mask(self, image: np.ndarray, results: list, isDebug: bool) -> np.ndarray:
        """検出結果からピザのマスクを生成"""
        # 空のマスクを初期化
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        pizza_count = 0
        
        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i, (box, mask_data) in enumerate(zip(result.boxes, result.masks)):
                    class_id = int(box.cls)
                    
                    if class_id == self.pizza_class_id:
                        pizza_count += 1
                        # マスクデータを取得してリサイズ
                        pizza_mask = mask_data.data[0].cpu().numpy()
                        pizza_mask_resized = cv2.resize(
                            pizza_mask, 
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        # バイナリマスクに変換
                        mask[pizza_mask_resized > 0.5] = 255
                        
                        if isDebug:
                            confidence = float(box.conf)
                            print(f"[DEBUG] ピザ {pizza_count} を検出 (信頼度: {confidence:.2f})")
        
        if isDebug:
            print(f"[DEBUG] 検出されたピザの総数: {pizza_count}")
            print(f"[DEBUG] マスクのピクセル数: {np.sum(mask == 255)}")
        
        # 縮小→膨張処理で細くつながっている箇所を断ち切る
        if np.sum(mask == 255) > 0:
            mask = self._apply_morphological_separation(mask, isDebug)
        
        return mask
    
    def _apply_morphological_separation(self, mask: np.ndarray, isDebug: bool = False) -> np.ndarray:
        """
        縮小→膨張処理で細くつながっている箇所を断ち切る
        
        Args:
            mask: 入力バイナリマスク
            isDebug: デバッグモード
            
        Returns:
            処理後のバイナリマスク
        """
        if isDebug:
            print(f"[DEBUG] 形態学的分離処理を開始")
            
        # 元のマスクのコピーを作成
        processed_mask = mask.copy()
        
        # カーネルサイズを設定（細い接続を切るために適切なサイズ）
        kernel_size = 32
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 縮小処理（エロージョン）- 細い接続を断ち切る
        eroded = cv2.erode(processed_mask, kernel, iterations=1)
        
        if isDebug:
            eroded_pixels = np.sum(eroded == 255)
            print(f"[DEBUG] 縮小後のピクセル数: {eroded_pixels}")
            
            # デバッグ画像を保存
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "pizza_mask_eroded.png"), eroded)
        
        # 膨張処理（ダイレーション）- 元のサイズに近づける
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        if isDebug:
            dilated_pixels = np.sum(dilated == 255)
            print(f"[DEBUG] 膨張後のピクセル数: {dilated_pixels}")
            cv2.imwrite(str(debug_dir / "pizza_mask_dilated.png"), dilated)
            
            # 連結成分の数を確認
            num_labels, labels = cv2.connectedComponents(dilated)
            print(f"[DEBUG] 分離後の連結成分数: {num_labels - 1}")  # 背景を除く
        
        return dilated
    
    def save_mask(self, mask: np.ndarray, output_path: str, isDebug: bool = False):
        """
        マスクを画像として保存
        
        Args:
            mask: 保存するマスク
            output_path: 出力先のパス
            isDebug: デバッグモード
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = cv2.imwrite(output_path, mask)
        
        if isDebug:
            if success:
                print(f"[DEBUG] マスクを保存しました: {output_path}")
            else:
                print(f"[DEBUG] マスクの保存に失敗しました: {output_path}")
        
        return success


if __name__ == "__main__":
    # サービスを初期化
    service = PizzaSegmentationService()
    
    # 入出力パスの設定
    input_image = "resource/pizza.jpg"
    output_mask = "result/pizza_mask.png"
    
    try:
        # デバッグモードでピザをセグメンテーション
        print("ピザのセグメンテーションを開始します...")
        mask = service.segment_pizza(input_image, isDebug=True)
        
        print(f"\nマスクの情報:")
        print(f"  - 形状: {mask.shape}")
        print(f"  - ユニーク値: {np.unique(mask)}")
        print(f"  - ピザ領域のピクセル数: {np.sum(mask == 255)}")
        print(f"  - ピザ領域の割合: {np.sum(mask == 255) / mask.size * 100:.1f}%")
        
        # マスクを保存
        service.save_mask(mask, output_mask, isDebug=True)
        print(f"\nセグメンテーションが正常に完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()