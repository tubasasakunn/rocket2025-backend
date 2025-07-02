import cv2
import numpy as np
from typing import Tuple, Optional
import os

try:
    from .pizza_segmentation_service import PizzaSegmentationService
except ImportError:
    from pizza_segmentation_service import PizzaSegmentationService


class PizzaCircleDetectionService:
    """ピザを円として検出するサービスクラス"""
    
    def __init__(self):
        """ピザセグメンテーションサービスを初期化"""
        self.segmentation_service = PizzaSegmentationService()
    
    def detect_circle_from_image(self, image_path: str, isDebug: bool = False) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        画像からピザの円を検出し、中心座標と半径を返す
        
        Args:
            image_path: 入力画像のパス
            isDebug: デバッグモード（Trueの場合、途中経過を保存）
            
        Returns:
            ((center_x, center_y), radius) または None（円が見つからない場合）
        """
        if isDebug:
            print(f"[DEBUG] 画像からピザの円を検出します: {image_path}")
        
        # セグメンテーションマスクを取得
        mask = self.segmentation_service.segment_pizza(image_path, isDebug=False)
        
        if isDebug:
            # マスクを保存
            debug_mask_path = os.path.join('debug', 'pizza_circle_mask.png')
            os.makedirs('debug', exist_ok=True)
            cv2.imwrite(debug_mask_path, mask)
            print(f"[DEBUG] セグメンテーションマスクを保存: {debug_mask_path}")
        
        # マスクから円を検出
        return self.detect_circle_from_mask(mask, isDebug)
    
    def detect_circle_from_mask(self, mask: np.ndarray, isDebug: bool = False) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        セグメンテーションマスクから円を検出
        
        Args:
            mask: バイナリマスク (255 = ピザ部分, 0 = 背景)
            isDebug: デバッグモード
            
        Returns:
            ((center_x, center_y), radius) または None（円が見つからない場合）
        """
        # 輪郭を検出
        contours = self._find_contours(mask, isDebug)
        
        if not contours:
            if isDebug:
                print("[DEBUG] 輪郭が見つかりませんでした")
            return None
        
        # 最大の輪郭を選択
        largest_contour = self._find_largest_contour(contours, isDebug)
        
        # 最小外接円を計算
        circle_info = self._compute_min_enclosing_circle(largest_contour, isDebug)
        
        return circle_info
    
    def _find_contours(self, mask: np.ndarray, isDebug: bool) -> list:
        """マスクから輪郭を検出"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if isDebug:
            print(f"[DEBUG] {len(contours)}個の輪郭を検出しました")
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                print(f"[DEBUG]   輪郭 {i+1}: 面積 = {area:.0f} ピクセル")
        
        return contours
    
    def _find_largest_contour(self, contours: list, isDebug: bool) -> np.ndarray:
        """最大の輪郭を選択"""
        largest_contour = max(contours, key=cv2.contourArea)
        
        if isDebug:
            area = cv2.contourArea(largest_contour)
            print(f"[DEBUG] 最大輪郭の面積: {area:.0f} ピクセル")
        
        return largest_contour
    
    def _compute_min_enclosing_circle(self, contour: np.ndarray, isDebug: bool) -> Tuple[Tuple[int, int], int]:
        """最小外接円を計算"""
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # 整数に変換
        center = (int(center_x), int(center_y))
        radius = int(radius)
        
        if isDebug:
            print(f"[DEBUG] 最小外接円:")
            print(f"[DEBUG]   中心: {center}")
            print(f"[DEBUG]   半径: {radius} ピクセル")
            
            # 円形度を計算（参考値）
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                print(f"[DEBUG]   円形度: {circularity:.3f} (1.0 = 完全な円)")
        
        return (center, radius)
    
    def draw_circle_on_image(self, image: np.ndarray, center: Tuple[int, int], radius: int, isDebug: bool = False) -> np.ndarray:
        """
        画像に円を描画
        
        Args:
            image: 入力画像
            center: 円の中心座標 (x, y)
            radius: 円の半径
            isDebug: デバッグモード
            
        Returns:
            円が描画された画像
        """
        # 画像をコピー
        result = image.copy()
        
        # 円を描画
        self._draw_circle(result, center, radius)
        
        # 中心点を描画
        self._draw_center_point(result, center)
        
        # 情報テキストを追加
        self._add_info_text(result, center, radius)
        
        if isDebug:
            print(f"[DEBUG] 円を描画しました:")
            print(f"[DEBUG]   中心: {center}")
            print(f"[DEBUG]   半径: {radius}")
        
        return result
    
    def _draw_circle(self, image: np.ndarray, center: Tuple[int, int], radius: int):
        """円の輪郭を描画"""
        # 緑色、線幅3ピクセル
        cv2.circle(image, center, radius, (0, 255, 0), 3)
    
    def _draw_center_point(self, image: np.ndarray, center: Tuple[int, int]):
        """中心点を描画"""
        # 赤色、半径5ピクセルの塗りつぶし
        cv2.circle(image, center, 5, (0, 0, 255), -1)
    
    def _add_info_text(self, image: np.ndarray, center: Tuple[int, int], radius: int):
        """情報テキストを追加"""
        text = f"中心: {center}, 半径: {radius}px"
        # テキストの背景を追加（読みやすくするため）
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (5, 5), (15 + text_size[0], 35), (255, 255, 255), -1)
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)


if __name__ == "__main__":
    import glob
    
    # サービスのインスタンスを作成
    service = PizzaCircleDetectionService()
    
    # resourceディレクトリ内のすべての画像を処理
    resource_dir = "resource"
    output_dir = "debug/pizza_circle"
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイルを取得
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(resource_dir, ext)))
    
    if not image_files:
        print(f"{resource_dir}に画像ファイルが見つかりません")
    else:
        print(f"{len(image_files)}個の画像ファイルが見つかりました:")
        for img_file in image_files:
            print(f"  {img_file}")
    
    # 各画像を処理
    for input_image_path in image_files:
        try:
            # ファイル名を取得
            filename = os.path.basename(input_image_path)
            name, ext = os.path.splitext(filename)
            preprocessed_image_path = os.path.join(output_dir, f"{name}_preprocessed{ext}")
            output_image_path = os.path.join(output_dir, f"{name}_circle{ext}")
            
            print(f"\nProcessing: {input_image_path}")
            
            # 前処理を実行
            print("  前処理を実行中...")
            preprocess_service = None
            try:
                from .preprocess import PreprocessService
            except ImportError:
                from preprocess import PreprocessService
            
            if preprocess_service is None:
                preprocess_service = PreprocessService()
            
            preprocessed_image, preprocess_info = preprocess_service.preprocess_pizza_image(
                input_image_path, preprocessed_image_path, isDebug=True
            )
            
            if preprocessed_image is None:
                print(f"  エラー: {filename}の前処理に失敗しました")
                continue
            
            print(f"  前処理が完了しました:")
            print(f"    情報: {preprocess_info}")
            
            # 前処理された画像から円を検出
            circle_info = service.detect_circle_from_image(preprocessed_image_path, isDebug=True)
            
            if circle_info is None:
                print(f"  {filename}で円が検出されませんでした")
                continue
            
            center, radius = circle_info
            print(f"  円を検出しました:")
            print(f"    中心: {center}")
            print(f"    半径: {radius}")
            
            # 前処理された画像を読み込み
            processed_image = cv2.imread(preprocessed_image_path)
            
            if processed_image is None:
                print(f"  エラー: 前処理された画像を読み込めませんでした {preprocessed_image_path}")
                continue
            
            # 円をオーバーレイ
            result_image = service.draw_circle_on_image(processed_image, center, radius, isDebug=True)
            
            # 結果を保存
            cv2.imwrite(output_image_path, result_image)
            print(f"  結果を保存しました: {output_image_path}")
            
        except Exception as e:
            print(f"  {input_image_path}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n処理が完了しました。")