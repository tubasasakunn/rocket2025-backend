import cv2
import numpy as np
from pathlib import Path
import sys
import os
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService
from preprocess import PreprocessService


class SalamiSegmentationService:
    # ===== 処理パラメータ定数 =====
    # カラー検出パラメータ
    BILATERAL_D = 9                      # バイラテラルフィルタの直径
    BILATERAL_SIGMA_COLOR = 50           # カラーシグマ値
    BILATERAL_SIGMA_SPACE = 50           # スペースシグマ値
    CLAHE_CLIP_LIMIT = 2.0              # CLAHE制限値
    CLAHE_TILE_GRID_SIZE = (8, 8)       # CLAHEタイルサイズ
    
    # HSV色検出の許容値
    HUE_TOLERANCE = 15                   # 色相の許容範囲
    SATURATION_TOLERANCE = 50            # 彩度の許容範囲
    VALUE_TOLERANCE = 50                 # 明度の許容範囲
    
    # 基本形態学的処理パラメータ
    BASIC_KERNEL_SIZE = 32             # 基本カーネルサイズ
    
 



    # サラミ直接検出用のHSV範囲
    H_MIN = 165  # 色相の最小値（330°÷2 = 紫赤の開始）
    H_MAX = 10   # 色相の最大値（20°÷2 = 赤の終了、0をまたぐ）
    S_MIN = 30   # 彩度の最小値（低めに設定）
    S_MAX = 255  # 彩度の最大値（制限なし）
    V_MIN = 50   # 明度の最小値（暗めのサラミも検出）
    V_MAX = 140  # 明度の最大値（明るすぎる部分は除外）
    
    # 面積フィルタリング
    MIN_CONTOUR_AREA = 1500              # サラミ片の最小面積（ピクセル）
    
    # サラミ色サンプル（RGB値：ヒストグラム平坦化前）
    PIZZA1_SALAMI_COLORS = [(116,53,45), (129,68,61), (104,51,37), (168,114,104)]
    PIZZA2_SALAMI_COLORS = [(153,81,81), (164,83,75), (164,88,78), (163,94,94)]
    
    def __init__(self):
        pass

    def _apply_preprocessing(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """画像前処理：ガウスフィルタ + バイラテラルフィルタ + ヒストグラム平坦化（マスク対応）"""
        
        # 1. ガウスフィルタでノイズ除去（最初のぼかし処理）
        # 注：クラス定数として以下を追加することを推奨
        # GAUSSIAN_KERNEL_SIZE = (5, 5)  # ガウスフィルタのカーネルサイズ
        # GAUSSIAN_SIGMA = 1.5            # ガウスフィルタの標準偏差
        gaussian_kernel_size = (25, 25)  # カーネルサイズ（奇数値を使用）
        gaussian_sigma = 3           # 標準偏差
        blurred = cv2.GaussianBlur(image, gaussian_kernel_size, gaussian_sigma)
        
        # 2. バイラテラルフィルタでエッジ保持平滑化
        smoothed = cv2.bilateralFilter(blurred, self.BILATERAL_D, 
                                    self.BILATERAL_SIGMA_COLOR, self.BILATERAL_SIGMA_SPACE)
        
        # 3. マスク対応のヒストグラム平坦化
        if mask is None:
            # マスクがない場合は画像全体に適用
            b, g, r = cv2.split(smoothed)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, 
                                tileGridSize=self.CLAHE_TILE_GRID_SIZE)
            b_eq = clahe.apply(b)
            g_eq = clahe.apply(g)
            r_eq = clahe.apply(r)
            return cv2.merge([b_eq, g_eq, r_eq])
        else:
            # マスク領域のみヒストグラム平坦化
            result = smoothed.copy()
            
            # マスク領域の境界ボックスを取得
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return result
                
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            # ROI抽出
            roi = smoothed[min_y:max_y+1, min_x:max_x+1]
            roi_mask = mask[min_y:max_y+1, min_x:max_x+1]
            
            # 各チャンネルに対してマスク付きCLAHE適用
            b, g, r = cv2.split(roi)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, 
                                tileGridSize=self.CLAHE_TILE_GRID_SIZE)
            
            b_eq = clahe.apply(b)
            g_eq = clahe.apply(g)
            r_eq = clahe.apply(r)
            
            roi_eq = cv2.merge([b_eq, g_eq, r_eq])
            
            # マスク領域のみ結果に反映
            result[min_y:max_y+1, min_x:max_x+1][roi_mask > 0] = roi_eq[roi_mask > 0]
            
            return result

    def _calculate_hsv_color_range(self, rgb_colors: list) -> tuple:
        """RGB色サンプルから前処理済みHSV範囲を計算"""
        hsv_values = []
        
        for r, g, b in rgb_colors:
            # 小さなパッチを作成（BGR形式）
            rgb_patch = np.full((10, 10, 3), [b, g, r], dtype=np.uint8)
            
            # 検出パイプラインと同じ前処理を適用
            processed_patch = self._apply_preprocessing(rgb_patch)
            
            # HSV変換
            hsv_patch = cv2.cvtColor(processed_patch, cv2.COLOR_BGR2HSV)
            # 中央ピクセルの値を取得
            hsv_values.append(hsv_patch[5, 5])
        
        hsv_array = np.array(hsv_values)
        
        # 最小・最大値を計算
        h_min, h_max = hsv_array[:, 0].min(), hsv_array[:, 0].max()
        s_min, s_max = hsv_array[:, 1].min(), hsv_array[:, 1].max()
        v_min, v_max = hsv_array[:, 2].min(), hsv_array[:, 2].max()
        
        print(f"  元のRGB色: {rgb_colors}")
        print(f"  処理後HSV値: {hsv_array}")
        
        # 許容値を追加
        h_min_adj = int(h_min) - self.HUE_TOLERANCE
        h_max_adj = int(h_max) + self.HUE_TOLERANCE
        s_min = max(0, int(s_min) - self.SATURATION_TOLERANCE)
        s_max = min(255, int(s_max) + self.SATURATION_TOLERANCE)
        v_min = max(0, int(v_min) - self.VALUE_TOLERANCE)
        v_max = min(255, int(v_max) + self.VALUE_TOLERANCE)
        
        return (h_min_adj, h_max_adj), (s_min, s_max), (v_min, v_max)

    def _create_color_mask(self, hsv_image: np.ndarray, h_range: tuple, s_range: tuple, v_range: tuple) -> np.ndarray:
        """HSV画像から色マスクを作成（色相のラップアラウンド対応）"""
        h_min, h_max = h_range
        s_min, s_max = s_range
        v_min, v_max = v_range
        
        # 色相が0をまたぐ場合の判定
        if h_min > h_max:
            # 2つのマスクを作成して結合
            # マスク1: h_min～179の範囲
            lower1 = np.array([h_min, s_min, v_min])
            upper1 = np.array([179, s_max, v_max])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            
            # マスク2: 0～h_maxの範囲
            lower2 = np.array([0, s_min, v_min])
            upper2 = np.array([h_max, s_max, v_max])
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # 通常の範囲（0をまたがない場合）
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # マスクを反転しない（サラミ部分が白になる）
        return mask

    def _fill_holes_only(self, mask: np.ndarray) -> np.ndarray:
        """穴だけを埋める（領域を接続しない）"""
        # 輪郭を階層付きで検出
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 結果用のマスクをコピー
        filled_mask = mask.copy()
        
        # 階層情報がある場合
        if hierarchy is not None:
            for i in range(len(contours)):
                # 内側の輪郭（穴）の場合
                if hierarchy[0][i][3] != -1:  # 親輪郭がある = 穴
                    # この穴を白で塗りつぶす
                    cv2.drawContours(filled_mask, contours, i, 255, -1)
        
        return filled_mask

    def _apply_basic_morphology(self, mask: np.ndarray) -> np.ndarray:
        """基本的な形態学的処理（穴埋め・ノイズ除去）"""
        # 先に穴だけを埋める
        mask = self._fill_holes_only(mask)
        
        # その後ノイズ除去
        kernel = np.ones((self.BASIC_KERNEL_SIZE, self.BASIC_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def _separate_by_erosion(self, mask: np.ndarray) -> np.ndarray:
        # より小さいカーネルで段階的に処理
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16,16))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32,32))
        
        # 段階的な侵食
        eroded = cv2.erode(mask, kernel_small, iterations=2)
        eroded = cv2.erode(eroded, kernel_large, iterations=1)
        
        # 各領域を処理
        num_labels, labels = cv2.connectedComponents(eroded)
        result = np.zeros_like(mask)
        
        for label in range(1, num_labels):
            region_mask = (labels == label).astype(np.uint8) * 255
            
            # 条件付き膨張（元のマスクに制限される）
            dilated = cv2.dilate(region_mask, kernel_large, iterations=1)
            dilated = cv2.dilate(dilated, kernel_small, iterations=2)
            
            # 元のマスクとの交差部分のみ取得
            dilated = cv2.bitwise_and(dilated, mask)
            result = cv2.bitwise_or(result, dilated)
        
        return result

    def _filter_by_area(self, mask: np.ndarray) -> np.ndarray:
        """面積による輪郭フィルタリング"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_CONTOUR_AREA:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask

    def detect_salami_by_color(self, image: np.ndarray, pizza_mask: np.ndarray, debug_output_dir: Path = None, base_name: str = "") -> tuple:
        """ピザ領域内でサラミ色検出によるセグメンテーション"""
        # 1. ピザマスクを適用して検出範囲を限定
        masked_image = cv2.bitwise_and(image, image, mask=pizza_mask)
        
        # 2. 前処理（バイラテラルフィルタ + ヒストグラム平坦化）
        preprocessed = self._apply_preprocessing(masked_image,pizza_mask)
        
        # 3. HSV色空間変換
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        
        # 4. サラミ色範囲の計算
        all_salami_colors = self.PIZZA1_SALAMI_COLORS + self.PIZZA2_SALAMI_COLORS
        

        # デバッグ用：色検出前の画像を保存
        if debug_output_dir and base_name:
            preprocessed_path = debug_output_dir / f"{base_name}_色相変換済み.png"
            cv2.imwrite(str(preprocessed_path), preprocessed)
        
        # 5. 色マスクの作成
        color_mask = cv2.bitwise_and(self._create_color_mask(hsv, (self.H_MIN,self.H_MAX),
                                                              (self.S_MIN,self.S_MAX), 
                                                              (self.V_MIN,self.V_MAX)), pizza_mask)
    
        
        # デバッグ用：色検出後のマスクを保存（サラミが白）
        if debug_output_dir and base_name:
            color_mask_path = debug_output_dir / f"{base_name}_色検出マスク.png"
            cv2.imwrite(str(color_mask_path), color_mask)
        
        # 6. 基本形態学的処理（元の色マスクを使用）
        processed_mask = self._apply_basic_morphology(color_mask)

        # デバッグ用：基本形態学的処理後のマスクを保存
        if debug_output_dir and base_name:
            processed_mask_path = debug_output_dir / f"{base_name}_基本形態学的処理後マスク.png"
            cv2.imwrite(str(processed_mask_path), processed_mask)
        
        # 7. 接触しているオブジェクトの分離
        # processed_mask = self._separate_by_erosion(processed_mask)

        # デバッグ用：分離後のマスクを保存
        if debug_output_dir and base_name:
            separated_mask_path = debug_output_dir / f"{base_name}_分離後マスク.png"
            cv2.imwrite(str(separated_mask_path), processed_mask)
        
        # 8. 面積フィルタリング
        filtered_mask = self._filter_by_area(processed_mask)
        
        # デバッグ用：フィルタリング後のマスクを保存
        if debug_output_dir and base_name:
            filtered_mask_path = debug_output_dir / f"{base_name}_フィルタリング後マスク.png"
            cv2.imwrite(str(filtered_mask_path), filtered_mask)
        
        # 9. ピザマスクを適用して境界確定（サラミ=白、背景=黒）
        final_mask = cv2.bitwise_and(filtered_mask, pizza_mask)
        
        # 中間結果も返す（色マスクは反転版を保存）
        intermediate_masks = {
            'color_mask': color_mask,
            'filtered_mask': filtered_mask
        }
        
        return final_mask, intermediate_masks
    
    def segment_salami(self, image_path: str, pizza_mask: np.ndarray = None, debug_output_dir: Path = None, base_name: str = "") -> np.ndarray:
        """画像からサラミをセグメンテーション（メイン関数）"""
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像が読み込めません: {image_path}")
        
        
        # ピザマスクが未提供の場合は取得
        if pizza_mask is None:
            from pizza_segmentation_service import PizzaSegmentationService
            pizza_service = PizzaSegmentationService()
            pizza_mask = pizza_service.segment_pizza(image_path)
        
        # デバッグ情報付きで実行
        if debug_output_dir and base_name:
            final_mask, intermediate_masks = self.detect_salami_by_color(image, pizza_mask, debug_output_dir, base_name)
            return final_mask, intermediate_masks
        else:
            # 通常実行（後方互換性維持）
            final_mask, _ = self.detect_salami_by_color(image, pizza_mask)
            return final_mask
    
    def save_mask(self, mask: np.ndarray, output_path: str):
        """セグメンテーションマスクを保存"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    # サービス初期化
    salami_service = SalamiSegmentationService()
    pizza_service = PizzaSegmentationService()
    preprocess_service = PreprocessService()
    
    # ディレクトリ設定
    resource_dir = Path("resource")
    output_dir = Path("debug/salami_segment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("resourceディレクトリに画像ファイルが見つかりません")
        exit(1)
    
    print(f"{len(image_files)}個の画像ファイルが見つかりました")
    
    # 各画像を処理
    for image_path in image_files:
        print(f"\n{image_path.name}を処理中...")
        
        try:
            # 1. 前処理
            temp_preprocessed_path = output_dir / f"temp_preprocessed_{image_path.stem}.jpg"
            preprocessed_image, info = preprocess_service.preprocess_pizza_image(str(image_path), str(temp_preprocessed_path))
            
            if info['is_transformed']:
                print(f"  楕円変換を適用しました")
                processed_image_path = str(temp_preprocessed_path)
            else:
                print("  変換は不要でした")
                processed_image_path = str(image_path)
            
            # 2. ピザマスク取得
            pizza_mask = pizza_service.segment_pizza(processed_image_path)
            
            # 3. ファイル名ベース設定
            base_name = image_path.stem
            
            # 4. サラミセグメンテーション実行（デバッグ情報付き）
            result = salami_service.segment_salami(processed_image_path, pizza_mask, output_dir, base_name)
            if isinstance(result, tuple):
                salami_mask, intermediate_masks = result
            else:
                salami_mask = result
                intermediate_masks = None
            
            # 5. 表示用画像設定
            if info['is_transformed']:
                display_image = preprocessed_image.copy()
            else:
                display_image = cv2.imread(str(image_path))
            
            # 6. オーバーレイ可視化作成
            overlay = display_image.copy()
            overlay[salami_mask == 255] = [0, 0, 255]  # サラミ領域を赤で表示
            result_overlay = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
            
            # 7. 結果保存
            
            # 前処理済み画像
            preprocessed_save_path = output_dir / f"{base_name}_前処理済み.jpg"
            cv2.imwrite(str(preprocessed_save_path), display_image)
            
            # ピザマスク
            pizza_mask_path = output_dir / f"{base_name}_ピザマスク.png"
            cv2.imwrite(str(pizza_mask_path), pizza_mask)
            
            # サラミマスク
            salami_mask_path = output_dir / f"{base_name}_サラミマスク.png"
            cv2.imwrite(str(salami_mask_path), salami_mask)
            
            # オーバーレイ画像
            overlay_path = output_dir / f"{base_name}_オーバーレイ.jpg"
            cv2.imwrite(str(overlay_path), result_overlay)
            
            # 一時ファイル削除
            if temp_preprocessed_path.exists():
                temp_preprocessed_path.unlink()
            
            # 統計情報表示
            print(f"  統計情報:")
            print(f"    ピザマスクピクセル数: {np.sum(pizza_mask == 255):,}")
            print(f"    サラミマスクピクセル数: {np.sum(salami_mask == 255):,}")
            
            print(f"  結果をdebug/salami_segment/に保存しました:")
            print(f"    - 前処理済み画像: {preprocessed_save_path.name}")
            print(f"    - ピザマスク: {pizza_mask_path.name}")
            print(f"    - サラミマスク: {salami_mask_path.name}")
            print(f"    - オーバーレイ: {overlay_path.name}")
            
            # 中間マスクの保存情報も表示
            if intermediate_masks:
                print(f"    - 色検出マスク: {base_name}_色検出マスク.png")
                print(f"    - フィルタリング後マスク: {base_name}_フィルタリング後マスク.png")
            
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            continue
    
    print(f"\n処理完了。debug/salami_segment/で結果を確認してください。")