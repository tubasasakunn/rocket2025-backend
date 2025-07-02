import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService
from salami_segmentation_service import SalamiSegmentationService
from preprocess import PreprocessService


class SalamiCircleDetectionService:
    """ピザ画像からサラミの円形領域を検出するサービス"""
    
    # 定数定義
    MIN_AREA_THRESHOLD = 200  # 最小面積閾値
    MIN_RADIUS = 12  # 最小半径
    MAX_RADIUS_RATIO = 0.15  # 画像サイズに対する最大半径の比率
    CIRCULARITY_THRESHOLD = 0.3  # 円形度の閾値
    DISTANCE_THRESHOLD_RATIO = 0.5  # 距離変換のピーク検出閾値比率
    
    def __init__(self):
        """サービスの初期化"""
        self.pizza_service = PizzaSegmentationService()
        self.salami_service = SalamiSegmentationService()
        self.preprocess_service = PreprocessService()
    
    def detect_circles_from_mask(
        self, 
        mask: np.ndarray, 
        debug_output_dir: Optional[Path] = None, 
        base_name: str = ""
    ) -> List[Tuple[Tuple[int, int], int]]:
        """
        バイナリマスクから円形領域を検出（縮小処理による分割判定）
        
        Args:
            mask: バイナリマスク（0または255の値）
            debug_output_dir: デバッグ画像を保存するディレクトリ
            base_name: デバッグ画像のベース名
            
        Returns:
            各要素が((中心X座標, 中心Y座標), 半径)のタプルのリスト
        """
        # マスクを8ビットに変換
        mask = self._ensure_uint8(mask)
        
        # 距離変換を実行
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # デバッグ画像の保存
        if debug_output_dir and base_name:
            self._save_distance_transform(dist_transform, debug_output_dir, base_name)
        
        # 連結成分を検出
        num_labels, labels = cv2.connectedComponents(mask)
        
        circles = []
        
        # 各連結成分を処理（ラベル0は背景）
        for label in range(1, num_labels):
            # 現在のラベルの領域のみを抽出
            label_mask = (labels == label).astype(np.uint8) * 255
            label_dist = dist_transform * (labels == label)
            
            # この領域の最大距離値を取得
            max_dist = label_dist.max()
            if max_dist < self.MIN_RADIUS:
                continue
            
            # 縮小処理による分割判定
            sub_circles = self._detect_circles_by_erosion(label_mask, label_dist, max_dist)
            circles.extend(sub_circles)
        
        return circles
    
    def _detect_circles_by_erosion(
        self, 
        mask: np.ndarray, 
        dist_transform: np.ndarray,
        max_dist: float
    ) -> List[Tuple[Tuple[int, int], int]]:
        """縮小処理により領域を分割して円を検出"""
        circles = []
        
        # 最大距離の80%以上の領域を抽出
        threshold = max_dist * 0.8
        high_dist_mask = (dist_transform >= threshold).astype(np.uint8) * 255
        
        # デバッグ用：80%以上の領域を保存
        # if debug_output_dir and base_name:
        #     self._save_debug_image(high_dist_mask, debug_output_dir, base_name, "80%領域")
        
        # 連結成分を検出（80%領域での分割判定）
        num_sub_labels, sub_labels = cv2.connectedComponents(high_dist_mask)
        
        if num_sub_labels <= 1:
            # 分割されなかった場合（背景のみ、または縮小で消えた場合）
            # 元のマスク全体を1つの円として処理
            circle = self._create_circle_from_single_region(mask, dist_transform)
            if circle:
                circles.append(circle)
        else:
            # 複数の領域に分割された場合
            # 各分割領域を個別の円として処理
            for sub_label in range(1, num_sub_labels):
                # 80%領域の各部分に対応する元のマスク領域を抽出
                sub_region_mask = self._expand_region_to_original(
                    mask, sub_labels, sub_label, dist_transform
                )
                
                if sub_region_mask is not None:
                    circle = self._create_circle_from_single_region(
                        sub_region_mask, 
                        dist_transform * (sub_region_mask > 0)
                    )
                    if circle:
                        circles.append(circle)
        
        return circles
    
    def _expand_region_to_original(
        self,
        original_mask: np.ndarray,
        sub_labels: np.ndarray,
        target_label: int,
        dist_transform: np.ndarray
    ) -> Optional[np.ndarray]:
        """80%領域から元のマスク領域に拡張"""
        # 目標ラベルの領域を取得
        seed_region = (sub_labels == target_label).astype(np.uint8) * 255
        
        # この領域の距離変換の最大値位置を記録（後で中心点として使用）
        seed_dist = dist_transform * (seed_region > 0)
        max_point = np.unravel_index(np.argmax(seed_dist), seed_dist.shape)
        
        # 領域成長法で元のマスクまで拡張
        expanded = seed_region.copy()
        kernel = np.ones((3, 3), np.uint8)
        
        # 反復的に拡張
        prev_expanded = None
        iteration = 0
        max_iterations = 100  # 無限ループ防止
        
        while not np.array_equal(expanded, prev_expanded) and iteration < max_iterations:
            prev_expanded = expanded.copy()
            iteration += 1
            
            # 1ピクセル膨張
            dilated = cv2.dilate(expanded, kernel, iterations=1)
            
            # 元のマスク内で拡張
            valid_expansion = (original_mask > 0) & (dilated > expanded)
            
            # 他の高距離値領域との境界をチェック
            for other_label in range(1, sub_labels.max() + 1):
                if other_label != target_label:
                    other_seed = (sub_labels == other_label).astype(np.uint8)
                    if np.any(other_seed):
                        # 他の領域の最大距離点
                        other_dist = dist_transform * other_seed
                        other_max_point = np.unravel_index(np.argmax(other_dist), other_dist.shape)
                        
                        # 拡張候補の各点について、どちらの中心に近いかチェック
                        expansion_points = np.where(valid_expansion)
                        for i in range(len(expansion_points[0])):
                            py, px = expansion_points[0][i], expansion_points[1][i]
                            
                            # 自分の中心までの距離
                            dist_to_self = np.sqrt((px - max_point[1])**2 + (py - max_point[0])**2)
                            # 他の中心までの距離
                            dist_to_other = np.sqrt((px - other_max_point[1])**2 + (py - other_max_point[0])**2)
                            
                            # より近い中心に属するようにする
                            if dist_to_other <= dist_to_self:
                                valid_expansion[py, px] = False
            
            expanded[valid_expansion] = 255
        
        # 領域が小さすぎる場合はNoneを返す
        if np.sum(expanded > 0) < self.MIN_AREA_THRESHOLD:
            return None
        
        return expanded
    
    def _create_circle_from_single_region(
        self, 
        mask: np.ndarray, 
        dist_transform: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """単一領域から円を作成（距離変換の最大値を中心とする）"""
        # 面積チェック
        if np.sum(mask > 0) < self.MIN_AREA_THRESHOLD:
            return None
        
        # 距離変換の最大値位置を円の中心とする
        if dist_transform.max() == 0:
            return None
        
        max_dist_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        center_y, center_x = max_dist_idx
        max_dist_value = dist_transform[max_dist_idx]
        
        # 中心から外縁までの距離を計測して半径を決定
        radius = self._calculate_radius_from_center(mask, center_x, center_y)
        
        if radius is None:
            # フォールバック：距離変換の最大値を半径として使用
            radius = max_dist_value
        
        # 画像サイズに対する最大半径チェック
        img_height, img_width = mask.shape
        max_radius = min(img_height, img_width) * self.MAX_RADIUS_RATIO
        
        if radius < self.MIN_RADIUS or radius > max_radius:
            return None
        
        return ((int(center_x), int(center_y)), int(radius))
    
    def _calculate_radius_from_center(
        self, 
        mask: np.ndarray, 
        center_x: int, 
        center_y: int
    ) -> Optional[float]:
        """中心点から外縁までの距離を計測して半径を計算"""
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 中心点から輪郭上の各点までの距離を計算
        distances = []
        for point in largest_contour:
            px, py = point[0]
            dist = np.sqrt((px - center_x)**2 + (py - center_y)**2)
            distances.append(dist)
        
        if not distances:
            return None
        
        # 半径として平均距離を使用（外れ値の影響を減らすため）
        # または最大距離を使用することも可能
        radius = np.mean(distances)
        
        return radius
    
    def detect_salami_circles(
        self, 
        image_path: str, 
        debug_output_dir: Optional[Path] = None, 
        base_name: str = ""
    ) -> List[Tuple[Tuple[int, int], int]]:
        """
        画像からサラミの円形領域を検出（ピザとサラミのセグメンテーションを組み合わせ）
        
        Args:
            image_path: 入力画像のパス
            debug_output_dir: デバッグ画像を保存するディレクトリ
            base_name: デバッグ画像のベース名
            
        Returns:
            各要素が((中心X座標, 中心Y座標), 半径)のタプルのリスト
        """
        # サラミマスクを取得
        salami_mask = self._get_salami_mask(image_path)
        
        if debug_output_dir and base_name:
            self._save_debug_image(salami_mask, debug_output_dir, base_name, "サラミマスク")
        
        # ピザマスクを取得
        pizza_mask = self.pizza_service.segment_pizza(image_path)
        
        if debug_output_dir and base_name:
            self._save_debug_image(pizza_mask, debug_output_dir, base_name, "ピザマスク")
        
        # マスクを結合（ピザ領域内のサラミのみを抽出）
        final_mask = cv2.bitwise_and(salami_mask, pizza_mask)
        
        if debug_output_dir and base_name:
            self._save_debug_image(final_mask, debug_output_dir, base_name, "結合マスク")
        
        # マスクから円を検出
        circles = self.detect_circles_from_mask(final_mask, debug_output_dir, base_name)
        
        return circles
    
    def draw_circles_on_image(
        self, 
        image: np.ndarray, 
        circles: List[Tuple[Tuple[int, int], int]]
    ) -> np.ndarray:
        """
        検出した円を画像上に描画
        
        Args:
            image: 入力画像
            circles: 描画する円のリスト
            
        Returns:
            円が描画された画像
        """
        result = image.copy()
        
        for (center_x, center_y), radius in circles:
            # 円の輪郭を描画（緑色）
            cv2.circle(result, (center_x, center_y), radius, (0, 255, 0), 2)
            # 中心点を描画（赤色）
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
            # 円の情報をテキストで表示
            self._draw_circle_info(result, center_x, center_y, radius)
        
        return result
    
    # プライベートメソッド（ヘルパー関数）
    
    def _ensure_uint8(self, mask: np.ndarray) -> np.ndarray:
        """マスクを8ビット符号なし整数型に変換"""
        if mask.dtype != np.uint8:
            return mask.astype(np.uint8)
        return mask
    
    def _find_local_maxima(self, dist_transform: np.ndarray, max_dist: float) -> List[Tuple[int, int]]:
        """距離変換の最大値点を検出"""
        # 最大値の90%以上の点を最大値点とみなす
        threshold = max_dist * 0.9
        max_points = np.where(dist_transform >= threshold)
        
        # 座標のリストに変換
        points = list(zip(max_points[1], max_points[0]))  # (x, y)形式
        
        # 近接する点をグループ化（同じ最大値領域として扱う）
        grouped_points = self._group_nearby_points(points)
        
        return grouped_points
    
    def _group_nearby_points(self, points: List[Tuple[int, int]], threshold: int = 10) -> List[Tuple[int, int]]:
        """近接する点をグループ化し、各グループの中心点を返す"""
        if not points:
            return []
        
        groups = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            # 新しいグループを開始
            group = [point]
            used.add(i)
            
            # 近接する点を同じグループに追加
            for j, other_point in enumerate(points):
                if j in used:
                    continue
                
                # グループ内のいずれかの点との距離をチェック
                for group_point in group:
                    dist = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                    if dist < threshold:
                        group.append(other_point)
                        used.add(j)
                        break
            
            # グループの中心点を計算
            center_x = int(np.mean([p[0] for p in group]))
            center_y = int(np.mean([p[1] for p in group]))
            groups.append((center_x, center_y))
        
        return groups
    
    def _split_by_maxima(
        self, 
        mask: np.ndarray, 
        dist_transform: np.ndarray, 
        max_points: List[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], int]]:
        """複数の最大値点に基づいて領域を分割し、円を検出"""
        circles = []
        
        # 各最大値点を中心とした領域を作成
        for point in max_points:
            # この点を中心とした領域を抽出
            sub_mask = self._create_region_around_point(mask, dist_transform, point)
            
            if sub_mask is not None:
                circle = self._create_circle_from_mask(sub_mask, dist_transform * (sub_mask > 0))
                if circle:
                    circles.append(circle)
        
        return circles
    
    def _create_region_around_point(
        self, 
        mask: np.ndarray, 
        dist_transform: np.ndarray, 
        center: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """指定した点を中心とした領域を作成"""
        # 中心点の距離値を取得
        center_dist = dist_transform[center[1], center[0]]
        
        # 領域成長法で領域を抽出
        region = np.zeros_like(mask)
        region[center[1], center[0]] = 255
        
        # 距離変換の勾配に基づいて領域を拡張
        kernel = np.ones((3, 3), np.uint8)
        prev_region = None
        
        while not np.array_equal(region, prev_region):
            prev_region = region.copy()
            dilated = cv2.dilate(region, kernel, iterations=1)
            
            # マスク内かつ距離値が減少する方向のみ拡張
            valid_expansion = (mask > 0) & (dilated > region) & (dist_transform > center_dist * 0.3)
            region[valid_expansion] = 255
        
        # 領域が小さすぎる場合はNoneを返す
        if np.sum(region > 0) < self.MIN_AREA_THRESHOLD:
            return None
        
        return region
    
    def _create_circle_from_mask(
        self, 
        mask: np.ndarray, 
        dist_transform: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """マスクと距離変換から円を作成"""
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 最小外接円を計算
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # 距離変換の最大値も考慮
        max_dist_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        dist_based_radius = dist_transform[max_dist_idx]
        
        # より適切な半径を選択（距離変換ベースと外接円の平均）
        final_radius = (radius + dist_based_radius) / 2
        
        # 画像サイズに対する最大半径チェック
        img_height, img_width = mask.shape
        max_radius = min(img_height, img_width) * self.MAX_RADIUS_RATIO
        
        if final_radius < self.MIN_RADIUS or final_radius > max_radius:
            return None
        
        return ((int(x), int(y)), int(final_radius))
    
    def _get_salami_mask(self, image_path: str) -> np.ndarray:
        """サラミのセグメンテーション結果を取得"""
        salami_result = self.salami_service.segment_salami(image_path)
        
        # 戻り値の型に応じて処理
        if isinstance(salami_result, tuple):
            salami_mask, _ = salami_result
        else:
            salami_mask = salami_result
        
        return salami_mask
    
    def _draw_circle_info(
        self, 
        image: np.ndarray, 
        center_x: int, 
        center_y: int, 
        radius: int
    ) -> None:
        """円の情報をテキストで画像に描画"""
        text = f"({center_x}, {center_y}) r={radius}"
        text_position = (center_x - 50, center_y - radius - 10)
        cv2.putText(
            image, text, text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    # デバッグ用のヘルパーメソッド
    
    def _save_debug_image(
        self, 
        image: np.ndarray, 
        output_dir: Path, 
        base_name: str, 
        suffix: str
    ) -> None:
        """デバッグ画像を保存"""
        file_path = output_dir / f"{base_name}_{suffix}.png"
        cv2.imwrite(str(file_path), image)
    
    def _save_distance_transform(
        self, 
        dist_transform: np.ndarray, 
        output_dir: Path, 
        base_name: str
    ) -> None:
        """距離変換の可視化画像を保存"""
        # 可視化用に正規化
        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_colored = cv2.applyColorMap(dist_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        self._save_debug_image(dist_colored, output_dir, base_name, "距離変換")
    
    def _save_watershed_visualization(
        self, 
        markers: np.ndarray, 
        shape: Tuple[int, int], 
        output_dir: Path, 
        base_name: str
    ) -> None:
        """分割結果の可視化画像を保存"""
        # 各領域に異なる色を割り当て
        visualization = np.zeros((*shape, 3), dtype=np.uint8)
        
        # 連結成分のラベル数を取得
        num_labels = int(markers.max())
        
        for label in range(1, num_labels + 1):
            # ランダムな色を生成
            color = np.random.randint(50, 255, size=3).tolist()
            visualization[markers == label] = color
        
        self._save_debug_image(visualization, output_dir, base_name, "領域分割")


def process_images() -> None:
    """メイン処理：リソースディレクトリ内のすべての画像を処理"""
    # サービスのインスタンスを作成
    service = SalamiCircleDetectionService()
    
    # ディレクトリの設定
    resource_dir = Path("resource")
    output_dir = Path("debug/salami_circle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルを検索
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("リソースディレクトリに画像ファイルが見つかりません")
        return
    
    print(f"{len(image_files)}個の画像を処理します")
    
    for image_path in image_files:
        print(f"\n{image_path.name}を処理中...")
        
        try:
            process_single_image(service, image_path, output_dir)
        except Exception as e:
            print(f"  {image_path.name}の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n処理が完了しました。結果はdebug/salami_circle/に保存されています。")


def process_single_image(
    service: SalamiCircleDetectionService, 
    image_path: Path, 
    output_dir: Path
) -> None:
    """単一の画像を処理"""
    base_name = image_path.stem
    
    # ステップ1: 画像の前処理
    temp_preprocessed_path = output_dir / f"temp_preprocessed_{base_name}.jpg"
    preprocessed_image, info = service.preprocess_service.preprocess_pizza_image(
        str(image_path), str(temp_preprocessed_path)
    )
    
    if info['is_transformed']:
        print(f"  変換を適用しました")
        processed_image_path = str(temp_preprocessed_path)
    else:
        print("  変換は不要でした")
        processed_image_path = str(image_path)
    
    # ステップ2: サラミの円を検出
    circles = service.detect_salami_circles(processed_image_path, output_dir, base_name)
    
    # ステップ3: 表示用の画像を準備
    display_image = preprocessed_image if info['is_transformed'] else cv2.imread(str(image_path))
    
    # ステップ4: 結果を保存
    save_results(service, display_image, circles, output_dir, base_name)
    
    # 一時ファイルをクリーンアップ
    if temp_preprocessed_path.exists():
        temp_preprocessed_path.unlink()
    
    # 結果を表示
    print_detection_results(circles, base_name)


def save_results(
    service: SalamiCircleDetectionService,
    display_image: np.ndarray,
    circles: List[Tuple[Tuple[int, int], int]],
    output_dir: Path,
    base_name: str
) -> None:
    """処理結果を保存"""
    # 前処理済み画像を保存
    preprocessed_save_path = output_dir / f"{base_name}_preprocessed.jpg"
    cv2.imwrite(str(preprocessed_save_path), display_image)
    
    # 円検出結果を保存
    circle_image = service.draw_circles_on_image(display_image, circles)
    circle_result_path = output_dir / f"{base_name}_circles.jpg"
    cv2.imwrite(str(circle_result_path), circle_image)


def print_detection_results(
    circles: List[Tuple[Tuple[int, int], int]], 
    base_name: str
) -> None:
    """検出結果をコンソールに出力"""
    print(f"  {len(circles)}個のサラミの円を検出しました:")
    for i, ((cx, cy), r) in enumerate(circles):
        print(f"    円{i+1}: 中心=({cx}, {cy}), 半径={r}")
    
    print(f"  結果はdebug/salami_circle/に保存されました")


if __name__ == "__main__":
    process_images()