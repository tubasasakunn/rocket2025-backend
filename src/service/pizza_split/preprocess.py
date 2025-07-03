import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService


class PreprocessService:
    def __init__(self):
        self.segmentation_service = PizzaSegmentationService()
        self.ellipse_threshold = 0.1  # 楕円判定の閾値：長軸と短軸の差が10%以上
    
    def detect_ellipse_from_mask(self, mask: np.ndarray, is_debug: bool = False) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        セグメンテーションマスクから楕円を検出
        
        Args:
            mask: バイナリマスク (255 = ピザ, 0 = 背景)
            is_debug: デバッグモードフラグ
            
        Returns:
            ((中心x座標, 中心y座標), (長軸半径, 短軸半径), 角度) または None
        """
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if is_debug:
                print("[DEBUG] 輪郭が検出されませんでした")
            return None
        
        # 最大の輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        
        if is_debug:
            print(f"[DEBUG] 検出された輪郭数: {len(contours)}")
            print(f"[DEBUG] 最大輪郭の点数: {len(largest_contour)}")
        
        # 輪郭が5点以上ある場合に楕円をフィッティング
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center_x, center_y), (width, height), angle = ellipse
            
            # cv2.fitEllipseでは：width が第一軸、height が第二軸に対応
            # angle は第一軸の水平からの回転角度を表す
            # 長軸が常により長い軸となるように調整し、角度も対応させる
            if width >= height:
                # width が長軸、height が短軸
                major_axis = width / 2
                minor_axis = height / 2
                # angle はすでに長軸の角度を表している
                major_axis_angle = angle
            else:
                # height が長軸、width が短軸
                major_axis = height / 2
                minor_axis = width / 2
                # 長軸は与えられた角度に対して垂直
                major_axis_angle = angle + 90
                if major_axis_angle >= 180:
                    major_axis_angle -= 180
            
            if is_debug:
                print(f"[DEBUG] 楕円フィッティング結果: 中心=({center_x:.1f}, {center_y:.1f}), 長軸={major_axis:.1f}, 短軸={minor_axis:.1f}, 角度={major_axis_angle:.1f}°")
            
            return ((int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), major_axis_angle)
        
        return None
    
    def is_elliptical(self, major_axis: int, minor_axis: int, is_debug: bool = False) -> bool:
        """
        軸の比率に基づいて形状が楕円かどうかを判定
        
        Args:
            major_axis: 長軸の長さ
            minor_axis: 短軸の長さ
            is_debug: デバッグモードフラグ
            
        Returns:
            楕円の場合True、円形の場合False
        """
        if minor_axis == 0:
            return True
        
        ratio = (major_axis - minor_axis) / major_axis
        is_ellipse = ratio > self.ellipse_threshold
        
        if is_debug:
            print(f"[DEBUG] 楕円判定: 長軸/短軸比率={ratio:.3f}, 閾値={self.ellipse_threshold}, 楕円={is_ellipse}")
        
        return is_ellipse
    
    def calculate_transform_scale(self, major_axis: int, minor_axis: int, is_debug: bool = False) -> Tuple[float, float]:
        """
        楕円を円に変換するためのスケール係数を計算
        短軸に合わせてx軸（長軸）のみをスケーリング
        
        Args:
            major_axis: 長軸の長さ（回転後は水平になる）
            minor_axis: 短軸の長さ（回転後は垂直になる）
            is_debug: デバッグモードフラグ
            
        Returns:
            (スケールx, スケールy) の係数
        """
        # x軸（長軸）のみをy軸（短軸）に合わせてスケーリング
        # より長い寸法を縮小することで楕円を円形にする
        scale_x = minor_axis / major_axis  # x軸をy軸に合わせて縮小
        scale_y = 1.0  # y軸は変更なし
        
        if is_debug:
            print(f"[DEBUG] スケール係数: x={scale_x:.4f}, y={scale_y:.4f}")
        
        return (scale_x, scale_y)
    
    def _create_transformation_matrix(self, center: Tuple[int, int], angle: float, scale: Tuple[float, float], is_debug: bool = False) -> np.ndarray:
        """
        変換行列を作成する内部メソッド
        
        Args:
            center: 中心座標 (x, y)
            angle: 回転角度（度）
            scale: スケール係数 (scale_x, scale_y)
            is_debug: デバッグモードフラグ
            
        Returns:
            2x3の変換行列
        """
        center_x, center_y = center
        scale_x, scale_y = scale
        
        # 原点への平行移動
        M1 = np.array([[1, 0, -center_x],
                       [0, 1, -center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # 軸に合わせるための回転（逆回転）
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        M2 = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # 円形にするためのスケーリング
        M3 = np.array([[scale_x, 0, 0],
                       [0, scale_y, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # 元の向きに戻す回転
        angle_rad_back = np.radians(angle)
        cos_b = np.cos(angle_rad_back)
        sin_b = np.sin(angle_rad_back)
        M4 = np.array([[cos_b, -sin_b, 0],
                       [sin_b, cos_b, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # 元の位置への平行移動
        M5 = np.array([[1, 0, center_x],
                       [0, 1, center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # 変換行列の結合
        M = M5 @ M4 @ M3 @ M2 @ M1
        M = M[:2, :]  # 同次座標を削除
        
        if is_debug:
            print(f"[DEBUG] 変換行列作成完了")
        
        return M
    
    def transform_image_to_circular(self, image: np.ndarray, ellipse_params: Tuple, is_debug: bool = False) -> np.ndarray:
        """
        楕円形のピザを円形に変換
        
        Args:
            image: 入力画像
            ellipse_params: ((中心x座標, 中心y座標), (長軸半径, 短軸半径), 角度)
            is_debug: デバッグモードフラグ
            
        Returns:
            変換後の画像
        """
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        
        if is_debug:
            print(f"[DEBUG] 変換開始: 中心=({center_x}, {center_y}), 長軸={major_axis}, 短軸={minor_axis}, 角度={angle:.1f}°")
        
        # スケール係数を計算
        scale_x, scale_y = self.calculate_transform_scale(major_axis, minor_axis, is_debug)
        
        # 変換行列を作成
        M = self._create_transformation_matrix(
            (center_x, center_y), angle, (scale_x, scale_y), is_debug
        )
        
        # 変換を適用
        transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        if is_debug:
            print(f"[DEBUG] 変換完了")
        
        return transformed
    
    def transform_image_step_by_step(self, image: np.ndarray, ellipse_params: Tuple, is_debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        デバッグ用に段階的に画像を変換
        
        Args:
            image: 入力画像
            ellipse_params: ((中心x座標, 中心y座標), (長軸半径, 短軸半径), 角度)
            is_debug: デバッグモードフラグ
            
        Returns:
            (回転後画像, スケール後画像, 最終画像)
        """
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        
        if is_debug:
            print(f"[DEBUG] 段階的変換開始")
        
        # スケール係数を計算
        scale_x, scale_y = self.calculate_transform_scale(major_axis, minor_axis, is_debug)
        
        # ステップ1: 回転のみ（楕円を軸に合わせる）
        M1 = np.array([[1, 0, -center_x],
                       [0, 1, -center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # 長軸をx軸に合わせるための回転
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        M2 = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        M5 = np.array([[1, 0, center_x],
                       [0, 1, center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # 回転のみの変換（楕円を軸に合わせた状態を保持）
        M_rotation = M5 @ M2 @ M1
        M_rotation = M_rotation[:2, :]
        rotated = cv2.warpAffine(image, M_rotation, (image.shape[1], image.shape[0]))
        
        if is_debug:
            print(f"[DEBUG] ステップ1完了: 回転のみ")
        
        # ステップ2: 回転した画像にスケーリングを追加
        # x軸のみスケーリング（長軸を短軸に合わせて縮小）
        M3 = np.array([[scale_x, 0, 0],
                       [0, scale_y, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # 完全な変換: 回転してからスケーリング（元に戻す回転なし）
        M_full = M5 @ M3 @ M2 @ M1
        M_full = M_full[:2, :]
        final = cv2.warpAffine(image, M_full, (image.shape[1], image.shape[0]))
        
        if is_debug:
            print(f"[DEBUG] ステップ2完了: スケーリング追加")
        
        return rotated, final, final
    
    def normalize_to_512x512(self, image: np.ndarray, center: Tuple[int, int], radius: int, is_debug: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        ピザを半径でクロップし、512x512にリサイズ
        
        Args:
            image: 入力画像（円形に変換済みであるべき）
            center: 中心座標 (x, y)
            radius: クロップ用のピザ半径
            is_debug: デバッグモードフラグ
            
        Returns:
            (512x512に正規化された画像, 正規化パラメータ)
        """
        center_x, center_y = center
        
        # クロップ境界を計算
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(image.shape[1], center_x + radius)
        y2 = min(image.shape[0], center_y + radius)
        
        if is_debug:
            print(f"[DEBUG] クロップ領域: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"[DEBUG] クロップサイズ: {x2-x1}x{y2-y1}")
        
        # 画像をクロップ
        cropped = image[y1:y2, x1:x2]
        
        # 元のクロップサイズを記録
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 512x512にリサイズ
        normalized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # 正規化パラメータを記録
        normalization_params = {
            'crop_region': (x1, y1, x2, y2),
            'crop_size': (crop_width, crop_height),
            'target_size': (512, 512),
            'scale_factor': 512.0 / max(crop_width, crop_height),
            'center_in_transformed': (center_x, center_y),  # 変換後画像での中心
            'radius_in_transformed': radius  # 変換後画像での半径
        }
        
        if is_debug:
            print(f"[DEBUG] 正規化完了: 512x512")
            print(f"[DEBUG] スケールファクター: {normalization_params['scale_factor']}")
        
        return normalized, normalization_params
    
    def preprocess_pizza_image(self, image_path: str, output_path: Optional[str] = None, is_debug: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        メイン前処理関数
        
        Args:
            image_path: 入力画像のパス
            output_path: 前処理済み画像の保存パス（オプション）
            is_debug: デバッグモードフラグ
            
        Returns:
            (前処理済み画像, 情報辞書)
        """
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        
        if is_debug:
            print(f"[DEBUG] 画像読み込み完了: {image_path}")
            print(f"[DEBUG] 画像サイズ: {image.shape}")
        
        # セグメンテーションマスクを取得
        mask = self.segmentation_service.segment_pizza(image_path)
        
        # 楕円を検出
        ellipse_params = self.detect_ellipse_from_mask(mask, is_debug)
        
        info = {
            'original_shape': image.shape,
            'is_transformed': False,
            'ellipse_params': None,
            'transformation_applied': None
        }
        
        if ellipse_params is None:
            if is_debug:
                print("[DEBUG] 画像から楕円が検出されませんでした")
            return image, info
        
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        info['ellipse_params'] = {
            'center': (center_x, center_y),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle
        }
        
        if is_debug:
            print(f"[DEBUG] 検出された楕円: 中心=({center_x}, {center_y}), 長軸={major_axis}, 短軸={minor_axis}, 角度={angle:.2f}°")
        
        # 常に変換を適用（すべてのピザを楕円として扱う）
        if is_debug:
            print(f"[DEBUG] ピザに変換を適用中: 中心=({center_x}, {center_y}), 長軸={major_axis}, 短軸={minor_axis}, 角度={angle:.2f}°")
        
        # 変換行列を生成
        scale_x, scale_y = self.calculate_transform_scale(major_axis, minor_axis, is_debug)
        transform_matrix = self._create_transformation_matrix(
            (center_x, center_y), angle, (scale_x, scale_y), is_debug
        )
        
        # 画像を変換
        transformed = self.transform_image_to_circular(image, ellipse_params, is_debug)

        # 512x512に正規化
        normalized, normalization_params = self.normalize_to_512x512(transformed, (center_x, center_y), minor_axis, is_debug)
        
        info['is_transformed'] = True
        info['transformation_applied'] = {
            'scale_x': scale_x,                  # x軸（長軸）を縮小
            'scale_y': scale_y,                  # y軸は変更なし
            'angle': angle                       # 元の楕円の角度
        }
        info['transform_matrix'] = transform_matrix
        info['crop_info'] = (
            normalization_params['crop_region'][0],  # crop_x
            normalization_params['crop_region'][1],  # crop_y  
            normalization_params['crop_region'][2] - normalization_params['crop_region'][0],  # crop_w
            normalization_params['crop_region'][3] - normalization_params['crop_region'][1]   # crop_h
        )
        info['scale_factor'] = normalization_params['scale_factor']
        info['normalization_params'] = normalization_params
        info['transformed_shape'] = transformed.shape
        
        # 出力パスが指定されている場合は保存
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, normalized)
            if is_debug:
                print(f"[DEBUG] 前処理済み画像を保存: {output_path}")
        
        return normalized, info


def process_and_save_debug_images(service: 'PreprocessService', input_path: Path, is_debug: bool = False) -> None:
    """
    デバッグ用に段階的な変換結果を保存
    
    Args:
        service: PreprocessServiceインスタンス
        input_path: 入力画像パス
        is_debug: デバッグモードフラグ
    """
    try:
        if is_debug:
            print(f"\n[DEBUG] 処理中: {input_path}...")
        
        # 画像を読み込み
        image = cv2.imread(str(input_path))
        if image is None:
            if is_debug:
                print(f"[DEBUG] 画像を読み込めませんでした: {input_path}")
            return
        
        # セグメンテーションマスクと楕円パラメータを取得
        mask = service.segmentation_service.segment_pizza(str(input_path))
        ellipse_params = service.detect_ellipse_from_mask(mask, is_debug)
        
        if ellipse_params is None:
            if is_debug:
                print("[DEBUG] 画像から楕円が検出されませんでした")
            return
        
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        if is_debug:
            print(f"[DEBUG] 検出された楕円: 中心=({center_x}, {center_y}), 長軸={major_axis}, 短軸={minor_axis}, 角度={angle:.2f}°")
        
        # 段階的変換
        rotated, final, _ = service.transform_image_step_by_step(image, ellipse_params, is_debug)
        
        # 512x512に正規化（ピザ半径でクロップしてリサイズ）
        # ピザは現在円形なので、短軸を半径として使用
        normalized = service.normalize_to_512x512(final, (center_x, center_y), minor_axis, is_debug)
        
        # デバッグ時のみ段階的な結果を保存
        if is_debug:
            base_name = input_path.stem
            
            # 0. オリジナルを保存
            original_path = f"debug/preprocess/{base_name}_original.jpg"
            Path(original_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(original_path, image)
            print(f"  ✓ オリジナル画像を保存: {original_path}")
            
            # 1. 回転のみを保存
            rotated_path = f"debug/preprocess/{base_name}_rotated.jpg"
            cv2.imwrite(rotated_path, rotated)
            print(f"  ✓ 回転後画像を保存: {rotated_path}")
            
            # 2. 最終結果（回転＋スケーリング）を保存
            final_path = f"debug/preprocess/{base_name}_final.jpg"
            cv2.imwrite(final_path, final)
            print(f"  ✓ 最終画像を保存: {final_path}")
            
            # 3. 正規化された512x512を保存
            normalized_path = f"debug/preprocess/{base_name}_normalized.jpg"
            cv2.imwrite(normalized_path, normalized)
            print(f"  ✓ 正規化された512x512画像を保存: {normalized_path}")
            
            # スケール係数を計算して表示
            scale_x = minor_axis / major_axis  # x軸のみ縮小
            scale_y = 1.0  # y軸は変更なし
            print(f"  スケール係数: x={scale_x:.4f} (長軸を縮小), y={scale_y:.4f} (変更なし)")
            print(f"  正規化: 半径{minor_axis}pxでクロップし、512x512にリサイズ")
            
    except Exception as e:
        if is_debug:
            print(f"[DEBUG] エラーが発生しました ({input_path}): {e}")
        else:
            raise


if __name__ == "__main__":
    # サービスインスタンスを作成
    service = PreprocessService()
    
    # デバッグモードを有効化
    is_debug = True
    
    # resourceディレクトリ内のすべての画像を検索
    resource_dir = Path("resource")
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    input_images = []
    
    for ext in supported_extensions:
        input_images.extend(resource_dir.glob(f"*{ext}"))
        input_images.extend(resource_dir.glob(f"*{ext.upper()}"))
    
    print(f"resourceディレクトリ内に{len(input_images)}枚の画像が見つかりました")
    
    for input_path in input_images:
        process_and_save_debug_images(service, input_path, is_debug)