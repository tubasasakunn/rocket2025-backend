# Pizza Division API Results Saver

## 概要
ピザ分割APIから取得した結果（SVG・PNG画像）を自動的に保存するシェルスクリプトです。

## 使用方法

### 基本的な使い方
```bash
./save_pizza_results.sh <画像パス> [分割数]
```

### 実行例
```bash
# pizza1.jpgを4分割（デフォルト）
./save_pizza_results.sh resource/pizza1.jpg

# pizza2.jpgを6分割
./save_pizza_results.sh resource/pizza2.jpg 6

# 絶対パスも使用可能
./save_pizza_results.sh /path/to/your/pizza.jpg 8
```

## 出力ファイル

すべてのファイルは `result/api_res/` ディレクトリに保存されます。

### ファイル命名規則
```
{画像名}_{分割数}pieces_{タイムスタンプ}_{種類}.{拡張子}
```

### 生成されるファイル

1. **SVGファイル（3種類）**
   - `*_before.svg` - 爆発前（通常の分割表示）
   - `*_after.svg` - 爆発後（ピース分離表示）
   - `*_animated.svg` - アニメーション付き

2. **PNGファイル**
   - `*_overlay.png` - 元画像にオーバーレイした結果

3. **個別ピースSVG**
   - `*_piece1.svg` ~ `*_piece{N}.svg` - 各ピースの個別SVG

4. **レスポンスJSON**
   - `*_full_response.json` - API完全レスポンス（デバッグ用）

## 例：pizza2.jpgを4分割した場合
```
result/api_res/
├── pizza2_4pieces_20250703_182903_before.svg      (15.6KB)
├── pizza2_4pieces_20250703_182903_after.svg       (15.8KB)
├── pizza2_4pieces_20250703_182903_animated.svg    (17.9KB)
├── pizza2_4pieces_20250703_182903_overlay.png     (6.3MB)
├── pizza2_4pieces_20250703_182903_piece1.svg      (4.8KB)
├── pizza2_4pieces_20250703_182903_piece2.svg      (4.6KB)
├── pizza2_4pieces_20250703_182903_piece3.svg      (3.2KB)
├── pizza2_4pieces_20250703_182903_piece4.svg      (3.3KB)
└── pizza2_4pieces_20250703_182903_full_response.json (8.5MB)
```

## 前提条件

1. **APIサーバーが起動中**
   ```bash
   python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **必要なコマンド**
   - `curl` - API呼び出し
   - `jq` - JSON処理
   - `base64` - base64デコード

3. **権限**
   ```bash
   chmod +x save_pizza_results.sh
   ```

## トラブルシューティング

### APIエラーの場合
- APIサーバーの起動状況を確認
- 画像ファイルの存在・形式を確認
- エラーメッセージをチェック

### ファイル権限エラー
```bash
chmod +x save_pizza_results.sh
```

### jqコマンドがない場合
```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq
```

## カスタマイズ

### APIエンドポイントの変更
スクリプト内の `API_URL` 変数を編集：
```bash
API_URL="http://your-server:port/api/pizza-cutter/divide"
```

### 出力ディレクトリの変更
```bash
OUTPUT_DIR="path/to/your/output"
```