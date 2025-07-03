#!/bin/bash

# ピザ分割API結果保存スクリプト
# Usage: ./save_pizza_results.sh <image_path> [n_pieces]

set -e

# 引数チェック
if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_path> [n_pieces]"
    echo "Example: $0 resource/pizza1.jpg 4"
    exit 1
fi

# 変数設定
INPUT_IMAGE="$1"
N_PIECES="${2:-4}"  # デフォルト4分割
API_URL="http://localhost:8000/api/pizza-cutter/divide"
OUTPUT_DIR="result/api_res"

# 入力画像の存在確認
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ Error: Image file not found: $INPUT_IMAGE"
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# ファイル名から拡張子を除いた部分を取得
BASENAME=$(basename "$INPUT_IMAGE" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PREFIX="${BASENAME}_${N_PIECES}pieces_${TIMESTAMP}"

echo "🍕 Starting Pizza Division API..."
echo "   Input: $INPUT_IMAGE"
echo "   Pieces: $N_PIECES"
echo "   Output: $OUTPUT_DIR/"

# APIを呼び出してレスポンスを取得
echo "📡 Calling API..."
RESPONSE_FILE="${OUTPUT_DIR}/${PREFIX}_response.json"

curl -X POST "$API_URL" \
  -F "file=@$INPUT_IMAGE" \
  -F "n_pieces=$N_PIECES" \
  -s -o "$RESPONSE_FILE"

# レスポンスの成功確認
SUCCESS=$(jq -r '.success' "$RESPONSE_FILE" 2>/dev/null || echo "false")

if [ "$SUCCESS" != "true" ]; then
    ERROR_MSG=$(jq -r '.error_message // "Unknown error"' "$RESPONSE_FILE" 2>/dev/null || echo "API call failed")
    echo "❌ API Error: $ERROR_MSG"
    exit 1
fi

echo "✅ API call successful!"

# 各データを個別ファイルに保存
echo "💾 Saving results..."

# 1. SVG Before Explosion
echo "   Saving svg_before_explosion..."
jq -r '.svg_before_explosion' "$RESPONSE_FILE" > "${OUTPUT_DIR}/${PREFIX}_before.svg"

# 2. SVG After Explosion  
echo "   Saving svg_after_explosion..."
jq -r '.svg_after_explosion' "$RESPONSE_FILE" > "${OUTPUT_DIR}/${PREFIX}_after.svg"

# 3. SVG Animated
echo "   Saving svg_animated..."
jq -r '.svg_animated' "$RESPONSE_FILE" > "${OUTPUT_DIR}/${PREFIX}_animated.svg"

# 4. Overlay Image (PNG)
echo "   Saving overlay_image..."
jq -r '.overlay_image' "$RESPONSE_FILE" | sed 's/data:image\/png;base64,//' | base64 -d > "${OUTPUT_DIR}/${PREFIX}_overlay.png"

# 5. Individual Piece SVGs
PIECE_COUNT=$(jq -r '.piece_svgs | length' "$RESPONSE_FILE")
echo "   Saving ${PIECE_COUNT} piece SVGs..."

for i in $(seq 0 $((PIECE_COUNT - 1))); do
    PIECE_NUM=$((i + 1))
    jq -r ".piece_svgs[$i]" "$RESPONSE_FILE" > "${OUTPUT_DIR}/${PREFIX}_piece${PIECE_NUM}.svg"
done

# 6. Response JSON (for reference)
echo "   Saving response.json..."
cp "$RESPONSE_FILE" "${OUTPUT_DIR}/${PREFIX}_full_response.json"

# サイズ情報を表示
echo ""
echo "📊 Generated files:"
echo "   📁 Directory: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/${PREFIX}* | while read -r line; do
    echo "   📄 $line"
done

echo ""
echo "🎉 All files saved successfully!"
echo ""
echo "📋 Summary:"
echo "   • SVG Before:    ${PREFIX}_before.svg"
echo "   • SVG After:     ${PREFIX}_after.svg" 
echo "   • SVG Animated:  ${PREFIX}_animated.svg"
echo "   • Overlay PNG:   ${PREFIX}_overlay.png"
echo "   • Piece SVGs:    ${PREFIX}_piece1.svg ~ ${PREFIX}_piece${PIECE_COUNT}.svg"
echo "   • Full Response: ${PREFIX}_full_response.json"
echo ""
echo "💡 To view SVGs, open them in a web browser or SVG viewer"
echo "💡 Overlay PNG can be viewed with any image viewer"