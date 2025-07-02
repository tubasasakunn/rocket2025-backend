#!/usr/bin/env python
"""
このスクリプトはユーザーレコードAPIをローカルでテストするためのものです。
Google Apps ScriptのWebアプリがデプロイされていることを前提としています。
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# 環境変数の読み込み
load_dotenv()

# プロジェクトのルートディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # project root
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# インポート
from src.service.user_record.client import UserRecordClient, UserRecord
from src.service.user_record.service import UserRecordService
from src.service.user_record.repository import UserRecordRepository


# テスト用アカウント名と初期スコア
TEST_ACCOUNT = "test_user"
INITIAL_SCORE = 100


async def test_user_record_api():
    """ユーザーレコードAPIのテスト"""
    print("===== ユーザーレコードAPIテスト開始 =====")
    
    # クライアントを初期化
    gas_url = os.getenv("GAS_WEBAPP_URL")
    
    if not gas_url:
        print("環境変数 GAS_WEBAPP_URL が設定されていません。")
        print(".env ファイルを作成し、GAS_WEBAPP_URL を設定してください。")
        return
    
    client = UserRecordClient(gas_url)
    repository = UserRecordRepository(client)
    service = UserRecordService(repository)
    
    # 1. 全ユーザーを取得
    try:
        print("\n----- すべてのユーザーを取得 -----")
        users = await service.get_all_users()
        print(f"取得されたユーザー数: {len(users)}")
        for user in users:
            print(f"ID: {user.id}, アカウント: {user.account}, スコア: {user.score}")
    except Exception as e:
        print(f"エラー: {e}")
    
    # 2. テストユーザーが存在するか確認
    print("\n----- テストユーザーを検索 -----")
    user = await service.get_user_by_account(TEST_ACCOUNT)
    
    if user:
        print(f"テストユーザーが見つかりました: ID: {user.id}, アカウント: {user.account}, スコア: {user.score}")
    else:
        print(f"テストユーザー '{TEST_ACCOUNT}' が見つかりません。新規作成します。")
        
        # 3. テストユーザーを作成
        try:
            user = await service.create_user(TEST_ACCOUNT, INITIAL_SCORE)
            print(f"テストユーザーを作成しました: ID: {user.id}, アカウント: {user.account}, スコア: {user.score}")
        except Exception as e:
            print(f"ユーザー作成エラー: {e}")
            return
    
    # 4. スコアを更新
    try:
        print("\n----- ユーザースコアを更新 -----")
        new_score = user.score + 50
        updated_user = await service.update_user_score(user.account, new_score)
        print(f"スコアを更新しました: ID: {updated_user.id}, アカウント: {updated_user.account}")
        print(f"旧スコア: {user.score} → 新スコア: {updated_user.score}")
    except Exception as e:
        print(f"スコア更新エラー: {e}")
    
    # 5. 再度ユーザーを取得して確認
    try:
        print("\n----- 更新後のユーザー情報を確認 -----")
        updated_user = await service.get_user_by_id(user.id)
        
        if updated_user:
            print(f"ID: {updated_user.id}, アカウント: {updated_user.account}, スコア: {updated_user.score}")
            print(f"更新日時: {updated_user.update_at}")
        else:
            print(f"エラー: ユーザーID {user.id} が見つかりません。")
    except Exception as e:
        print(f"ユーザー取得エラー: {e}")
    
    print("\n===== テスト完了 =====")


if __name__ == "__main__":
    asyncio.run(test_user_record_api())
