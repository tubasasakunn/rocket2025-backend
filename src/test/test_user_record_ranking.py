#!/usr/bin/env python
"""
このスクリプトはユーザーランキングAPIをローカルでテストするためのものです。
Google Apps ScriptのWebアプリがデプロイされていることを前提としています。
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import unittest.mock as mock

# 環境変数の読み込み
load_dotenv()

# プロジェクトのルートディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # project root
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# インポート
from src.service.user_record.client import UserRecord
from src.service.user_record.service import UserRecordService
from src.api.user_record_router import get_ranking, UserRankingResponse


async def test_ranking_api():
    """ユーザーランキングAPIのテスト"""
    print("===== ユーザーランキングAPIテスト開始 =====")
    
    # モックユーザーデータを作成
    mock_users = [
        UserRecord(id=1, account="user1", score=100, update_at="2025-07-01"),
        UserRecord(id=2, account="user2", score=100, update_at="2025-07-01"),
        UserRecord(id=3, account="user3", score=90, update_at="2025-07-01"),
        UserRecord(id=4, account="user4", score=80, update_at="2025-07-01"),
        UserRecord(id=5, account="user5", score=80, update_at="2025-07-01"),
        UserRecord(id=6, account="user6", score=70, update_at="2025-07-01"),
        UserRecord(id=7, account="user7", score=60, update_at="2025-07-01"),
        UserRecord(id=8, account="user8", score=50, update_at="2025-07-01"),
        UserRecord(id=9, account="user9", score=40, update_at="2025-07-01"),
        UserRecord(id=10, account="user10", score=30, update_at="2025-07-01"),
        UserRecord(id=11, account="user11", score=20, update_at="2025-07-01"),
        UserRecord(id=12, account="user12", score=10, update_at="2025-07-01"),
    ]
    
    # サービスをモック
    mock_service = mock.AsyncMock(spec=UserRecordService)
    mock_service.get_all_users.return_value = mock_users
    
    # テスト1: デフォルト制限（10件）
    print("\n----- デフォルト制限（10件）のテスト -----")
    result = await get_ranking(limit=10, service=mock_service)
    
    # 結果を検証
    print(f"取得されたランキング数: {len(result)}")
    assert len(result) == 10, f"期待: 10件, 実際: {len(result)}件"
    
    # ランキングの検証
    expected_ranks = [1, 1, 3, 4, 4, 6, 7, 8, 9, 10]
    actual_ranks = [user.rank for user in result]
    
    print("ランキング検証:")
    for i, (user, expected_rank) in enumerate(zip(result, expected_ranks)):
        print(f"{i+1}. ID: {user.id}, アカウント: {user.account}, スコア: {user.score}, ランク: {user.rank}")
        assert user.rank == expected_rank, f"ユーザー {user.account} のランク - 期待: {expected_rank}, 実際: {user.rank}"
    
    # テスト2: カスタム制限（5件）
    print("\n----- カスタム制限（5件）のテスト -----")
    result = await get_ranking(limit=5, service=mock_service)
    
    # 結果を検証
    print(f"取得されたランキング数: {len(result)}")
    assert len(result) == 5, f"期待: 5件, 実際: {len(result)}件"
    
    # ランキングの検証
    expected_ranks = [1, 1, 3, 4, 4]
    actual_ranks = [user.rank for user in result]
    
    print("ランキング検証:")
    for i, (user, expected_rank) in enumerate(zip(result, expected_ranks)):
        print(f"{i+1}. ID: {user.id}, アカウント: {user.account}, スコア: {user.score}, ランク: {user.rank}")
        assert user.rank == expected_rank, f"ユーザー {user.account} のランク - 期待: {expected_rank}, 実際: {user.rank}"
    
    print("\n===== テスト完了 =====")


if __name__ == "__main__":
    asyncio.run(test_ranking_api())
