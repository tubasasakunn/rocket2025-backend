from typing import List, Optional, Union
from .client import UserRecordClient, UserRecord


class UserRecordRepository:
    """
    ユーザーレコードリポジトリ
    
    Google Apps Scriptで管理されるユーザーレコードへのアクセスを抽象化します。
    """
    
    def __init__(self, client: UserRecordClient):
        """
        リポジトリを初期化
        
        Args:
            client: GASクライアントインスタンス
        """
        self.client = client
    
    async def get_all_records(self) -> List[UserRecord]:
        """
        すべてのユーザーレコードを取得
        
        Returns:
            List[UserRecord]: ユーザーレコードのリスト
        """
        return await self.client.get_all_records()
    
    async def find_by_account(self, account: str) -> Optional[UserRecord]:
        """
        アカウント名でユーザーを検索
        
        Args:
            account: 検索するアカウント名
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        return await self.client.find_by_account(account)
    
    async def find_by_id(self, user_id: int) -> Optional[UserRecord]:
        """
        IDでユーザーを検索
        
        Args:
            user_id: 検索するユーザーID
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        return await self.client.find_by_id(user_id)
    
    async def create_record(self, account: str, score: int = 0) -> UserRecord:
        """
        新規ユーザーレコードを作成
        
        Args:
            account: ユーザーアカウント名
            score: 初期スコア（デフォルト：0）
            
        Returns:
            UserRecord: 作成されたユーザーレコード
        """
        return await self.client.create_record(account, score)
    
    async def update_score(self, 
                          identifier: Union[int, str], 
                          score: int, 
                          create_if_not_exist: bool = True) -> UserRecord:
        """
        ユーザースコアを更新
        
        Args:
            identifier: ユーザーIDまたはアカウント名
            score: 新しいスコア
            create_if_not_exist: ユーザーが存在しない場合に作成するかどうか
            
        Returns:
            UserRecord: 更新されたユーザーレコード
        """
        return await self.client.update_score(identifier, score, create_if_not_exist)
