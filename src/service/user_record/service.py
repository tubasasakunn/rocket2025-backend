from typing import List, Optional, Union
from .client import UserRecordClient, UserRecord
from .repository import UserRecordRepository
from .settings import get_user_record_settings


class UserRecordService:
    """
    ユーザーレコードサービス
    
    Google Apps Scriptで管理されているユーザーレコードへのアクセスと操作の
    ビジネスロジックを提供します。
    """
    
    def __init__(self, repository: UserRecordRepository):
        """
        サービスを初期化
        
        Args:
            repository: ユーザーレコードリポジトリ
        """
        self.repository = repository
    
    async def get_all_users(self) -> List[UserRecord]:
        """
        すべてのユーザーレコードを取得
        
        Returns:
            List[UserRecord]: ユーザーレコードのリスト
        """
        return await self.repository.get_all_records()
    
    async def get_user_by_account(self, account: str) -> Optional[UserRecord]:
        """
        アカウント名でユーザーを検索
        
        Args:
            account: 検索するアカウント名
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        return await self.repository.find_by_account(account)
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserRecord]:
        """
        IDでユーザーを検索
        
        Args:
            user_id: 検索するユーザーID
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        return await self.repository.find_by_id(user_id)
    
    async def create_user(self, account: str, score: int = 0) -> UserRecord:
        """
        新規ユーザーを作成
        
        Args:
            account: ユーザーアカウント名
            score: 初期スコア（デフォルト：0）
            
        Returns:
            UserRecord: 作成されたユーザーレコード
        """
        return await self.repository.create_record(account, score)
    
    async def update_user_score(self, 
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
        return await self.repository.update_score(identifier, score, create_if_not_exist)


def get_user_record_service() -> UserRecordService:
    """
    ユーザーレコードサービスのインスタンスを取得
    
    FastAPI依存関係注入システムで使用するためのファクトリ関数です。
    
    Returns:
        UserRecordService: サービスインスタンス
    """
    settings = get_user_record_settings()
    client = UserRecordClient(settings.gas_webapp_url)
    repository = UserRecordRepository(client)
    return UserRecordService(repository)
