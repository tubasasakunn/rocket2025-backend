import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class UserRecordSettings(BaseSettings):
    """
    ユーザーレコード関連の設定
    
    環境変数から設定を読み込みます。
    """
    
    gas_webapp_url: str = os.environ.get("GAS_WEBAPP_URL", "")
    
    class Config:
        env_prefix = "USER_RECORD_"
        case_sensitive = False


@lru_cache()
def get_user_record_settings() -> UserRecordSettings:
    """
    ユーザーレコード設定を取得
    
    キャッシュされたシングルトンインスタンスを返します。
    
    Returns:
        UserRecordSettings: 設定インスタンス
    """
    return UserRecordSettings()
