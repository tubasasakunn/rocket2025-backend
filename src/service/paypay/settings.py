import os
from pydantic import BaseSettings
from functools import lru_cache


class PayPaySettings(BaseSettings):
    """
    PayPay API 設定
    環境変数から PayPay API の設定を読み込む
    """
    
    # PayPay API 認証情報
    paypay_api_key: str = ""
    paypay_api_secret: str = ""
    paypay_merchant_id: str = ""
    
    # 環境設定
    paypay_is_production: bool = False
    
    # アプリケーション設定
    app_host: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_paypay_settings() -> PayPaySettings:
    """
    PayPay 設定のシングルトンインスタンスを取得
    
    Returns:
        PayPaySettings インスタンス
    """
    return PayPaySettings()
