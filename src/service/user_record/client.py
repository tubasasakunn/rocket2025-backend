import httpx
from typing import Dict, Any, List, Optional, Union
import os
from pydantic import BaseModel, Field


class UserRecord(BaseModel):
    """ユーザーレコードモデル"""
    id: int
    account: str
    score: int
    update_at: str


class GasClientException(Exception):
    """GASクライアント例外"""
    pass


class UserRecordClient:
    """
    Google Apps Script Webアプリと通信するクライアント
    
    GASで公開されたWebアプリにリクエストを送信し、ユーザーレコードを操作します。
    """
    
    def __init__(self, gas_webapp_url: Optional[str] = None):
        """
        クライアントを初期化
        
        Args:
            gas_webapp_url: Google Apps ScriptのWebアプリURL
                           未指定の場合は環境変数 GAS_WEBAPP_URL から取得
        """
        self.gas_webapp_url = gas_webapp_url or os.environ.get("GAS_WEBAPP_URL")
        
        if not self.gas_webapp_url:
            raise ValueError("GAS_WEBAPP_URL environment variable or gas_webapp_url parameter is required")
    
    async def _handle_response(self, response) -> Dict[str, Any]:
        """レスポンスを処理して結果を返す"""
        try:
            print(f"Handling response with status: {response.status_code}")
            print(f"Response content: {response.text}")
            
            # Try to parse as JSON
            data = response.json()
            
            if response.status_code != 200 or data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                raise GasClientException(f"GAS API error: {error_msg}")
            
            return data.get("data", {})
        except Exception as e:
            print(f"Error parsing JSON: {type(e).__name__}: {str(e)}")
            raise GasClientException(f"Invalid JSON response: {response.text}")
    
    async def get_all_records(self) -> List[UserRecord]:
        """
        すべてのユーザーレコードを取得
        
        Returns:
            List[UserRecord]: ユーザーレコードのリスト
        """
        params = {"action": "getAll"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        try:
            print(f"Making request to GAS URL: {self.gas_webapp_url}")
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                print(f"Requesting URL: {self.gas_webapp_url} with params: {params}")
                response = await client.get(
                    self.gas_webapp_url, 
                    params=params, 
                    headers=headers
                )
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response body: {response.text}")
                data = await self._handle_response(response)
                
                # レスポンスデータをUserRecordモデルに変換
                return [UserRecord(**record) for record in data]
        except (httpx.HTTPError, Exception) as e:
            print(f"Error in get_all_records: {type(e).__name__}: {str(e)}")
            raise GasClientException(f"Failed to get all records: {str(e)}")
    
    async def find_by_account(self, account: str) -> Optional[UserRecord]:
        """
        アカウント名でユーザーを検索
        
        Args:
            account: 検索するアカウント名
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        params = {
            "action": "findByAccount",
            "account": account
        }
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(self.gas_webapp_url, params=params)
                data = await self._handle_response(response)
                
                # レスポンスデータをUserRecordモデルに変換
                return UserRecord(**data)
        except GasClientException as e:
            if "ユーザーが見つかりません" in str(e):
                return None
            raise
        except (httpx.HTTPError, Exception) as e:
            raise GasClientException(f"Failed to find user by account: {str(e)}")
    
    async def find_by_id(self, user_id: int) -> Optional[UserRecord]:
        """
        IDでユーザーを検索
        
        Args:
            user_id: 検索するユーザーID
            
        Returns:
            Optional[UserRecord]: 見つかった場合はユーザーレコード、見つからない場合はNone
        """
        params = {
            "action": "findById",
            "id": user_id
        }
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(self.gas_webapp_url, params=params)
                data = await self._handle_response(response)
                
                # レスポンスデータをUserRecordモデルに変換
                return UserRecord(**data)
        except GasClientException as e:
            if "ユーザーが見つかりません" in str(e):
                return None
            raise
        except (httpx.HTTPError, Exception) as e:
            raise GasClientException(f"Failed to find user by ID: {str(e)}")
    
    async def create_record(self, account: str, score: int = 0) -> UserRecord:
        """
        新規ユーザーレコードを作成
        
        Args:
            account: ユーザーアカウント名
            score: 初期スコア（デフォルト：0）
            
        Returns:
            UserRecord: 作成されたユーザーレコード
        """
        data = {
            "action": "create",
            "account": account,
            "score": score
        }
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.post(self.gas_webapp_url, json=data)
                result = await self._handle_response(response)
                
                # レスポンスデータをUserRecordモデルに変換
                return UserRecord(**result)
        except (httpx.HTTPError, Exception) as e:
            raise GasClientException(f"Failed to create record: {str(e)}")
    
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
        data = {
            "action": "update",
            "identifier": identifier,
            "score": score,
            "createIfNotExist": create_if_not_exist
        }
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.post(self.gas_webapp_url, json=data)
                result = await self._handle_response(response)
                
                # レスポンスデータをUserRecordモデルに変換
                return UserRecord(**result)
        except (httpx.HTTPError, Exception) as e:
            raise GasClientException(f"Failed to update score: {str(e)}")
