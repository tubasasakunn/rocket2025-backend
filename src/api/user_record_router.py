from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.service.user_record.service import UserRecordService, get_user_record_service
from src.service.user_record.client import UserRecord, GasClientException


# モデル定義
class UserRecordResponse(BaseModel):
    """ユーザーレコードのレスポンス"""
    id: int
    account: str
    score: int
    update_at: str


class CreateUserRequest(BaseModel):
    """ユーザー作成リクエスト"""
    account: str = Field(..., description="ユーザーのアカウント名")
    score: int = Field(0, description="初期スコア")


class UpdateScoreRequest(BaseModel):
    """スコア更新リクエスト"""
    score: int = Field(..., description="新しいスコア")
    create_if_not_exist: bool = Field(True, description="ユーザーが存在しない場合に作成するか")


# ルーター作成
router = APIRouter(prefix="/user-records", tags=["user-records"])


# エンドポイント定義
@router.get("/", response_model=List[UserRecordResponse])
async def get_all_users(
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    すべてのユーザーレコードを取得
    
    スプレッドシートに保存されているすべてのユーザーレコードを返します。
    """
    try:
        users = await service.get_all_users()
        return users
    except GasClientException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")


@router.get("/health", response_model=Dict[str, str])
async def health_check(
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    ユーザーレコードAPIのヘルスチェック
    
    Google Apps Script Webアプリとの接続が正常であるかを確認します。
    """
    try:
        # とりあえずユーザー一覧を取得してみる（接続テスト）
        await service.get_all_users()
        return {"status": "ok", "message": "User Record API is running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/{user_id}", response_model=UserRecordResponse)
async def get_user_by_id(
    user_id: int,
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    IDによるユーザー検索
    
    指定されたIDのユーザーレコードを取得します。
    """
    try:
        user = await service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
        return user
    except GasClientException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")


@router.get("/by-account/{account}", response_model=UserRecordResponse)
async def get_user_by_account(
    account: str,
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    アカウント名によるユーザー検索
    
    指定されたアカウント名のユーザーレコードを取得します。
    """
    try:
        user = await service.get_user_by_account(account)
        if not user:
            raise HTTPException(status_code=404, detail=f"User with account '{account}' not found")
        return user
    except GasClientException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")


@router.post("/", response_model=UserRecordResponse)
async def create_user(
    request: CreateUserRequest,
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    新規ユーザー作成
    
    新しいユーザーレコードを作成します。
    """
    try:
        user = await service.create_user(request.account, request.score)
        return user
    except GasClientException as e:
        if "既に存在します" in str(e):
            raise HTTPException(status_code=409, detail=f"User with account '{request.account}' already exists")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@router.put("/{identifier}/score", response_model=UserRecordResponse)
async def update_user_score(
    identifier: str,
    request: UpdateScoreRequest,
    service: UserRecordService = Depends(get_user_record_service)
):
    """
    ユーザースコア更新
    
    ユーザーのスコアを更新します。
    identifierには、ユーザーIDまたはアカウント名を指定できます。
    """
    try:
        # identifierが数値であれば整数に変換、そうでなければ文字列のまま
        if identifier.isdigit():
            identifier = int(identifier)
        
        user = await service.update_user_score(
            identifier, 
            request.score, 
            request.create_if_not_exist
        )
        return user
    except GasClientException as e:
        if "が見つかりません" in str(e) and not request.create_if_not_exist:
            raise HTTPException(status_code=404, detail=f"User with identifier '{identifier}' not found")
        raise HTTPException(status_code=500, detail=f"Failed to update score: {str(e)}")
