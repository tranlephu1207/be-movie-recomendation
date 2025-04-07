from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any

class UserSignUp(BaseModel):
    """Schema for user registration."""
    name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password (min 8 characters)")

class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")

class TokenRefresh(BaseModel):
    """Schema for refreshing tokens."""
    refresh_token: str = Field(..., description="Refresh token")

class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    refresh_token: str
    token_type: str

class UserResponse(BaseModel):
    """Schema for user data response."""
    id: int
    name: str
    email: EmailStr
    created_at: str

class TokenData(BaseModel):
    """Schema for data contained in a token."""
    user_id: int

class LoginResponse(BaseModel):
    """Schema for login response."""
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str