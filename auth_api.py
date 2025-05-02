from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os

from models.auth import AuthManager
from schemas.auth import UserSignUp, UserLogin, Token, UserResponse, LoginResponse, TokenRefresh
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Initialize router
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={401: {"description": "Unauthorized"}},
)

# Initialize Auth Manager
bucket_name = os.getenv("GCS_BUCKET_NAME")
if not bucket_name:
    raise ValueError("GCS_BUCKET_NAME environment variable must be set")
auth_manager = AuthManager(bucket_name=bucket_name)

# OAuth2 password bearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from access token."""
    user = auth_manager.validate_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignUp):
    """
    Register a new user.
    
    - **name**: Full name (required)
    - **email**: Email address (required)
    - **password**: Password (minimum 8 characters, required)
    """
    user = auth_manager.register_user(
        name=user_data.name,
        email=user_data.email,
        password=user_data.password
    )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    return user

@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login to get access token.
    
    - **username**: Email address
    - **password**: Password
    """
    user_data = auth_manager.authenticate_user(
        email=form_data.username,  # OAuth2 uses username field for email
        password=form_data.password
    )
    
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_data

@router.post("/login/email", response_model=LoginResponse)
async def login_with_email(user_data: UserLogin):
    """
    Login with email and password to get access token.
    
    - **email**: Email address
    - **password**: Password
    """
    authenticated = auth_manager.authenticate_user(
        email=user_data.email,
        password=user_data.password
    )
    
    if authenticated is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return authenticated

@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh):
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Refresh token received during login
    """
    new_tokens = auth_manager.refresh_tokens(token_data.refresh_token)
    
    if new_tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "access_token": new_tokens["access_token"],
        "refresh_token": new_tokens["refresh_token"],
        "token_type": "bearer"
    }

@router.post("/logout")
async def logout(current_user: UserResponse = Depends(get_current_user)):
    """
    Logout current user by invalidating all tokens.
    """
    auth_manager.invalidate_tokens(current_user["id"])
    return {"detail": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    """
    Get details of currently authenticated user.
    """
    return current_user