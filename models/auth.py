import os
import csv
import json
import hashlib
import secrets
import datetime
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from passlib.context import CryptContext

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    """
    Handle user authentication, registration, and token management.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize authentication manager.
        
        Args:
            data_dir: Directory for storing user data
        """
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.csv")
        self.tokens_file = os.path.join(data_dir, "tokens.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # Load user data
        self.users_df = pd.read_csv(self.users_file) if os.path.exists(self.users_file) else pd.DataFrame(
            columns=["id", "name", "email", "password_hash", "created_at"]
        )
        
        # Load token data
        self.tokens_df = pd.read_csv(self.tokens_file) if os.path.exists(self.tokens_file) else pd.DataFrame(
            columns=["user_id", "access_token", "refresh_token", "access_expiry", "refresh_expiry"]
        )
    
    def _initialize_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        
        # Users file
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "name", "email", "password_hash", "created_at"
                ])
        
        # Tokens file
        if not os.path.exists(self.tokens_file):
            with open(self.tokens_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "user_id", "access_token", "refresh_token", "access_expiry", "refresh_expiry"
                ])
    
    def _save_users(self) -> None:
        """Save users dataframe to CSV."""
        self.users_df.to_csv(self.users_file, index=False)
    
    def _save_tokens(self) -> None:
        """Save tokens dataframe to CSV."""
        self.tokens_df.to_csv(self.tokens_file, index=False)
    
    def register_user(self, name: str, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Register a new user.
        
        Args:
            name: User's full name
            email: User's email
            password: User's password
            
        Returns:
            User data if registration successful, None if email already exists
        """
        # Check if email already exists
        if len(self.users_df[self.users_df["email"] == email]) > 0:
            return None
        
        # Generate user ID
        user_id = len(self.users_df) + 1
        
        # Hash password
        password_hash = pwd_context.hash(password)
        
        # Create user record
        new_user = {
            "id": user_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Add to dataframe
        self.users_df = pd.concat([self.users_df, pd.DataFrame([new_user])], ignore_index=True)
        
        # Save updated users
        self._save_users()
        
        # Create user response without password hash
        user_data = new_user.copy()
        user_data.pop("password_hash")
        
        return user_data
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user and generate tokens.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Dict with user data and tokens if authentication successful, None otherwise
        """
        # Find user by email
        user_rows = self.users_df[self.users_df["email"] == email]
        if len(user_rows) == 0:
            return None
        
        user = user_rows.iloc[0].to_dict()
        
        # Verify password
        if not pwd_context.verify(password, user["password_hash"]):
            return None
        
        # Generate tokens
        tokens = self._create_tokens(user["id"])
        
        # Create user response without password hash
        user_data = user.copy()
        user_data.pop("password_hash")
        
        return {
            "user": user_data,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer"
        }
    
    def _create_tokens(self, user_id: int) -> Dict[str, str]:
        """
        Create access and refresh tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with access and refresh tokens
        """
        # Generate tokens
        access_token = secrets.token_hex(32)
        refresh_token = secrets.token_hex(32)
        
        # Calculate expiry times
        now = datetime.datetime.now()
        access_expiry = (now + datetime.timedelta(minutes=60)).isoformat()  # 60 minutes
        refresh_expiry = (now + datetime.timedelta(days=14)).isoformat()  # 2 weeks
        
        # Remove any existing tokens for this user
        self.tokens_df = self.tokens_df[self.tokens_df["user_id"] != user_id]
        
        # Add new tokens
        new_token = {
            "user_id": user_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "access_expiry": access_expiry,
            "refresh_expiry": refresh_expiry
        }
        
        self.tokens_df = pd.concat([self.tokens_df, pd.DataFrame([new_token])], ignore_index=True)
        
        # Save updated tokens
        self._save_tokens()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    
    def refresh_tokens(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Create new access and refresh tokens using a valid refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Dict with new tokens if refresh token is valid, None otherwise
        """
        # Find token
        token_rows = self.tokens_df[self.tokens_df["refresh_token"] == refresh_token]
        if len(token_rows) == 0:
            return None
        
        token_data = token_rows.iloc[0].to_dict()
        
        # Check if refresh token is expired
        refresh_expiry = datetime.datetime.fromisoformat(token_data["refresh_expiry"])
        if refresh_expiry < datetime.datetime.now():
            # Remove expired token
            self.tokens_df = self.tokens_df[self.tokens_df["refresh_token"] != refresh_token]
            self._save_tokens()
            return None
        
        # Create new tokens
        user_id = token_data["user_id"]
        return self._create_tokens(user_id)
    
    def validate_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate access token and return user data.
        
        Args:
            access_token: Access token
            
        Returns:
            User data if token is valid, None otherwise
        """
        # Find token
        token_rows = self.tokens_df[self.tokens_df["access_token"] == access_token]
        if len(token_rows) == 0:
            return None
        
        token_data = token_rows.iloc[0].to_dict()
        
        # Check if access token is expired
        access_expiry = datetime.datetime.fromisoformat(token_data["access_expiry"])
        if access_expiry < datetime.datetime.now():
            return None
        
        # Get user data
        user_id = token_data["user_id"]
        user_rows = self.users_df[self.users_df["id"] == user_id]
        if len(user_rows) == 0:
            return None
        
        user = user_rows.iloc[0].to_dict()
        
        # Create user response without password hash
        user_data = user.copy()
        user_data.pop("password_hash")
        
        return user_data
    
    def invalidate_tokens(self, user_id: int) -> None:
        """
        Invalidate all tokens for a user (logout).
        
        Args:
            user_id: User ID
        """
        self.tokens_df = self.tokens_df[self.tokens_df["user_id"] != user_id]
        self._save_tokens()