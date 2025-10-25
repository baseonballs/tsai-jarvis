"""
TSAI Jarvis - Enterprise Security API
Phase 2.6: Enterprise Security Implementation

This module implements comprehensive enterprise security features for hockey analytics,
including multi-factor authentication, role-based access control, data encryption,
audit logging, API security, and data privacy protection.
"""

import asyncio
import logging
import json
import time
import hashlib
import secrets
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import aiohttp
from aiofiles import open as aio_open
import sqlite3
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import math
import uuid
import jwt
import bcrypt
import pyotp
import qrcode
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
import uvicorn

# Security imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ Cryptography not available. Encryption features will use basic implementations.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security models
class User(BaseModel):
    """User model for authentication"""
    user_id: str
    username: str
    email: EmailStr
    password_hash: str
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

class Role(BaseModel):
    """Role model for RBAC"""
    role_id: str
    role_name: str
    description: str
    permissions: List[str] = []
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)

class Permission(BaseModel):
    """Permission model for RBAC"""
    permission_id: str
    permission_name: str
    resource: str
    action: str
    description: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)

class MFASetup(BaseModel):
    """MFA setup model"""
    user_id: str
    secret_key: str
    qr_code: str
    backup_codes: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

class AuditLog(BaseModel):
    """Audit log model"""
    log_id: str
    user_id: Optional[str] = None
    action: str
    resource: str
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str = "info"  # info, warning, error, critical

class SecurityEvent(BaseModel):
    """Security event model"""
    event_id: str
    event_type: str  # login_attempt, permission_denied, data_access, etc.
    user_id: Optional[str] = None
    details: Dict[str, Any] = {}
    severity: str = "medium"  # low, medium, high, critical
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False

class EncryptionKey(BaseModel):
    """Encryption key model"""
    key_id: str
    key_type: str  # symmetric, asymmetric
    key_data: str  # encrypted key data
    algorithm: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True

class DataPrivacyRule(BaseModel):
    """Data privacy rule model"""
    rule_id: str
    rule_name: str
    data_type: str
    privacy_level: str  # public, internal, confidential, restricted
    retention_period: int  # days
    encryption_required: bool = True
    access_controls: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

# Enterprise Security Engine
class EnterpriseSecurityEngine:
    """Enterprise Security Engine for hockey analytics"""
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Enterprise Security API",
            description="Comprehensive enterprise security features for hockey analytics",
            version="2.6.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add trusted host middleware
        self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        
        # Initialize components
        self.authentication = AuthenticationService()
        self.authorization = AuthorizationService()
        self.encryption = EncryptionService()
        self.audit_logging = AuditLoggingService()
        self.api_security = APISecurityService()
        self.data_privacy = DataPrivacyService()
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self.setup_routes()
        
        # Initialize database
        self.init_database()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Enterprise Security API",
                "version": "2.6.0",
                "status": "operational",
                "phase": "2.6 - Enterprise Security"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "authentication": "operational",
                    "authorization": "operational",
                    "encryption": "operational",
                    "audit_logging": "operational",
                    "api_security": "operational",
                    "data_privacy": "operational"
                }
            }
        
        # Authentication endpoints
        @self.app.post("/api/security/auth/register")
        async def register_user(user_data: Dict[str, Any]):
            """Register new user"""
            try:
                result = await self.authentication.register_user(user_data)
                return {"status": "success", "user_id": result["user_id"]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/security/auth/login")
        async def login_user(credentials: Dict[str, Any]):
            """Login user"""
            try:
                result = await self.authentication.login_user(credentials)
                return {"status": "success", "access_token": result["access_token"]}
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))
        
        @self.app.post("/api/security/auth/logout")
        async def logout_user(token: str):
            """Logout user"""
            try:
                result = await self.authentication.logout_user(token)
                return {"status": "success", "message": "Logged out successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/security/auth/refresh")
        async def refresh_token(refresh_token: str):
            """Refresh access token"""
            try:
                result = await self.authentication.refresh_token(refresh_token)
                return {"status": "success", "access_token": result["access_token"]}
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))
        
        # MFA endpoints
        @self.app.post("/api/security/mfa/setup")
        async def setup_mfa(user_id: str):
            """Setup MFA for user"""
            try:
                result = await self.authentication.setup_mfa(user_id)
                return {"status": "success", "mfa_setup": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/security/mfa/verify")
        async def verify_mfa(user_id: str, token: str):
            """Verify MFA token"""
            try:
                result = await self.authentication.verify_mfa(user_id, token)
                return {"status": "success", "verified": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Authorization endpoints
        @self.app.post("/api/security/rbac/roles")
        async def create_role(role_data: Dict[str, Any]):
            """Create new role"""
            try:
                result = await self.authorization.create_role(role_data)
                return {"status": "success", "role_id": result["role_id"]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/security/rbac/roles")
        async def get_roles():
            """Get all roles"""
            try:
                roles = await self.authorization.get_roles()
                return {"status": "success", "roles": roles}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/rbac/permissions")
        async def create_permission(permission_data: Dict[str, Any]):
            """Create new permission"""
            try:
                result = await self.authorization.create_permission(permission_data)
                return {"status": "success", "permission_id": result["permission_id"]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/security/rbac/permissions")
        async def get_permissions():
            """Get all permissions"""
            try:
                permissions = await self.authorization.get_permissions()
                return {"status": "success", "permissions": permissions}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/rbac/assign-role")
        async def assign_role(user_id: str, role_id: str):
            """Assign role to user"""
            try:
                result = await self.authorization.assign_role(user_id, role_id)
                return {"status": "success", "assignment": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/security/rbac/check-permission")
        async def check_permission(user_id: str, resource: str, action: str):
            """Check user permission"""
            try:
                result = await self.authorization.check_permission(user_id, resource, action)
                return {"status": "success", "permitted": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Encryption endpoints
        @self.app.post("/api/security/encryption/encrypt")
        async def encrypt_data(data: str, key_id: str = None):
            """Encrypt data"""
            try:
                result = await self.encryption.encrypt_data(data, key_id)
                return {"status": "success", "encrypted_data": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/encryption/decrypt")
        async def decrypt_data(encrypted_data: str, key_id: str = None):
            """Decrypt data"""
            try:
                result = await self.encryption.decrypt_data(encrypted_data, key_id)
                return {"status": "success", "decrypted_data": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/encryption/generate-key")
        async def generate_encryption_key(key_type: str = "symmetric"):
            """Generate new encryption key"""
            try:
                result = await self.encryption.generate_key(key_type)
                return {"status": "success", "key": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Audit logging endpoints
        @self.app.post("/api/security/audit/log")
        async def create_audit_log(log_data: Dict[str, Any]):
            """Create audit log entry"""
            try:
                result = await self.audit_logging.create_log(log_data)
                return {"status": "success", "log_id": result["log_id"]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/security/audit/logs")
        async def get_audit_logs(user_id: str = None, action: str = None, limit: int = 100):
            """Get audit logs"""
            try:
                logs = await self.audit_logging.get_logs(user_id, action, limit)
                return {"status": "success", "logs": logs}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/security/audit/security-events")
        async def get_security_events(severity: str = None, limit: int = 100):
            """Get security events"""
            try:
                events = await self.audit_logging.get_security_events(severity, limit)
                return {"status": "success", "events": events}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # API Security endpoints
        @self.app.get("/api/security/api/rate-limits")
        async def get_rate_limits():
            """Get API rate limits"""
            try:
                limits = await self.api_security.get_rate_limits()
                return {"status": "success", "rate_limits": limits}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/api/validate-request")
        async def validate_request(request_data: Dict[str, Any]):
            """Validate API request"""
            try:
                result = await self.api_security.validate_request(request_data)
                return {"status": "success", "valid": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Data Privacy endpoints
        @self.app.post("/api/security/privacy/rules")
        async def create_privacy_rule(rule_data: Dict[str, Any]):
            """Create data privacy rule"""
            try:
                result = await self.data_privacy.create_privacy_rule(rule_data)
                return {"status": "success", "rule_id": result["rule_id"]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/security/privacy/rules")
        async def get_privacy_rules():
            """Get data privacy rules"""
            try:
                rules = await self.data_privacy.get_privacy_rules()
                return {"status": "success", "rules": rules}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/security/privacy/classify-data")
        async def classify_data(data: Dict[str, Any]):
            """Classify data privacy level"""
            try:
                result = await self.data_privacy.classify_data(data)
                return {"status": "success", "classification": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint
        @self.app.websocket("/ws/security")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_connection(websocket)
    
    async def websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send security updates
                await self.send_security_updates(websocket)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def send_security_updates(self, websocket: WebSocket):
        """Send security updates"""
        try:
            # Get real-time security data
            security_data = await self.get_security_data()
            
            await websocket.send_json({
                "type": "security_update",
                "timestamp": datetime.now().isoformat(),
                "data": security_data
            })
        except Exception as e:
            logger.error(f"Error sending security updates: {e}")
    
    async def get_security_data(self):
        """Get real-time security data"""
        return {
            "authentication": await self.authentication.get_status(),
            "authorization": await self.authorization.get_status(),
            "encryption": await self.encryption.get_status(),
            "audit_logging": await self.audit_logging.get_status(),
            "api_security": await self.api_security.get_status(),
            "data_privacy": await self.data_privacy.get_status()
        }
    
    def init_database(self):
        """Initialize database for enterprise security"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    is_active BOOLEAN,
                    is_verified BOOLEAN,
                    mfa_enabled BOOLEAN,
                    mfa_secret TEXT,
                    roles TEXT,
                    permissions TEXT,
                    created_at TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER,
                    locked_until TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS roles (
                    role_id TEXT PRIMARY KEY,
                    role_name TEXT UNIQUE,
                    description TEXT,
                    permissions TEXT,
                    is_active BOOLEAN,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS permissions (
                    permission_id TEXT PRIMARY KEY,
                    permission_name TEXT UNIQUE,
                    resource TEXT,
                    action TEXT,
                    description TEXT,
                    is_active BOOLEAN,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    action TEXT,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP,
                    severity TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    user_id TEXT,
                    details TEXT,
                    severity TEXT,
                    timestamp TIMESTAMP,
                    resolved BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    key_id TEXT PRIMARY KEY,
                    key_type TEXT,
                    key_data TEXT,
                    algorithm TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_privacy_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT,
                    data_type TEXT,
                    privacy_level TEXT,
                    retention_period INTEGER,
                    encryption_required BOOLEAN,
                    access_controls TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enterprise security database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

# Authentication Service
class AuthenticationService:
    """Authentication Service"""
    
    def __init__(self):
        self.status = "operational"
        self.active_sessions = {}
        self.jwt_secret = "your-secret-key"  # In production, use environment variable
        self.jwt_algorithm = "HS256"
        self.session_timeout = 3600  # 1 hour
    
    async def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register new user"""
        try:
            user_id = str(uuid.uuid4())
            password_hash = self._hash_password(user_data["password"])
            
            # Create user record
            user = User(
                user_id=user_id,
                username=user_data["username"],
                email=user_data["email"],
                password_hash=password_hash,
                roles=["user"],  # Default role
                permissions=["read:profile"]  # Default permissions
            )
            
            # Store user in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (user_id, username, email, password_hash, is_active, 
                                 is_verified, mfa_enabled, roles, permissions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.username, user.email, user.password_hash,
                user.is_active, user.is_verified, user.mfa_enabled,
                json.dumps(user.roles), json.dumps(user.permissions),
                user.created_at
            ))
            
            conn.commit()
            conn.close()
            
            return {"user_id": user_id, "status": "registered"}
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise e
    
    async def login_user(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Login user"""
        try:
            username = credentials["username"]
            password = credentials["password"]
            
            # Get user from database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("Invalid credentials")
            
            user_data = {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "password_hash": row[3],
                "is_active": bool(row[4]),
                "is_verified": bool(row[5]),
                "mfa_enabled": bool(row[6]),
                "roles": json.loads(row[8]) if row[8] else [],
                "permissions": json.loads(row[9]) if row[9] else [],
                "failed_login_attempts": row[12] or 0,
                "locked_until": row[13]
            }
            
            conn.close()
            
            # Check if account is locked
            if user_data["locked_until"] and datetime.now() < datetime.fromisoformat(user_data["locked_until"]):
                raise Exception("Account is locked due to too many failed login attempts")
            
            # Verify password
            if not self._verify_password(password, user_data["password_hash"]):
                # Increment failed login attempts
                await self._increment_failed_attempts(user_data["user_id"])
                raise Exception("Invalid credentials")
            
            # Reset failed login attempts
            await self._reset_failed_attempts(user_data["user_id"])
            
            # Generate JWT token
            access_token = self._generate_jwt_token(user_data["user_id"], user_data["roles"])
            refresh_token = self._generate_refresh_token(user_data["user_id"])
            
            # Update last login
            await self._update_last_login(user_data["user_id"])
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user_data["user_id"],
                "roles": user_data["roles"],
                "permissions": user_data["permissions"]
            }
            
        except Exception as e:
            logger.error(f"Error logging in user: {e}")
            raise e
    
    async def logout_user(self, token: str) -> Dict[str, Any]:
        """Logout user"""
        try:
            # Decode and validate token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload["sub"]
            
            # Remove from active sessions
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            
            return {"status": "logged_out"}
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            raise e
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload["sub"]
            
            # Get user data
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT roles FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("User not found")
            
            roles = json.loads(row[0]) if row[0] else []
            conn.close()
            
            # Generate new access token
            access_token = self._generate_jwt_token(user_id, roles)
            
            return {"access_token": access_token}
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise e
    
    async def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """Setup MFA for user"""
        try:
            # Generate MFA secret
            secret = pyotp.random_base32()
            
            # Generate QR code
            totp = pyotp.TOTP(secret)
            qr_code = totp.provisioning_uri(
                name=user_id,
                issuer_name="TSAI Jarvis"
            )
            
            # Generate backup codes
            backup_codes = [secrets.token_hex(4) for _ in range(10)]
            
            # Store MFA setup
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET mfa_enabled = ?, mfa_secret = ? WHERE user_id = ?
            ''', (True, secret, user_id))
            
            conn.commit()
            conn.close()
            
            return {
                "secret": secret,
                "qr_code": qr_code,
                "backup_codes": backup_codes
            }
            
        except Exception as e:
            logger.error(f"Error setting up MFA: {e}")
            raise e
    
    async def verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify MFA token"""
        try:
            # Get user MFA secret
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT mfa_secret FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if not row or not row[0]:
                raise Exception("MFA not enabled for user")
            
            secret = row[0]
            conn.close()
            
            # Verify token
            totp = pyotp.TOTP(secret)
            return totp.verify(token)
            
        except Exception as e:
            logger.error(f"Error verifying MFA: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_jwt_token(self, user_id: str, roles: List[str]) -> str:
        """Generate JWT access token"""
        payload = {
            "sub": user_id,
            "roles": roles,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _generate_refresh_token(self, user_id: str) -> str:
        """Generate JWT refresh token"""
        payload = {
            "sub": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=30)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _increment_failed_attempts(self, user_id: str):
        """Increment failed login attempts"""
        conn = sqlite3.connect('enterprise_security.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET failed_login_attempts = failed_login_attempts + 1,
                           locked_until = ? WHERE user_id = ?
        ''', (datetime.now() + timedelta(minutes=15), user_id))
        
        conn.commit()
        conn.close()
    
    async def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts"""
        conn = sqlite3.connect('enterprise_security.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET failed_login_attempts = 0, locked_until = NULL 
            WHERE user_id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    async def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        conn = sqlite3.connect('enterprise_security.db')
        cursor = conn.cursor()
        
        cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?', 
                      (datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get authentication service status"""
        return {
            "status": self.status,
            "active_sessions": len(self.active_sessions),
            "last_updated": datetime.now().isoformat()
        }

# Authorization Service
class AuthorizationService:
    """Authorization Service for RBAC"""
    
    def __init__(self):
        self.status = "operational"
        self.role_cache = {}
        self.permission_cache = {}
    
    async def create_role(self, role_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new role"""
        try:
            role_id = str(uuid.uuid4())
            
            role = Role(
                role_id=role_id,
                role_name=role_data["role_name"],
                description=role_data["description"],
                permissions=role_data.get("permissions", [])
            )
            
            # Store role in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO roles (role_id, role_name, description, permissions, 
                                 is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                role.role_id, role.role_name, role.description,
                json.dumps(role.permissions), role.is_active, role.created_at
            ))
            
            conn.commit()
            conn.close()
            
            return {"role_id": role_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating role: {e}")
            raise e
    
    async def get_roles(self) -> List[Dict[str, Any]]:
        """Get all roles"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM roles WHERE is_active = 1')
            rows = cursor.fetchall()
            conn.close()
            
            roles = []
            for row in rows:
                roles.append({
                    "role_id": row[0],
                    "role_name": row[1],
                    "description": row[2],
                    "permissions": json.loads(row[3]) if row[3] else [],
                    "is_active": bool(row[4]),
                    "created_at": row[5]
                })
            
            return roles
            
        except Exception as e:
            logger.error(f"Error getting roles: {e}")
            raise e
    
    async def create_permission(self, permission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new permission"""
        try:
            permission_id = str(uuid.uuid4())
            
            permission = Permission(
                permission_id=permission_id,
                permission_name=permission_data["permission_name"],
                resource=permission_data["resource"],
                action=permission_data["action"],
                description=permission_data["description"]
            )
            
            # Store permission in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO permissions (permission_id, permission_name, resource, 
                                      action, description, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                permission.permission_id, permission.permission_name,
                permission.resource, permission.action, permission.description,
                permission.is_active, permission.created_at
            ))
            
            conn.commit()
            conn.close()
            
            return {"permission_id": permission_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating permission: {e}")
            raise e
    
    async def get_permissions(self) -> List[Dict[str, Any]]:
        """Get all permissions"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM permissions WHERE is_active = 1')
            rows = cursor.fetchall()
            conn.close()
            
            permissions = []
            for row in rows:
                permissions.append({
                    "permission_id": row[0],
                    "permission_name": row[1],
                    "resource": row[2],
                    "action": row[3],
                    "description": row[4],
                    "is_active": bool(row[5]),
                    "created_at": row[6]
                })
            
            return permissions
            
        except Exception as e:
            logger.error(f"Error getting permissions: {e}")
            raise e
    
    async def assign_role(self, user_id: str, role_id: str) -> Dict[str, Any]:
        """Assign role to user"""
        try:
            # Get role permissions
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT permissions FROM roles WHERE role_id = ?', (role_id,))
            role_row = cursor.fetchone()
            
            if not role_row:
                raise Exception("Role not found")
            
            role_permissions = json.loads(role_row[0]) if role_row[0] else []
            
            # Get current user roles and permissions
            cursor.execute('SELECT roles, permissions FROM users WHERE user_id = ?', (user_id,))
            user_row = cursor.fetchone()
            
            if not user_row:
                raise Exception("User not found")
            
            current_roles = json.loads(user_row[0]) if user_row[0] else []
            current_permissions = json.loads(user_row[1]) if user_row[1] else []
            
            # Add role and permissions
            if role_id not in current_roles:
                current_roles.append(role_id)
            
            for permission in role_permissions:
                if permission not in current_permissions:
                    current_permissions.append(permission)
            
            # Update user
            cursor.execute('''
                UPDATE users SET roles = ?, permissions = ? WHERE user_id = ?
            ''', (json.dumps(current_roles), json.dumps(current_permissions), user_id))
            
            conn.commit()
            conn.close()
            
            return {"user_id": user_id, "role_id": role_id, "status": "assigned"}
            
        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            raise e
    
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission"""
        try:
            # Get user permissions
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT permissions FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            user_permissions = json.loads(row[0]) if row[0] else []
            conn.close()
            
            # Check if user has the required permission
            required_permission = f"{action}:{resource}"
            return required_permission in user_permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get authorization service status"""
        return {
            "status": self.status,
            "cached_roles": len(self.role_cache),
            "cached_permissions": len(self.permission_cache),
            "last_updated": datetime.now().isoformat()
        }

# Encryption Service
class EncryptionService:
    """Encryption Service"""
    
    def __init__(self):
        self.status = "operational"
        self.encryption_keys = {}
        self._initialize_default_key()
    
    def _initialize_default_key(self):
        """Initialize default encryption key"""
        if CRYPTO_AVAILABLE:
            # Generate Fernet key
            key = Fernet.generate_key()
            self.encryption_keys["default"] = Fernet(key)
        else:
            # Basic encryption fallback
            self.encryption_keys["default"] = None
    
    async def encrypt_data(self, data: str, key_id: str = None) -> str:
        """Encrypt data"""
        try:
            key_id = key_id or "default"
            
            if CRYPTO_AVAILABLE and key_id in self.encryption_keys:
                # Use Fernet encryption
                encrypted_data = self.encryption_keys[key_id].encrypt(data.encode())
                return base64.b64encode(encrypted_data).decode()
            else:
                # Basic encryption fallback
                return self._basic_encrypt(data)
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise e
    
    async def decrypt_data(self, encrypted_data: str, key_id: str = None) -> str:
        """Decrypt data"""
        try:
            key_id = key_id or "default"
            
            if CRYPTO_AVAILABLE and key_id in self.encryption_keys:
                # Use Fernet decryption
                decoded_data = base64.b64decode(encrypted_data.encode())
                decrypted_data = self.encryption_keys[key_id].decrypt(decoded_data)
                return decrypted_data.decode()
            else:
                # Basic decryption fallback
                return self._basic_decrypt(encrypted_data)
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise e
    
    async def generate_key(self, key_type: str = "symmetric") -> Dict[str, Any]:
        """Generate new encryption key"""
        try:
            key_id = str(uuid.uuid4())
            
            if CRYPTO_AVAILABLE and key_type == "symmetric":
                # Generate Fernet key
                key = Fernet.generate_key()
                self.encryption_keys[key_id] = Fernet(key)
                
                # Store key in database
                conn = sqlite3.connect('enterprise_security.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO encryption_keys (key_id, key_type, key_data, 
                                              algorithm, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    key_id, key_type, base64.b64encode(key).decode(),
                    "Fernet", datetime.now(), True
                ))
                
                conn.commit()
                conn.close()
                
                return {"key_id": key_id, "key_type": key_type, "algorithm": "Fernet"}
            else:
                # Basic key generation
                key_data = secrets.token_hex(32)
                return {"key_id": key_id, "key_type": key_type, "algorithm": "basic"}
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise e
    
    def _basic_encrypt(self, data: str) -> str:
        """Basic encryption fallback"""
        # Simple XOR encryption (not secure for production)
        key = "tsai-jarvis-security-key"
        encrypted = ""
        for i, char in enumerate(data):
            encrypted += chr(ord(char) ^ ord(key[i % len(key)]))
        return base64.b64encode(encrypted.encode()).decode()
    
    def _basic_decrypt(self, encrypted_data: str) -> str:
        """Basic decryption fallback"""
        # Simple XOR decryption
        key = "tsai-jarvis-security-key"
        decoded = base64.b64decode(encrypted_data.encode()).decode()
        decrypted = ""
        for i, char in enumerate(decoded):
            decrypted += chr(ord(char) ^ ord(key[i % len(key)]))
        return decrypted
    
    async def get_status(self) -> Dict[str, Any]:
        """Get encryption service status"""
        return {
            "status": self.status,
            "available_keys": len(self.encryption_keys),
            "crypto_available": CRYPTO_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }

# Audit Logging Service
class AuditLoggingService:
    """Audit Logging Service"""
    
    def __init__(self):
        self.status = "operational"
        self.log_count = 0
        self.security_event_count = 0
    
    async def create_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit log entry"""
        try:
            log_id = str(uuid.uuid4())
            
            log = AuditLog(
                log_id=log_id,
                user_id=log_data.get("user_id"),
                action=log_data["action"],
                resource=log_data["resource"],
                details=log_data.get("details", {}),
                ip_address=log_data.get("ip_address"),
                user_agent=log_data.get("user_agent"),
                severity=log_data.get("severity", "info")
            )
            
            # Store log in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs (log_id, user_id, action, resource, details,
                                     ip_address, user_agent, timestamp, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log.log_id, log.user_id, log.action, log.resource,
                json.dumps(log.details), log.ip_address, log.user_agent,
                log.timestamp, log.severity
            ))
            
            conn.commit()
            conn.close()
            
            self.log_count += 1
            return {"log_id": log_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            raise e
    
    async def get_logs(self, user_id: str = None, action: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            logs = []
            for row in rows:
                logs.append({
                    "log_id": row[0],
                    "user_id": row[1],
                    "action": row[2],
                    "resource": row[3],
                    "details": json.loads(row[4]) if row[4] else {},
                    "ip_address": row[5],
                    "user_agent": row[6],
                    "timestamp": row[7],
                    "severity": row[8]
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            raise e
    
    async def get_security_events(self, severity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security events"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            query = "SELECT * FROM security_events WHERE 1=1"
            params = []
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                events.append({
                    "event_id": row[0],
                    "event_type": row[1],
                    "user_id": row[2],
                    "details": json.loads(row[3]) if row[3] else {},
                    "severity": row[4],
                    "timestamp": row[5],
                    "resolved": bool(row[6])
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            raise e
    
    async def create_security_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create security event"""
        try:
            event_id = str(uuid.uuid4())
            
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_data["event_type"],
                user_id=event_data.get("user_id"),
                details=event_data.get("details", {}),
                severity=event_data.get("severity", "medium")
            )
            
            # Store event in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events (event_id, event_type, user_id, details,
                                           severity, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.event_type, event.user_id,
                json.dumps(event.details), event.severity, event.timestamp, event.resolved
            ))
            
            conn.commit()
            conn.close()
            
            self.security_event_count += 1
            return {"event_id": event_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating security event: {e}")
            raise e
    
    async def get_status(self) -> Dict[str, Any]:
        """Get audit logging service status"""
        return {
            "status": self.status,
            "log_count": self.log_count,
            "security_event_count": self.security_event_count,
            "last_updated": datetime.now().isoformat()
        }

# API Security Service
class APISecurityService:
    """API Security Service"""
    
    def __init__(self):
        self.status = "operational"
        self.rate_limits = {}
        self.request_counts = defaultdict(int)
        self.blocked_ips = set()
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get API rate limits"""
        return {
            "default": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            },
            "authenticated": {
                "requests_per_minute": 120,
                "requests_per_hour": 2000,
                "requests_per_day": 20000
            },
            "admin": {
                "requests_per_minute": 300,
                "requests_per_hour": 5000,
                "requests_per_day": 50000
            }
        }
    
    async def validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate API request"""
        try:
            # Check rate limits
            client_ip = request_data.get("ip_address", "unknown")
            user_id = request_data.get("user_id")
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                return False
            
            # Check rate limits
            if not await self._check_rate_limit(client_ip, user_id):
                # Block IP if rate limit exceeded
                self.blocked_ips.add(client_ip)
                return False
            
            # Validate request format
            if not self._validate_request_format(request_data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return False
    
    async def _check_rate_limit(self, client_ip: str, user_id: str = None) -> bool:
        """Check rate limit for client"""
        try:
            current_time = time.time()
            minute_key = f"{client_ip}:{int(current_time // 60)}"
            hour_key = f"{client_ip}:{int(current_time // 3600)}"
            
            # Check minute rate limit
            if self.request_counts[minute_key] > 60:
                return False
            
            # Check hour rate limit
            if self.request_counts[hour_key] > 1000:
                return False
            
            # Increment counters
            self.request_counts[minute_key] += 1
            self.request_counts[hour_key] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    def _validate_request_format(self, request_data: Dict[str, Any]) -> bool:
        """Validate request format"""
        required_fields = ["endpoint", "method", "timestamp"]
        
        for field in required_fields:
            if field not in request_data:
                return False
        
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get API security service status"""
        return {
            "status": self.status,
            "blocked_ips": len(self.blocked_ips),
            "active_rate_limits": len(self.request_counts),
            "last_updated": datetime.now().isoformat()
        }

# Data Privacy Service
class DataPrivacyService:
    """Data Privacy Service"""
    
    def __init__(self):
        self.status = "operational"
        self.privacy_rules = {}
        self.data_classifications = {}
    
    async def create_privacy_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data privacy rule"""
        try:
            rule_id = str(uuid.uuid4())
            
            rule = DataPrivacyRule(
                rule_id=rule_id,
                rule_name=rule_data["rule_name"],
                data_type=rule_data["data_type"],
                privacy_level=rule_data["privacy_level"],
                retention_period=rule_data["retention_period"],
                encryption_required=rule_data.get("encryption_required", True),
                access_controls=rule_data.get("access_controls", [])
            )
            
            # Store rule in database
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_privacy_rules (rule_id, rule_name, data_type, 
                                              privacy_level, retention_period,
                                              encryption_required, access_controls, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id, rule.rule_name, rule.data_type, rule.privacy_level,
                rule.retention_period, rule.encryption_required,
                json.dumps(rule.access_controls), rule.created_at
            ))
            
            conn.commit()
            conn.close()
            
            return {"rule_id": rule_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating privacy rule: {e}")
            raise e
    
    async def get_privacy_rules(self) -> List[Dict[str, Any]]:
        """Get data privacy rules"""
        try:
            conn = sqlite3.connect('enterprise_security.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM data_privacy_rules')
            rows = cursor.fetchall()
            conn.close()
            
            rules = []
            for row in rows:
                rules.append({
                    "rule_id": row[0],
                    "rule_name": row[1],
                    "data_type": row[2],
                    "privacy_level": row[3],
                    "retention_period": row[4],
                    "encryption_required": bool(row[5]),
                    "access_controls": json.loads(row[6]) if row[6] else [],
                    "created_at": row[7]
                })
            
            return rules
            
        except Exception as e:
            logger.error(f"Error getting privacy rules: {e}")
            raise e
    
    async def classify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify data privacy level"""
        try:
            # Simple data classification based on content
            privacy_level = "public"
            encryption_required = False
            retention_period = 365  # days
            
            # Check for sensitive data patterns
            sensitive_patterns = [
                "password", "ssn", "credit_card", "bank_account",
                "personal_id", "medical", "financial"
            ]
            
            data_content = json.dumps(data).lower()
            
            for pattern in sensitive_patterns:
                if pattern in data_content:
                    privacy_level = "restricted"
                    encryption_required = True
                    retention_period = 30
                    break
            
            # Check for internal data patterns
            internal_patterns = [
                "internal", "confidential", "proprietary", "trade_secret"
            ]
            
            for pattern in internal_patterns:
                if pattern in data_content:
                    if privacy_level == "public":
                        privacy_level = "internal"
                        encryption_required = True
                        retention_period = 90
                    break
            
            return {
                "privacy_level": privacy_level,
                "encryption_required": encryption_required,
                "retention_period": retention_period,
                "classification_confidence": 0.85,
                "classification_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error classifying data: {e}")
            raise e
    
    async def get_status(self) -> Dict[str, Any]:
        """Get data privacy service status"""
        return {
            "status": self.status,
            "privacy_rules": len(self.privacy_rules),
            "data_classifications": len(self.data_classifications),
            "last_updated": datetime.now().isoformat()
        }

# Main execution
if __name__ == "__main__":
    # Initialize enterprise security engine
    engine = EnterpriseSecurityEngine()
    
    # Run the application
    uvicorn.run(
        engine.app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )
