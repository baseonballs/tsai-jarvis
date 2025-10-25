"""
TSAI Platform Foundation Service: Session Management

This service provides cross-application session management with SSO and CORS support
for the TSAI platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4
from enum import Enum

import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel, Field
from jose import JWTError, jwt
import secrets

logger = logging.getLogger(__name__)

class SessionStatus(str, Enum):
    """Session status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

class SessionType(str, Enum):
    """Session types"""
    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    SSO = "sso"

class SessionData(BaseModel):
    """Session data model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    refresh_token: Optional[str] = Field(default_factory=lambda: secrets.token_urlsafe(32))
    session_type: SessionType = SessionType.WEB
    status: SessionStatus = SessionStatus.ACTIVE
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    app_context: str = "tsai-platform"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None

class SessionValidation(BaseModel):
    """Session validation result"""
    is_valid: bool
    session_data: Optional[SessionData] = None
    error_message: Optional[str] = None
    requires_refresh: bool = False

class SSOProvider(BaseModel):
    """SSO provider configuration"""
    provider_name: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    user_info_url: str
    redirect_uri: str
    scope: List[str] = ["openid", "profile", "email"]

class SessionManagementService:
    """
    Foundation service for session management.
    
    Provides cross-application session management with SSO and CORS support.
    """
    
    def __init__(self, db_connection: asyncpg.Connection, redis_connection: redis.Redis):
        self.db = db_connection
        self.redis = redis_connection
        self.jwt_secret = "tsai-session-secret-key"  # Should be from environment
        self.jwt_algorithm = "HS256"
        self.session_timeout = timedelta(hours=24)
        self.refresh_timeout = timedelta(days=30)
        self.sso_providers: Dict[str, SSOProvider] = {}
        
        logger.info("🔐 SessionManagementService initialized: Ready for cross-application session management")
    
    async def create_session(self, user_id: str, app_context: str, session_type: SessionType = SessionType.WEB, 
                           ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> SessionData:
        """
        Create cross-application session.
        
        Args:
            user_id: User ID
            app_context: Application context (autopilot, spotlight, etc.)
            session_type: Type of session
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session data
        """
        try:
            logger.info(f"🔑 Creating session for user: {user_id}, context: {app_context}")
            
            # Generate session tokens
            session_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)
            
            # Calculate expiration times
            expires_at = datetime.utcnow() + self.session_timeout
            refresh_expires_at = datetime.utcnow() + self.refresh_timeout
            
            # Create session data
            session_data = SessionData(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                session_type=session_type,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                app_context=app_context
            )
            
            # Store session in database
            await self.db.execute("""
                INSERT INTO users.user_sessions (id, user_id, session_token, refresh_token, 
                                               expires_at, ip_address, user_agent, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, session_data.id, session_data.user_id, session_data.session_token, 
                session_data.refresh_token, session_data.expires_at, session_data.ip_address,
                session_data.user_agent, session_data.is_active, session_data.created_at, 
                session_data.updated_at)
            
            # Store session in Redis for fast access
            await self._cache_session(session_data)
            
            # Generate JWT token for the session
            jwt_token = await self._generate_session_jwt(session_data)
            
            logger.info(f"✅ Session created successfully: {session_data.id}")
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise
    
    async def validate_session(self, session_token: str) -> SessionValidation:
        """
        Validate session token.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Session validation result
        """
        try:
            logger.info("🔍 Validating session token")
            
            # First check Redis cache
            cached_session = await self._get_cached_session(session_token)
            if cached_session:
                # Update last accessed time
                await self._update_session_access(session_token)
                return SessionValidation(
                    is_valid=True,
                    session_data=cached_session,
                    requires_refresh=False
                )
            
            # Check database
            session_row = await self.db.fetchrow("""
                SELECT * FROM users.user_sessions 
                WHERE session_token = $1 AND is_active = true AND expires_at > NOW()
            """, session_token)
            
            if not session_row:
                logger.warning("⚠️ Session not found or expired")
                return SessionValidation(
                    is_valid=False,
                    error_message="Session not found or expired"
                )
            
            # Convert to SessionData
            session_data = SessionData(
                id=session_row['id'],
                user_id=session_row['user_id'],
                session_token=session_row['session_token'],
                refresh_token=session_row['refresh_token'],
                session_type=SessionType(session_row.get('session_type', 'web')),
                status=SessionStatus(session_row.get('status', 'active')),
                expires_at=session_row['expires_at'],
                ip_address=session_row['ip_address'],
                user_agent=session_row['user_agent'],
                app_context=session_row.get('app_context', 'tsai-platform'),
                is_active=session_row['is_active'],
                created_at=session_row['created_at'],
                updated_at=session_row['updated_at'],
                last_accessed=session_row.get('last_accessed')
            )
            
            # Cache session for future requests
            await self._cache_session(session_data)
            
            # Update last accessed time
            await self._update_session_access(session_token)
            
            logger.info(f"✅ Session validated successfully: {session_data.id}")
            return SessionValidation(
                is_valid=True,
                session_data=session_data,
                requires_refresh=False
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to validate session: {e}")
            return SessionValidation(
                is_valid=False,
                error_message=f"Session validation error: {str(e)}"
            )
    
    async def refresh_session(self, refresh_token: str) -> Optional[SessionData]:
        """
        Refresh session using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New session data if refresh successful
        """
        try:
            logger.info("🔄 Refreshing session")
            
            # Validate refresh token
            session_row = await self.db.fetchrow("""
                SELECT * FROM users.user_sessions 
                WHERE refresh_token = $1 AND is_active = true AND expires_at > NOW()
            """, refresh_token)
            
            if not session_row:
                logger.warning("⚠️ Invalid refresh token")
                return None
            
            # Generate new session token
            new_session_token = secrets.token_urlsafe(32)
            new_refresh_token = secrets.token_urlsafe(32)
            new_expires_at = datetime.utcnow() + self.session_timeout
            
            # Update session in database
            await self.db.execute("""
                UPDATE users.user_sessions 
                SET session_token = $1, refresh_token = $2, expires_at = $3, updated_at = $4
                WHERE id = $5
            """, new_session_token, new_refresh_token, new_expires_at, datetime.utcnow(), session_row['id'])
            
            # Create new session data
            session_data = SessionData(
                id=session_row['id'],
                user_id=session_row['user_id'],
                session_token=new_session_token,
                refresh_token=new_refresh_token,
                session_type=SessionType(session_row.get('session_type', 'web')),
                status=SessionStatus.ACTIVE,
                expires_at=new_expires_at,
                ip_address=session_row['ip_address'],
                user_agent=session_row['user_agent'],
                app_context=session_row.get('app_context', 'tsai-platform'),
                is_active=True,
                created_at=session_row['created_at'],
                updated_at=datetime.utcnow()
            )
            
            # Update cache
            await self._cache_session(session_data)
            
            logger.info(f"✅ Session refreshed successfully: {session_data.id}")
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh session: {e}")
            return None
    
    async def terminate_session(self, session_token: str) -> bool:
        """
        Terminate user session.
        
        Args:
            session_token: Session token to terminate
            
        Returns:
            True if session terminated successfully
        """
        try:
            logger.info(f"🛑 Terminating session: {session_token}")
            
            # Update session status in database
            result = await self.db.execute("""
                UPDATE users.user_sessions 
                SET is_active = false, status = $1, updated_at = $2
                WHERE session_token = $3
            """, SessionStatus.REVOKED.value, datetime.utcnow(), session_token)
            
            # Remove from cache
            await self._remove_cached_session(session_token)
            
            if result == "UPDATE 1":
                logger.info(f"✅ Session terminated successfully: {session_token}")
                return True
            else:
                logger.warning(f"⚠️ Session not found: {session_token}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to terminate session: {e}")
            return False
    
    async def terminate_all_user_sessions(self, user_id: str) -> int:
        """
        Terminate all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions terminated
        """
        try:
            logger.info(f"🛑 Terminating all sessions for user: {user_id}")
            
            # Get all active sessions for user
            sessions = await self.db.fetch("""
                SELECT session_token FROM users.user_sessions 
                WHERE user_id = $1 AND is_active = true
            """, user_id)
            
            # Terminate each session
            terminated_count = 0
            for session in sessions:
                if await self.terminate_session(session['session_token']):
                    terminated_count += 1
            
            logger.info(f"✅ Terminated {terminated_count} sessions for user: {user_id}")
            return terminated_count
            
        except Exception as e:
            logger.error(f"❌ Failed to terminate all user sessions: {e}")
            return 0
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active sessions
        """
        try:
            logger.info(f"📋 Getting sessions for user: {user_id}")
            
            # Get sessions from database
            session_rows = await self.db.fetch("""
                SELECT * FROM users.user_sessions 
                WHERE user_id = $1 AND is_active = true AND expires_at > NOW()
                ORDER BY created_at DESC
            """, user_id)
            
            # Convert to SessionData objects
            sessions = []
            for row in session_rows:
                session_data = SessionData(
                    id=row['id'],
                    user_id=row['user_id'],
                    session_token=row['session_token'],
                    refresh_token=row['refresh_token'],
                    session_type=SessionType(row.get('session_type', 'web')),
                    status=SessionStatus(row.get('status', 'active')),
                    expires_at=row['expires_at'],
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    app_context=row.get('app_context', 'tsai-platform'),
                    is_active=row['is_active'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_accessed=row.get('last_accessed')
                )
                sessions.append(session_data)
            
            logger.info(f"✅ Found {len(sessions)} active sessions for user: {user_id}")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get user sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            logger.info("🧹 Cleaning up expired sessions")
            
            # Get expired sessions
            expired_sessions = await self.db.fetch("""
                SELECT session_token FROM users.user_sessions 
                WHERE expires_at < NOW() AND is_active = true
            """)
            
            # Terminate expired sessions
            cleaned_count = 0
            for session in expired_sessions:
                if await self.terminate_session(session['session_token']):
                    cleaned_count += 1
            
            logger.info(f"✅ Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired sessions: {e}")
            return 0
    
    async def _cache_session(self, session_data: SessionData):
        """Cache session in Redis"""
        try:
            cache_key = f"session:{session_data.session_token}"
            cache_data = session_data.dict()
            
            # Cache for session duration
            ttl = int((session_data.expires_at - datetime.utcnow()).total_seconds())
            await self.redis.setex(cache_key, ttl, str(cache_data))
            
        except Exception as e:
            logger.error(f"❌ Failed to cache session: {e}")
    
    async def _get_cached_session(self, session_token: str) -> Optional[SessionData]:
        """Get session from Redis cache"""
        try:
            cache_key = f"session:{session_token}"
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                session_dict = eval(cached_data)  # In production, use proper JSON parsing
                return SessionData(**session_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached session: {e}")
            return None
    
    async def _remove_cached_session(self, session_token: str):
        """Remove session from Redis cache"""
        try:
            cache_key = f"session:{session_token}"
            await self.redis.delete(cache_key)
        except Exception as e:
            logger.error(f"❌ Failed to remove cached session: {e}")
    
    async def _update_session_access(self, session_token: str):
        """Update session last accessed time"""
        try:
            await self.db.execute("""
                UPDATE users.user_sessions 
                SET last_accessed = $1, updated_at = $2
                WHERE session_token = $3
            """, datetime.utcnow(), datetime.utcnow(), session_token)
        except Exception as e:
            logger.error(f"❌ Failed to update session access: {e}")
    
    async def _generate_session_jwt(self, session_data: SessionData) -> str:
        """Generate JWT token for session"""
        try:
            payload = {
                "session_id": session_data.id,
                "user_id": session_data.user_id,
                "session_token": session_data.session_token,
                "app_context": session_data.app_context,
                "exp": session_data.expires_at,
                "iat": datetime.utcnow()
            }
            
            return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate session JWT: {e}")
            raise
