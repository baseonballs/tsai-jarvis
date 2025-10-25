"""
TSAI Platform Foundation Service: User Management & Accounts

This service provides comprehensive user lifecycle management, entitlements,
access control, security, and payment/subscription management for the TSAI platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4
from enum import Enum

import asyncpg
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from oauthlib.oauth2 import WebApplicationClient

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(str, Enum):
    """User roles in the TSAI platform"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"

class LicenseType(str, Enum):
    """License types for TSAI platform"""
    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional"
    INDIVIDUAL = "individual"
    TRIAL = "trial"
    OPEN_SOURCE = "open_source"

class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"

class UserData(BaseModel):
    """User data model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    email: EmailStr
    username: str
    password_hash: Optional[str] = None
    profile_data: Dict[str, Any] = Field(default_factory=dict)
    role: UserRole = UserRole.VIEWER
    license_type: LicenseType = LicenseType.TRIAL
    subscription_status: SubscriptionStatus = SubscriptionStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False

class EntitlementData(BaseModel):
    """User entitlement data"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    feature_name: str
    license_type: LicenseType
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class SubscriptionData(BaseModel):
    """Subscription data"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    plan_name: str
    status: SubscriptionStatus = SubscriptionStatus.PENDING
    billing_cycle: str = "monthly"
    next_billing_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class UserManagementService:
    """
    Foundation service for user management and accounts.
    
    Provides comprehensive user lifecycle management, entitlements,
    access control, security, and payment/subscription management.
    """
    
    def __init__(self, db_connection: asyncpg.Connection):
        self.db = db_connection
        self.oauth2_client = WebApplicationClient("tsai-platform")
        self.jwt_secret = "tsai-jwt-secret-key"  # Should be from environment
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        logger.info("🔐 UserManagementService initialized: Ready for user lifecycle management")
    
    async def create_user(self, user_data: UserData, password: str) -> UserData:
        """
        Create new user account with entitlements and initial subscription.
        
        Args:
            user_data: User data to create
            password: Plain text password to hash
            
        Returns:
            Created user data with hashed password
        """
        try:
            logger.info(f"👤 Creating user account: {user_data.email}")
            
            # Hash password
            password_hash = pwd_context.hash(password)
            user_data.password_hash = password_hash
            
            # Create user in database
            await self.db.execute("""
                INSERT INTO users (id, email, username, password_hash, profile_data, 
                                 role, license_type, subscription_status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, user_data.id, user_data.email, user_data.username, user_data.password_hash,
                user_data.profile_data, user_data.role.value, user_data.license_type.value,
                user_data.subscription_status.value, user_data.created_at, user_data.updated_at)
            
            # Create initial entitlements based on license type
            await self._create_initial_entitlements(user_data.id, user_data.license_type)
            
            # Create initial subscription
            await self._create_initial_subscription(user_data.id, user_data.license_type)
            
            logger.info(f"✅ User account created successfully: {user_data.email}")
            return user_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create user account: {e}")
            raise
    
    async def authenticate_user(self, email: str, password: str) -> Optional[UserData]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email
            password: Plain text password
            
        Returns:
            User data if authentication successful, None otherwise
        """
        try:
            logger.info(f"🔑 Authenticating user: {email}")
            
            # Get user from database
            user_row = await self.db.fetchrow("""
                SELECT * FROM users WHERE email = $1 AND is_active = true
            """, email)
            
            if not user_row:
                logger.warning(f"⚠️ User not found or inactive: {email}")
                return None
            
            # Verify password
            if not pwd_context.verify(password, user_row['password_hash']):
                logger.warning(f"⚠️ Invalid password for user: {email}")
                return None
            
            # Update last login
            await self.db.execute("""
                UPDATE users SET last_login = $1, updated_at = $2 WHERE id = $3
            """, datetime.utcnow(), datetime.utcnow(), user_row['id'])
            
            # Convert to UserData
            user_data = UserData(
                id=user_row['id'],
                email=user_row['email'],
                username=user_row['username'],
                password_hash=user_row['password_hash'],
                profile_data=user_row['profile_data'],
                role=UserRole(user_row['role']),
                license_type=LicenseType(user_row['license_type']),
                subscription_status=SubscriptionStatus(user_row['subscription_status']),
                created_at=user_row['created_at'],
                updated_at=user_row['updated_at'],
                last_login=user_row['last_login'],
                is_active=user_row['is_active'],
                is_verified=user_row.get('is_verified', False)
            )
            
            logger.info(f"✅ User authenticated successfully: {email}")
            return user_data
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return None
    
    async def check_entitlements(self, user_id: str, feature: str) -> bool:
        """
        Check user entitlements for specific features.
        
        Args:
            user_id: User ID
            feature: Feature name to check
            
        Returns:
            True if user has entitlement, False otherwise
        """
        try:
            logger.info(f"🔍 Checking entitlements for user {user_id}, feature: {feature}")
            
            # Check active entitlements
            entitlement = await self.db.fetchrow("""
                SELECT * FROM entitlements 
                WHERE user_id = $1 AND feature_name = $2 AND is_active = true
                AND (expires_at IS NULL OR expires_at > NOW())
            """, user_id, feature)
            
            has_entitlement = entitlement is not None
            logger.info(f"📋 Entitlement check result: {has_entitlement}")
            
            return has_entitlement
            
        except Exception as e:
            logger.error(f"❌ Failed to check entitlements: {e}")
            return False
    
    async def manage_subscription(self, user_id: str, plan: str, billing_cycle: str = "monthly") -> bool:
        """
        Manage user subscription and billing.
        
        Args:
            user_id: User ID
            plan: Subscription plan name
            billing_cycle: Billing cycle (monthly, yearly)
            
        Returns:
            True if subscription updated successfully
        """
        try:
            logger.info(f"💳 Managing subscription for user {user_id}, plan: {plan}")
            
            # Calculate next billing date
            next_billing = datetime.utcnow() + timedelta(days=30 if billing_cycle == "monthly" else 365)
            
            # Update subscription
            await self.db.execute("""
                UPDATE subscriptions 
                SET plan_name = $1, billing_cycle = $2, next_billing_date = $3, 
                    status = $4, updated_at = $5
                WHERE user_id = $6
            """, plan, billing_cycle, next_billing, SubscriptionStatus.ACTIVE.value, 
                datetime.utcnow(), user_id)
            
            # Update user license type based on plan
            license_type = self._map_plan_to_license(plan)
            await self.db.execute("""
                UPDATE users 
                SET license_type = $1, subscription_status = $2, updated_at = $3
                WHERE id = $4
            """, license_type.value, SubscriptionStatus.ACTIVE.value, datetime.utcnow(), user_id)
            
            logger.info(f"✅ Subscription updated successfully for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to manage subscription: {e}")
            return False
    
    async def generate_jwt_token(self, user_data: UserData) -> str:
        """
        Generate JWT token for user authentication.
        
        Args:
            user_data: User data
            
        Returns:
            JWT token string
        """
        try:
            logger.info(f"🎫 Generating JWT token for user: {user_data.email}")
            
            # Create JWT payload
            payload = {
                "user_id": user_data.id,
                "email": user_data.email,
                "role": user_data.role.value,
                "license_type": user_data.license_type.value,
                "exp": datetime.utcnow() + self.jwt_expiration,
                "iat": datetime.utcnow()
            }
            
            # Generate JWT token
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
            logger.info(f"✅ JWT token generated successfully for user: {user_data.email}")
            return token
            
        except Exception as e:
            logger.error(f"❌ Failed to generate JWT token: {e}")
            raise
    
    async def validate_jwt_token(self, token: str) -> Optional[UserData]:
        """
        Validate JWT token and return user data.
        
        Args:
            token: JWT token string
            
        Returns:
            User data if token valid, None otherwise
        """
        try:
            logger.info("🔍 Validating JWT token")
            
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Get user from database
            user_row = await self.db.fetchrow("""
                SELECT * FROM users WHERE id = $1 AND is_active = true
            """, payload['user_id'])
            
            if not user_row:
                logger.warning("⚠️ User not found or inactive")
                return None
            
            # Convert to UserData
            user_data = UserData(
                id=user_row['id'],
                email=user_row['email'],
                username=user_row['username'],
                password_hash=user_row['password_hash'],
                profile_data=user_row['profile_data'],
                role=UserRole(user_row['role']),
                license_type=LicenseType(user_row['license_type']),
                subscription_status=SubscriptionStatus(user_row['subscription_status']),
                created_at=user_row['created_at'],
                updated_at=user_row['updated_at'],
                last_login=user_row['last_login'],
                is_active=user_row['is_active'],
                is_verified=user_row.get('is_verified', False)
            )
            
            logger.info(f"✅ JWT token validated successfully for user: {user_data.email}")
            return user_data
            
        except JWTError as e:
            logger.warning(f"⚠️ JWT token validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ JWT token validation error: {e}")
            return None
    
    async def _create_initial_entitlements(self, user_id: str, license_type: LicenseType):
        """Create initial entitlements based on license type"""
        try:
            # Define entitlements by license type
            entitlements_map = {
                LicenseType.ENTERPRISE: [
                    "autopilot_training", "autopilot_inference", "spotlight_events",
                    "spotlight_highlights", "toolchain_development", "watson_nlp",
                    "holmes_curation", "advanced_analytics", "api_access"
                ],
                LicenseType.PROFESSIONAL: [
                    "autopilot_training", "autopilot_inference", "spotlight_events",
                    "spotlight_highlights", "toolchain_development", "watson_nlp",
                    "holmes_curation", "basic_analytics"
                ],
                LicenseType.INDIVIDUAL: [
                    "autopilot_inference", "spotlight_events", "spotlight_highlights",
                    "toolchain_development", "basic_analytics"
                ],
                LicenseType.TRIAL: [
                    "autopilot_inference", "spotlight_events", "basic_analytics"
                ],
                LicenseType.OPEN_SOURCE: [
                    "toolchain_development", "basic_analytics"
                ]
            }
            
            entitlements = entitlements_map.get(license_type, [])
            
            # Create entitlements
            for feature in entitlements:
                await self.db.execute("""
                    INSERT INTO entitlements (id, user_id, feature_name, license_type, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, str(uuid4()), user_id, feature, license_type.value, datetime.utcnow())
            
            logger.info(f"✅ Created {len(entitlements)} entitlements for license type: {license_type.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create initial entitlements: {e}")
            raise
    
    async def _create_initial_subscription(self, user_id: str, license_type: LicenseType):
        """Create initial subscription based on license type"""
        try:
            # Map license type to plan
            plan_map = {
                LicenseType.ENTERPRISE: "enterprise",
                LicenseType.PROFESSIONAL: "professional",
                LicenseType.INDIVIDUAL: "individual",
                LicenseType.TRIAL: "trial",
                LicenseType.OPEN_SOURCE: "open_source"
            }
            
            plan_name = plan_map.get(license_type, "trial")
            
            # Create subscription
            await self.db.execute("""
                INSERT INTO subscriptions (id, user_id, plan_name, status, billing_cycle, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, str(uuid4()), user_id, plan_name, SubscriptionStatus.ACTIVE.value, 
                "monthly", datetime.utcnow())
            
            logger.info(f"✅ Created initial subscription: {plan_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create initial subscription: {e}")
            raise
    
    def _map_plan_to_license(self, plan: str) -> LicenseType:
        """Map subscription plan to license type"""
        plan_mapping = {
            "enterprise": LicenseType.ENTERPRISE,
            "professional": LicenseType.PROFESSIONAL,
            "individual": LicenseType.INDIVIDUAL,
            "trial": LicenseType.TRIAL,
            "open_source": LicenseType.OPEN_SOURCE
        }
        return plan_mapping.get(plan, LicenseType.TRIAL)
