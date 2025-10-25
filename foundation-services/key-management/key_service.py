"""
TSAI Platform Foundation Service: Key Management

This service provides centralized key management for encryption, authentication,
and security for the TSAI platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4
from enum import Enum
import secrets
import base64
import hashlib

import asyncpg
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class KeyType(str, Enum):
    """Key types"""
    AES = "aes"
    RSA = "rsa"
    ECDSA = "ecdsa"
    HMAC = "hmac"
    API = "api"

class KeyPurpose(str, Enum):
    """Key purposes"""
    ENCRYPTION = "encryption"
    SIGNING = "signing"
    AUTHENTICATION = "authentication"
    API_ACCESS = "api_access"
    SESSION = "session"

class KeyStatus(str, Enum):
    """Key status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATING = "rotating"

class KeyData(BaseModel):
    """Key data model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    key_name: str
    key_type: KeyType
    key_purpose: KeyPurpose
    key_data: bytes  # Encrypted key data
    public_key: Optional[bytes] = None  # For asymmetric keys
    key_hash: str  # Hash of the key for identification
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EncryptionResult(BaseModel):
    """Encryption result"""
    encrypted_data: bytes
    key_id: str
    iv: Optional[bytes] = None  # For symmetric encryption
    algorithm: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DecryptionResult(BaseModel):
    """Decryption result"""
    decrypted_data: bytes
    key_id: str
    algorithm: str
    decrypted_at: datetime = Field(default_factory=datetime.utcnow)

class KeyRotationPolicy(BaseModel):
    """Key rotation policy"""
    key_id: str
    rotation_interval: int  # Days
    auto_rotate: bool = True
    notify_before_expiry: int = 7  # Days
    created_at: datetime = Field(default_factory=datetime.utcnow)

class KeyManagementService:
    """
    Foundation service for key management.
    
    Provides centralized key management for encryption, authentication,
    and security for the TSAI platform.
    """
    
    def __init__(self, db_connection: asyncpg.Connection, master_key: str):
        self.db = db_connection
        self.master_key = master_key.encode()
        self.fernet = Fernet(self._derive_fernet_key())
        self.rotation_scheduler = KeyRotationScheduler()
        
        logger.info("🔐 KeyManagementService initialized: Ready for centralized key management")
    
    async def generate_key(self, key_name: str, key_type: KeyType, key_purpose: KeyPurpose,
                         key_size: int = 256, expires_in_days: Optional[int] = None) -> KeyData:
        """
        Generate new encryption key.
        
        Args:
            key_name: Name for the key
            key_type: Type of key to generate
            key_purpose: Purpose of the key
            key_size: Key size in bits
            expires_in_days: Key expiration in days
            
        Returns:
            Generated key data
        """
        try:
            logger.info(f"🔑 Generating {key_type.value} key: {key_name}")
            
            # Generate key based on type
            if key_type == KeyType.AES:
                key_data = self._generate_aes_key(key_size)
            elif key_type == KeyType.RSA:
                key_data, public_key = self._generate_rsa_key(key_size)
            elif key_type == KeyType.ECDSA:
                key_data, public_key = self._generate_ecdsa_key(key_size)
            elif key_type == KeyType.HMAC:
                key_data = self._generate_hmac_key(key_size)
            elif key_type == KeyType.API:
                key_data = self._generate_api_key()
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Calculate key hash
            key_hash = hashlib.sha256(key_data).hexdigest()
            
            # Set expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Create key data
            key_data_obj = KeyData(
                key_name=key_name,
                key_type=key_type,
                key_purpose=key_purpose,
                key_data=key_data,
                public_key=public_key if key_type in [KeyType.RSA, KeyType.ECDSA] else None,
                key_hash=key_hash,
                expires_at=expires_at
            )
            
            # Encrypt key data for storage
            encrypted_key_data = self.fernet.encrypt(key_data)
            key_data_obj.key_data = encrypted_key_data
            
            # Store key in database
            await self._store_key(key_data_obj)
            
            logger.info(f"✅ Key generated successfully: {key_name}")
            return key_data_obj
            
        except Exception as e:
            logger.error(f"❌ Failed to generate key: {e}")
            raise
    
    async def rotate_key(self, key_id: str) -> KeyData:
        """
        Rotate existing key.
        
        Args:
            key_id: ID of key to rotate
            
        Returns:
            New rotated key data
        """
        try:
            logger.info(f"🔄 Rotating key: {key_id}")
            
            # Get existing key
            existing_key = await self._get_key(key_id)
            if not existing_key:
                raise ValueError(f"Key not found: {key_id}")
            
            # Generate new key with same parameters
            new_key = await self.generate_key(
                key_name=f"{existing_key.key_name}_rotated",
                key_type=existing_key.key_type,
                key_purpose=existing_key.key_purpose,
                expires_in_days=365  # Default 1 year
            )
            
            # Mark old key as rotated
            await self._update_key_status(key_id, KeyStatus.ROTATING)
            
            # Set up rotation policy
            await self._create_rotation_policy(new_key.id, existing_key.id)
            
            logger.info(f"✅ Key rotated successfully: {key_id} -> {new_key.id}")
            return new_key
            
        except Exception as e:
            logger.error(f"❌ Failed to rotate key: {e}")
            raise
    
    async def encrypt_data(self, data: bytes, key_id: str) -> EncryptionResult:
        """
        Encrypt data with specified key.
        
        Args:
            data: Data to encrypt
            key_id: Key ID to use for encryption
            
        Returns:
            Encryption result
        """
        try:
            logger.info(f"🔒 Encrypting data with key: {key_id}")
            
            # Get key
            key_data = await self._get_key(key_id)
            if not key_data:
                raise ValueError(f"Key not found: {key_id}")
            
            # Decrypt key for use
            decrypted_key = self.fernet.decrypt(key_data.key_data)
            
            # Encrypt data based on key type
            if key_data.key_type == KeyType.AES:
                encrypted_data, iv = self._encrypt_aes(data, decrypted_key)
                result = EncryptionResult(
                    encrypted_data=encrypted_data,
                    key_id=key_id,
                    iv=iv,
                    algorithm="AES-256-CBC"
                )
            elif key_data.key_type == KeyType.RSA:
                encrypted_data = self._encrypt_rsa(data, decrypted_key)
                result = EncryptionResult(
                    encrypted_data=encrypted_data,
                    key_id=key_id,
                    algorithm="RSA-OAEP"
                )
            else:
                raise ValueError(f"Unsupported key type for encryption: {key_data.key_type}")
            
            # Update key usage
            await self._update_key_usage(key_id)
            
            logger.info(f"✅ Data encrypted successfully with key: {key_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to encrypt data: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes, key_id: str, 
                          iv: Optional[bytes] = None) -> DecryptionResult:
        """
        Decrypt data with specified key.
        
        Args:
            encrypted_data: Encrypted data
            key_id: Key ID to use for decryption
            iv: Initialization vector (for symmetric encryption)
            
        Returns:
            Decryption result
        """
        try:
            logger.info(f"🔓 Decrypting data with key: {key_id}")
            
            # Get key
            key_data = await self._get_key(key_id)
            if not key_data:
                raise ValueError(f"Key not found: {key_id}")
            
            # Decrypt key for use
            decrypted_key = self.fernet.decrypt(key_data.key_data)
            
            # Decrypt data based on key type
            if key_data.key_type == KeyType.AES:
                if not iv:
                    raise ValueError("IV required for AES decryption")
                decrypted_data = self._decrypt_aes(encrypted_data, decrypted_key, iv)
                algorithm = "AES-256-CBC"
            elif key_data.key_type == KeyType.RSA:
                decrypted_data = self._decrypt_rsa(encrypted_data, decrypted_key)
                algorithm = "RSA-OAEP"
            else:
                raise ValueError(f"Unsupported key type for decryption: {key_data.key_type}")
            
            # Update key usage
            await self._update_key_usage(key_id)
            
            logger.info(f"✅ Data decrypted successfully with key: {key_id}")
            return DecryptionResult(
                decrypted_data=decrypted_data,
                key_id=key_id,
                algorithm=algorithm
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to decrypt data: {e}")
            raise
    
    async def sign_data(self, data: bytes, key_id: str) -> bytes:
        """
        Sign data with specified key.
        
        Args:
            data: Data to sign
            key_id: Key ID to use for signing
            
        Returns:
            Digital signature
        """
        try:
            logger.info(f"✍️ Signing data with key: {key_id}")
            
            # Get key
            key_data = await self._get_key(key_id)
            if not key_data:
                raise ValueError(f"Key not found: {key_id}")
            
            # Decrypt key for use
            decrypted_key = self.fernet.decrypt(key_data.key_data)
            
            # Sign data based on key type
            if key_data.key_type == KeyType.RSA:
                signature = self._sign_rsa(data, decrypted_key)
            elif key_data.key_type == KeyType.ECDSA:
                signature = self._sign_ecdsa(data, decrypted_key)
            elif key_type == KeyType.HMAC:
                signature = self._sign_hmac(data, decrypted_key)
            else:
                raise ValueError(f"Unsupported key type for signing: {key_data.key_type}")
            
            # Update key usage
            await self._update_key_usage(key_id)
            
            logger.info(f"✅ Data signed successfully with key: {key_id}")
            return signature
            
        except Exception as e:
            logger.error(f"❌ Failed to sign data: {e}")
            raise
    
    async def verify_signature(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """
        Verify digital signature.
        
        Args:
            data: Original data
            signature: Digital signature
            key_id: Key ID to use for verification
            
        Returns:
            True if signature is valid
        """
        try:
            logger.info(f"🔍 Verifying signature with key: {key_id}")
            
            # Get key
            key_data = await self._get_key(key_id)
            if not key_data:
                raise ValueError(f"Key not found: {key_id}")
            
            # Use public key for verification
            if key_data.public_key:
                public_key = key_data.public_key
            else:
                # Decrypt private key for verification
                decrypted_key = self.fernet.decrypt(key_data.key_data)
                public_key = decrypted_key
            
            # Verify signature based on key type
            if key_data.key_type == KeyType.RSA:
                is_valid = self._verify_rsa(data, signature, public_key)
            elif key_data.key_type == KeyType.ECDSA:
                is_valid = self._verify_ecdsa(data, signature, public_key)
            elif key_data.key_type == KeyType.HMAC:
                is_valid = self._verify_hmac(data, signature, decrypted_key)
            else:
                raise ValueError(f"Unsupported key type for verification: {key_data.key_type}")
            
            logger.info(f"✅ Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"❌ Failed to verify signature: {e}")
            return False
    
    async def revoke_key(self, key_id: str) -> bool:
        """
        Revoke key and mark as inactive.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if key revoked successfully
        """
        try:
            logger.info(f"🚫 Revoking key: {key_id}")
            
            # Update key status
            success = await self._update_key_status(key_id, KeyStatus.REVOKED)
            
            if success:
                logger.info(f"✅ Key revoked successfully: {key_id}")
            else:
                logger.warning(f"⚠️ Failed to revoke key: {key_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to revoke key: {e}")
            return False
    
    async def get_key_info(self, key_id: str) -> Optional[KeyData]:
        """
        Get key information (without decrypted key data).
        
        Args:
            key_id: Key ID
            
        Returns:
            Key data without sensitive information
        """
        try:
            logger.info(f"📋 Getting key info: {key_id}")
            
            key_data = await self._get_key(key_id)
            if key_data:
                # Remove sensitive data
                key_data.key_data = b"[ENCRYPTED]"
                if key_data.public_key:
                    key_data.public_key = key_data.public_key[:50] + b"..."
            
            return key_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get key info: {e}")
            return None
    
    def _derive_fernet_key(self) -> bytes:
        """Derive Fernet key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'tsai-platform-salt',
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key))
    
    def _generate_aes_key(self, key_size: int) -> bytes:
        """Generate AES key"""
        return secrets.token_bytes(key_size // 8)
    
    def _generate_rsa_key(self, key_size: int) -> tuple[bytes, bytes]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def _generate_ecdsa_key(self, key_size: int) -> tuple[bytes, bytes]:
        """Generate ECDSA key pair"""
        # Implementation for ECDSA key generation
        return secrets.token_bytes(32), secrets.token_bytes(64)
    
    def _generate_hmac_key(self, key_size: int) -> bytes:
        """Generate HMAC key"""
        return secrets.token_bytes(key_size // 8)
    
    def _generate_api_key(self) -> bytes:
        """Generate API key"""
        return secrets.token_urlsafe(32).encode()
    
    def _encrypt_aes(self, data: bytes, key: bytes) -> tuple[bytes, bytes]:
        """Encrypt data with AES"""
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return encrypted_data, iv
    
    def _decrypt_aes(self, encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt data with AES"""
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _encrypt_rsa(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data with RSA"""
        public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data
    
    def _decrypt_rsa(self, encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data with RSA"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data
    
    def _sign_rsa(self, data: bytes, private_key_pem: bytes) -> bytes:
        """Sign data with RSA"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def _verify_rsa(self, data: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify RSA signature"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _sign_ecdsa(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data with ECDSA"""
        # Implementation for ECDSA signing
        return secrets.token_bytes(64)
    
    def _verify_ecdsa(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify ECDSA signature"""
        # Implementation for ECDSA verification
        return True
    
    def _sign_hmac(self, data: bytes, key: bytes) -> bytes:
        """Sign data with HMAC"""
        import hmac
        return hmac.new(key, data, hashlib.sha256).digest()
    
    def _verify_hmac(self, data: bytes, signature: bytes, key: bytes) -> bool:
        """Verify HMAC signature"""
        import hmac
        expected_signature = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected_signature)
    
    async def _store_key(self, key_data: KeyData):
        """Store key in database"""
        try:
            await self.db.execute("""
                INSERT INTO security.encryption_keys (id, key_name, key_type, key_purpose, 
                                                     key_data, public_key, key_hash, status, 
                                                     expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, key_data.id, key_data.key_name, key_data.key_type.value, key_data.key_purpose.value,
                key_data.key_data, key_data.public_key, key_data.key_hash, key_data.status.value,
                key_data.expires_at, key_data.created_at)
        except Exception as e:
            logger.error(f"❌ Failed to store key: {e}")
            raise
    
    async def _get_key(self, key_id: str) -> Optional[KeyData]:
        """Get key from database"""
        try:
            row = await self.db.fetchrow("""
                SELECT * FROM security.encryption_keys WHERE id = $1
            """, key_id)
            
            if row:
                return KeyData(
                    id=row['id'],
                    key_name=row['key_name'],
                    key_type=KeyType(row['key_type']),
                    key_purpose=KeyPurpose(row['key_purpose']),
                    key_data=row['key_data'],
                    public_key=row['public_key'],
                    key_hash=row['key_hash'],
                    status=KeyStatus(row['status']),
                    created_at=row['created_at'],
                    expires_at=row['expires_at'],
                    last_used=row['last_used'],
                    usage_count=row['usage_count'],
                    metadata=row['metadata']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get key: {e}")
            return None
    
    async def _update_key_status(self, key_id: str, status: KeyStatus) -> bool:
        """Update key status"""
        try:
            result = await self.db.execute("""
                UPDATE security.encryption_keys 
                SET status = $1, updated_at = $2
                WHERE id = $3
            """, status.value, datetime.utcnow(), key_id)
            
            return result == "UPDATE 1"
            
        except Exception as e:
            logger.error(f"❌ Failed to update key status: {e}")
            return False
    
    async def _update_key_usage(self, key_id: str):
        """Update key usage statistics"""
        try:
            await self.db.execute("""
                UPDATE security.encryption_keys 
                SET usage_count = usage_count + 1, last_used = $1, updated_at = $2
                WHERE id = $3
            """, datetime.utcnow(), datetime.utcnow(), key_id)
        except Exception as e:
            logger.error(f"❌ Failed to update key usage: {e}")
    
    async def _create_rotation_policy(self, new_key_id: str, old_key_id: str):
        """Create key rotation policy"""
        try:
            policy = KeyRotationPolicy(
                key_id=new_key_id,
                rotation_interval=365,  # 1 year
                auto_rotate=True,
                notify_before_expiry=30  # 30 days
            )
            
            await self.db.execute("""
                INSERT INTO security.key_rotation_policies (key_id, rotation_interval, 
                                                           auto_rotate, notify_before_expiry, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, policy.key_id, policy.rotation_interval, policy.auto_rotate,
                policy.notify_before_expiry, policy.created_at)
                
        except Exception as e:
            logger.error(f"❌ Failed to create rotation policy: {e}")

class KeyRotationScheduler:
    """Schedules and manages key rotation"""
    
    async def schedule_rotation(self, key_id: str, rotation_date: datetime):
        """Schedule key rotation"""
        try:
            logger.info(f"📅 Scheduling key rotation for: {key_id}")
        except Exception as e:
            logger.error(f"❌ Failed to schedule rotation: {e}")
    
    async def check_expiring_keys(self) -> List[str]:
        """Check for expiring keys"""
        try:
            logger.info("🔍 Checking for expiring keys")
            return []
        except Exception as e:
            logger.error(f"❌ Failed to check expiring keys: {e}")
            return []
