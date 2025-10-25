"""
TSAI Platform Foundation Service: Asset Management

This service provides comprehensive media asset management including storage,
metadata, lifecycle management, and access control for the TSAI platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
from enum import Enum
import hashlib
import mimetypes
import os
from pathlib import Path

import asyncpg
from pydantic import BaseModel, Field
import aiofiles
import aiohttp
from PIL import Image
import ffmpeg

logger = logging.getLogger(__name__)

class AssetType(str, Enum):
    """Asset types in the TSAI platform"""
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    MODEL = "model"
    DATASET = "dataset"
    OTHER = "other"

class StorageTier(str, Enum):
    """Storage tiers for asset management"""
    HOT = "hot"      # Frequently accessed (Redis, SSD)
    WARM = "warm"    # Occasionally accessed (S3, HDD)
    COLD = "cold"    # Archived (Glacier, Tape)

class AssetStatus(str, Enum):
    """Asset status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ARCHIVED = "archived"
    DELETED = "deleted"
    ERROR = "error"

class MediaAsset(BaseModel):
    """Media asset data model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str
    asset_type: AssetType
    storage_tier: StorageTier = StorageTier.HOT
    status: AssetStatus = AssetStatus.UPLOADING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    owner_id: str
    is_public: bool = False
    access_count: int = 0

class AssetMetadata(BaseModel):
    """Asset metadata model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    asset_id: str
    metadata_type: str
    metadata_value: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class OptimizationParams(BaseModel):
    """Asset optimization parameters"""
    quality: int = 85
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    format: Optional[str] = None
    compression: str = "standard"
    thumbnail: bool = True
    preview: bool = True

class LifecyclePolicy(BaseModel):
    """Asset lifecycle policy"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    asset_id: str
    policy_type: str  # "archival", "deletion", "migration"
    trigger_condition: str  # "age", "access_count", "size"
    trigger_value: Any
    action: str  # "archive", "delete", "migrate"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AssetManagementService:
    """
    Foundation service for asset management.
    
    Provides comprehensive media asset management including storage,
    metadata, lifecycle management, and access control.
    """
    
    def __init__(self, db_connection: asyncpg.Connection, storage_config: Dict[str, Any]):
        self.db = db_connection
        self.storage_config = storage_config
        self.storage_backend = self._initialize_storage_backend()
        self.optimization_engine = AssetOptimizationEngine()
        self.lifecycle_manager = AssetLifecycleManager()
        
        logger.info("📁 AssetManagementService initialized: Ready for media asset management")
    
    async def store_asset(self, asset: MediaAsset, file_data: bytes, metadata: Optional[AssetMetadata] = None) -> str:
        """
        Store media asset with metadata.
        
        Args:
            asset: Media asset data
            file_data: Raw file data
            metadata: Optional asset metadata
            
        Returns:
            Asset ID
        """
        try:
            logger.info(f"📁 Storing asset: {asset.filename}")
            
            # Calculate file hash for deduplication
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            # Check for existing asset with same hash
            existing_asset = await self._check_duplicate_asset(file_hash)
            if existing_asset:
                logger.info(f"🔄 Duplicate asset found, linking to existing: {existing_asset['id']}")
                return existing_asset['id']
            
            # Determine storage tier based on asset type and size
            storage_tier = self._determine_storage_tier(asset)
            asset.storage_tier = storage_tier
            
            # Store file in appropriate storage backend
            file_path = await self._store_file_data(asset, file_data, storage_tier)
            asset.file_path = file_path
            
            # Update asset status
            asset.status = AssetStatus.PROCESSING
            
            # Store asset in database
            await self.db.execute("""
                INSERT INTO assets (id, filename, original_filename, file_path, file_size,
                                 mime_type, asset_type, storage_tier, status, metadata,
                                 tags, created_at, updated_at, owner_id, is_public, file_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """, asset.id, asset.filename, asset.original_filename, asset.file_path,
                asset.file_size, asset.mime_type, asset.asset_type.value, asset.storage_tier.value,
                asset.status.value, asset.metadata, asset.tags, asset.created_at, asset.updated_at,
                asset.owner_id, asset.is_public, file_hash)
            
            # Store metadata if provided
            if metadata:
                await self._store_asset_metadata(asset.id, metadata)
            
            # Process asset for optimization
            await self._process_asset_optimization(asset)
            
            # Update status to ready
            await self.db.execute("""
                UPDATE assets SET status = $1, updated_at = $2 WHERE id = $3
            """, AssetStatus.READY.value, datetime.utcnow(), asset.id)
            
            logger.info(f"✅ Asset stored successfully: {asset.id}")
            return asset.id
            
        except Exception as e:
            logger.error(f"❌ Failed to store asset: {e}")
            raise
    
    async def retrieve_asset(self, asset_id: str, user_id: str) -> Optional[MediaAsset]:
        """
        Retrieve media asset with access control.
        
        Args:
            asset_id: Asset ID
            user_id: User ID requesting the asset
            
        Returns:
            Media asset if accessible, None otherwise
        """
        try:
            logger.info(f"📥 Retrieving asset: {asset_id}")
            
            # Check access permissions
            if not await self._check_access_permissions(asset_id, user_id):
                logger.warning(f"⚠️ Access denied for asset: {asset_id}")
                return None
            
            # Get asset from database
            asset_row = await self.db.fetchrow("""
                SELECT * FROM assets WHERE id = $1 AND status != $2
            """, asset_id, AssetStatus.DELETED.value)
            
            if not asset_row:
                logger.warning(f"⚠️ Asset not found: {asset_id}")
                return None
            
            # Update access count and timestamp
            await self.db.execute("""
                UPDATE assets SET access_count = access_count + 1, accessed_at = $1, updated_at = $2
                WHERE id = $3
            """, datetime.utcnow(), datetime.utcnow(), asset_id)
            
            # Convert to MediaAsset
            asset = MediaAsset(
                id=asset_row['id'],
                filename=asset_row['filename'],
                original_filename=asset_row['original_filename'],
                file_path=asset_row['file_path'],
                file_size=asset_row['file_size'],
                mime_type=asset_row['mime_type'],
                asset_type=AssetType(asset_row['asset_type']),
                storage_tier=StorageTier(asset_row['storage_tier']),
                status=AssetStatus(asset_row['status']),
                metadata=asset_row['metadata'],
                tags=asset_row['tags'],
                created_at=asset_row['created_at'],
                updated_at=asset_row['updated_at'],
                accessed_at=asset_row['accessed_at'],
                expires_at=asset_row['expires_at'],
                owner_id=asset_row['owner_id'],
                is_public=asset_row['is_public'],
                access_count=asset_row['access_count']
            )
            
            logger.info(f"✅ Asset retrieved successfully: {asset_id}")
            return asset
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve asset: {e}")
            return None
    
    async def optimize_asset(self, asset_id: str, optimization_params: OptimizationParams) -> str:
        """
        Optimize asset for different use cases.
        
        Args:
            asset_id: Asset ID to optimize
            optimization_params: Optimization parameters
            
        Returns:
            Optimized asset ID
        """
        try:
            logger.info(f"⚡ Optimizing asset: {asset_id}")
            
            # Get original asset
            asset = await self.retrieve_asset(asset_id, "system")
            if not asset:
                raise ValueError(f"Asset not found: {asset_id}")
            
            # Create optimized version
            optimized_asset = await self.optimization_engine.optimize_asset(
                asset, optimization_params
            )
            
            # Store optimized asset
            optimized_id = await self.store_asset(optimized_asset, b"", None)
            
            # Link original to optimized version
            await self._link_asset_versions(asset_id, optimized_id)
            
            logger.info(f"✅ Asset optimized successfully: {optimized_id}")
            return optimized_id
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize asset: {e}")
            raise
    
    async def manage_lifecycle(self, asset_id: str, policy: LifecyclePolicy) -> bool:
        """
        Manage asset lifecycle (archival, deletion, migration).
        
        Args:
            asset_id: Asset ID
            policy: Lifecycle policy
            
        Returns:
            True if lifecycle managed successfully
        """
        try:
            logger.info(f"🔄 Managing lifecycle for asset: {asset_id}")
            
            # Apply lifecycle policy
            success = await self.lifecycle_manager.apply_policy(asset_id, policy)
            
            if success:
                logger.info(f"✅ Lifecycle managed successfully for asset: {asset_id}")
            else:
                logger.warning(f"⚠️ Lifecycle management failed for asset: {asset_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to manage lifecycle: {e}")
            return False
    
    async def search_assets(self, query: Dict[str, Any], user_id: str) -> List[MediaAsset]:
        """
        Search assets with filters and access control.
        
        Args:
            query: Search query parameters
            user_id: User ID performing search
            
        Returns:
            List of accessible assets
        """
        try:
            logger.info(f"🔍 Searching assets with query: {query}")
            
            # Build search query
            search_sql = """
                SELECT * FROM assets 
                WHERE status != $1 AND (is_public = true OR owner_id = $2)
            """
            params = [AssetStatus.DELETED.value, user_id]
            param_count = 2
            
            # Add filters
            if query.get("asset_type"):
                search_sql += f" AND asset_type = ${param_count + 1}"
                params.append(query["asset_type"])
                param_count += 1
            
            if query.get("tags"):
                search_sql += f" AND tags && ${param_count + 1}"
                params.append(query["tags"])
                param_count += 1
            
            if query.get("date_range"):
                start_date, end_date = query["date_range"]
                search_sql += f" AND created_at BETWEEN ${param_count + 1} AND ${param_count + 2}"
                params.extend([start_date, end_date])
                param_count += 2
            
            # Add ordering and limit
            search_sql += " ORDER BY created_at DESC"
            if query.get("limit"):
                search_sql += f" LIMIT ${param_count + 1}"
                params.append(query["limit"])
            
            # Execute search
            asset_rows = await self.db.fetch(search_sql, *params)
            
            # Convert to MediaAsset objects
            assets = []
            for row in asset_rows:
                asset = MediaAsset(
                    id=row['id'],
                    filename=row['filename'],
                    original_filename=row['original_filename'],
                    file_path=row['file_path'],
                    file_size=row['file_size'],
                    mime_type=row['mime_type'],
                    asset_type=AssetType(row['asset_type']),
                    storage_tier=StorageTier(row['storage_tier']),
                    status=AssetStatus(row['status']),
                    metadata=row['metadata'],
                    tags=row['tags'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    accessed_at=row['accessed_at'],
                    expires_at=row['expires_at'],
                    owner_id=row['owner_id'],
                    is_public=row['is_public'],
                    access_count=row['access_count']
                )
                assets.append(asset)
            
            logger.info(f"✅ Found {len(assets)} assets matching query")
            return assets
            
        except Exception as e:
            logger.error(f"❌ Failed to search assets: {e}")
            return []
    
    async def _check_duplicate_asset(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Check for existing asset with same hash"""
        try:
            row = await self.db.fetchrow("""
                SELECT id, filename, file_path FROM assets WHERE file_hash = $1
            """, file_hash)
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to check duplicate asset: {e}")
            return None
    
    def _determine_storage_tier(self, asset: MediaAsset) -> StorageTier:
        """Determine appropriate storage tier for asset"""
        # Simple logic - can be enhanced with ML-based predictions
        if asset.file_size > 100 * 1024 * 1024:  # > 100MB
            return StorageTier.COLD
        elif asset.asset_type in [AssetType.VIDEO, AssetType.MODEL]:
            return StorageTier.WARM
        else:
            return StorageTier.HOT
    
    async def _store_file_data(self, asset: MediaAsset, file_data: bytes, storage_tier: StorageTier) -> str:
        """Store file data in appropriate storage backend"""
        try:
            # Generate file path
            file_path = f"{storage_tier.value}/{asset.asset_type.value}/{asset.id}"
            
            # Store in appropriate backend based on tier
            if storage_tier == StorageTier.HOT:
                # Store in Redis or local SSD
                await self.storage_backend.store_hot(file_path, file_data)
            elif storage_tier == StorageTier.WARM:
                # Store in S3 or similar
                await self.storage_backend.store_warm(file_path, file_data)
            else:  # COLD
                # Store in Glacier or similar
                await self.storage_backend.store_cold(file_path, file_data)
            
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to store file data: {e}")
            raise
    
    async def _store_asset_metadata(self, asset_id: str, metadata: AssetMetadata):
        """Store asset metadata"""
        try:
            await self.db.execute("""
                INSERT INTO asset_metadata (id, asset_id, metadata_type, metadata_value, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, metadata.id, asset_id, metadata.metadata_type, metadata.metadata_value,
                metadata.created_at, metadata.updated_at)
        except Exception as e:
            logger.error(f"❌ Failed to store asset metadata: {e}")
            raise
    
    async def _process_asset_optimization(self, asset: MediaAsset):
        """Process asset for optimization"""
        try:
            # Generate thumbnails for images/videos
            if asset.asset_type in [AssetType.IMAGE, AssetType.VIDEO]:
                await self.optimization_engine.generate_thumbnails(asset)
            
            # Extract metadata
            await self.optimization_engine.extract_metadata(asset)
            
        except Exception as e:
            logger.error(f"❌ Failed to process asset optimization: {e}")
    
    async def _check_access_permissions(self, asset_id: str, user_id: str) -> bool:
        """Check if user has access to asset"""
        try:
            # Check if asset is public or user is owner
            row = await self.db.fetchrow("""
                SELECT is_public, owner_id FROM assets WHERE id = $1
            """, asset_id)
            
            if not row:
                return False
            
            return row['is_public'] or row['owner_id'] == user_id
            
        except Exception as e:
            logger.error(f"❌ Failed to check access permissions: {e}")
            return False
    
    async def _link_asset_versions(self, original_id: str, optimized_id: str):
        """Link original asset to optimized version"""
        try:
            await self.db.execute("""
                INSERT INTO asset_versions (original_id, optimized_id, created_at)
                VALUES ($1, $2, $3)
            """, original_id, optimized_id, datetime.utcnow())
        except Exception as e:
            logger.error(f"❌ Failed to link asset versions: {e}")
    
    def _initialize_storage_backend(self):
        """Initialize storage backend based on configuration"""
        # This would be implemented based on actual storage configuration
        return StorageBackend(self.storage_config)

class AssetOptimizationEngine:
    """Asset optimization engine for media processing"""
    
    def __init__(self):
        self.supported_formats = {
            AssetType.IMAGE: ['jpg', 'png', 'webp', 'gif'],
            AssetType.VIDEO: ['mp4', 'avi', 'mov', 'webm'],
            AssetType.AUDIO: ['mp3', 'wav', 'aac', 'ogg']
        }
    
    async def optimize_asset(self, asset: MediaAsset, params: OptimizationParams) -> MediaAsset:
        """Optimize asset based on parameters"""
        try:
            # Create optimized version
            optimized_asset = MediaAsset(
                filename=f"optimized_{asset.filename}",
                original_filename=asset.original_filename,
                file_path="",  # Will be set by storage
                file_size=0,  # Will be calculated
                mime_type=asset.mime_type,
                asset_type=asset.asset_type,
                storage_tier=asset.storage_tier,
                status=AssetStatus.PROCESSING,
                metadata=asset.metadata.copy(),
                tags=asset.tags.copy(),
                owner_id=asset.owner_id,
                is_public=asset.is_public
            )
            
            # Apply optimization based on asset type
            if asset.asset_type == AssetType.IMAGE:
                await self._optimize_image(asset, optimized_asset, params)
            elif asset.asset_type == AssetType.VIDEO:
                await self._optimize_video(asset, optimized_asset, params)
            elif asset.asset_type == AssetType.AUDIO:
                await self._optimize_audio(asset, optimized_asset, params)
            
            return optimized_asset
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize asset: {e}")
            raise
    
    async def _optimize_image(self, original: MediaAsset, optimized: MediaAsset, params: OptimizationParams):
        """Optimize image asset"""
        # Implementation for image optimization
        pass
    
    async def _optimize_video(self, original: MediaAsset, optimized: MediaAsset, params: OptimizationParams):
        """Optimize video asset"""
        # Implementation for video optimization
        pass
    
    async def _optimize_audio(self, original: MediaAsset, optimized: MediaAsset, params: OptimizationParams):
        """Optimize audio asset"""
        # Implementation for audio optimization
        pass
    
    async def generate_thumbnails(self, asset: MediaAsset):
        """Generate thumbnails for asset"""
        # Implementation for thumbnail generation
        pass
    
    async def extract_metadata(self, asset: MediaAsset):
        """Extract metadata from asset"""
        # Implementation for metadata extraction
        pass

class AssetLifecycleManager:
    """Asset lifecycle management"""
    
    async def apply_policy(self, asset_id: str, policy: LifecyclePolicy) -> bool:
        """Apply lifecycle policy to asset"""
        try:
            if policy.policy_type == "archival":
                return await self._archive_asset(asset_id)
            elif policy.policy_type == "deletion":
                return await self._delete_asset(asset_id)
            elif policy.policy_type == "migration":
                return await self._migrate_asset(asset_id, policy)
            else:
                logger.warning(f"⚠️ Unknown policy type: {policy.policy_type}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to apply lifecycle policy: {e}")
            return False
    
    async def _archive_asset(self, asset_id: str) -> bool:
        """Archive asset to cold storage"""
        # Implementation for asset archival
        return True
    
    async def _delete_asset(self, asset_id: str) -> bool:
        """Delete asset permanently"""
        # Implementation for asset deletion
        return True
    
    async def _migrate_asset(self, asset_id: str, policy: LifecyclePolicy) -> bool:
        """Migrate asset between storage tiers"""
        # Implementation for asset migration
        return True

class StorageBackend:
    """Storage backend for different tiers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def store_hot(self, file_path: str, data: bytes):
        """Store in hot storage (Redis, SSD)"""
        # Implementation for hot storage
        pass
    
    async def store_warm(self, file_path: str, data: bytes):
        """Store in warm storage (S3, HDD)"""
        # Implementation for warm storage
        pass
    
    async def store_cold(self, file_path: str, data: bytes):
        """Store in cold storage (Glacier, Tape)"""
        # Implementation for cold storage
        pass
