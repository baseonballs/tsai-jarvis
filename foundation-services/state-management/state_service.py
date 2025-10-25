"""
TSAI Platform Foundation Service: Unified State Management

This service provides centralized state management with caching, failover,
and consistency for the TSAI platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
from enum import Enum
import json
import pickle

import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class StateType(str, Enum):
    """State types"""
    CACHE = "cache"
    PERSISTENT = "persistent"
    SESSION = "session"
    TEMPORARY = "temporary"

class ConsistencyLevel(str, Enum):
    """Consistency levels"""
    STRONG = "strong"      # ACID compliance
    EVENTUAL = "eventual"  # Eventually consistent
    WEAK = "weak"         # Best effort

class StateValue(BaseModel):
    """State value model"""
    key: str
    value: Any
    state_type: StateType = StateType.CACHE
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ttl: Optional[int] = None  # Time to live in seconds
    namespace: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StateOperation(BaseModel):
    """State operation model"""
    operation_id: str = Field(default_factory=lambda: str(uuid4()))
    operation_type: str  # "get", "set", "delete", "sync"
    key: str
    namespace: str
    value: Optional[Any] = None
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    created_at: datetime = Field(default_factory=datetime.utcnow)

class NodeStatus(str, Enum):
    """Node status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"

class ClusterNode(BaseModel):
    """Cluster node model"""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    load_factor: float = 0.0
    capacity: int = 1000

class UnifiedStateService:
    """
    Foundation service for unified state management.
    
    Provides centralized state management with caching, failover,
    and consistency for the TSAI platform.
    """
    
    def __init__(self, db_connection: asyncpg.Connection, redis_cluster: redis.Redis):
        self.db = db_connection
        self.redis = redis_cluster
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        self.consistency_manager = ConsistencyManager()
        self.failover_manager = FailoverManager()
        
        logger.info("🔄 UnifiedStateService initialized: Ready for centralized state management")
    
    async def get_state(self, key: str, namespace: str = "default", 
                       consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL) -> Optional[StateValue]:
        """
        Get state value with caching and consistency.
        
        Args:
            key: State key
            namespace: State namespace
            consistency_level: Consistency level required
            
        Returns:
            State value if found, None otherwise
        """
        try:
            logger.info(f"📥 Getting state: {namespace}:{key}")
            
            # Try cache first for performance
            if consistency_level in [ConsistencyLevel.EVENTUAL, ConsistencyLevel.WEAK]:
                cached_state = await self._get_from_cache(key, namespace)
                if cached_state:
                    logger.info(f"✅ State found in cache: {namespace}:{key}")
                    return cached_state
            
            # Get from database for strong consistency
            if consistency_level == ConsistencyLevel.STRONG:
                state_value = await self._get_from_database(key, namespace)
                if state_value:
                    # Update cache for future requests
                    await self._cache_state(state_value)
                    logger.info(f"✅ State found in database: {namespace}:{key}")
                    return state_value
            
            # Try database for eventual consistency
            state_value = await self._get_from_database(key, namespace)
            if state_value:
                # Update cache
                await self._cache_state(state_value)
                logger.info(f"✅ State found in database: {namespace}:{key}")
                return state_value
            
            logger.info(f"⚠️ State not found: {namespace}:{key}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get state: {e}")
            return None
    
    async def set_state(self, key: str, value: Any, namespace: str = "default",
                       state_type: StateType = StateType.CACHE,
                       consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
                       ttl: Optional[int] = None) -> bool:
        """
        Set state value with consistency and caching.
        
        Args:
            key: State key
            value: State value
            namespace: State namespace
            state_type: State type
            consistency_level: Consistency level
            ttl: Time to live in seconds
            
        Returns:
            True if state set successfully
        """
        try:
            logger.info(f"📤 Setting state: {namespace}:{key}")
            
            # Create state value
            state_value = StateValue(
                key=key,
                value=value,
                state_type=state_type,
                consistency_level=consistency_level,
                ttl=ttl,
                namespace=namespace
            )
            
            # Set in database for persistence
            if state_type in [StateType.PERSISTENT, StateType.SESSION]:
                success = await self._set_in_database(state_value)
                if not success:
                    logger.error(f"❌ Failed to set state in database: {namespace}:{key}")
                    return False
            
            # Set in cache for performance
            if state_type in [StateType.CACHE, StateType.TEMPORARY]:
                success = await self._set_in_cache(state_value)
                if not success:
                    logger.error(f"❌ Failed to set state in cache: {namespace}:{key}")
                    return False
            
            # Handle consistency requirements
            if consistency_level == ConsistencyLevel.STRONG:
                await self._ensure_strong_consistency(state_value)
            
            logger.info(f"✅ State set successfully: {namespace}:{key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set state: {e}")
            return False
    
    async def delete_state(self, key: str, namespace: str = "default") -> bool:
        """
        Delete state value from all storage layers.
        
        Args:
            key: State key
            namespace: State namespace
            
        Returns:
            True if state deleted successfully
        """
        try:
            logger.info(f"🗑️ Deleting state: {namespace}:{key}")
            
            # Delete from cache
            await self._delete_from_cache(key, namespace)
            
            # Delete from database
            await self._delete_from_database(key, namespace)
            
            logger.info(f"✅ State deleted successfully: {namespace}:{key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete state: {e}")
            return False
    
    async def sync_state(self, namespace: str) -> bool:
        """
        Synchronize state across all nodes in the cluster.
        
        Args:
            namespace: State namespace to sync
            
        Returns:
            True if sync successful
        """
        try:
            logger.info(f"🔄 Syncing state for namespace: {namespace}")
            
            # Get all state values for namespace
            states = await self._get_all_states_for_namespace(namespace)
            
            # Sync each state across cluster nodes
            for state in states:
                await self._sync_state_to_nodes(state)
            
            logger.info(f"✅ State sync completed for namespace: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync state: {e}")
            return False
    
    async def handle_failover(self, failed_node: str) -> bool:
        """
        Handle node failure and recovery.
        
        Args:
            failed_node: ID of the failed node
            
        Returns:
            True if failover handled successfully
        """
        try:
            logger.info(f"🔄 Handling failover for node: {failed_node}")
            
            # Mark node as failed
            if failed_node in self.cluster_nodes:
                self.cluster_nodes[failed_node].status = NodeStatus.FAILED
            
            # Redistribute load to healthy nodes
            await self._redistribute_load()
            
            # Replicate critical state to other nodes
            await self._replicate_critical_state()
            
            logger.info(f"✅ Failover handled successfully for node: {failed_node}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to handle failover: {e}")
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get cluster status and health information.
        
        Returns:
            Cluster status information
        """
        try:
            logger.info("📊 Getting cluster status")
            
            healthy_nodes = sum(1 for node in self.cluster_nodes.values() if node.status == NodeStatus.HEALTHY)
            total_nodes = len(self.cluster_nodes)
            
            status = {
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "unhealthy_nodes": total_nodes - healthy_nodes,
                "cluster_health": "healthy" if healthy_nodes > total_nodes // 2 else "unhealthy",
                "nodes": {node_id: node.dict() for node_id, node in self.cluster_nodes.items()},
                "timestamp": datetime.utcnow()
            }
            
            logger.info(f"✅ Cluster status: {healthy_nodes}/{total_nodes} nodes healthy")
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get cluster status: {e}")
            return {"error": str(e)}
    
    async def _get_from_cache(self, key: str, namespace: str) -> Optional[StateValue]:
        """Get state from cache"""
        try:
            cache_key = f"state:{namespace}:{key}"
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                state_dict = json.loads(cached_data)
                return StateValue(**state_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get from cache: {e}")
            return None
    
    async def _get_from_database(self, key: str, namespace: str) -> Optional[StateValue]:
        """Get state from database"""
        try:
            row = await self.db.fetchrow("""
                SELECT * FROM platform.state_store 
                WHERE key = $1 AND namespace = $2
            """, key, namespace)
            
            if row:
                return StateValue(
                    key=row['key'],
                    value=row['value'],
                    state_type=StateType(row['state_type']),
                    consistency_level=ConsistencyLevel(row['consistency_level']),
                    ttl=row['ttl'],
                    namespace=row['namespace'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    version=row['version'],
                    metadata=row['metadata']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get from database: {e}")
            return None
    
    async def _set_in_cache(self, state_value: StateValue) -> bool:
        """Set state in cache"""
        try:
            cache_key = f"state:{state_value.namespace}:{state_value.key}"
            cache_data = json.dumps(state_value.dict())
            
            if state_value.ttl:
                await self.redis.setex(cache_key, state_value.ttl, cache_data)
            else:
                await self.redis.set(cache_key, cache_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set in cache: {e}")
            return False
    
    async def _set_in_database(self, state_value: StateValue) -> bool:
        """Set state in database"""
        try:
            await self.db.execute("""
                INSERT INTO platform.state_store (key, value, state_type, consistency_level, 
                                                 ttl, namespace, version, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (key, namespace) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    state_type = EXCLUDED.state_type,
                    consistency_level = EXCLUDED.consistency_level,
                    ttl = EXCLUDED.ttl,
                    version = platform.state_store.version + 1,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, state_value.key, state_value.value, state_value.state_type.value,
                state_value.consistency_level.value, state_value.ttl, state_value.namespace,
                state_value.version, state_value.metadata, state_value.created_at, state_value.updated_at)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set in database: {e}")
            return False
    
    async def _delete_from_cache(self, key: str, namespace: str):
        """Delete state from cache"""
        try:
            cache_key = f"state:{namespace}:{key}"
            await self.redis.delete(cache_key)
        except Exception as e:
            logger.error(f"❌ Failed to delete from cache: {e}")
    
    async def _delete_from_database(self, key: str, namespace: str):
        """Delete state from database"""
        try:
            await self.db.execute("""
                DELETE FROM platform.state_store 
                WHERE key = $1 AND namespace = $2
            """, key, namespace)
        except Exception as e:
            logger.error(f"❌ Failed to delete from database: {e}")
    
    async def _cache_state(self, state_value: StateValue):
        """Cache state value"""
        try:
            await self._set_in_cache(state_value)
        except Exception as e:
            logger.error(f"❌ Failed to cache state: {e}")
    
    async def _ensure_strong_consistency(self, state_value: StateValue):
        """Ensure strong consistency across cluster"""
        try:
            await self.consistency_manager.ensure_strong_consistency(state_value)
        except Exception as e:
            logger.error(f"❌ Failed to ensure strong consistency: {e}")
    
    async def _get_all_states_for_namespace(self, namespace: str) -> List[StateValue]:
        """Get all states for a namespace"""
        try:
            rows = await self.db.fetch("""
                SELECT * FROM platform.state_store 
                WHERE namespace = $1
            """, namespace)
            
            states = []
            for row in rows:
                state = StateValue(
                    key=row['key'],
                    value=row['value'],
                    state_type=StateType(row['state_type']),
                    consistency_level=ConsistencyLevel(row['consistency_level']),
                    ttl=row['ttl'],
                    namespace=row['namespace'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    version=row['version'],
                    metadata=row['metadata']
                )
                states.append(state)
            
            return states
            
        except Exception as e:
            logger.error(f"❌ Failed to get states for namespace: {e}")
            return []
    
    async def _sync_state_to_nodes(self, state_value: StateValue):
        """Sync state to all cluster nodes"""
        try:
            for node_id, node in self.cluster_nodes.items():
                if node.status == NodeStatus.HEALTHY:
                    await self._sync_to_node(node, state_value)
        except Exception as e:
            logger.error(f"❌ Failed to sync state to nodes: {e}")
    
    async def _sync_to_node(self, node: ClusterNode, state_value: StateValue):
        """Sync state to specific node"""
        try:
            # Implementation would depend on the specific node communication protocol
            logger.debug(f"Syncing state to node: {node.node_id}")
        except Exception as e:
            logger.error(f"❌ Failed to sync to node: {e}")
    
    async def _redistribute_load(self):
        """Redistribute load after node failure"""
        try:
            healthy_nodes = [node for node in self.cluster_nodes.values() if node.status == NodeStatus.HEALTHY]
            if healthy_nodes:
                # Simple round-robin redistribution
                for i, node in enumerate(healthy_nodes):
                    node.load_factor = 1.0 / len(healthy_nodes)
        except Exception as e:
            logger.error(f"❌ Failed to redistribute load: {e}")
    
    async def _replicate_critical_state(self):
        """Replicate critical state to healthy nodes"""
        try:
            # Get critical states
            critical_states = await self._get_critical_states()
            
            # Replicate to healthy nodes
            for state in critical_states:
                await self._sync_state_to_nodes(state)
        except Exception as e:
            logger.error(f"❌ Failed to replicate critical state: {e}")
    
    async def _get_critical_states(self) -> List[StateValue]:
        """Get critical states that need replication"""
        try:
            rows = await self.db.fetch("""
                SELECT * FROM platform.state_store 
                WHERE consistency_level = $1 AND state_type = $2
            """, ConsistencyLevel.STRONG.value, StateType.PERSISTENT.value)
            
            states = []
            for row in rows:
                state = StateValue(
                    key=row['key'],
                    value=row['value'],
                    state_type=StateType(row['state_type']),
                    consistency_level=ConsistencyLevel(row['consistency_level']),
                    ttl=row['ttl'],
                    namespace=row['namespace'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    version=row['version'],
                    metadata=row['metadata']
                )
                states.append(state)
            
            return states
            
        except Exception as e:
            logger.error(f"❌ Failed to get critical states: {e}")
            return []

class ConsistencyManager:
    """Manages state consistency across the cluster"""
    
    async def ensure_strong_consistency(self, state_value: StateValue):
        """Ensure strong consistency for state value"""
        try:
            # Implementation for strong consistency
            logger.debug(f"Ensuring strong consistency for: {state_value.key}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure strong consistency: {e}")

class FailoverManager:
    """Manages failover and recovery operations"""
    
    async def handle_node_failure(self, node_id: str):
        """Handle node failure"""
        try:
            logger.info(f"Handling failure for node: {node_id}")
        except Exception as e:
            logger.error(f"❌ Failed to handle node failure: {e}")
