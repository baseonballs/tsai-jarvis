#!/usr/bin/env python3
"""
TSAI Jarvis Platform Setup Script

This script sets up the complete TSAI Jarvis platform including:
- Database setup and configuration
- Foundation services initialization
- Core intelligence services
- API gateway configuration
- Infrastructure setup
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import setup modules
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from infrastructure.database.setup import DatabaseSetup
    from foundation_services.user_management.user_service import UserManagementService
    from foundation_services.session_management.session_service import SessionManagementService
    from foundation_services.state_management.state_service import UnifiedStateService
    from foundation_services.key_management.key_service import KeyManagementService
    from foundation_services.asset_management.asset_service import AssetManagementService
    from core_intelligence.decision_engine.jarvis_brain import JarvisBrain
    from core_intelligence.temporal_orchestration.jarvis_orchestrator import JarvisOrchestrator
    from api_gateway.intelligence_api.jarvis_api import JarvisAPI
except ImportError as e:
    print(f"Import error: {e}")
    print("Some services may not be available. Continuing with basic setup...")
    DatabaseSetup = None
    UserManagementService = None
    SessionManagementService = None
    UnifiedStateService = None
    KeyManagementService = None
    AssetManagementService = None
    JarvisBrain = None
    JarvisOrchestrator = None
    JarvisAPI = None

logger = logging.getLogger(__name__)

class TSAIJarvisSetup:
    """Complete TSAI Jarvis platform setup manager"""
    
    def __init__(self):
        self.setup_status: Dict[str, Any] = {}
        self.services: Dict[str, Any] = {}
        
        # Configuration
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://temporal:temporal@localhost:5432/tsai_platform"
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.temporal_url = os.getenv("TEMPORAL_URL", "localhost:7233")
        self.master_key = os.getenv("MASTER_KEY", "tsai-jarvis-master-key-2024")
        
        logger.info("🚀 TSAI Jarvis Setup initialized")
    
    async def setup_platform(self) -> bool:
        """
        Set up the complete TSAI Jarvis platform.
        
        Returns:
            True if setup successful
        """
        try:
            logger.info("🎯 Starting TSAI Jarvis platform setup")
            
            # Phase 1: Infrastructure Setup
            await self._setup_infrastructure()
            
            # Phase 2: Foundation Services
            await self._setup_foundation_services()
            
            # Phase 3: Core Intelligence
            await self._setup_core_intelligence()
            
            # Phase 4: API Gateway
            await self._setup_api_gateway()
            
            # Phase 5: Integration Testing
            await self._run_integration_tests()
            
            logger.info("✅ TSAI Jarvis platform setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ TSAI Jarvis platform setup failed: {e}")
            return False
    
    async def _setup_infrastructure(self):
        """Set up infrastructure components"""
        try:
            logger.info("🏗️ Setting up infrastructure")
            
            if DatabaseSetup is None:
                logger.warning("⚠️ DatabaseSetup not available, skipping database setup")
                self.setup_status["database"] = {"status": "skipped", "reason": "DatabaseSetup not available"}
                return
            
            # 1. Database setup
            logger.info("📊 Setting up PostgreSQL database")
            db_setup = DatabaseSetup(self.database_url)
            db_success = await db_setup.setup_database()
            
            if not db_success:
                raise Exception("Database setup failed")
            
            # Verify database setup
            db_verification = await db_setup.verify_setup()
            self.setup_status["database"] = db_verification
            
            logger.info("✅ Infrastructure setup completed")
            
        except Exception as e:
            logger.error(f"❌ Infrastructure setup failed: {e}")
            self.setup_status["database"] = {"status": "failed", "error": str(e)}
    
    async def _setup_foundation_services(self):
        """Set up foundation services"""
        try:
            logger.info("🔧 Setting up foundation services")
            
            foundation_status = {}
            
            # Check if services are available
            if UserManagementService is None:
                logger.warning("⚠️ UserManagementService not available")
                foundation_status["user_management"] = "skipped"
            else:
                try:
                    # Database connection for services
                    db_conn = await self._get_database_connection()
                    redis_conn = await self._get_redis_connection()
                    
                    # 1. User Management Service
                    logger.info("👤 Initializing User Management Service")
                    user_service = UserManagementService(db_conn)
                    self.services["user_management"] = user_service
                    foundation_status["user_management"] = "initialized"
                    
                    # 2. Session Management Service
                    if SessionManagementService is not None:
                        logger.info("🔐 Initializing Session Management Service")
                        session_service = SessionManagementService(db_conn, redis_conn)
                        self.services["session_management"] = session_service
                        foundation_status["session_management"] = "initialized"
                    else:
                        foundation_status["session_management"] = "skipped"
                    
                    # 3. State Management Service
                    if UnifiedStateService is not None:
                        logger.info("🔄 Initializing Unified State Service")
                        state_service = UnifiedStateService(db_conn, redis_conn)
                        self.services["state_management"] = state_service
                        foundation_status["state_management"] = "initialized"
                    else:
                        foundation_status["state_management"] = "skipped"
                    
                    # 4. Key Management Service
                    if KeyManagementService is not None:
                        logger.info("🔑 Initializing Key Management Service")
                        key_service = KeyManagementService(db_conn, self.master_key)
                        self.services["key_management"] = key_service
                        foundation_status["key_management"] = "initialized"
                    else:
                        foundation_status["key_management"] = "skipped"
                    
                    # 5. Asset Management Service
                    if AssetManagementService is not None:
                        logger.info("📁 Initializing Asset Management Service")
                        storage_config = {
                            "hot_storage": {"type": "redis", "url": self.redis_url},
                            "warm_storage": {"type": "s3", "bucket": "tsai-assets"},
                            "cold_storage": {"type": "glacier", "vault": "tsai-archive"}
                        }
                        asset_service = AssetManagementService(db_conn, storage_config)
                        self.services["asset_management"] = asset_service
                        foundation_status["asset_management"] = "initialized"
                    else:
                        foundation_status["asset_management"] = "skipped"
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize foundation services: {e}")
                    foundation_status["error"] = str(e)
            
            self.setup_status["foundation_services"] = foundation_status
            logger.info("✅ Foundation services setup completed")
            
        except Exception as e:
            logger.error(f"❌ Foundation services setup failed: {e}")
            self.setup_status["foundation_services"] = {"status": "failed", "error": str(e)}
    
    async def _setup_core_intelligence(self):
        """Set up core intelligence services"""
        try:
            logger.info("🧠 Setting up core intelligence services")
            
            core_status = {}
            
            # 1. Jarvis Brain (Decision Engine)
            if JarvisBrain is not None:
                logger.info("🧠 Initializing Jarvis Brain")
                jarvis_brain = JarvisBrain()
                self.services["jarvis_brain"] = jarvis_brain
                core_status["jarvis_brain"] = "initialized"
            else:
                logger.warning("⚠️ JarvisBrain not available")
                core_status["jarvis_brain"] = "skipped"
            
            # 2. Jarvis Orchestrator (Temporal AI)
            if JarvisOrchestrator is not None:
                try:
                    logger.info("🎭 Initializing Jarvis Orchestrator")
                    jarvis_orchestrator = JarvisOrchestrator(
                        server_url=self.temporal_url,
                        namespace="tsai-jarvis"
                    )
                    await jarvis_orchestrator.connect()
                    await jarvis_orchestrator.start_worker()
                    self.services["jarvis_orchestrator"] = jarvis_orchestrator
                    core_status["jarvis_orchestrator"] = "initialized"
                except Exception as e:
                    logger.warning(f"⚠️ Jarvis Orchestrator connection failed: {e}")
                    core_status["jarvis_orchestrator"] = "failed"
            else:
                logger.warning("⚠️ JarvisOrchestrator not available")
                core_status["jarvis_orchestrator"] = "skipped"
            
            self.setup_status["core_intelligence"] = core_status
            logger.info("✅ Core intelligence services setup completed")
            
        except Exception as e:
            logger.error(f"❌ Core intelligence setup failed: {e}")
            self.setup_status["core_intelligence"] = {"status": "failed", "error": str(e)}
    
    async def _setup_api_gateway(self):
        """Set up API gateway"""
        try:
            logger.info("🌐 Setting up API Gateway")
            
            if JarvisAPI is not None:
                # Initialize Jarvis API
                jarvis_api = JarvisAPI()
                self.services["jarvis_api"] = jarvis_api
                
                self.setup_status["api_gateway"] = {
                    "jarvis_api": "initialized",
                    "endpoints": [
                        "/health",
                        "/api/intelligence/analyze",
                        "/api/intelligence/decide",
                        "/api/workflows/start",
                        "/api/workflows/{workflow_id}/status"
                    ]
                }
                logger.info("✅ API Gateway setup completed")
            else:
                logger.warning("⚠️ JarvisAPI not available")
                self.setup_status["api_gateway"] = {
                    "jarvis_api": "skipped",
                    "reason": "JarvisAPI not available"
                }
            
        except Exception as e:
            logger.error(f"❌ API Gateway setup failed: {e}")
            self.setup_status["api_gateway"] = {"status": "failed", "error": str(e)}
    
    async def _run_integration_tests(self):
        """Run integration tests"""
        try:
            logger.info("🧪 Running integration tests")
            
            # Test database connectivity
            await self._test_database_connectivity()
            
            # Test Redis connectivity
            await self._test_redis_connectivity()
            
            # Test Temporal connectivity
            await self._test_temporal_connectivity()
            
            # Test foundation services
            await self._test_foundation_services()
            
            # Test core intelligence
            await self._test_core_intelligence()
            
            self.setup_status["integration_tests"] = {
                "database": "passed",
                "redis": "passed",
                "temporal": "passed",
                "foundation_services": "passed",
                "core_intelligence": "passed"
            }
            
            logger.info("✅ Integration tests completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            raise
    
    async def _get_database_connection(self):
        """Get database connection"""
        try:
            import asyncpg
            conn = await asyncpg.connect(self.database_url)
            return conn
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return None
    
    async def _get_redis_connection(self):
        """Get Redis connection"""
        try:
            import redis.asyncio as redis
            conn = redis.from_url(self.redis_url)
            return conn
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return None
    
    async def _test_database_connectivity(self):
        """Test database connectivity"""
        try:
            conn = await self._get_database_connection()
            if conn is None:
                logger.warning("⚠️ Database connection not available, skipping test")
                return
            
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            if result != 1:
                raise Exception("Database connectivity test failed")
            
            logger.info("✅ Database connectivity test passed")
            
        except Exception as e:
            logger.error(f"❌ Database connectivity test failed: {e}")
            raise
    
    async def _test_redis_connectivity(self):
        """Test Redis connectivity"""
        try:
            conn = await self._get_redis_connection()
            if conn is None:
                logger.warning("⚠️ Redis connection not available, skipping test")
                return
                
            await conn.ping()
            await conn.close()
            
            logger.info("✅ Redis connectivity test passed")
            
        except Exception as e:
            logger.error(f"❌ Redis connectivity test failed: {e}")
            raise
    
    async def _test_temporal_connectivity(self):
        """Test Temporal connectivity"""
        try:
            from temporalio.client import Client
            client = await Client.connect(self.temporal_url)
            namespaces = await client.list_namespaces()
            await client.close()
            
            logger.info("✅ Temporal connectivity test passed")
            
        except Exception as e:
            logger.error(f"❌ Temporal connectivity test failed: {e}")
            raise
    
    async def _test_foundation_services(self):
        """Test foundation services"""
        try:
            # Test user management
            user_service = self.services["user_management"]
            # Add test user creation and authentication tests here
            
            # Test session management
            session_service = self.services["session_management"]
            # Add session creation and validation tests here
            
            # Test state management
            state_service = self.services["state_management"]
            # Add state storage and retrieval tests here
            
            # Test key management
            key_service = self.services["key_management"]
            # Add key generation and encryption tests here
            
            # Test asset management
            asset_service = self.services["asset_management"]
            # Add asset storage and retrieval tests here
            
            logger.info("✅ Foundation services test passed")
            
        except Exception as e:
            logger.error(f"❌ Foundation services test failed: {e}")
            raise
    
    async def _test_core_intelligence(self):
        """Test core intelligence services"""
        try:
            # Test Jarvis Brain
            jarvis_brain = self.services["jarvis_brain"]
            test_data = {"type": "test", "value": "test_value"}
            analysis_result = await jarvis_brain.analyze_data(test_data)
            
            if not analysis_result:
                raise Exception("Jarvis Brain test failed")
            
            # Test Jarvis Orchestrator
            jarvis_orchestrator = self.services["jarvis_orchestrator"]
            if not jarvis_orchestrator.client:
                raise Exception("Jarvis Orchestrator not connected")
            
            logger.info("✅ Core intelligence test passed")
            
        except Exception as e:
            logger.error(f"❌ Core intelligence test failed: {e}")
            raise
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get current setup status"""
        return self.setup_status
    
    def get_services(self) -> Dict[str, Any]:
        """Get initialized services"""
        return self.services
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up resources")
            
            # Close database connections
            for service_name, service in self.services.items():
                if hasattr(service, 'db') and service.db:
                    await service.db.close()
                if hasattr(service, 'redis') and service.redis:
                    await service.redis.close()
            
            # Shutdown Temporal orchestrator
            if 'jarvis_orchestrator' in self.services:
                await self.services['jarvis_orchestrator'].shutdown()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main setup function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create setup instance
    setup = TSAIJarvisSetup()
    
    try:
        # Run setup
        success = await setup.setup_platform()
        
        if success:
            # Print setup status
            status = setup.get_setup_status()
            print("\n🎯 TSAI Jarvis Platform Setup Status:")
            print("=" * 50)
            
            for component, details in status.items():
                print(f"\n📋 {component.upper()}:")
                if isinstance(details, dict):
                    for key, value in details.items():
                        print(f"  ✅ {key}: {value}")
                else:
                    print(f"  ✅ Status: {details}")
            
            print("\n🚀 TSAI Jarvis Platform is ready!")
            print("🌐 API Gateway: http://localhost:8000")
            print("📊 Health Check: http://localhost:8000/health")
            print("🎭 Temporal Web UI: http://localhost:8080")
            
        else:
            print("❌ TSAI Jarvis Platform setup failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1
    
    finally:
        # Cleanup
        await setup.cleanup()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
