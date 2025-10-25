"""
TSAI Platform Database Setup Script

This script sets up the PostgreSQL v15 database with enterprise configuration
for the TSAI platform.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and configuration manager"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info("🗄️ DatabaseSetup initialized: Ready for PostgreSQL v15 setup")
    
    async def setup_database(self) -> bool:
        """
        Set up the complete database schema and configuration.
        
        Returns:
            True if setup successful
        """
        try:
            logger.info("🚀 Starting TSAI Platform database setup")
            
            # 1. Create database if it doesn't exist
            await self._create_database()
            
            # 2. Load and execute schema
            await self._load_schema()
            
            # 3. Create indexes for performance
            await self._create_indexes()
            
            # 4. Set up partitioning
            await self._setup_partitioning()
            
            # 5. Configure PostgreSQL settings
            await self._configure_postgresql()
            
            # 6. Insert initial data
            await self._insert_initial_data()
            
            # 7. Set up monitoring
            await self._setup_monitoring()
            
            logger.info("✅ TSAI Platform database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    async def _create_database(self):
        """Create database if it doesn't exist"""
        try:
            logger.info("📁 Creating database if not exists")
            
            # Connect to postgres database to create our database
            admin_conn_string = self.connection_string.replace('/tsai_platform', '/postgres')
            conn = await asyncpg.connect(admin_conn_string)
            
            # Check if database exists
            result = await conn.fetchval("""
                SELECT 1 FROM pg_database WHERE datname = 'tsai_platform'
            """)
            
            if not result:
                # Create database
                await conn.execute("CREATE DATABASE tsai_platform")
                logger.info("✅ Database 'tsai_platform' created")
            else:
                logger.info("✅ Database 'tsai_platform' already exists")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to create database: {e}")
            raise
    
    async def _load_schema(self):
        """Load and execute database schema"""
        try:
            logger.info("📋 Loading database schema")
            
            # Read schema file
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema
            conn = await asyncpg.connect(self.connection_string)
            await conn.execute(schema_sql)
            await conn.close()
            
            logger.info("✅ Database schema loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load schema: {e}")
            raise
    
    async def _create_indexes(self):
        """Create performance indexes"""
        try:
            logger.info("📊 Creating performance indexes")
            
            indexes = [
                # Users indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users.users(email);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username ON users.users(username);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_license_type ON users.users(license_type);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_subscription_status ON users.users(subscription_status);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users.users(created_at);",
                
                # User sessions indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id ON users.user_sessions(user_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_token ON users.user_sessions(session_token);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_at ON users.user_sessions(expires_at);",
                
                # Assets indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_owner_id ON assets.assets(owner_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_asset_type ON assets.assets(asset_type);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_storage_tier ON assets.assets(storage_tier);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_status ON assets.assets(status);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_created_at ON assets.assets(created_at);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_tags ON assets.assets USING GIN(tags);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assets_metadata ON assets.assets USING GIN(metadata);",
                
                # Workflow indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_executions_workflow_id ON workflows.workflow_executions(workflow_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_executions_status ON workflows.workflow_executions(status);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_executions_created_by ON workflows.workflow_executions(created_by);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_executions_started_at ON workflows.workflow_executions(started_at);",
                
                # Pipeline indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pipeline_executions_pipeline_id ON pipelines.pipeline_executions(pipeline_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pipeline_executions_status ON pipelines.pipeline_executions(status);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pipeline_executions_created_by ON pipelines.pipeline_executions(created_by);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pipeline_executions_started_at ON pipelines.pipeline_executions(started_at);",
                
                # Security indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id ON security.api_keys(user_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_hash ON security.api_keys(key_hash);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_user_id ON security.security_events(user_id);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_event_type ON security.security_events(event_type);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_severity ON security.security_events(severity);",
            ]
            
            conn = await asyncpg.connect(self.connection_string)
            
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"⚠️ Index creation warning: {e}")
            
            await conn.close()
            
            logger.info("✅ Performance indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            raise
    
    async def _setup_partitioning(self):
        """Set up table partitioning"""
        try:
            logger.info("📊 Setting up table partitioning")
            
            # Create additional partitions for audit logs
            partitions = [
                ("2024_03", "2024-03-01", "2024-04-01"),
                ("2024_04", "2024-04-01", "2024-05-01"),
                ("2024_05", "2024-05-01", "2024-06-01"),
                ("2024_06", "2024-06-01", "2024-07-01"),
            ]
            
            conn = await asyncpg.connect(self.connection_string)
            
            for partition_name, start_date, end_date in partitions:
                # Create audit log partition
                audit_partition_sql = f"""
                    CREATE TABLE IF NOT EXISTS platform.audit_logs_{partition_name}
                    PARTITION OF platform.audit_logs
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """
                
                # Create asset access log partition
                asset_partition_sql = f"""
                    CREATE TABLE IF NOT EXISTS assets.asset_access_logs_{partition_name}
                    PARTITION OF assets.asset_access_logs
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """
                
                # Create security events partition
                security_partition_sql = f"""
                    CREATE TABLE IF NOT EXISTS security.security_events_{partition_name}
                    PARTITION OF security.security_events
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """
                
                try:
                    await conn.execute(audit_partition_sql)
                    await conn.execute(asset_partition_sql)
                    await conn.execute(security_partition_sql)
                except Exception as e:
                    logger.warning(f"⚠️ Partition creation warning: {e}")
            
            await conn.close()
            
            logger.info("✅ Table partitioning set up successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup partitioning: {e}")
            raise
    
    async def _configure_postgresql(self):
        """Configure PostgreSQL for enterprise use"""
        try:
            logger.info("⚙️ Configuring PostgreSQL for enterprise use")
            
            # PostgreSQL configuration for enterprise
            config_settings = {
                'shared_preload_libraries': "'pg_stat_statements'",
                'max_connections': '1000',
                'shared_buffers': '4GB',
                'effective_cache_size': '12GB',
                'maintenance_work_mem': '1GB',
                'checkpoint_completion_target': '0.9',
                'wal_buffers': '64MB',
                'default_statistics_target': '100',
                'random_page_cost': '1.1',
                'effective_io_concurrency': '200',
                'log_statement': "'all'",
                'log_min_duration_statement': '1000',
                'log_line_prefix': "'%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '",
                'log_checkpoints': 'on',
                'log_connections': 'on',
                'log_disconnections': 'on',
                'log_lock_waits': 'on',
                'log_temp_files': '0',
                'log_autovacuum_min_duration': '0',
                'log_error_verbosity': 'verbose',
                'log_min_messages': 'warning',
                'log_min_error_statement': 'error',
                'log_min_duration_statement': '1000',
                'track_activities': 'on',
                'track_counts': 'on',
                'track_io_timing': 'on',
                'track_functions': 'all',
                'stats_temp_directory': "'/tmp'",
                'autovacuum': 'on',
                'autovacuum_max_workers': '3',
                'autovacuum_naptime': '1min',
                'autovacuum_vacuum_threshold': '50',
                'autovacuum_analyze_threshold': '50',
                'autovacuum_vacuum_scale_factor': '0.1',
                'autovacuum_analyze_scale_factor': '0.05',
                'autovacuum_freeze_max_age': '200000000',
                'autovacuum_multixact_freeze_max_age': '400000000',
                'autovacuum_vacuum_cost_delay': '20ms',
                'autovacuum_vacuum_cost_limit': '200',
                'deadlock_timeout': '1s',
                'lock_timeout': '0',
                'statement_timeout': '0',
                'idle_in_transaction_session_timeout': '0',
                'vacuum_cost_delay': '0',
                'vacuum_cost_page_hit': '1',
                'vacuum_cost_page_miss': '10',
                'vacuum_cost_page_dirty': '20',
                'vacuum_cost_limit': '200',
                'bgwriter_delay': '200ms',
                'bgwriter_lru_maxpages': '100',
                'bgwriter_lru_multiplier': '2.0',
                'bgwriter_flush_after': '512kB',
                'checkpoint_timeout': '5min',
                'checkpoint_completion_target': '0.9',
                'checkpoint_flush_after': '256kB',
                'checkpoint_warning': '30s',
                'max_wal_size': '1GB',
                'min_wal_size': '80MB',
                'wal_compression': 'off',
                'wal_buffers': '16MB',
                'wal_writer_delay': '200ms',
                'wal_writer_flush_after': '1MB',
                'commit_delay': '0',
                'commit_siblings': '5',
                'synchronous_commit': 'on',
                'wal_sync_method': 'fsync',
                'full_page_writes': 'on',
                'wal_log_hints': 'off',
                'wal_compression': 'off',
                'wal_init_zero': 'on',
                'wal_recycle': 'on',
                'hot_standby': 'on',
                'hot_standby_feedback': 'off',
                'max_standby_archive_delay': '30s',
                'max_standby_streaming_delay': '30s',
                'wal_receiver_status_interval': '10s',
                'hot_standby_feedback': 'off',
                'wal_receiver_timeout': '60s',
                'wal_retrieve_retry_interval': '5s',
                'wal_sender_timeout': '60s',
                'replication_timeout': '60s',
                'replication_slot_timeout': '60s',
                'max_replication_slots': '10',
                'max_wal_senders': '10',
                'track_commit_timestamp': 'off',
                'default_transaction_isolation': 'read committed',
                'transaction_isolation': 'read committed',
                'default_transaction_read_only': 'off',
                'transaction_read_only': 'off',
                'default_transaction_deferrable': 'off',
                'transaction_deferrable': 'off',
                'synchronous_standby_names': '',
                'vacuum_defer_cleanup_age': '0',
                'max_standby_archive_delay': '30s',
                'max_standby_streaming_delay': '30s',
                'wal_receiver_status_interval': '10s',
                'hot_standby_feedback': 'off',
                'wal_receiver_timeout': '60s',
                'wal_retrieve_retry_interval': '5s',
                'wal_sender_timeout': '60s',
                'replication_timeout': '60s',
                'replication_slot_timeout': '60s',
                'max_replication_slots': '10',
                'max_wal_senders': '10',
                'track_commit_timestamp': 'off',
            }
            
            conn = await asyncpg.connect(self.connection_string)
            
            for setting, value in config_settings.items():
                try:
                    await conn.execute(f"ALTER SYSTEM SET {setting} = {value};")
                except Exception as e:
                    logger.warning(f"⚠️ Configuration warning for {setting}: {e}")
            
            # Reload configuration
            await conn.execute("SELECT pg_reload_conf();")
            await conn.close()
            
            logger.info("✅ PostgreSQL configured for enterprise use")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure PostgreSQL: {e}")
            raise
    
    async def _insert_initial_data(self):
        """Insert initial data and configuration"""
        try:
            logger.info("📝 Inserting initial data")
            
            conn = await asyncpg.connect(self.connection_string)
            
            # Check if initial data already exists
            existing_config = await conn.fetchval("""
                SELECT COUNT(*) FROM platform.system_config
            """)
            
            if existing_config > 0:
                logger.info("✅ Initial data already exists")
                await conn.close()
                return
            
            # Insert system configuration
            system_configs = [
                ('platform_name', '"TSAI Platform"', 'Platform name'),
                ('platform_version', '"1.0.0"', 'Platform version'),
                ('max_file_size', '1073741824', 'Maximum file size in bytes (1GB)'),
                ('allowed_file_types', '["image", "video", "audio", "document"]', 'Allowed file types'),
                ('storage_retention_days', '365', 'Default storage retention in days'),
                ('session_timeout_hours', '24', 'Default session timeout in hours'),
                ('max_concurrent_sessions', '10', 'Maximum concurrent sessions per user'),
                ('api_rate_limit_per_minute', '100', 'API rate limit per minute'),
                ('encryption_algorithm', 'AES-256-GCM', 'Default encryption algorithm'),
                ('key_rotation_days', '365', 'Default key rotation interval'),
            ]
            
            for config_key, config_value, description in system_configs:
                await conn.execute("""
                    INSERT INTO platform.system_config (config_key, config_value, description)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (config_key) DO NOTHING
                """, config_key, config_value, description)
            
            # Insert feature flags
            feature_flags = [
                ('hockey_analytics_enabled', True, 'Enable hockey analytics features'),
                ('ai_pipeline_autonomous', True, 'Enable autonomous AI pipelines'),
                ('media_curation_advanced', False, 'Enable advanced media curation'),
                ('user_registration_open', True, 'Allow new user registration'),
                ('api_rate_limiting', True, 'Enable API rate limiting'),
                ('session_management', True, 'Enable session management'),
                ('key_rotation_automatic', True, 'Enable automatic key rotation'),
                ('audit_logging', True, 'Enable audit logging'),
                ('performance_monitoring', True, 'Enable performance monitoring'),
                ('security_scanning', True, 'Enable security scanning'),
            ]
            
            for flag_name, flag_value, description in feature_flags:
                await conn.execute("""
                    INSERT INTO platform.feature_flags (flag_name, flag_value, description)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (flag_name) DO NOTHING
                """, flag_name, flag_value, description)
            
            # Insert default workflow definitions
            workflow_definitions = [
                ('hockey_highlight_pipeline', '1.0.0', '{"stages": ["video_ingestion", "event_detection", "highlight_generation", "media_curation"], "timeout": 3600, "retry_policy": {"max_retries": 3, "backoff": "exponential"}}'),
                ('model_training_pipeline', '1.0.0', '{"stages": ["data_preparation", "model_training", "validation", "deployment"], "timeout": 7200, "retry_policy": {"max_retries": 2, "backoff": "linear"}}'),
                ('media_curation_pipeline', '1.0.0', '{"stages": ["asset_ingestion", "content_analysis", "organization", "cataloguing"], "timeout": 1800, "retry_policy": {"max_retries": 3, "backoff": "exponential"}}'),
                ('user_authentication_workflow', '1.0.0', '{"stages": ["credential_validation", "session_creation", "token_generation"], "timeout": 300, "retry_policy": {"max_retries": 1, "backoff": "none"}}'),
                ('asset_optimization_workflow', '1.0.0', '{"stages": ["asset_analysis", "optimization_planning", "processing", "storage"], "timeout": 2400, "retry_policy": {"max_retries": 2, "backoff": "linear"}}'),
            ]
            
            for workflow_name, workflow_version, workflow_definition in workflow_definitions:
                await conn.execute("""
                    INSERT INTO workflows.workflow_definitions (workflow_name, workflow_version, workflow_definition)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (workflow_name) DO NOTHING
                """, workflow_name, workflow_version, workflow_definition)
            
            # Insert default pipeline definitions
            pipeline_definitions = [
                ('hockey_highlight_pipeline', '1.0.0', 'hockey_analytics', '{"description": "Generate hockey highlights from video", "stages": ["video_ingestion", "event_detection", "highlight_generation", "media_curation"], "business_rules": {"quality_threshold": 0.8, "max_duration": 300}}'),
                ('model_training_pipeline', '1.0.0', 'ai_ml', '{"description": "Train AI models for sports analytics", "stages": ["data_preparation", "model_training", "validation", "deployment"], "business_rules": {"min_accuracy": 0.85, "max_training_time": 7200}}'),
                ('media_curation_pipeline', '1.0.0', 'media_management', '{"description": "Curate and organize media assets", "stages": ["asset_ingestion", "content_analysis", "organization", "cataloguing"], "business_rules": {"quality_threshold": 0.7, "max_file_size": 1073741824}}'),
                ('user_onboarding_pipeline', '1.0.0', 'user_management', '{"description": "Onboard new users to the platform", "stages": ["registration", "verification", "profile_setup", "entitlement_assignment"], "business_rules": {"verification_required": true, "default_license": "trial"}}'),
                ('security_audit_pipeline', '1.0.0', 'security', '{"description": "Perform security audits and monitoring", "stages": ["threat_detection", "vulnerability_scanning", "compliance_checking", "reporting"], "business_rules": {"scan_frequency": "daily", "alert_threshold": 0.8}}'),
            ]
            
            for pipeline_name, pipeline_version, pipeline_type, pipeline_definition in pipeline_definitions:
                await conn.execute("""
                    INSERT INTO pipelines.pipeline_definitions (pipeline_name, pipeline_version, pipeline_type, pipeline_definition)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (pipeline_name) DO NOTHING
                """, pipeline_name, pipeline_version, pipeline_type, pipeline_definition)
            
            await conn.close()
            
            logger.info("✅ Initial data inserted successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert initial data: {e}")
            raise
    
    async def _setup_monitoring(self):
        """Set up database monitoring and statistics"""
        try:
            logger.info("📊 Setting up database monitoring")
            
            conn = await asyncpg.connect(self.connection_string)
            
            # Enable pg_stat_statements extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
            
            # Create monitoring views
            monitoring_views = [
                """
                CREATE OR REPLACE VIEW platform.database_stats AS
                SELECT 
                    'connections' as metric,
                    (SELECT count(*) FROM pg_stat_activity) as current_value,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_value
                UNION ALL
                SELECT 
                    'database_size',
                    pg_database_size(current_database()),
                    NULL
                UNION ALL
                SELECT 
                    'cache_hit_ratio',
                    round(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2),
                    NULL
                FROM pg_stat_database 
                WHERE datname = current_database();
                """,
                """
                CREATE OR REPLACE VIEW platform.table_stats AS
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC;
                """,
                """
                CREATE OR REPLACE VIEW platform.index_stats AS
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC;
                """,
                """
                CREATE OR REPLACE VIEW platform.query_stats AS
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements
                ORDER BY total_time DESC
                LIMIT 50;
                """
            ]
            
            for view_sql in monitoring_views:
                try:
                    await conn.execute(view_sql)
                except Exception as e:
                    logger.warning(f"⚠️ Monitoring view creation warning: {e}")
            
            await conn.close()
            
            logger.info("✅ Database monitoring set up successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup monitoring: {e}")
            raise
    
    async def verify_setup(self) -> Dict[str, Any]:
        """
        Verify database setup and return status.
        
        Returns:
            Setup verification results
        """
        try:
            logger.info("🔍 Verifying database setup")
            
            conn = await asyncpg.connect(self.connection_string)
            
            # Check schemas
            schemas = await conn.fetch("""
                SELECT schema_name FROM information_schema.schemata 
                WHERE schema_name IN ('platform', 'users', 'assets', 'workflows', 'pipelines', 'security')
                ORDER BY schema_name;
            """)
            
            # Check tables
            tables = await conn.fetch("""
                SELECT schemaname, tablename FROM pg_tables 
                WHERE schemaname IN ('platform', 'users', 'assets', 'workflows', 'pipelines', 'security')
                ORDER BY schemaname, tablename;
            """)
            
            # Check indexes
            indexes = await conn.fetch("""
                SELECT schemaname, tablename, indexname FROM pg_indexes 
                WHERE schemaname IN ('platform', 'users', 'assets', 'workflows', 'pipelines', 'security')
                ORDER BY schemaname, tablename, indexname;
            """)
            
            # Check initial data
            config_count = await conn.fetchval("SELECT COUNT(*) FROM platform.system_config")
            feature_count = await conn.fetchval("SELECT COUNT(*) FROM platform.feature_flags")
            workflow_count = await conn.fetchval("SELECT COUNT(*) FROM workflows.workflow_definitions")
            pipeline_count = await conn.fetchval("SELECT COUNT(*) FROM pipelines.pipeline_definitions")
            
            await conn.close()
            
            verification_result = {
                "schemas_created": len(schemas),
                "tables_created": len(tables),
                "indexes_created": len(indexes),
                "initial_data": {
                    "system_config": config_count,
                    "feature_flags": feature_count,
                    "workflow_definitions": workflow_count,
                    "pipeline_definitions": pipeline_count
                },
                "status": "success" if len(schemas) >= 6 and len(tables) >= 20 else "incomplete"
            }
            
            logger.info(f"✅ Database setup verification: {verification_result['status']}")
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Database setup verification failed: {e}")
            return {"status": "failed", "error": str(e)}

async def main():
    """Main setup function"""
    # Database connection string
    connection_string = os.getenv(
        "DATABASE_URL", 
        "postgresql://temporal:temporal@localhost:5432/tsai_platform"
    )
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database setup instance
    db_setup = DatabaseSetup(connection_string)
    
    # Run setup
    success = await db_setup.setup_database()
    
    if success:
        # Verify setup
        verification = await db_setup.verify_setup()
        print(f"Database setup completed: {verification}")
    else:
        print("Database setup failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
