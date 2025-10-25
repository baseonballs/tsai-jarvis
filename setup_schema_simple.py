#!/usr/bin/env python3
"""
TSAI Jarvis Database Schema Setup (Simplified)

This script sets up the core database schema for TSAI Jarvis without requiring superuser privileges.
"""

import asyncio
import logging
import asyncpg

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_tsai_schema():
    """Set up the TSAI Jarvis database schema"""
    try:
        logger.info("🚀 Setting up TSAI Jarvis database schema")
        
        # Connect to the temporal database
        conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
        
        # Core schema without extensions
        schema_sql = """
        -- TSAI Jarvis Core Database Schema
        
        -- 1. User Management
        CREATE TABLE IF NOT EXISTS users (
            user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            status VARCHAR(20) DEFAULT 'active' NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id UUID PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
            full_name VARCHAR(255),
            date_of_birth DATE,
            country VARCHAR(100),
            preferences JSONB DEFAULT '{}',
            profile_picture_url VARCHAR(255)
        );

        CREATE TABLE IF NOT EXISTS roles (
            role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            role_name VARCHAR(50) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_roles (
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            role_id UUID REFERENCES roles(role_id) ON DELETE CASCADE,
            PRIMARY KEY (user_id, role_id)
        );

        -- 2. Session Management
        CREATE TABLE IF NOT EXISTS sessions (
            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            token VARCHAR(512) UNIQUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            ip_address INET,
            user_agent TEXT,
            is_valid BOOLEAN DEFAULT TRUE
        );

        -- 3. Asset Management
        CREATE TABLE IF NOT EXISTS assets (
            asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            uploader_user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
            file_name VARCHAR(255) NOT NULL,
            file_type VARCHAR(100) NOT NULL,
            size_bytes BIGINT NOT NULL,
            storage_path TEXT NOT NULL,
            status VARCHAR(50) DEFAULT 'active' NOT NULL,
            uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS asset_metadata (
            asset_id UUID PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            sport_event_id VARCHAR(255),
            event_type VARCHAR(100),
            duration_seconds NUMERIC(10, 2),
            resolution VARCHAR(50),
            codec VARCHAR(50),
            additional_data JSONB DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS asset_tags (
            tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tag_name VARCHAR(100) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS asset_asset_tags (
            asset_id UUID REFERENCES assets(asset_id) ON DELETE CASCADE,
            tag_id UUID REFERENCES asset_tags(tag_id) ON DELETE CASCADE,
            PRIMARY KEY (asset_id, tag_id)
        );

        -- 4. Security & Key Management
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            hashed_key VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(100),
            permissions JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE,
            is_active BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS secrets (
            secret_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) UNIQUE NOT NULL,
            encrypted_value TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at TIMESTAMP WITH TIME ZONE,
            version INT DEFAULT 1
        );

        -- 5. Audit Logging
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
            action VARCHAR(255) NOT NULL,
            resource_type VARCHAR(100),
            resource_id UUID,
            details JSONB DEFAULT '{}',
            ip_address INET
        );

        -- 6. TSAI Platform Specific Tables
        CREATE TABLE IF NOT EXISTS tsai_services (
            service_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            service_name VARCHAR(100) UNIQUE NOT NULL,
            service_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) DEFAULT 'active',
            endpoint_url VARCHAR(255),
            configuration JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_name VARCHAR(255) NOT NULL,
            workflow_type VARCHAR(100) NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            input_data JSONB DEFAULT '{}',
            output_data JSONB DEFAULT '{}',
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE,
            created_by UUID REFERENCES users(user_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS hockey_analytics (
            analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            game_id VARCHAR(255),
            event_type VARCHAR(100),
            player_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE,
            coordinates JSONB,
            confidence_score NUMERIC(5, 4),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
        CREATE INDEX IF NOT EXISTS idx_assets_uploader_user_id ON assets(uploader_user_id);
        CREATE INDEX IF NOT EXISTS idx_asset_metadata_sport_event_id ON asset_metadata(sport_event_id);
        CREATE INDEX IF NOT EXISTS idx_asset_metadata_event_type ON asset_metadata(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
        CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
        CREATE INDEX IF NOT EXISTS idx_hockey_analytics_game_id ON hockey_analytics(game_id);
        CREATE INDEX IF NOT EXISTS idx_hockey_analytics_event_type ON hockey_analytics(event_type);

        -- Insert default roles
        INSERT INTO roles (role_name) VALUES 
            ('admin'),
            ('user'),
            ('editor'),
            ('viewer')
        ON CONFLICT (role_name) DO NOTHING;

        -- Insert default TSAI services
        INSERT INTO tsai_services (service_name, service_type, status, endpoint_url) VALUES 
            ('autopilot', 'ai_pipeline', 'active', 'http://autopilot:4001'),
            ('spotlight', 'video_processing', 'active', 'http://spotlight:4002'),
            ('toolchain', 'development', 'active', 'http://toolchain:4003'),
            ('watson', 'nlp', 'active', 'http://watson:4004'),
            ('holmes', 'media_curation', 'active', 'http://holmes:4005')
        ON CONFLICT (service_name) DO NOTHING;

        -- Create a default admin user
        INSERT INTO users (username, email, password_hash, status) VALUES 
            ('admin', 'admin@tsai-platform.com', 'hashed_admin_password', 'active')
        ON CONFLICT (username) DO NOTHING;

        -- Assign admin role to admin user
        INSERT INTO user_roles (user_id, role_id) 
        SELECT u.user_id, r.role_id 
        FROM users u, roles r 
        WHERE u.username = 'admin' AND r.role_name = 'admin'
        ON CONFLICT DO NOTHING;
        """
        
        # Execute the schema
        await conn.execute(schema_sql)
        
        logger.info("✅ TSAI Jarvis database schema created successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        logger.info(f"📊 Created {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        # Check for data
        user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        role_count = await conn.fetchval("SELECT COUNT(*) FROM roles")
        service_count = await conn.fetchval("SELECT COUNT(*) FROM tsai_services")
        
        logger.info(f"📈 Database populated with:")
        logger.info(f"  - {user_count} users")
        logger.info(f"  - {role_count} roles")
        logger.info(f"  - {service_count} TSAI services")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema setup error: {e}")
        return False

async def main():
    """Main setup function"""
    success = await setup_tsai_schema()
    
    if success:
        print("\n🎯 TSAI Jarvis Database Setup Complete!")
        print("=" * 50)
        print("✅ Database schema created")
        print("✅ Indexes created for performance")
        print("✅ Default roles and services inserted")
        print("✅ Admin user created")
        print("\n🚀 TSAI Jarvis database is ready!")
    else:
        print("\n❌ Database setup failed")
        print("📋 Check the logs above for specific issues")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
