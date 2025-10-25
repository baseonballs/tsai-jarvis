-- TSAI Platform Database Schema
-- PostgreSQL v15 Enterprise Configuration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS platform;
CREATE SCHEMA IF NOT EXISTS users;
CREATE SCHEMA IF NOT EXISTS assets;
CREATE SCHEMA IF NOT EXISTS workflows;
CREATE SCHEMA IF NOT EXISTS pipelines;
CREATE SCHEMA IF NOT EXISTS security;

-- ============================================================================
-- PLATFORM SCHEMA - Core platform tables
-- ============================================================================

-- Audit logs table with partitioning
CREATE TABLE platform.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for audit logs
CREATE TABLE platform.audit_logs_2024_01 
PARTITION OF platform.audit_logs 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE platform.audit_logs_2024_02 
PARTITION OF platform.audit_logs 
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- System configuration table
CREATE TABLE platform.system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature flags table
CREATE TABLE platform.feature_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    flag_name VARCHAR(255) UNIQUE NOT NULL,
    flag_value BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT,
    target_users JSONB, -- User targeting rules
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- USERS SCHEMA - User management and authentication
-- ============================================================================

-- Users table
CREATE TABLE users.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    profile_data JSONB DEFAULT '{}',
    role VARCHAR(50) DEFAULT 'viewer',
    license_type VARCHAR(50) DEFAULT 'trial',
    subscription_status VARCHAR(20) DEFAULT 'pending',
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions table
CREATE TABLE users.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    session_token VARCHAR(500) UNIQUE NOT NULL,
    refresh_token VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User entitlements table
CREATE TABLE users.entitlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    license_type VARCHAR(50) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, feature_name)
);

-- User subscriptions table
CREATE TABLE users.subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    plan_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    billing_cycle VARCHAR(20) DEFAULT 'monthly',
    next_billing_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User preferences table
CREATE TABLE users.user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    preference_key VARCHAR(100) NOT NULL,
    preference_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, preference_key)
);

-- ============================================================================
-- ASSETS SCHEMA - Media asset management
-- ============================================================================

-- Assets table
CREATE TABLE assets.assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash VARCHAR(64) UNIQUE, -- SHA256 hash for deduplication
    mime_type VARCHAR(100) NOT NULL,
    asset_type VARCHAR(50) NOT NULL,
    storage_tier VARCHAR(20) DEFAULT 'hot',
    status VARCHAR(20) DEFAULT 'uploading',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    owner_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Asset metadata table
CREATE TABLE assets.asset_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES assets.assets(id) ON DELETE CASCADE,
    metadata_type VARCHAR(100) NOT NULL,
    metadata_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Asset versions table (for optimized versions)
CREATE TABLE assets.asset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_id UUID NOT NULL REFERENCES assets.assets(id) ON DELETE CASCADE,
    optimized_id UUID NOT NULL REFERENCES assets.assets(id) ON DELETE CASCADE,
    optimization_params JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Asset lifecycle policies table
CREATE TABLE assets.lifecycle_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES assets.assets(id) ON DELETE CASCADE,
    policy_type VARCHAR(50) NOT NULL, -- 'archival', 'deletion', 'migration'
    trigger_condition VARCHAR(50) NOT NULL, -- 'age', 'access_count', 'size'
    trigger_value JSONB NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'archive', 'delete', 'migrate'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Asset access logs table
CREATE TABLE assets.asset_access_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES assets.assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    access_type VARCHAR(50) NOT NULL, -- 'read', 'write', 'delete'
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for asset access logs
CREATE TABLE assets.asset_access_logs_2024_01 
PARTITION OF assets.asset_access_logs 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ============================================================================
-- WORKFLOWS SCHEMA - Temporal workflow management
-- ============================================================================

-- Workflow definitions table
CREATE TABLE workflows.workflow_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(255) UNIQUE NOT NULL,
    workflow_version VARCHAR(50) NOT NULL,
    workflow_definition JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow executions table
CREATE TABLE workflows.workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL, -- Temporal workflow ID
    workflow_name VARCHAR(255) NOT NULL,
    workflow_version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed', 'cancelled'
    input_data JSONB,
    output_data JSONB,
    error_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID NOT NULL REFERENCES users.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow activities table
CREATE TABLE workflows.workflow_activities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_execution_id UUID NOT NULL REFERENCES workflows.workflow_executions(id) ON DELETE CASCADE,
    activity_name VARCHAR(255) NOT NULL,
    activity_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'pending', 'running', 'completed', 'failed'
    input_data JSONB,
    output_data JSONB,
    error_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- ============================================================================
-- PIPELINES SCHEMA - Business pipeline definitions
-- ============================================================================

-- Pipeline definitions table
CREATE TABLE pipelines.pipeline_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_name VARCHAR(255) UNIQUE NOT NULL,
    pipeline_version VARCHAR(50) NOT NULL,
    pipeline_type VARCHAR(100) NOT NULL, -- 'hockey_analytics', 'media_curation', etc.
    pipeline_definition JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID NOT NULL REFERENCES users.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pipeline executions table
CREATE TABLE pipelines.pipeline_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines.pipeline_definitions(id),
    workflow_execution_id UUID REFERENCES workflows.workflow_executions(id),
    status VARCHAR(50) NOT NULL, -- 'pending', 'running', 'completed', 'failed'
    input_data JSONB,
    output_data JSONB,
    error_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID NOT NULL REFERENCES users.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pipeline stages table
CREATE TABLE pipelines.pipeline_stages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_execution_id UUID NOT NULL REFERENCES pipelines.pipeline_executions(id) ON DELETE CASCADE,
    stage_name VARCHAR(255) NOT NULL,
    stage_order INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'pending', 'running', 'completed', 'failed'
    input_data JSONB,
    output_data JSONB,
    error_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- ============================================================================
-- SECURITY SCHEMA - Security and encryption
-- ============================================================================

-- API keys table
CREATE TABLE security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL, -- Hashed API key
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Encryption keys table
CREATE TABLE security.encryption_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) UNIQUE NOT NULL,
    key_type VARCHAR(50) NOT NULL, -- 'AES', 'RSA', 'ECDSA'
    key_purpose VARCHAR(100) NOT NULL, -- 'encryption', 'signing', 'authentication'
    key_data BYTEA NOT NULL, -- Encrypted key data
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Security events table
CREATE TABLE security.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL, -- 'login', 'logout', 'failed_auth', 'suspicious_activity'
    user_id UUID REFERENCES users.users(id) ON DELETE SET NULL,
    ip_address INET,
    user_agent TEXT,
    event_data JSONB,
    severity VARCHAR(20) DEFAULT 'info', -- 'info', 'warning', 'error', 'critical'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for security events
CREATE TABLE security.security_events_2024_01 
PARTITION OF security.security_events 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Users indexes
CREATE INDEX idx_users_email ON users.users(email);
CREATE INDEX idx_users_username ON users.users(username);
CREATE INDEX idx_users_license_type ON users.users(license_type);
CREATE INDEX idx_users_subscription_status ON users.users(subscription_status);
CREATE INDEX idx_users_created_at ON users.users(created_at);

-- User sessions indexes
CREATE INDEX idx_user_sessions_user_id ON users.user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON users.user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires_at ON users.user_sessions(expires_at);

-- Assets indexes
CREATE INDEX idx_assets_owner_id ON assets.assets(owner_id);
CREATE INDEX idx_assets_asset_type ON assets.assets(asset_type);
CREATE INDEX idx_assets_storage_tier ON assets.assets(storage_tier);
CREATE INDEX idx_assets_status ON assets.assets(status);
CREATE INDEX idx_assets_created_at ON assets.assets(created_at);
CREATE INDEX idx_assets_tags ON assets.assets USING GIN(tags);
CREATE INDEX idx_assets_metadata ON assets.assets USING GIN(metadata);

-- Workflow indexes
CREATE INDEX idx_workflow_executions_workflow_id ON workflows.workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_status ON workflows.workflow_executions(status);
CREATE INDEX idx_workflow_executions_created_by ON workflows.workflow_executions(created_by);
CREATE INDEX idx_workflow_executions_started_at ON workflows.workflow_executions(started_at);

-- Pipeline indexes
CREATE INDEX idx_pipeline_executions_pipeline_id ON pipelines.pipeline_executions(pipeline_id);
CREATE INDEX idx_pipeline_executions_status ON pipelines.pipeline_executions(status);
CREATE INDEX idx_pipeline_executions_created_by ON pipelines.pipeline_executions(created_by);
CREATE INDEX idx_pipeline_executions_started_at ON pipelines.pipeline_executions(started_at);

-- Security indexes
CREATE INDEX idx_api_keys_user_id ON security.api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON security.api_keys(key_hash);
CREATE INDEX idx_security_events_user_id ON security.security_events(user_id);
CREATE INDEX idx_security_events_event_type ON security.security_events(event_type);
CREATE INDEX idx_security_events_severity ON security.security_events(severity);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON users.user_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_entitlements_updated_at BEFORE UPDATE ON users.entitlements FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON users.subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_assets_updated_at BEFORE UPDATE ON assets.assets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_definitions_updated_at BEFORE UPDATE ON workflows.workflow_definitions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_pipeline_definitions_updated_at BEFORE UPDATE ON pipelines.pipeline_definitions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- User summary view
CREATE VIEW users.user_summary AS
SELECT 
    u.id,
    u.email,
    u.username,
    u.role,
    u.license_type,
    u.subscription_status,
    u.is_active,
    u.is_verified,
    u.last_login,
    u.created_at,
    COUNT(DISTINCT s.id) as active_sessions,
    COUNT(DISTINCT e.id) as entitlements_count
FROM users.users u
LEFT JOIN users.user_sessions s ON u.id = s.user_id AND s.is_active = TRUE
LEFT JOIN users.entitlements e ON u.id = e.user_id AND e.is_active = TRUE
GROUP BY u.id, u.email, u.username, u.role, u.license_type, u.subscription_status, 
         u.is_active, u.is_verified, u.last_login, u.created_at;

-- Asset summary view
CREATE VIEW assets.asset_summary AS
SELECT 
    a.id,
    a.filename,
    a.asset_type,
    a.storage_tier,
    a.status,
    a.file_size,
    a.access_count,
    a.is_public,
    a.owner_id,
    u.username as owner_username,
    a.created_at,
    a.accessed_at
FROM assets.assets a
JOIN users.users u ON a.owner_id = u.id;

-- Workflow execution summary view
CREATE VIEW workflows.workflow_execution_summary AS
SELECT 
    we.id,
    we.workflow_id,
    we.workflow_name,
    we.status,
    we.started_at,
    we.completed_at,
    u.username as created_by_username,
    COUNT(wa.id) as activities_count,
    COUNT(CASE WHEN wa.status = 'completed' THEN 1 END) as completed_activities,
    COUNT(CASE WHEN wa.status = 'failed' THEN 1 END) as failed_activities
FROM workflows.workflow_executions we
JOIN users.users u ON we.created_by = u.id
LEFT JOIN workflows.workflow_activities wa ON we.id = wa.workflow_execution_id
GROUP BY we.id, we.workflow_id, we.workflow_name, we.status, we.started_at, 
         we.completed_at, u.username;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE users.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE assets.assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.api_keys ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY users_own_data ON users.users
    FOR ALL TO authenticated
    USING (id = current_setting('app.current_user_id')::uuid);

-- Assets access policy
CREATE POLICY assets_access ON assets.assets
    FOR ALL TO authenticated
    USING (
        owner_id = current_setting('app.current_user_id')::uuid 
        OR is_public = TRUE
    );

-- API keys access policy
CREATE POLICY api_keys_own_data ON security.api_keys
    FOR ALL TO authenticated
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default system configuration
INSERT INTO platform.system_config (config_key, config_value, description) VALUES
('platform_name', '"TSAI Platform"', 'Platform name'),
('platform_version', '"1.0.0"', 'Platform version'),
('max_file_size', '1073741824', 'Maximum file size in bytes (1GB)'),
('allowed_file_types', '["image", "video", "audio", "document"]', 'Allowed file types'),
('storage_retention_days', '365', 'Default storage retention in days');

-- Insert default feature flags
INSERT INTO platform.feature_flags (flag_name, flag_value, description) VALUES
('hockey_analytics_enabled', TRUE, 'Enable hockey analytics features'),
('ai_pipeline_autonomous', TRUE, 'Enable autonomous AI pipelines'),
('media_curation_advanced', FALSE, 'Enable advanced media curation'),
('user_registration_open', TRUE, 'Allow new user registration'),
('api_rate_limiting', TRUE, 'Enable API rate limiting');

-- Insert default workflow definitions
INSERT INTO workflows.workflow_definitions (workflow_name, workflow_version, workflow_definition) VALUES
('hockey_highlight_pipeline', '1.0.0', '{"stages": ["video_ingestion", "event_detection", "highlight_generation", "media_curation"]}'),
('model_training_pipeline', '1.0.0', '{"stages": ["data_preparation", "model_training", "validation", "deployment"]}'),
('media_curation_pipeline', '1.0.0', '{"stages": ["asset_ingestion", "content_analysis", "organization", "cataloguing"]}');

-- Insert default pipeline definitions
INSERT INTO pipelines.pipeline_definitions (pipeline_name, pipeline_version, pipeline_type, pipeline_definition) VALUES
('hockey_highlight_pipeline', '1.0.0', 'hockey_analytics', '{"description": "Generate hockey highlights from video", "stages": ["video_ingestion", "event_detection", "highlight_generation", "media_curation"]}'),
('model_training_pipeline', '1.0.0', 'ai_ml', '{"description": "Train AI models for sports analytics", "stages": ["data_preparation", "model_training", "validation", "deployment"]}'),
('media_curation_pipeline', '1.0.0', 'media_management', '{"description": "Curate and organize media assets", "stages": ["asset_ingestion", "content_analysis", "organization", "cataloguing"]}');
