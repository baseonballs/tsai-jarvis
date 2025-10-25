# TSAI Platform Architecture

## Executive Summary

This document defines the comprehensive architecture for the TSAI platform ecosystem, clarifying the semantics between pipelines and workflows, and establishing the foundational managed services that support the entire platform.

## 1. Pipeline vs Workflow Semantics

### 1.1 Pipelines: Business Logic & Intent

**Definition**: A pipeline is a specific, intentful collection of stages and units of work that fulfills a particular business outcome.

**Characteristics**:
- **Business-focused**: Designed to achieve specific business goals
- **Outcome-oriented**: Produces tangible results (e.g., hockey highlights)
- **Domain-specific**: Tailored to specific use cases (sports analytics, media processing)
- **User-facing**: What users interact with and understand

**Examples**:
- **Hockey Highlight Pipeline**: Processes video → detects events → generates highlights
- **Model Training Pipeline**: Prepares data → trains model → validates → deploys
- **Media Curation Pipeline**: Ingests assets → analyzes content → organizes → catalogs

### 1.2 Workflows: Infrastructure Plumbing & Orchestration

**Definition**: A workflow is the infrastructure plumbing that substantiates pipeline execution through orchestration engines, frameworks, and technical coordination.

**Characteristics**:
- **Infrastructure-focused**: Technical orchestration and coordination
- **Execution-oriented**: How work gets done, not what gets done
- **Technology-agnostic**: Can use Temporal, Airflow, or other orchestration engines
- **System-facing**: Internal technical implementation

**Examples**:
- **Temporal Workflows**: Durable, fault-tolerant execution
- **Kubernetes Jobs**: Container orchestration
- **Apache Airflow DAGs**: Data pipeline orchestration
- **Custom Orchestration**: Service coordination and state management

### 1.3 Relationship

```
Business Pipeline (What)
├── Stage 1: Video Ingestion
├── Stage 2: Event Detection  
├── Stage 3: Highlight Generation
└── Stage 4: Media Curation

Infrastructure Workflow (How)
├── Temporal Workflow Engine
├── Service Orchestration
├── State Management
├── Error Handling
└── Monitoring & Observability
```

## 2. TSAI Platform Architecture

### 2.1 Platform Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    TSAI Platform Layers                    │
├─────────────────────────────────────────────────────────────┤
│  Business Pipelines (Domain-Specific Business Logic)     │
│  ├── Hockey Analytics Pipeline                            │
│  ├── Media Curation Pipeline                              │
│  ├── Model Training Pipeline                              │
│  └── Content Generation Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer (Workflow Infrastructure)            │
│  ├── TSAI Jarvis (Core Intelligence)                      │
│  ├── Temporal AI Orchestration                           │
│  ├── Service Coordination                                │
│  └── State Management                                    │
├─────────────────────────────────────────────────────────────┤
│  Application Services (TSAI Product Ecosystem)            │
│  ├── Autopilot (AI/ML Pipelines)                         │
│  ├── Spotlight (Events & Highlights)                     │
│  ├── Toolchain (Development Framework)                   │
│  ├── Watson (NLP Reasoning)                              │
│  └── Holmes (Media Curation)                            │
├─────────────────────────────────────────────────────────────┤
│  Foundation Services (Platform Infrastructure)            │
│  ├── User Management & Security                          │
│  ├── Asset Management & Storage                           │
│  ├── Unified State Management                            │
│  ├── Key Management & Encryption                         │
│  └── Session Management                                  │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer (Technical Foundation)             │
│  ├── PostgreSQL v15 (Enterprise Database)               │
│  ├── Distributed Storage (Media Assets)                 │
│  ├── Redis (Caching & Sessions)                          │
│  ├── Temporal Server (Workflow Engine)                   │
│  └── Kubernetes (Container Orchestration)                │
└─────────────────────────────────────────────────────────────┘
```

## 3. Foundation Services (Managed Services)

### 3.1 User Management & Accounts Service

**Purpose**: Comprehensive user lifecycle management, entitlements, and access control

**Capabilities**:
- **User Registration & Authentication**: Account creation, verification, profile management
- **Entitlement Management**: License validation, feature access, usage limits
- **Access Control Lists (ACL)**: Role-based permissions, resource access control
- **Security**: OAuth2, SSO, multi-factor authentication, password policies
- **Payments & Subscriptions**: Billing, subscription management, usage tracking

**Technical Implementation**:
```python
class UserManagementService:
    """Foundation service for user management and accounts"""
    
    def __init__(self):
        self.db = PostgreSQLConnection()
        self.auth_provider = OAuth2Provider()
        self.entitlement_engine = EntitlementEngine()
        self.payment_processor = PaymentProcessor()
    
    async def create_user(self, user_data: UserData) -> User:
        """Create new user account with entitlements"""
        pass
    
    async def authenticate_user(self, credentials: Credentials) -> AuthResult:
        """Authenticate user with SSO/OAuth2"""
        pass
    
    async def check_entitlements(self, user_id: str, feature: str) -> bool:
        """Check user entitlements for specific features"""
        pass
    
    async def manage_subscription(self, user_id: str, plan: SubscriptionPlan) -> bool:
        """Manage user subscription and billing"""
        pass
```

**Database Schema**:
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    profile_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Entitlements table
CREATE TABLE entitlements (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    feature_name VARCHAR(100) NOT NULL,
    license_type VARCHAR(50),
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Subscriptions table
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    plan_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    billing_cycle VARCHAR(20),
    next_billing_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 3.2 Session Management Service

**Purpose**: Cross-application session management with SSO and CORS support

**Capabilities**:
- **Cross-Application Sessions**: Unified session across TSAI services
- **SSO Integration**: Single sign-on across platform
- **CORS Management**: Cross-origin resource sharing
- **Session Security**: Token management, session validation, timeout handling
- **Browser Compatibility**: Web session management

**Technical Implementation**:
```python
class SessionManagementService:
    """Foundation service for session management"""
    
    def __init__(self):
        self.redis = RedisConnection()
        self.jwt_handler = JWTHandler()
        self.sso_provider = SSOProvider()
    
    async def create_session(self, user_id: str, app_context: str) -> Session:
        """Create cross-application session"""
        pass
    
    async def validate_session(self, session_token: str) -> SessionValidation:
        """Validate session token"""
        pass
    
    async def refresh_session(self, session_token: str) -> Session:
        """Refresh session token"""
        pass
    
    async def terminate_session(self, session_token: str) -> bool:
        """Terminate user session"""
        pass
```

### 3.3 Unified State Management Service

**Purpose**: Centralized state management with caching, failover, and consistency

**Capabilities**:
- **Distributed Caching**: Redis-based caching with clustering
- **State Synchronization**: Cross-service state consistency
- **Failover Management**: Automatic failover and recovery
- **Data Consistency**: ACID compliance and eventual consistency
- **Performance Optimization**: Caching strategies and optimization

**Technical Implementation**:
```python
class UnifiedStateService:
    """Foundation service for unified state management"""
    
    def __init__(self):
        self.redis_cluster = RedisCluster()
        self.postgres = PostgreSQLConnection()
        self.consistency_manager = ConsistencyManager()
    
    async def get_state(self, key: str, namespace: str) -> StateValue:
        """Get state value with caching"""
        pass
    
    async def set_state(self, key: str, value: StateValue, namespace: str) -> bool:
        """Set state value with consistency"""
        pass
    
    async def sync_state(self, namespace: str) -> bool:
        """Synchronize state across services"""
        pass
    
    async def handle_failover(self, failed_node: str) -> bool:
        """Handle node failure and recovery"""
        pass
```

### 3.4 Key Management Service

**Purpose**: Centralized key management for encryption, authentication, and security

**Capabilities**:
- **Encryption Key Management**: AES, RSA key generation and rotation
- **API Key Management**: Service-to-service authentication
- **Certificate Management**: SSL/TLS certificate lifecycle
- **Secret Management**: Secure storage and retrieval of secrets
- **Key Rotation**: Automated key rotation and updates

**Technical Implementation**:
```python
class KeyManagementService:
    """Foundation service for key management"""
    
    def __init__(self):
        self.vault = HashiCorpVault()
        self.kms = AWSKMS()
        self.rotation_scheduler = KeyRotationScheduler()
    
    async def generate_key(self, key_type: KeyType, purpose: str) -> Key:
        """Generate new encryption key"""
        pass
    
    async def rotate_key(self, key_id: str) -> Key:
        """Rotate existing key"""
        pass
    
    async def encrypt_data(self, data: bytes, key_id: str) -> EncryptedData:
        """Encrypt data with specified key"""
        pass
    
    async def decrypt_data(self, encrypted_data: EncryptedData, key_id: str) -> bytes:
        """Decrypt data with specified key"""
        pass
```

### 3.5 Security Service

**Purpose**: Comprehensive security management including OAuth2, SSL, SSO, and encryption

**Capabilities**:
- **OAuth2 Implementation**: Authorization server and client management
- **SSL/TLS Management**: Certificate lifecycle and validation
- **SSO Integration**: Single sign-on across platform
- **Encryption Services**: Data encryption and decryption
- **Security Monitoring**: Threat detection and response

**Technical Implementation**:
```python
class SecurityService:
    """Foundation service for security management"""
    
    def __init__(self):
        self.oauth2_server = OAuth2Server()
        self.ssl_manager = SSLManager()
        self.encryption_service = EncryptionService()
        self.threat_detector = ThreatDetector()
    
    async def validate_oauth2_token(self, token: str) -> TokenValidation:
        """Validate OAuth2 token"""
        pass
    
    async def generate_ssl_certificate(self, domain: str) -> SSLCertificate:
        """Generate SSL certificate for domain"""
        pass
    
    async def encrypt_sensitive_data(self, data: str) -> EncryptedString:
        """Encrypt sensitive data"""
        pass
    
    async def detect_security_threats(self, request: SecurityRequest) -> ThreatAssessment:
        """Detect and assess security threats"""
        pass
```

### 3.6 Asset Management Service

**Purpose**: Comprehensive media asset management including storage, metadata, and lifecycle

**Capabilities**:
- **Media Storage**: Images, videos, audio files, documents
- **Metadata Management**: Asset tagging, categorization, search
- **Storage Optimization**: Compression, format conversion, CDN integration
- **Lifecycle Management**: Archival, deletion, retention policies
- **Access Control**: Asset-level permissions and sharing

**Technical Implementation**:
```python
class AssetManagementService:
    """Foundation service for asset management"""
    
    def __init__(self):
        self.storage_backend = DistributedStorage()
        self.metadata_db = PostgreSQLConnection()
        self.cdn = CloudFrontCDN()
        self.optimization_engine = MediaOptimizationEngine()
    
    async def store_asset(self, asset: MediaAsset, metadata: AssetMetadata) -> AssetID:
        """Store media asset with metadata"""
        pass
    
    async def retrieve_asset(self, asset_id: AssetID) -> MediaAsset:
        """Retrieve media asset"""
        pass
    
    async def optimize_asset(self, asset_id: AssetID, optimization_params: OptimizationParams) -> AssetID:
        """Optimize asset for different use cases"""
        pass
    
    async def manage_lifecycle(self, asset_id: AssetID, policy: LifecyclePolicy) -> bool:
        """Manage asset lifecycle (archival, deletion)"""
        pass
```

## 4. Infrastructure Layer

### 4.1 Database Architecture

**PostgreSQL v15 Enterprise Configuration**:
```yaml
# postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
max_connections = 1000
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 64MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

**Database Schema Design**:
```sql
-- Core platform tables
CREATE SCHEMA platform;
CREATE SCHEMA users;
CREATE SCHEMA assets;
CREATE SCHEMA workflows;
CREATE SCHEMA pipelines;

-- Partitioning for scale
CREATE TABLE platform.audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID,
    action VARCHAR(100),
    resource VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE platform.audit_logs_2024_01 
PARTITION OF platform.audit_logs 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 4.2 Distributed Storage Architecture

**Storage Tiers**:
- **Hot Storage**: Frequently accessed assets (Redis, SSD)
- **Warm Storage**: Occasionally accessed assets (S3, HDD)
- **Cold Storage**: Archived assets (Glacier, Tape)

**Storage Configuration**:
```yaml
# storage-config.yaml
storage:
  tiers:
    hot:
      type: "redis"
      ttl: "7d"
      capacity: "100GB"
    warm:
      type: "s3"
      ttl: "90d"
      capacity: "10TB"
    cold:
      type: "glacier"
      ttl: "1y"
      capacity: "100TB"
  
  replication:
    factor: 3
    regions: ["us-east-1", "us-west-2", "eu-west-1"]
  
  encryption:
    algorithm: "AES-256"
    key_rotation: "90d"
```

### 4.3 Caching Architecture

**Multi-Level Caching**:
```python
class CachingArchitecture:
    """Multi-level caching for TSAI platform"""
    
    def __init__(self):
        # L1: Application-level cache
        self.l1_cache = LRUCache(maxsize=10000)
        
        # L2: Redis cluster cache
        self.l2_cache = RedisCluster()
        
        # L3: CDN cache
        self.l3_cache = CloudFrontCDN()
        
        # L4: Database cache
        self.l4_cache = PostgreSQLConnection()
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Multi-level cache retrieval"""
        # Try L1 cache first
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        # Try L3 cache (CDN)
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value)
            self.l1_cache[key] = value
            return value
        
        # Fallback to database
        value = await self.l4_cache.get(key)
        if value:
            await self.l3_cache.set(key, value)
            await self.l2_cache.set(key, value)
            self.l1_cache[key] = value
        
        return value
```

## 5. Pipeline Examples

### 5.1 Hockey Highlight Pipeline

**Business Pipeline (What)**:
```python
class HockeyHighlightPipeline:
    """Business pipeline for hockey highlight generation"""
    
    async def execute(self, video_input: VideoInput) -> HighlightOutput:
        """Execute hockey highlight generation pipeline"""
        
        # Stage 1: Video Ingestion
        video_asset = await self.ingest_video(video_input)
        
        # Stage 2: Event Detection
        events = await self.detect_events(video_asset)
        
        # Stage 3: Highlight Generation
        highlights = await self.generate_highlights(events)
        
        # Stage 4: Media Curation
        curated_highlights = await self.curate_highlights(highlights)
        
        return HighlightOutput(
            highlights=curated_highlights,
            metadata=self.generate_metadata(events),
            analytics=self.generate_analytics(events)
        )
```

**Infrastructure Workflow (How)**:
```python
@workflow.defn
class HockeyHighlightWorkflow:
    """Temporal workflow for hockey highlight generation"""
    
    @workflow.run
    async def run(self, pipeline_request: HockeyHighlightRequest) -> HockeyHighlightResult:
        """Execute workflow with fault tolerance and monitoring"""
        
        # Orchestrate pipeline stages with Temporal
        video_result = await workflow.execute_activity(
            ingest_hockey_video,
            pipeline_request.video_input,
            start_to_close_timeout=timedelta(minutes=30)
        )
        
        events_result = await workflow.execute_activity(
            detect_hockey_events,
            video_result,
            start_to_close_timeout=timedelta(hours=2)
        )
        
        highlights_result = await workflow.execute_activity(
            generate_hockey_highlights,
            events_result,
            start_to_close_timeout=timedelta(hours=1)
        )
        
        curation_result = await workflow.execute_activity(
            curate_hockey_highlights,
            highlights_result,
            start_to_close_timeout=timedelta(minutes=30)
        )
        
        return HockeyHighlightResult(
            highlights=curation_result.highlights,
            metadata=events_result.metadata,
            analytics=highlights_result.analytics
        )
```

## 6. Platform Benefits

### 6.1 Clear Separation of Concerns

- **Business Logic**: Pipelines focus on business outcomes
- **Infrastructure**: Workflows handle technical orchestration
- **Foundation**: Managed services provide platform capabilities
- **Infrastructure**: Technical foundation supports everything

### 6.2 Scalability & Performance

- **Horizontal Scaling**: Services can scale independently
- **Caching**: Multi-level caching for performance
- **Storage**: Distributed storage for media assets
- **Database**: PostgreSQL v15 for enterprise scale

### 6.3 Security & Compliance

- **User Management**: Comprehensive user lifecycle
- **Security**: OAuth2, SSL, encryption, SSO
- **Access Control**: Role-based permissions
- **Audit**: Complete audit trail

### 6.4 Developer Experience

- **Clear Semantics**: Pipelines vs workflows
- **Foundation Services**: Reusable platform capabilities
- **API Consistency**: Unified API across services
- **Documentation**: Clear architecture and patterns

## 7. Implementation Roadmap

### Phase 1: Foundation Services (Weeks 1-4)
- [ ] User Management & Security Service
- [ ] Session Management Service
- [ ] Key Management Service
- [ ] Database setup (PostgreSQL v15)

### Phase 2: Asset & State Management (Weeks 5-8)
- [ ] Asset Management Service
- [ ] Unified State Management Service
- [ ] Distributed Storage setup
- [ ] Caching architecture

### Phase 3: Pipeline & Workflow Integration (Weeks 9-12)
- [ ] Pipeline semantics implementation
- [ ] Workflow orchestration
- [ ] TSAI service integration
- [ ] End-to-end testing

### Phase 4: Production Deployment (Weeks 13-16)
- [ ] Production infrastructure
- [ ] Monitoring and observability
- [ ] Security hardening
- [ ] Performance optimization

## Conclusion

This architecture provides a clear, scalable foundation for the TSAI platform ecosystem with:

- **Clear Semantics**: Pipelines (business logic) vs Workflows (infrastructure)
- **Foundation Services**: Comprehensive managed services for platform capabilities
- **Enterprise Scale**: PostgreSQL v15, distributed storage, multi-level caching
- **Security**: OAuth2, SSL, encryption, user management
- **Developer Experience**: Clear patterns and reusable services

The result is a robust, enterprise-ready platform that can scale to support the entire TSAI ecosystem while maintaining clear separation of concerns and excellent developer experience.
