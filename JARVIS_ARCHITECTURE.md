# Jarvis Core Architecture

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Services](#core-services)
3. [Service Capabilities](#service-capabilities)
4. [TSAI Integration](#tsai-integration)
5. [Implementation Strategy](#implementation-strategy)
6. [Deployment Architecture](#deployment-architecture)

---

## Architecture Overview

### **Jarvis as TSAI Core Foundation**
Jarvis serves as the **central nervous system** for the entire TSAI ecosystem, providing shared services that all TSAI components (Toolchain, Spotlight, Autopilot, etc.) can leverage.

```
┌─────────────────────────────────────────────────────────────────┐
│                        JARVIS CORE                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  STORAGE        │  │  CLIENT STORAGE │  │  EXPERIMENT     │  │
│  │  SERVICE        │  │  SERVICE        │  │  SERVICE        │  │
│  │  (MinIO)        │  │  (GDrive/iCloud)│  │  (MLflow)       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  WORKFLOW       │  │  ANALYTICS      │  │  SECURITY       │  │
│  │  SERVICE        │  │  SERVICE        │  │  SERVICE        │  │
│  │  (Temporal)     │  │  (Metrics)      │  │  (Auth/RBAC)    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  NOTIFICATION   │  │  CONFIG         │  │  MONITORING    │  │
│  │  SERVICE        │  │  SERVICE        │  │  SERVICE       │  │
│  │  (Alerts)       │  │  (Settings)     │  │  (Health)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### **TSAI Ecosystem Integration**
```
┌─────────────────────────────────────────────────────────────────┐
│                    TSAI ECOSYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   TOOLCHAIN     │  │    SPOTLIGHT    │  │    AUTOPILOT    │  │
│  │   (Hockey AI)   │  │   (Computer     │  │   (Autonomous   │  │
│  │                 │  │    Vision)      │  │    Systems)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    HOLMES      │  │    WATSON       │  │    SHERLOCK     │  │
│  │   (Analytics)  │  │   (NLP/AI)      │  │   (Security)    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        JARVIS CORE                             │
│                    (Shared Services)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Services

### **1. Storage Service (MinIO Backend)**
**Purpose**: Durable backend storage for artifacts, models, datasets, and logs.

```python
class JarvisStorageService:
    """Jarvis storage service for durable backend storage"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.minio_client = self._create_minio_client()
        self.buckets = self._initialize_buckets()
    
    def store_artifact(self, artifact_path: str, metadata: Dict = None) -> str:
        """Store artifact with metadata"""
        artifact_id = self._generate_artifact_id()
        remote_path = f"artifacts/{artifact_id}/{Path(artifact_path).name}"
        
        # Upload to MinIO
        self.minio_client.fput_object(
            self.buckets["artifacts"], 
            remote_path, 
            artifact_path
        )
        
        # Store metadata
        if metadata:
            self._store_metadata(artifact_id, metadata)
        
        return artifact_id
    
    def retrieve_artifact(self, artifact_id: str, local_path: str) -> bool:
        """Retrieve artifact from storage"""
        try:
            self.minio_client.fget_object(
                self.buckets["artifacts"],
                f"artifacts/{artifact_id}",
                local_path
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            return False
    
    def store_model(self, model_path: str, model_metadata: Dict) -> str:
        """Store trained model with metadata"""
        model_id = self._generate_model_id()
        remote_path = f"models/{model_id}/{Path(model_path).name}"
        
        # Upload model
        self.minio_client.fput_object(
            self.buckets["models"],
            remote_path,
            model_path
        )
        
        # Store metadata
        self._store_model_metadata(model_id, model_metadata)
        
        return model_id
    
    def list_artifacts(self, component: str = None, experiment_id: str = None) -> List[Dict]:
        """List artifacts with optional filtering"""
        objects = self.minio_client.list_objects(
            self.buckets["artifacts"],
            prefix="artifacts/",
            recursive=True
        )
        
        artifacts = []
        for obj in objects:
            artifact_info = {
                "id": obj.object_name.split("/")[1],
                "path": obj.object_name,
                "size": obj.size,
                "modified": obj.last_modified
            }
            artifacts.append(artifact_info)
        
        return artifacts
```

### **2. Client Storage Service (GDrive/iCloud)**
**Purpose**: User-facing storage for personal media, user-generated content, and sharing.

```python
class JarvisClientStorageService:
    """Jarvis client storage service for user-facing storage"""
    
    def __init__(self, config: ClientStorageConfig):
        self.config = config
        self.drives = self._initialize_drives()
    
    def import_user_media(self, drive_name: str, folder_id: str = None, 
                         file_types: List[str] = None) -> List[str]:
        """Import user media from cloud drives"""
        drive = self.drives.get(drive_name)
        if not drive:
            raise ValueError(f"Drive {drive_name} not configured")
        
        # List files from drive
        files = drive.list_files(folder_id, file_types)
        
        # Download files
        downloaded_files = []
        for file_info in files:
            local_path = f"./imports/{file_info['name']}"
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            success = drive.download_file(file_info['id'], local_path)
            if success:
                downloaded_files.append(local_path)
        
        return downloaded_files
    
    def export_results(self, results: List[str], drive_name: str, 
                      folder_id: str = None) -> List[str]:
        """Export results to user's cloud drive"""
        drive = self.drives.get(drive_name)
        if not drive:
            raise ValueError(f"Drive {drive_name} not configured")
        
        # Upload results
        uploaded_files = []
        for result_path in results:
            file_id = drive.upload_file(result_path, folder_id)
            if file_id:
                uploaded_files.append(file_id)
        
        return uploaded_files
    
    def create_shared_folder(self, folder_name: str, drive_name: str) -> str:
        """Create shared folder for results"""
        drive = self.drives.get(drive_name)
        if not drive:
            raise ValueError(f"Drive {drive_name} not configured")
        
        folder_id = drive.create_folder(folder_name)
        return folder_id
```

### **3. Experiment Service (MLflow)**
**Purpose**: Experiment tracking, model registry, and job analysis.

```python
class JarvisExperimentService:
    """Jarvis experiment service for MLflow integration"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.mlflow_client = self._create_mlflow_client()
        self.experiment_id = self._get_or_create_experiment()
    
    def start_experiment(self, experiment_name: str, parameters: Dict, 
                        component: str = None) -> str:
        """Start new experiment"""
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=experiment_name,
            tags={"component": component}
        ):
            # Log parameters
            mlflow.log_params(parameters)
            
            # Log system info
            mlflow.log_params({
                "python_version": sys.version,
                "platform": platform.platform(),
                "component": component
            })
            
            return mlflow.active_run().info.run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to experiment"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model_path: str, model_name: str, 
                  model_metadata: Dict = None):
        """Log model to experiment"""
        mlflow.pytorch.log_model(
            pytorch_model=model_path,
            artifact_path="model",
            registered_model_name=model_name,
            metadata=model_metadata
        )
    
    def get_best_model(self, experiment_name: str, metric: str) -> str:
        """Get best model by metric"""
        runs = self.mlflow_client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.component = '{experiment_name}'",
            order_by=[f"metrics.{metric} DESC"]
        )
        
        if runs:
            return runs[0].info.run_id
        return None
    
    def compare_experiments(self, run_ids: List[str]) -> Dict:
        """Compare multiple experiments"""
        runs = [self.mlflow_client.get_run(run_id) for run_id in run_ids]
        
        comparison = {
            "parameters": self._compare_parameters(runs),
            "metrics": self._compare_metrics(runs),
            "artifacts": self._compare_artifacts(runs)
        }
        
        return comparison
```

### **4. Workflow Service (Temporal)**
**Purpose**: Workflow orchestration, job scheduling, and distributed execution.

```python
class JarvisWorkflowService:
    """Jarvis workflow service for Temporal integration"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.temporal_client = self._create_temporal_client()
    
    def start_workflow(self, workflow_name: str, input_data: Dict, 
                      component: str = None) -> str:
        """Start workflow execution"""
        workflow_options = WorkflowOptions(
            id=f"{component}-{workflow_name}-{int(time.time())}",
            task_queue=self.config.task_queue,
            execution_timeout=timedelta(hours=24)
        )
        
        workflow_handle = self.temporal_client.start_workflow(
            workflow_name,
            input_data,
            id=workflow_options.id,
            task_queue=workflow_options.task_queue,
            execution_timeout=workflow_options.execution_timeout
        )
        
        return workflow_handle.id
    
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get workflow execution status"""
        try:
            workflow_handle = self.temporal_client.get_workflow_handle(workflow_id)
            workflow_info = workflow_handle.describe()
            
            return {
                "id": workflow_id,
                "status": workflow_info.status.name,
                "start_time": workflow_info.start_time,
                "execution_time": workflow_info.execution_time,
                "result": workflow_info.result if workflow_info.status == WorkflowExecutionStatus.COMPLETED else None
            }
        except Exception as e:
            return {
                "id": workflow_id,
                "status": "error",
                "error": str(e)
            }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
        try:
            workflow_handle = self.temporal_client.get_workflow_handle(workflow_id)
            workflow_handle.cancel()
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
```

### **5. Analytics Service (Metrics & Monitoring)**
**Purpose**: System metrics, performance monitoring, and alerting.

```python
class JarvisAnalyticsService:
    """Jarvis analytics service for metrics and monitoring"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.prometheus_client = self._create_prometheus_client()
        self.grafana_client = self._create_grafana_client()
    
    def log_metric(self, metric_name: str, value: float, 
                   labels: Dict[str, str] = None):
        """Log metric to Prometheus"""
        metric = self.prometheus_client.get_metric(metric_name)
        if metric:
            metric.labels(**(labels or {})).set(value)
    
    def log_counter(self, counter_name: str, value: float = 1.0, 
                    labels: Dict[str, str] = None):
        """Log counter metric"""
        counter = self.prometheus_client.get_counter(counter_name)
        if counter:
            counter.labels(**(labels or {})).inc(value)
    
    def log_histogram(self, histogram_name: str, value: float, 
                      labels: Dict[str, str] = None):
        """Log histogram metric"""
        histogram = self.prometheus_client.get_histogram(histogram_name)
        if histogram:
            histogram.labels(**(labels or {})).observe(value)
    
    def create_dashboard(self, dashboard_config: Dict) -> str:
        """Create Grafana dashboard"""
        dashboard = self.grafana_client.create_dashboard(dashboard_config)
        return dashboard["id"]
    
    def send_alert(self, alert_config: Dict):
        """Send alert notification"""
        # Implementation for alerting (Slack, email, etc.)
        pass
```

---

## Service Capabilities

### **Storage Service Capabilities**
- **Artifact Storage**: Models, datasets, logs, reports
- **Versioning**: Model versioning and rollback
- **Backup**: Automated backup and recovery
- **Compression**: Efficient storage with compression
- **Encryption**: Data encryption at rest and in transit

### **Client Storage Service Capabilities**
- **Multi-Provider**: Google Drive, iCloud Drive, OneDrive
- **File Types**: Images, videos, documents, datasets
- **Sharing**: Secure sharing with users
- **Sync**: Bidirectional synchronization
- **Search**: Full-text search across files

### **Experiment Service Capabilities**
- **Tracking**: Parameters, metrics, artifacts
- **Registry**: Model versioning and staging
- **Comparison**: Experiment comparison and analysis
- **Reproducibility**: Environment and dependency tracking
- **Collaboration**: Team experiment sharing

### **Workflow Service Capabilities**
- **Orchestration**: Complex workflow execution
- **Scheduling**: Cron-based and event-driven scheduling
- **Retry**: Automatic retry with exponential backoff
- **Monitoring**: Real-time workflow monitoring
- **Scaling**: Auto-scaling based on workload

### **Analytics Service Capabilities**
- **Metrics**: System and application metrics
- **Dashboards**: Real-time monitoring dashboards
- **Alerting**: Proactive alerting and notifications
- **Reporting**: Automated report generation
- **Trends**: Performance trend analysis

---

## TSAI Integration

### **TSAI Component Base Class**
```python
class TSAIComponent:
    """Base class for all TSAI components"""
    
    def __init__(self, component_name: str, jarvis_core: JarvisCoreServices):
        self.component_name = component_name
        self.jarvis = jarvis_core
        self.logger = logging.getLogger(f"TSAI.{component_name}")
    
    # Storage Operations
    def store_artifact(self, artifact_path: str, metadata: Dict = None) -> str:
        """Store artifact using Jarvis storage service"""
        return self.jarvis.storage_service.store_artifact(
            artifact_path, 
            metadata=metadata,
            component=self.component_name
        )
    
    def retrieve_artifact(self, artifact_id: str, local_path: str) -> bool:
        """Retrieve artifact from Jarvis storage service"""
        return self.jarvis.storage_service.retrieve_artifact(artifact_id, local_path)
    
    # Client Storage Operations
    def import_user_media(self, drive_name: str, folder_id: str = None) -> List[str]:
        """Import user media using Jarvis client storage service"""
        return self.jarvis.client_storage_service.import_user_media(
            drive_name, folder_id, component=self.component_name
        )
    
    def export_results(self, results: List[str], drive_name: str) -> List[str]:
        """Export results using Jarvis client storage service"""
        return self.jarvis.client_storage_service.export_results(
            results, drive_name, component=self.component_name
        )
    
    # Experiment Operations
    def start_experiment(self, experiment_name: str, parameters: Dict) -> str:
        """Start experiment using Jarvis experiment service"""
        return self.jarvis.experiment_service.start_experiment(
            experiment_name, parameters, component=self.component_name
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics using Jarvis experiment service"""
        self.jarvis.experiment_service.log_metrics(metrics, step)
    
    # Workflow Operations
    def start_workflow(self, workflow_name: str, input_data: Dict) -> str:
        """Start workflow using Jarvis workflow service"""
        return self.jarvis.workflow_service.start_workflow(
            workflow_name, input_data, component=self.component_name
        )
    
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get workflow status using Jarvis workflow service"""
        return self.jarvis.workflow_service.get_workflow_status(workflow_id)
    
    # Analytics Operations
    def log_metric(self, metric_name: str, value: float, labels: Dict = None):
        """Log metric using Jarvis analytics service"""
        self.jarvis.analytics_service.log_metric(metric_name, value, labels)
```

### **Component-Specific Implementations**

#### **Toolchain Component**
```python
class ToolchainComponent(TSAIComponent):
    """TSAI Toolchain component for hockey detection"""
    
    def __init__(self, jarvis_core: JarvisCoreServices):
        super().__init__("toolchain", jarvis_core)
    
    def run_hockey_detection_pipeline(self, config: Dict) -> str:
        """Run hockey detection pipeline"""
        # Start experiment
        experiment_id = self.start_experiment(
            "hockey-role-classification", 
            config["parameters"]
        )
        
        # Start workflow
        workflow_id = self.start_workflow("hockey-detection", config)
        
        # Monitor execution
        while True:
            status = self.get_workflow_status(workflow_id)
            if status["status"] == "completed":
                break
            
            # Log metrics
            self.log_metrics(status["metrics"])
            self.log_metric("pipeline_progress", status["progress"])
        
        return workflow_id
```

#### **Spotlight Component**
```python
class SpotlightComponent(TSAIComponent):
    """TSAI Spotlight component for computer vision"""
    
    def __init__(self, jarvis_core: JarvisCoreServices):
        super().__init__("spotlight", jarvis_core)
    
    def analyze_hockey_video(self, video_path: str) -> Dict:
        """Analyze hockey video using Spotlight"""
        # Import from user's cloud drive
        local_video = self.import_user_media("google_drive", video_path)
        
        # Process video
        results = self.process_video(local_video)
        
        # Store results
        artifact_id = self.store_artifact(results["output_path"])
        
        # Export to user's cloud drive
        self.export_results([results["output_path"]], "google_drive")
        
        return results
```

#### **Autopilot Component**
```python
class AutopilotComponent(TSAIComponent):
    """TSAI Autopilot component for autonomous systems"""
    
    def __init__(self, jarvis_core: JarvisCoreServices):
        super().__init__("autopilot", jarvis_core)
    
    def autonomous_hockey_analysis(self, config: Dict) -> str:
        """Autonomous hockey analysis pipeline"""
        # Start autonomous workflow
        workflow_id = self.start_workflow("autonomous-hockey-analysis", config)
        
        # Monitor and adapt
        while True:
            status = self.get_workflow_status(workflow_id)
            if status["status"] == "completed":
                break
            
            # Adaptive behavior
            if status["performance"] < 0.8:
                self.adapt_parameters(status["metrics"])
                self.log_metric("autonomous_adaptation", 1.0)
        
        return workflow_id
```

---

## Implementation Strategy

### **Phase 1: Core Services (Week 1-2)**
1. **Storage Service** - MinIO backend implementation
2. **Client Storage Service** - GDrive/iCloud integration
3. **Experiment Service** - MLflow integration
4. **Workflow Service** - Temporal integration
5. **Analytics Service** - Prometheus/Grafana integration

### **Phase 2: TSAI Integration (Week 3)**
1. **TSAI Component Base Class** - Shared interface
2. **Component-Specific Implementations** - Toolchain, Spotlight, Autopilot
3. **Integration Testing** - End-to-end testing
4. **Documentation** - API documentation and guides

### **Phase 3: Production Deployment (Week 4)**
1. **Docker Compose Stack** - Complete service orchestration
2. **Configuration Management** - Environment-specific configs
3. **Monitoring Setup** - Health checks and alerting
4. **CI/CD Pipeline** - Automated deployment

---

## Deployment Architecture

### **Docker Compose Stack**
```yaml
# jarvis-core/docker-compose.yml
version: '3.8'

services:
  # Core Infrastructure
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=jarvis
    volumes:
      - postgres-data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data

  temporal:
    image: temporalio/auto-setup:1.20.0
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=postgres
      - POSTGRES_PWD=password
    depends_on:
      - postgres

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.2
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://jarvis-artifacts/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000
    depends_on:
      - postgres
      - minio

  # Jarvis Core Services
  storage-service:
    build: ./services/storage
    environment:
      - MINIO_ENDPOINT=http://minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    depends_on:
      - minio

  client-storage-service:
    build: ./services/client-storage
    environment:
      - GOOGLE_CREDENTIALS_FILE=/app/config/google-credentials.json
      - ICLOUD_USERNAME=${ICLOUD_USERNAME}
      - ICLOUD_PASSWORD=${ICLOUD_PASSWORD}
    volumes:
      - ./config:/app/config
    depends_on:
      - storage-service

  experiment-service:
    build: ./services/experiment
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_REGISTRY_URI=postgresql://postgres:password@postgres:5432/mlflow
    depends_on:
      - mlflow

  workflow-service:
    build: ./services/workflow
    environment:
      - TEMPORAL_ENDPOINT=temporal:7233
      - TEMPORAL_NAMESPACE=jarvis-workflows
    depends_on:
      - temporal

  analytics-service:
    build: ./services/analytics
    environment:
      - PROMETHEUS_ENDPOINT=http://prometheus:9090
      - GRAFANA_ENDPOINT=http://grafana:3000
    depends_on:
      - prometheus
      - grafana

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  postgres-data:
  minio-data:
  grafana-data:

networks:
  default:
    name: jarvis-network
```

### **Service Configuration**
```yaml
# jarvis-core/config/jarvis-core.yaml
jarvis_core:
  services:
    storage:
      type: "minio"
      endpoint: "http://minio:9000"
      access_key: "minioadmin"
      secret_key: "minioadmin"
      buckets:
        - "jarvis-artifacts"
        - "jarvis-models"
        - "jarvis-datasets"
        - "jarvis-logs"
    
    client_storage:
      type: "cloud-drive"
      providers:
        google_drive:
          enabled: true
          credentials_file: "/app/config/google-credentials.json"
        icloud_drive:
          enabled: true
          username: "${ICLOUD_USERNAME}"
          password: "${ICLOUD_PASSWORD}"
    
    experiment:
      type: "mlflow"
      tracking_uri: "http://mlflow:5000"
      registry_uri: "postgresql://postgres:password@postgres:5432/mlflow"
      artifacts_uri: "s3://jarvis-artifacts/mlflow"
    
    workflow:
      type: "temporal"
      endpoint: "temporal:7233"
      namespace: "jarvis-workflows"
      task_queue: "jarvis-pipeline-queue"
    
    analytics:
      type: "prometheus"
      endpoint: "http://prometheus:9090"
      dashboard: "http://grafana:3000"
```

---

## Summary

Jarvis Core Architecture provides:

1. **Unified Service Layer** - Shared services across TSAI ecosystem
2. **Clear Separation** - Backend storage vs. client-facing storage
3. **Component Integration** - Consistent interfaces for all TSAI components
4. **Scalable Architecture** - Independent scaling of services
5. **Production Ready** - Complete monitoring, alerting, and deployment

**Jarvis serves as the foundation for the entire TSAI ecosystem! 🏒**
