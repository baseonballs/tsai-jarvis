"""
TSAI Jarvis - Pydantic AI Models
Structured AI data models for Jarvis intelligence platform
"""
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid

class IntelligenceType(str, Enum):
    """Types of intelligence requests"""
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    DECISION_MAKING = "decision_making"
    PATTERN_RECOGNITION = "pattern_recognition"
    LEARNING = "learning"
    COORDINATION = "coordination"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"

class Priority(str, Enum):
    """Priority levels for requests"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Confidence(str, Enum):
    """Confidence levels for AI decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ServiceType(str, Enum):
    """TSAI service types"""
    AUTOPILOT = "autopilot"
    SPOTLIGHT = "spotlight"
    TOOLCHAIN = "toolchain"
    WATSON = "watson"
    HOLMES = "holmes"
    JARVIS = "jarvis"

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

# Core Intelligence Models
class IntelligenceRequest(BaseModel):
    """Pydantic AI model for intelligence requests to Jarvis"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    intelligence_type: IntelligenceType = Field(description="Type of intelligence required")
    context: Dict[str, Any] = Field(description="Request context and parameters")
    priority: Priority = Field(default=Priority.MEDIUM, description="Request priority level")
    expected_outcome: str = Field(description="Expected outcome description")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Request constraints")
    service_origin: Optional[ServiceType] = Field(None, description="Originating service")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    
    @validator('priority')
    def validate_priority(cls, v):
        if not isinstance(v, Priority):
            raise ValueError('Priority must be a valid Priority enum')
        return v
    
    @validator('intelligence_type')
    def validate_intelligence_type(cls, v):
        if not isinstance(v, IntelligenceType):
            raise ValueError('Intelligence type must be a valid IntelligenceType enum')
        return v

class IntelligenceResult(BaseModel):
    """Pydantic AI model for intelligence results from Jarvis"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique result identifier")
    request_id: str = Field(description="Original request identifier")
    intelligence_confidence: float = Field(ge=0.0, le=1.0, description="AI confidence score")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="AI recommendations")
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="Generated insights")
    next_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended next actions")
    learning_opportunities: List[Dict[str, Any]] = Field(default_factory=list, description="Learning opportunities")
    execution_plan: Dict[str, Any] = Field(default_factory=dict, description="Execution plan for recommendations")
    processing_time: float = Field(gt=0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result_id": "intelligence_001",
                "request_id": "req_123",
                "intelligence_confidence": 0.95,
                "recommendations": [
                    {
                        "type": "model_optimization",
                        "description": "Optimize YOLO model for better hockey detection",
                        "confidence": 0.92,
                        "impact": "high",
                        "estimated_improvement": "15-20%"
                    }
                ],
                "insights": [
                    {
                        "type": "performance_insight",
                        "description": "Model accuracy improved by 15% with new training data",
                        "confidence": 0.88,
                        "actionable": True
                    }
                ],
                "next_actions": [
                    {
                        "action": "retrain_model",
                        "priority": "high",
                        "estimated_impact": "significant",
                        "timeline": "2-4 hours"
                    }
                ],
                "learning_opportunities": [
                    {
                        "opportunity": "data_augmentation",
                        "description": "Apply advanced data augmentation techniques",
                        "potential_improvement": "10-15%",
                        "complexity": "medium"
                    }
                ],
                "execution_plan": {
                    "steps": [
                        "Prepare training data",
                        "Configure model parameters",
                        "Execute training workflow",
                        "Validate model performance",
                        "Deploy if successful"
                    ],
                    "estimated_duration": "4-6 hours",
                    "required_resources": ["GPU", "Storage", "Compute"]
                },
                "processing_time": 2.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

# Workflow Orchestration Models
class OrchestrationRequest(BaseModel):
    """Pydantic AI model for orchestration requests"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    workflow_type: str = Field(description="Type of workflow to execute")
    services: List[ServiceType] = Field(description="Services involved in the workflow")
    parameters: Dict[str, Any] = Field(description="Workflow parameters")
    priority: Priority = Field(default=Priority.MEDIUM, description="Workflow priority")
    timeout: int = Field(default=3600, description="Workflow timeout in seconds")
    dependencies: List[str] = Field(default_factory=list, description="Workflow dependencies")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy configuration")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        if v > 86400:  # 24 hours
            raise ValueError('Timeout cannot exceed 24 hours')
        return v

class OrchestrationResult(BaseModel):
    """Pydantic AI model for orchestration results"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique result identifier")
    request_id: str = Field(description="Original request identifier")
    workflow_id: str = Field(description="Temporal workflow ID")
    status: WorkflowStatus = Field(description="Workflow execution status")
    services_executed: List[ServiceType] = Field(description="Services that were executed")
    execution_time: float = Field(description="Total execution time in seconds")
    results: Dict[str, Any] = Field(default_factory=dict, description="Results from each service")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Any errors encountered")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="AI recommendations")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result_id": "orchestration_001",
                "request_id": "req_123",
                "workflow_id": "workflow_abc123",
                "status": "completed",
                "services_executed": ["autopilot", "spotlight"],
                "execution_time": 45.2,
                "results": {
                    "autopilot": {"model_trained": True, "accuracy": 0.92},
                    "spotlight": {"analytics_generated": True, "insights_count": 15}
                },
                "errors": [],
                "recommendations": [
                    {
                        "type": "optimization",
                        "description": "Model performance can be improved by 10%",
                        "confidence": 0.88
                    }
                ],
                "performance_metrics": {
                    "cpu_usage": 75.5,
                    "memory_usage": 60.2,
                    "gpu_utilization": 85.0
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

# Service Integration Models
class ServiceRequest(BaseModel):
    """Pydantic AI model for service requests"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    service_type: ServiceType = Field(description="Target service type")
    operation: str = Field(description="Operation to perform")
    parameters: Dict[str, Any] = Field(description="Operation parameters")
    priority: Priority = Field(default=Priority.MEDIUM, description="Request priority")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    retry_count: int = Field(default=0, description="Current retry count")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    
    @validator('retry_count')
    def validate_retry_count(cls, v, values):
        max_retries = values.get('max_retries', 3)
        if v > max_retries:
            raise ValueError('Retry count cannot exceed max retries')
        return v

class ServiceResponse(BaseModel):
    """Pydantic AI model for service responses"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique response identifier")
    request_id: str = Field(description="Original request identifier")
    service_type: ServiceType = Field(description="Responding service type")
    status: str = Field(description="Response status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Any errors encountered")
    processing_time: float = Field(description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp_001",
                "request_id": "req_123",
                "service_type": "autopilot",
                "status": "success",
                "data": {
                    "model_trained": True,
                    "accuracy": 0.92,
                    "training_time": 45.2
                },
                "errors": [],
                "processing_time": 2.1,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

# Learning and Knowledge Models
class LearningExperience(BaseModel):
    """Pydantic AI model for learning experiences"""
    experience_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique experience identifier")
    request_id: str = Field(description="Original request identifier")
    result_id: str = Field(description="Result identifier")
    experience_type: str = Field(description="Type of learning experience")
    context: Dict[str, Any] = Field(description="Experience context")
    outcome: Dict[str, Any] = Field(description="Experience outcome")
    success: bool = Field(description="Whether the experience was successful")
    learning_points: List[str] = Field(default_factory=list, description="Key learning points")
    confidence_impact: float = Field(description="Impact on confidence score")
    timestamp: datetime = Field(default_factory=datetime.now, description="Experience timestamp")
    
    @validator('confidence_impact')
    def validate_confidence_impact(cls, v):
        if not -1.0 <= v <= 1.0:
            raise ValueError('Confidence impact must be between -1.0 and 1.0')
        return v

class KnowledgeEntry(BaseModel):
    """Pydantic AI model for knowledge base entries"""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique entry identifier")
    knowledge_type: str = Field(description="Type of knowledge")
    content: Dict[str, Any] = Field(description="Knowledge content")
    confidence: float = Field(ge=0.0, le=1.0, description="Knowledge confidence")
    source: str = Field(description="Knowledge source")
    tags: List[str] = Field(default_factory=list, description="Knowledge tags")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

# Performance and Monitoring Models
class PerformanceMetrics(BaseModel):
    """Pydantic AI model for performance metrics"""
    metrics_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique metrics identifier")
    service_type: ServiceType = Field(description="Service type")
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(ge=0.0, le=100.0, description="Memory usage percentage")
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization percentage")
    response_time: float = Field(gt=0, description="Average response time in seconds")
    throughput: float = Field(gt=0, description="Requests per second")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate")
    availability: float = Field(ge=0.0, le=1.0, description="Service availability")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    @validator('cpu_usage', 'memory_usage')
    def validate_usage_percentage(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('Usage percentage must be between 0.0 and 100.0')
        return v

class SystemHealth(BaseModel):
    """Pydantic AI model for system health status"""
    health_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique health identifier")
    overall_status: str = Field(description="Overall system status")
    services: Dict[ServiceType, str] = Field(description="Individual service status")
    performance_metrics: Dict[ServiceType, PerformanceMetrics] = Field(description="Service performance metrics")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active alerts")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Health recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    @validator('overall_status')
    def validate_overall_status(cls, v):
        valid_statuses = ['healthy', 'degraded', 'critical', 'unknown']
        if v not in valid_statuses:
            raise ValueError(f'Overall status must be one of: {valid_statuses}')
        return v

# Decision and Recommendation Models
class Decision(BaseModel):
    """Pydantic AI model for AI decisions"""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique decision identifier")
    decision_type: str = Field(description="Type of decision")
    context: Dict[str, Any] = Field(description="Decision context")
    options: List[Dict[str, Any]] = Field(description="Available options")
    selected_option: Dict[str, Any] = Field(description="Selected option")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    reasoning: List[str] = Field(description="Decision reasoning")
    impact_assessment: Dict[str, Any] = Field(description="Impact assessment")
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

class Recommendation(BaseModel):
    """Pydantic AI model for AI recommendations"""
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique recommendation identifier")
    recommendation_type: str = Field(description="Type of recommendation")
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Recommendation description")
    confidence: float = Field(ge=0.0, le=1.0, description="Recommendation confidence")
    impact: str = Field(description="Expected impact level")
    effort: str = Field(description="Implementation effort level")
    priority: Priority = Field(description="Recommendation priority")
    implementation_steps: List[str] = Field(description="Implementation steps")
    success_criteria: List[str] = Field(description="Success criteria")
    estimated_improvement: Optional[str] = Field(None, description="Estimated improvement")
    timestamp: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

# Validation and Error Models
class ValidationError(BaseModel):
    """Pydantic AI model for validation errors"""
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique error identifier")
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Error message")
    code: str = Field(description="Error code")
    severity: str = Field(description="Error severity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['low', 'medium', 'high', 'critical']
        if v not in valid_severities:
            raise ValueError(f'Severity must be one of: {valid_severities}')
        return v

class ProcessingError(BaseModel):
    """Pydantic AI model for processing errors"""
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique error identifier")
    request_id: str = Field(description="Request that caused the error")
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    recoverable: bool = Field(description="Whether the error is recoverable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    @validator('recoverable')
    def validate_recoverable(cls, v):
        return bool(v)

# Configuration Models
class JarvisConfig(BaseModel):
    """Pydantic AI model for Jarvis configuration"""
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique config identifier")
    intelligence_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Learning rate")
    max_concurrent_workflows: int = Field(default=10, gt=0, description="Maximum concurrent workflows")
    default_timeout: int = Field(default=3600, gt=0, description="Default timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy configuration")
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    learning_enabled: bool = Field(default=True, description="Enable learning")
    timestamp: datetime = Field(default_factory=datetime.now, description="Configuration timestamp")
    
    @validator('intelligence_confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v

# Utility Models
class TimestampedModel(BaseModel):
    """Base model with timestamp"""
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()

class IdentifiedModel(BaseModel):
    """Base model with unique identifier"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")

class VersionedModel(BaseModel):
    """Base model with versioning"""
    version: str = Field(default="1.0.0", description="Model version")
    version_history: List[str] = Field(default_factory=list, description="Version history")
    
    def increment_version(self):
        """Increment the version number"""
        version_parts = self.version.split('.')
        patch = int(version_parts[2]) + 1
        self.version = f"{version_parts[0]}.{version_parts[1]}.{patch}"
        self.version_history.append(self.version)
