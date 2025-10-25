"""
TSAI Jarvis - Intelligence API Gateway
The main API gateway for Jarvis intelligence platform
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
import uvicorn

# Import Jarvis components
from core_intelligence.decision_engine.jarvis_brain import JarvisBrain, IntelligenceRequest, IntelligenceResult
from core_intelligence.temporal_orchestration.jarvis_orchestrator import JarvisOrchestrator, OrchestrationRequest, OrchestrationResult
from core_intelligence.pydantic_ai_models.jarvis_models import (
    ServiceType, Priority, WorkflowStatus, PerformanceMetrics, 
    SystemHealth, Decision, Recommendation, JarvisConfig
)

# Import Temporal AI client
from temporalio.client import Client

logger = logging.getLogger(__name__)

class JarvisAPI:
    """Main API gateway for TSAI Jarvis"""
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Intelligence API",
            description="Core intelligence platform for the TSAI ecosystem",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.temporal_client = None
        self.jarvis_brain = None
        self.jarvis_orchestrator = None
        self.config = JarvisConfig()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup API middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health and status endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "tsai-jarvis",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.get("/status")
        async def system_status():
            """System status endpoint"""
            return {
                "jarvis_brain": "active" if self.jarvis_brain else "inactive",
                "orchestrator": "active" if self.jarvis_orchestrator else "inactive",
                "temporal_client": "connected" if self.temporal_client else "disconnected",
                "timestamp": datetime.now().isoformat()
            }
        
        # Intelligence endpoints
        @self.app.post("/api/intelligence/analyze")
        async def analyze_intelligence(request: IntelligenceRequest) -> IntelligenceResult:
            """Analyze request with Jarvis intelligence"""
            try:
                if not self.jarvis_brain:
                    raise HTTPException(status_code=503, detail="Jarvis brain not initialized")
                
                result = await self.jarvis_brain.analyze_request(request)
                return result
            except Exception as e:
                logger.error(f"Intelligence analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/intelligence/decide")
        async def make_decision(request: IntelligenceRequest) -> Decision:
            """Make intelligent decision"""
            try:
                if not self.jarvis_brain:
                    raise HTTPException(status_code=503, detail="Jarvis brain not initialized")
                
                # Analyze the request
                analysis = await self.jarvis_brain.analyze_request(request)
                
                # Create decision based on analysis
                decision = Decision(
                    decision_type="intelligent_decision",
                    context=request.context,
                    options=analysis.recommendations,
                    selected_option=analysis.recommendations[0] if analysis.recommendations else {},
                    confidence=analysis.intelligence_confidence,
                    reasoning=analysis.insights,
                    impact_assessment=analysis.execution_plan
                )
                
                return decision
            except Exception as e:
                logger.error(f"Decision making failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/intelligence/learn")
        async def learn_from_experience(experience: Dict[str, Any]):
            """Learn from experience to improve Jarvis intelligence"""
            try:
                if not self.jarvis_brain:
                    raise HTTPException(status_code=503, detail="Jarvis brain not initialized")
                
                # Process learning experience
                await self.jarvis_brain._learn_from_analysis(
                    IntelligenceRequest(**experience.get("request", {})),
                    IntelligenceResult(**experience.get("result", {}))
                )
                
                return {
                    "status": "success",
                    "message": "Learning experience processed",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Learning failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Workflow orchestration endpoints
        @self.app.post("/api/workflows/start")
        async def start_workflow(request: OrchestrationRequest) -> OrchestrationResult:
            """Start a new workflow"""
            try:
                if not self.jarvis_orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")
                
                result = await self.jarvis_orchestrator.execute_workflow(request)
                return result
            except Exception as e:
                logger.error(f"Workflow start failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/workflows/{workflow_id}/status")
        async def get_workflow_status(workflow_id: str):
            """Get workflow status"""
            try:
                if not self.jarvis_orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")
                
                status = await self.jarvis_orchestrator.get_workflow_status(workflow_id)
                return status
            except Exception as e:
                logger.error(f"Workflow status check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/{workflow_id}/cancel")
        async def cancel_workflow(workflow_id: str):
            """Cancel a running workflow"""
            try:
                if not self.jarvis_orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")
                
                success = await self.jarvis_orchestrator.cancel_workflow(workflow_id)
                return {
                    "workflow_id": workflow_id,
                    "cancelled": success,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Workflow cancellation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Service coordination endpoints
        @self.app.post("/api/services/coordinate")
        async def coordinate_services(request: Dict[str, Any]):
            """Coordinate between TSAI services"""
            try:
                services = request.get("services", [])
                operation = request.get("operation", "coordinate")
                parameters = request.get("parameters", {})
                
                # Create orchestration request
                orchestration_request = OrchestrationRequest(
                    workflow_type=f"{operation}_workflow",
                    services=[ServiceType(service) for service in services],
                    parameters=parameters,
                    priority=Priority(request.get("priority", "medium"))
                )
                
                if not self.jarvis_orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not initialized")
                
                result = await self.jarvis_orchestrator.execute_workflow(orchestration_request)
                return result
            except Exception as e:
                logger.error(f"Service coordination failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/services/status")
        async def get_services_status():
            """Get status of all TSAI services"""
            try:
                # This would check actual service status in production
                services_status = {
                    "autopilot": "healthy",
                    "spotlight": "healthy",
                    "toolchain": "healthy",
                    "watson": "healthy",
                    "holmes": "healthy",
                    "jarvis": "healthy"
                }
                
                return {
                    "services": services_status,
                    "overall_status": "healthy",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Service status check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Analytics and insights endpoints
        @self.app.get("/api/analytics/performance")
        async def get_performance_analytics():
            """Get performance analytics"""
            try:
                # This would gather actual performance metrics in production
                performance_metrics = {
                    "jarvis_brain": {
                        "intelligence_accuracy": 0.95,
                        "decision_confidence": 0.92,
                        "learning_rate": 0.1,
                        "experience_count": 1000
                    },
                    "orchestrator": {
                        "workflow_success_rate": 0.99,
                        "average_execution_time": 45.2,
                        "active_workflows": 5,
                        "completed_workflows": 150
                    },
                    "services": {
                        "autopilot": {"status": "healthy", "response_time": 1.2},
                        "spotlight": {"status": "healthy", "response_time": 0.8},
                        "toolchain": {"status": "healthy", "response_time": 0.5},
                        "watson": {"status": "healthy", "response_time": 0.3},
                        "holmes": {"status": "healthy", "response_time": 0.7}
                    }
                }
                
                return {
                    "metrics": performance_metrics,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Performance analytics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/analytics/insights")
        async def get_system_insights():
            """Get system insights"""
            try:
                insights = [
                    {
                        "type": "performance_insight",
                        "description": "System performance is optimal with 99.9% uptime",
                        "confidence": 0.95,
                        "actionable": True,
                        "impact": "high"
                    },
                    {
                        "type": "optimization_insight",
                        "description": "Resource utilization can be optimized by 15%",
                        "confidence": 0.88,
                        "actionable": True,
                        "impact": "medium"
                    },
                    {
                        "type": "learning_insight",
                        "description": "AI models show 10% improvement over last month",
                        "confidence": 0.92,
                        "actionable": False,
                        "impact": "high"
                    }
                ]
                
                return {
                    "insights": insights,
                    "total_insights": len(insights),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"System insights failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Configuration endpoints
        @self.app.get("/api/config")
        async def get_configuration():
            """Get Jarvis configuration"""
            return self.config.dict()
        
        @self.app.put("/api/config")
        async def update_configuration(config: JarvisConfig):
            """Update Jarvis configuration"""
            try:
                self.config = config
                return {
                    "status": "success",
                    "message": "Configuration updated",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Configuration update failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Error handling
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler"""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(exc),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def initialize(self):
        """Initialize Jarvis components"""
        logger.info("🚀 Initializing TSAI Jarvis API...")
        
        try:
            # Initialize Temporal AI client
            self.temporal_client = await Client.connect("localhost:7233")
            logger.info("✅ Temporal AI client connected")
            
            # Initialize Jarvis brain
            self.jarvis_brain = JarvisBrain(self.temporal_client)
            logger.info("✅ Jarvis brain initialized")
            
            # Initialize orchestrator
            self.jarvis_orchestrator = JarvisOrchestrator(self.temporal_client)
            logger.info("✅ Jarvis orchestrator initialized")
            
            logger.info("🎉 TSAI Jarvis API initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Jarvis initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown Jarvis components"""
        logger.info("🛑 Shutting down TSAI Jarvis API...")
        
        try:
            if self.temporal_client:
                await self.temporal_client.close()
                logger.info("✅ Temporal AI client closed")
            
            logger.info("✅ TSAI Jarvis API shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Jarvis shutdown failed: {e}")

# Create FastAPI app instance
jarvis_api = JarvisAPI()
app = jarvis_api.app

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    await jarvis_api.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    await jarvis_api.shutdown()

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "jarvis_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
