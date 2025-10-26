#!/usr/bin/env python3
"""
TSAI Jarvis - Service Mesh API
Inter-service communication and orchestration for TSAI ecosystem
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TSAI Jarvis Service Mesh API",
    description="Inter-service communication and orchestration for TSAI ecosystem",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Enums ---
class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"

class MessageType(str, Enum):
    SERVICE_DISCOVERY = "service_discovery"
    HEALTH_CHECK = "health_check"
    DATA_SYNC = "data_sync"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    ERROR_PROPAGATION = "error_propagation"
    PERFORMANCE_METRICS = "performance_metrics"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# --- Pydantic Models ---
class ServiceRegistration(BaseModel):
    service_id: str
    service_name: str
    service_type: str
    endpoint: str
    port: int
    status: ServiceStatus
    capabilities: List[str] = []
    dependencies: List[str] = []
    metadata: Dict[str, Any] = {}
    registered_at: datetime
    last_heartbeat: datetime

class ServiceMessage(BaseModel):
    message_id: str
    from_service: str
    to_service: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: int = 300  # Time to live in seconds

class WorkflowStep(BaseModel):
    step_id: str
    workflow_id: str
    service_name: str
    action: str
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []
    status: WorkflowStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class Workflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class CircuitBreaker(BaseModel):
    service_name: str
    failure_count: int = 0
    failure_threshold: int = 5
    timeout: int = 60  # seconds
    state: str = "closed"  # closed, open, half-open
    last_failure: Optional[datetime] = None

class LoadBalancer(BaseModel):
    service_name: str
    strategy: str = "round_robin"  # round_robin, least_connections, weighted
    instances: List[Dict[str, Any]] = []
    current_index: int = 0

# --- Database Setup ---
def init_database():
    """Initialize the Service Mesh database"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS service_registry (
            service_id TEXT PRIMARY KEY,
            service_name TEXT NOT NULL,
            service_type TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            port INTEGER NOT NULL,
            status TEXT NOT NULL,
            capabilities TEXT DEFAULT '[]',
            dependencies TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS service_messages (
            message_id TEXT PRIMARY KEY,
            from_service TEXT NOT NULL,
            to_service TEXT NOT NULL,
            message_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ttl INTEGER DEFAULT 300,
            delivered BOOLEAN DEFAULT FALSE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflow_steps (
            step_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            service_name TEXT NOT NULL,
            action TEXT NOT NULL,
            parameters TEXT DEFAULT '{}',
            dependencies TEXT DEFAULT '[]',
            status TEXT DEFAULT 'pending',
            result TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS circuit_breakers (
            service_name TEXT PRIMARY KEY,
            failure_count INTEGER DEFAULT 0,
            failure_threshold INTEGER DEFAULT 5,
            timeout INTEGER DEFAULT 60,
            state TEXT DEFAULT 'closed',
            last_failure TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS load_balancers (
            service_name TEXT PRIMARY KEY,
            strategy TEXT DEFAULT 'round_robin',
            instances TEXT DEFAULT '[]',
            current_index INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Service Mesh database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.service_connections: Dict[str, WebSocket] = {}
        self.monitoring_connections: List[WebSocket] = []
        self.orchestration_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, service_name: str = "unknown", connection_type: str = "service"):
        await websocket.accept()
        if connection_type == "monitoring":
            self.monitoring_connections.append(websocket)
        elif connection_type == "orchestration":
            self.orchestration_connections.append(websocket)
        else:
            self.service_connections[service_name] = websocket
        logger.info(f"WebSocket connected: {service_name} ({connection_type})")

    def disconnect(self, websocket: WebSocket, service_name: str = None):
        if service_name and service_name in self.service_connections:
            del self.service_connections[service_name]
        if websocket in self.monitoring_connections:
            self.monitoring_connections.remove(websocket)
        if websocket in self.orchestration_connections:
            self.orchestration_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {service_name}")

    async def send_to_service(self, service_name: str, message: str):
        if service_name in self.service_connections:
            await self.service_connections[service_name].send_text(message)

    async def broadcast_to_monitors(self, message: str):
        for connection in self.monitoring_connections:
            await connection.send_text(message)

    async def broadcast_to_orchestrators(self, message: str):
        for connection in self.orchestration_connections:
            await connection.send_text(message)

    async def broadcast_to_all_services(self, message: str):
        for service_name, connection in self.service_connections.items():
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Service Mesh API"""
    return {"status": "ok", "message": "TSAI Jarvis Service Mesh API is running"}

@app.get("/api/service-mesh/status", summary="Get Service Mesh Status")
async def get_service_mesh_status():
    """Get the current status of the service mesh"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    # Get registered services
    cursor.execute("SELECT COUNT(*) FROM service_registry WHERE status = 'healthy'")
    healthy_services = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM service_registry WHERE status = 'unhealthy'")
    unhealthy_services = cursor.fetchone()[0]
    
    # Get active workflows
    cursor.execute("SELECT COUNT(*) FROM workflows WHERE status IN ('pending', 'running')")
    active_workflows = cursor.fetchone()[0]
    
    # Get circuit breakers
    cursor.execute("SELECT COUNT(*) FROM circuit_breakers WHERE state = 'open'")
    open_circuit_breakers = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "healthy_services": healthy_services,
        "unhealthy_services": unhealthy_services,
        "active_workflows": active_workflows,
        "open_circuit_breakers": open_circuit_breakers,
        "connected_services": len(manager.service_connections),
        "monitoring_connections": len(manager.monitoring_connections),
        "orchestration_connections": len(manager.orchestration_connections)
    }

@app.post("/api/service-mesh/register", summary="Register Service")
async def register_service(service: ServiceRegistration):
    """Register a service in the service mesh"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO service_registry (service_id, service_name, service_type, endpoint, 
                                                   port, status, capabilities, dependencies, metadata, 
                                                   registered_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            service.service_id,
            service.service_name,
            service.service_type,
            service.endpoint,
            service.port,
            service.status,
            json.dumps(service.capabilities),
            json.dumps(service.dependencies),
            json.dumps(service.metadata),
            service.registered_at,
            service.last_heartbeat
        ))
        
        conn.commit()
        logger.info(f"Registered service: {service.service_name}")
        
        # Notify monitoring connections about service registration
        await manager.broadcast_to_monitors(json.dumps({
            "type": "service_registered",
            "service_name": service.service_name,
            "service_type": service.service_type,
            "status": service.status,
            "endpoint": service.endpoint
        }))
        
        return {"message": "Service registered successfully", "service_id": service.service_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to register service: {str(e)}")
    finally:
        conn.close()

@app.get("/api/service-mesh/services", summary="Get All Services")
async def get_services():
    """Get all registered services"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT service_id, service_name, service_type, endpoint, port, status, 
               capabilities, dependencies, metadata, registered_at, last_heartbeat
        FROM service_registry
        ORDER BY registered_at DESC
    ''')
    
    services = []
    for row in cursor.fetchall():
        services.append({
            "service_id": row[0],
            "service_name": row[1],
            "service_type": row[2],
            "endpoint": row[3],
            "port": row[4],
            "status": row[5],
            "capabilities": json.loads(row[6]),
            "dependencies": json.loads(row[7]),
            "metadata": json.loads(row[8]),
            "registered_at": row[9],
            "last_heartbeat": row[10]
        })
    
    conn.close()
    return {"services": services}

@app.post("/api/service-mesh/messages", summary="Send Service Message")
async def send_service_message(message: ServiceMessage):
    """Send a message between services"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO service_messages (message_id, from_service, to_service, message_type, 
                                        payload, timestamp, ttl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.message_id,
            message.from_service,
            message.to_service,
            message.message_type,
            json.dumps(message.payload),
            message.timestamp,
            message.ttl
        ))
        
        conn.commit()
        logger.info(f"Sent message: {message.message_id}")
        
        # Try to deliver the message immediately if the target service is connected
        if message.to_service in manager.service_connections:
            await manager.send_to_service(message.to_service, json.dumps({
                "type": "service_message",
                "message_id": message.message_id,
                "from_service": message.from_service,
                "message_type": message.message_type,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }))
            
            # Mark as delivered
            cursor.execute('''
                UPDATE service_messages 
                SET delivered = TRUE 
                WHERE message_id = ?
            ''', (message.message_id,))
            conn.commit()
        
        return {"message": "Service message sent successfully", "message_id": message.message_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
    finally:
        conn.close()

@app.post("/api/service-mesh/workflows", summary="Create Workflow")
async def create_workflow(workflow: Workflow):
    """Create a new workflow"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO workflows (workflow_id, name, description, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            workflow.workflow_id,
            workflow.name,
            workflow.description,
            workflow.status,
            workflow.created_at
        ))
        
        # Insert workflow steps
        for step in workflow.steps:
            cursor.execute('''
                INSERT INTO workflow_steps (step_id, workflow_id, service_name, action, 
                                         parameters, dependencies, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                step.step_id,
                step.workflow_id,
                step.service_name,
                step.action,
                json.dumps(step.parameters),
                json.dumps(step.dependencies),
                step.status,
                step.created_at
            ))
        
        conn.commit()
        logger.info(f"Created workflow: {workflow.workflow_id}")
        
        # Notify orchestration connections about new workflow
        await manager.broadcast_to_orchestrators(json.dumps({
            "type": "workflow_created",
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status,
            "step_count": len(workflow.steps)
        }))
        
        return {"message": "Workflow created successfully", "workflow_id": workflow.workflow_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")
    finally:
        conn.close()

@app.get("/api/service-mesh/workflows", summary="Get All Workflows")
async def get_workflows():
    """Get all workflows"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT w.workflow_id, w.name, w.description, w.status, w.created_at, w.started_at, w.completed_at,
               COUNT(ws.step_id) as step_count
        FROM workflows w
        LEFT JOIN workflow_steps ws ON w.workflow_id = ws.workflow_id
        GROUP BY w.workflow_id
        ORDER BY w.created_at DESC
    ''')
    
    workflows = []
    for row in cursor.fetchall():
        workflows.append({
            "workflow_id": row[0],
            "name": row[1],
            "description": row[2],
            "status": row[3],
            "created_at": row[4],
            "started_at": row[5],
            "completed_at": row[6],
            "step_count": row[7]
        })
    
    conn.close()
    return {"workflows": workflows}

@app.post("/api/service-mesh/circuit-breakers", summary="Update Circuit Breaker")
async def update_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Update circuit breaker state"""
    conn = sqlite3.connect('service_mesh.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO circuit_breakers (service_name, failure_count, failure_threshold, 
                                                   timeout, state, last_failure)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            circuit_breaker.service_name,
            circuit_breaker.failure_count,
            circuit_breaker.failure_threshold,
            circuit_breaker.timeout,
            circuit_breaker.state,
            circuit_breaker.last_failure
        ))
        
        conn.commit()
        logger.info(f"Updated circuit breaker: {circuit_breaker.service_name}")
        
        # Notify monitoring connections about circuit breaker state change
        await manager.broadcast_to_monitors(json.dumps({
            "type": "circuit_breaker_update",
            "service_name": circuit_breaker.service_name,
            "state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count
        }))
        
        return {"message": "Circuit breaker updated successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update circuit breaker: {str(e)}")
    finally:
        conn.close()

# --- WebSocket Endpoints ---
@app.websocket("/ws/service-mesh/{service_name}")
async def service_websocket_endpoint(websocket: WebSocket, service_name: str):
    """WebSocket endpoint for service communication"""
    await manager.connect(websocket, service_name, "service")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from services
            message = json.loads(data)
            logger.info(f"Received message from {service_name}: {message}")
            
            # Process different message types
            if message.get("type") == "heartbeat":
                # Update service heartbeat
                conn = sqlite3.connect('service_mesh.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE service_registry 
                    SET last_heartbeat = ?, status = ? 
                    WHERE service_name = ?
                ''', (datetime.now().isoformat(), "healthy", service_name))
                conn.commit()
                conn.close()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, service_name)
    except Exception as e:
        logger.error(f"Service WebSocket error: {e}")
        manager.disconnect(websocket, service_name)

@app.websocket("/ws/service-mesh/monitoring")
async def monitoring_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for monitoring"""
    await manager.connect(websocket, "monitoring", "monitoring")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from monitoring clients
            message = json.loads(data)
            logger.info(f"Received monitoring message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Monitoring WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/service-mesh/orchestration")
async def orchestration_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for orchestration"""
    await manager.connect(websocket, "orchestration", "orchestration")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from orchestration clients
            message = json.loads(data)
            logger.info(f"Received orchestration message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Orchestration WebSocket error: {e}")
        manager.disconnect(websocket)

# --- Background Tasks ---
async def cleanup_expired_messages():
    """Clean up expired service messages"""
    while True:
        try:
            conn = sqlite3.connect('service_mesh.db')
            cursor = conn.cursor()
            
            # Find expired messages
            cursor.execute('''
                SELECT message_id FROM service_messages 
                WHERE timestamp < datetime('now', '-' || ttl || ' seconds')
            ''')
            
            expired_messages = cursor.fetchall()
            
            for (message_id,) in expired_messages:
                cursor.execute('''
                    DELETE FROM service_messages 
                    WHERE message_id = ?
                ''', (message_id,))
                
                logger.info(f"Cleaned up expired message: {message_id}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up expired messages: {e}")
        
        await asyncio.sleep(60)  # Check every minute

async def health_check_services():
    """Periodically check service health"""
    while True:
        try:
            conn = sqlite3.connect('service_mesh.db')
            cursor = conn.cursor()
            
            # Find services that haven't sent heartbeat in 5 minutes
            cursor.execute('''
                SELECT service_name FROM service_registry 
                WHERE last_heartbeat < datetime('now', '-5 minutes')
                AND status = 'healthy'
            ''')
            
            stale_services = cursor.fetchall()
            
            for (service_name,) in stale_services:
                # Mark as unhealthy
                cursor.execute('''
                    UPDATE service_registry 
                    SET status = 'unhealthy' 
                    WHERE service_name = ?
                ''', (service_name,))
                
                logger.warning(f"Service {service_name} marked as unhealthy due to stale heartbeat")
                
                # Notify monitoring connections
                await manager.broadcast_to_monitors(json.dumps({
                    "type": "service_unhealthy",
                    "service_name": service_name,
                    "reason": "stale_heartbeat"
                }))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Service Mesh API background tasks...")
    asyncio.create_task(cleanup_expired_messages())
    asyncio.create_task(health_check_services())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8018)
