#!/usr/bin/env python3
"""
TSAI Jarvis - Toolchain Integration API
Human-driven hockey analytics and AI detection workflows
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
    title="TSAI Jarvis Toolchain Integration API",
    description="Human-driven hockey analytics and AI detection workflows",
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
class WorkflowStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"

class ApprovalType(str, Enum):
    ANALYTICS_RESULT = "analytics_result"
    MODEL_DEPLOYMENT = "model_deployment"
    PARAMETER_CHANGE = "parameter_change"
    DATA_LABELING = "data_labeling"
    WORKFLOW_EXECUTION = "workflow_execution"

class HumanFeedbackType(str, Enum):
    APPROVAL = "approval"
    REJECTION = "rejection"
    PARAMETER_TUNE = "parameter_tune"
    ANNOTATION = "annotation"
    QUALITY_ASSURANCE = "quality_assurance"

# --- Pydantic Models ---
class HumanApprovalRequest(BaseModel):
    request_id: str
    workflow_id: str
    approval_type: ApprovalType
    title: str
    description: str
    data: Dict[str, Any]
    priority: int = Field(default=1, ge=1, le=5)
    created_at: datetime
    expires_at: Optional[datetime] = None

class HumanFeedback(BaseModel):
    feedback_id: str
    request_id: str
    feedback_type: HumanFeedbackType
    user_id: str
    comment: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    annotations: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

class WorkflowStep(BaseModel):
    step_id: str
    workflow_id: str
    step_name: str
    step_type: str
    status: WorkflowStatus
    requires_human_approval: bool = False
    human_approval_request_id: Optional[str] = None
    parameters: Dict[str, Any] = {}
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class HumanDrivenWorkflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    created_by: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

class ParameterTuningRequest(BaseModel):
    tuning_id: str
    workflow_id: str
    parameter_name: str
    current_value: Any
    suggested_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step_size: Optional[Any] = None
    description: str
    created_at: datetime

class AnnotationTask(BaseModel):
    task_id: str
    workflow_id: str
    data_type: str
    data: Dict[str, Any]
    labels: List[str]
    annotations: List[Dict[str, Any]] = []
    created_at: datetime
    completed_at: Optional[datetime] = None

# --- Database Setup ---
def init_database():
    """Initialize the Toolchain database"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_approval_requests (
            request_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            approval_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            data TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_feedback (
            feedback_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            comment TEXT,
            parameters TEXT,
            annotations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_by TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflow_steps (
            step_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            step_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            requires_human_approval BOOLEAN DEFAULT FALSE,
            human_approval_request_id TEXT,
            parameters TEXT,
            results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameter_tuning_requests (
            tuning_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            parameter_name TEXT NOT NULL,
            current_value TEXT NOT NULL,
            suggested_value TEXT NOT NULL,
            min_value TEXT,
            max_value TEXT,
            step_size TEXT,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotation_tasks (
            task_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data TEXT NOT NULL,
            labels TEXT NOT NULL,
            annotations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Toolchain database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.human_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "human":
            self.human_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {connection_type}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.human_connections:
            self.human_connections.remove(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_humans(self, message: str):
        for connection in self.human_connections:
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Toolchain Integration API"""
    return {"status": "ok", "message": "TSAI Jarvis Toolchain Integration API is running"}

@app.get("/api/toolchain/status", summary="Get Toolchain Status")
async def get_toolchain_status():
    """Get the current status of the Toolchain integration"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    # Get pending approval requests
    cursor.execute("SELECT COUNT(*) FROM human_approval_requests WHERE status = 'pending'")
    pending_approvals = cursor.fetchone()[0]
    
    # Get active workflows
    cursor.execute("SELECT COUNT(*) FROM workflows WHERE status IN ('pending', 'in_progress', 'awaiting_approval')")
    active_workflows = cursor.fetchone()[0]
    
    # Get parameter tuning requests
    cursor.execute("SELECT COUNT(*) FROM parameter_tuning_requests")
    tuning_requests = cursor.fetchone()[0]
    
    # Get annotation tasks
    cursor.execute("SELECT COUNT(*) FROM annotation_tasks WHERE completed_at IS NULL")
    annotation_tasks = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "pending_approvals": pending_approvals,
        "active_workflows": active_workflows,
        "tuning_requests": tuning_requests,
        "annotation_tasks": annotation_tasks,
        "human_connections": len(manager.human_connections),
        "total_connections": len(manager.active_connections)
    }

@app.post("/api/toolchain/workflows", summary="Create Human-Driven Workflow")
async def create_workflow(workflow: HumanDrivenWorkflow):
    """Create a new human-driven workflow"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO workflows (workflow_id, name, description, status, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            workflow.workflow_id,
            workflow.name,
            workflow.description,
            workflow.status,
            workflow.created_by,
            workflow.created_at,
            workflow.updated_at
        ))
        
        # Insert workflow steps
        for step in workflow.steps:
            cursor.execute('''
                INSERT INTO workflow_steps (step_id, workflow_id, step_name, step_type, status, 
                                          requires_human_approval, human_approval_request_id, parameters, results, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                step.step_id,
                step.workflow_id,
                step.step_name,
                step.step_type,
                step.status,
                step.requires_human_approval,
                step.human_approval_request_id,
                json.dumps(step.parameters),
                json.dumps(step.results) if step.results else None,
                step.created_at
            ))
        
        conn.commit()
        logger.info(f"Created workflow: {workflow.workflow_id}")
        
        # Notify human connections about new workflow
        await manager.broadcast_to_humans(json.dumps({
            "type": "workflow_created",
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status
        }))
        
        return {"message": "Workflow created successfully", "workflow_id": workflow.workflow_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")
    finally:
        conn.close()

@app.get("/api/toolchain/workflows", summary="Get All Workflows")
async def get_workflows():
    """Get all human-driven workflows"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT w.workflow_id, w.name, w.description, w.status, w.created_by, w.created_at, w.updated_at, w.completed_at,
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
            "created_by": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "completed_at": row[7],
            "step_count": row[8]
        })
    
    conn.close()
    return {"workflows": workflows}

@app.post("/api/toolchain/approval-requests", summary="Create Approval Request")
async def create_approval_request(request: HumanApprovalRequest):
    """Create a human approval request"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO human_approval_requests (request_id, workflow_id, approval_type, title, description, 
                                               data, priority, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.request_id,
            request.workflow_id,
            request.approval_type,
            request.title,
            request.description,
            json.dumps(request.data),
            request.priority,
            request.created_at,
            request.expires_at
        ))
        
        conn.commit()
        logger.info(f"Created approval request: {request.request_id}")
        
        # Notify human connections about new approval request
        await manager.broadcast_to_humans(json.dumps({
            "type": "approval_request",
            "request_id": request.request_id,
            "workflow_id": request.workflow_id,
            "approval_type": request.approval_type,
            "title": request.title,
            "priority": request.priority
        }))
        
        return {"message": "Approval request created successfully", "request_id": request.request_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create approval request: {str(e)}")
    finally:
        conn.close()

@app.get("/api/toolchain/approval-requests", summary="Get Pending Approval Requests")
async def get_approval_requests():
    """Get all pending approval requests"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT request_id, workflow_id, approval_type, title, description, data, priority, created_at, expires_at
        FROM human_approval_requests
        WHERE status = 'pending'
        ORDER BY priority DESC, created_at ASC
    ''')
    
    requests = []
    for row in cursor.fetchall():
        requests.append({
            "request_id": row[0],
            "workflow_id": row[1],
            "approval_type": row[2],
            "title": row[3],
            "description": row[4],
            "data": json.loads(row[5]),
            "priority": row[6],
            "created_at": row[7],
            "expires_at": row[8]
        })
    
    conn.close()
    return {"approval_requests": requests}

@app.post("/api/toolchain/human-feedback", summary="Submit Human Feedback")
async def submit_human_feedback(feedback: HumanFeedback):
    """Submit human feedback for approval requests or workflows"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO human_feedback (feedback_id, request_id, feedback_type, user_id, comment, parameters, annotations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id,
            feedback.request_id,
            feedback.feedback_type,
            feedback.user_id,
            feedback.comment,
            json.dumps(feedback.parameters) if feedback.parameters else None,
            json.dumps(feedback.annotations) if feedback.annotations else None,
            feedback.created_at
        ))
        
        # Update approval request status if it's an approval/rejection
        if feedback.feedback_type in ["approval", "rejection"]:
            status = "approved" if feedback.feedback_type == "approval" else "rejected"
            cursor.execute('''
                UPDATE human_approval_requests 
                SET status = ? 
                WHERE request_id = ?
            ''', (status, feedback.request_id))
        
        conn.commit()
        logger.info(f"Received human feedback: {feedback.feedback_id}")
        
        # Notify all connections about the feedback
        await manager.broadcast(json.dumps({
            "type": "human_feedback",
            "feedback_id": feedback.feedback_id,
            "request_id": feedback.request_id,
            "feedback_type": feedback.feedback_type,
            "user_id": feedback.user_id
        }))
        
        return {"message": "Human feedback submitted successfully", "feedback_id": feedback.feedback_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
    finally:
        conn.close()

@app.post("/api/toolchain/parameter-tuning", summary="Request Parameter Tuning")
async def request_parameter_tuning(tuning: ParameterTuningRequest):
    """Request human parameter tuning for a workflow"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO parameter_tuning_requests (tuning_id, workflow_id, parameter_name, current_value, 
                                                 suggested_value, min_value, max_value, step_size, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tuning.tuning_id,
            tuning.workflow_id,
            tuning.parameter_name,
            json.dumps(tuning.current_value),
            json.dumps(tuning.suggested_value),
            json.dumps(tuning.min_value) if tuning.min_value else None,
            json.dumps(tuning.max_value) if tuning.max_value else None,
            json.dumps(tuning.step_size) if tuning.step_size else None,
            tuning.description,
            tuning.created_at
        ))
        
        conn.commit()
        logger.info(f"Created parameter tuning request: {tuning.tuning_id}")
        
        # Notify human connections about parameter tuning request
        await manager.broadcast_to_humans(json.dumps({
            "type": "parameter_tuning_request",
            "tuning_id": tuning.tuning_id,
            "workflow_id": tuning.workflow_id,
            "parameter_name": tuning.parameter_name,
            "current_value": tuning.current_value,
            "suggested_value": tuning.suggested_value
        }))
        
        return {"message": "Parameter tuning request created successfully", "tuning_id": tuning.tuning_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create tuning request: {str(e)}")
    finally:
        conn.close()

@app.post("/api/toolchain/annotation-tasks", summary="Create Annotation Task")
async def create_annotation_task(task: AnnotationTask):
    """Create a human annotation task"""
    conn = sqlite3.connect('toolchain_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO annotation_tasks (task_id, workflow_id, data_type, data, labels, annotations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id,
            task.workflow_id,
            task.data_type,
            json.dumps(task.data),
            json.dumps(task.labels),
            json.dumps(task.annotations),
            task.created_at
        ))
        
        conn.commit()
        logger.info(f"Created annotation task: {task.task_id}")
        
        # Notify human connections about annotation task
        await manager.broadcast_to_humans(json.dumps({
            "type": "annotation_task",
            "task_id": task.task_id,
            "workflow_id": task.workflow_id,
            "data_type": task.data_type,
            "labels": task.labels
        }))
        
        return {"message": "Annotation task created successfully", "task_id": task.task_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create annotation task: {str(e)}")
    finally:
        conn.close()

# --- WebSocket Endpoints ---
@app.websocket("/ws/toolchain")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general Toolchain communication"""
    await manager.connect(websocket, "general")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from clients
            message = json.loads(data)
            logger.info(f"Received message: {message}")
            
            # Echo back the message to all connected clients
            await manager.broadcast(json.dumps({
                "type": "echo",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/toolchain/human")
async def human_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for human operators"""
    await manager.connect(websocket, "human")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from human operators
            message = json.loads(data)
            logger.info(f"Received human message: {message}")
            
            # Process human feedback
            if message.get("type") == "human_feedback":
                feedback = HumanFeedback(**message["feedback"])
                # Process the feedback (this would typically update the database)
                logger.info(f"Processing human feedback: {feedback.feedback_id}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Human WebSocket error: {e}")
        manager.disconnect(websocket)

# --- Background Tasks ---
async def cleanup_expired_requests():
    """Clean up expired approval requests"""
    while True:
        try:
            conn = sqlite3.connect('toolchain_integration.db')
            cursor = conn.cursor()
            
            # Find expired requests
            cursor.execute('''
                SELECT request_id FROM human_approval_requests 
                WHERE expires_at IS NOT NULL AND expires_at < ? AND status = 'pending'
            ''', (datetime.now().isoformat(),))
            
            expired_requests = cursor.fetchall()
            
            for (request_id,) in expired_requests:
                # Mark as expired
                cursor.execute('''
                    UPDATE human_approval_requests 
                    SET status = 'expired' 
                    WHERE request_id = ?
                ''', (request_id,))
                
                logger.info(f"Expired approval request: {request_id}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")
        
        await asyncio.sleep(60)  # Check every minute

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Toolchain Integration API background tasks...")
    asyncio.create_task(cleanup_expired_requests())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013)
