#!/usr/bin/env python3
"""
TSAI Jarvis - Autopilot Integration API
Autonomous ML lifecycle and model management for hockey analytics
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
    title="TSAI Jarvis Autopilot Integration API",
    description="Autonomous ML lifecycle and model management for hockey analytics",
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
class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelStatus(str, Enum):
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    SERVING = "serving"
    RETIRED = "retired"
    FAILED = "failed"

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

class ModelType(str, Enum):
    PLAYER_PERFORMANCE = "player_performance"
    TEAM_STRATEGY = "team_strategy"
    GAME_OUTCOME = "game_outcome"
    INJURY_RISK = "injury_risk"
    EVENT_DETECTION = "event_detection"
    ANOMALY_DETECTION = "anomaly_detection"

# --- Pydantic Models ---
class MLExperiment(BaseModel):
    experiment_id: str
    name: str
    description: str
    model_type: ModelType
    status: ExperimentStatus
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float] = {}
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "autopilot"

class ModelVersion(BaseModel):
    version_id: str
    model_id: str
    experiment_id: str
    model_type: ModelType
    status: ModelStatus
    version: str
    metrics: Dict[str, float] = {}
    model_path: str
    created_at: datetime
    deployed_at: Optional[datetime] = None
    performance_score: float = 0.0

class ModelDeployment(BaseModel):
    deployment_id: str
    model_version_id: str
    deployment_name: str
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    replicas: int = 1
    created_at: datetime
    deployed_at: Optional[datetime] = None
    health_check_url: Optional[str] = None

class HyperparameterOptimization(BaseModel):
    optimization_id: str
    experiment_id: str
    parameter_space: Dict[str, Any]
    optimization_algorithm: str = "random_search"
    max_trials: int = 100
    current_trial: int = 0
    best_score: float = 0.0
    best_parameters: Dict[str, Any] = {}
    created_at: datetime
    completed_at: Optional[datetime] = None

class ModelPerformance(BaseModel):
    model_id: str
    version_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    timestamp: datetime

class TrainingJob(BaseModel):
    job_id: str
    experiment_id: str
    model_type: ModelType
    dataset_path: str
    hyperparameters: Dict[str, Any]
    status: str = "pending"
    progress: float = 0.0
    estimated_completion: Optional[datetime] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# --- Database Setup ---
def init_database():
    """Initialize the Autopilot database"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_experiments (
            experiment_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            model_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            hyperparameters TEXT NOT NULL,
            metrics TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_by TEXT DEFAULT 'autopilot'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            status TEXT DEFAULT 'training',
            version TEXT NOT NULL,
            metrics TEXT DEFAULT '{}',
            model_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed_at TIMESTAMP,
            performance_score REAL DEFAULT 0.0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_deployments (
            deployment_id TEXT PRIMARY KEY,
            model_version_id TEXT NOT NULL,
            deployment_name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            endpoint_url TEXT,
            replicas INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed_at TIMESTAMP,
            health_check_url TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hyperparameter_optimizations (
            optimization_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            parameter_space TEXT NOT NULL,
            optimization_algorithm TEXT DEFAULT 'random_search',
            max_trials INTEGER DEFAULT 100,
            current_trial INTEGER DEFAULT 0,
            best_score REAL DEFAULT 0.0,
            best_parameters TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            inference_time REAL NOT NULL,
            throughput REAL NOT NULL,
            memory_usage REAL NOT NULL,
            cpu_usage REAL NOT NULL,
            gpu_usage REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_jobs (
            job_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            progress REAL DEFAULT 0.0,
            estimated_completion TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Autopilot database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitoring_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "monitoring":
            self.monitoring_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {connection_type}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.monitoring_connections:
            self.monitoring_connections.remove(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_monitors(self, message: str):
        for connection in self.monitoring_connections:
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Autopilot Integration API"""
    return {"status": "ok", "message": "TSAI Jarvis Autopilot Integration API is running"}

@app.get("/api/autopilot/status", summary="Get Autopilot Status")
async def get_autopilot_status():
    """Get the current status of the Autopilot integration"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    # Get running experiments
    cursor.execute("SELECT COUNT(*) FROM ml_experiments WHERE status = 'running'")
    running_experiments = cursor.fetchone()[0]
    
    # Get deployed models
    cursor.execute("SELECT COUNT(*) FROM model_deployments WHERE status = 'deployed'")
    deployed_models = cursor.fetchone()[0]
    
    # Get training jobs
    cursor.execute("SELECT COUNT(*) FROM training_jobs WHERE status IN ('pending', 'running')")
    active_jobs = cursor.fetchone()[0]
    
    # Get optimization jobs
    cursor.execute("SELECT COUNT(*) FROM hyperparameter_optimizations WHERE completed_at IS NULL")
    optimization_jobs = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "running_experiments": running_experiments,
        "deployed_models": deployed_models,
        "active_jobs": active_jobs,
        "optimization_jobs": optimization_jobs,
        "monitoring_connections": len(manager.monitoring_connections),
        "total_connections": len(manager.active_connections)
    }

@app.post("/api/autopilot/experiments", summary="Create ML Experiment")
async def create_experiment(experiment: MLExperiment):
    """Create a new ML experiment"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO ml_experiments (experiment_id, name, description, model_type, status, 
                                      hyperparameters, metrics, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment.experiment_id,
            experiment.name,
            experiment.description,
            experiment.model_type,
            experiment.status,
            json.dumps(experiment.hyperparameters),
            json.dumps(experiment.metrics),
            experiment.created_at,
            experiment.created_by
        ))
        
        conn.commit()
        logger.info(f"Created ML experiment: {experiment.experiment_id}")
        
        # Notify monitoring connections about new experiment
        await manager.broadcast_to_monitors(json.dumps({
            "type": "experiment_created",
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "model_type": experiment.model_type,
            "status": experiment.status
        }))
        
        return {"message": "ML experiment created successfully", "experiment_id": experiment.experiment_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")
    finally:
        conn.close()

@app.get("/api/autopilot/experiments", summary="Get All Experiments")
async def get_experiments():
    """Get all ML experiments"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT experiment_id, name, description, model_type, status, hyperparameters, 
               metrics, created_at, started_at, completed_at, created_by
        FROM ml_experiments
        ORDER BY created_at DESC
    ''')
    
    experiments = []
    for row in cursor.fetchall():
        experiments.append({
            "experiment_id": row[0],
            "name": row[1],
            "description": row[2],
            "model_type": row[3],
            "status": row[4],
            "hyperparameters": json.loads(row[5]),
            "metrics": json.loads(row[6]),
            "created_at": row[7],
            "started_at": row[8],
            "completed_at": row[9],
            "created_by": row[10]
        })
    
    conn.close()
    return {"experiments": experiments}

@app.post("/api/autopilot/experiments/{experiment_id}/start", summary="Start Experiment")
async def start_experiment(experiment_id: str):
    """Start a ML experiment"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        # Update experiment status
        cursor.execute('''
            UPDATE ml_experiments 
            SET status = 'running', started_at = ? 
            WHERE experiment_id = ?
        ''', (datetime.now().isoformat(), experiment_id))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        conn.commit()
        logger.info(f"Started experiment: {experiment_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "experiment_started",
            "experiment_id": experiment_id,
            "started_at": datetime.now().isoformat()
        }))
        
        return {"message": f"Experiment {experiment_id} started successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")
    finally:
        conn.close()

@app.post("/api/autopilot/models", summary="Create Model Version")
async def create_model_version(model: ModelVersion):
    """Create a new model version"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO model_versions (version_id, model_id, experiment_id, model_type, status, 
                                      version, metrics, model_path, created_at, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model.version_id,
            model.model_id,
            model.experiment_id,
            model.model_type,
            model.status,
            model.version,
            json.dumps(model.metrics),
            model.model_path,
            model.created_at,
            model.performance_score
        ))
        
        conn.commit()
        logger.info(f"Created model version: {model.version_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "model_created",
            "version_id": model.version_id,
            "model_id": model.model_id,
            "model_type": model.model_type,
            "performance_score": model.performance_score
        }))
        
        return {"message": "Model version created successfully", "version_id": model.version_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create model version: {str(e)}")
    finally:
        conn.close()

@app.post("/api/autopilot/deployments", summary="Deploy Model")
async def deploy_model(deployment: ModelDeployment):
    """Deploy a model version"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO model_deployments (deployment_id, model_version_id, deployment_name, 
                                        status, endpoint_url, replicas, created_at, health_check_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment.deployment_id,
            deployment.model_version_id,
            deployment.deployment_name,
            deployment.status,
            deployment.endpoint_url,
            deployment.replicas,
            deployment.created_at,
            deployment.health_check_url
        ))
        
        conn.commit()
        logger.info(f"Created model deployment: {deployment.deployment_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "model_deployment",
            "deployment_id": deployment.deployment_id,
            "model_version_id": deployment.model_version_id,
            "deployment_name": deployment.deployment_name,
            "status": deployment.status
        }))
        
        return {"message": "Model deployment created successfully", "deployment_id": deployment.deployment_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {str(e)}")
    finally:
        conn.close()

@app.post("/api/autopilot/hyperparameter-optimization", summary="Start Hyperparameter Optimization")
async def start_hyperparameter_optimization(optimization: HyperparameterOptimization):
    """Start hyperparameter optimization for an experiment"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO hyperparameter_optimizations (optimization_id, experiment_id, parameter_space, 
                                                   optimization_algorithm, max_trials, current_trial, 
                                                   best_score, best_parameters, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            optimization.optimization_id,
            optimization.experiment_id,
            json.dumps(optimization.parameter_space),
            optimization.optimization_algorithm,
            optimization.max_trials,
            optimization.current_trial,
            optimization.best_score,
            json.dumps(optimization.best_parameters),
            optimization.created_at
        ))
        
        conn.commit()
        logger.info(f"Started hyperparameter optimization: {optimization.optimization_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "optimization_started",
            "optimization_id": optimization.optimization_id,
            "experiment_id": optimization.experiment_id,
            "max_trials": optimization.max_trials
        }))
        
        return {"message": "Hyperparameter optimization started successfully", "optimization_id": optimization.optimization_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")
    finally:
        conn.close()

@app.post("/api/autopilot/training-jobs", summary="Create Training Job")
async def create_training_job(job: TrainingJob):
    """Create a new training job"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO training_jobs (job_id, experiment_id, model_type, dataset_path, 
                                     hyperparameters, status, progress, estimated_completion, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id,
            job.experiment_id,
            job.model_type,
            job.dataset_path,
            json.dumps(job.hyperparameters),
            job.status,
            job.progress,
            job.estimated_completion,
            job.created_at
        ))
        
        conn.commit()
        logger.info(f"Created training job: {job.job_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "training_job_created",
            "job_id": job.job_id,
            "experiment_id": job.experiment_id,
            "model_type": job.model_type,
            "status": job.status
        }))
        
        return {"message": "Training job created successfully", "job_id": job.job_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")
    finally:
        conn.close()

@app.get("/api/autopilot/models/performance", summary="Get Model Performance")
async def get_model_performance():
    """Get performance metrics for all deployed models"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT model_id, version_id, accuracy, precision, recall, f1_score, 
               inference_time, throughput, memory_usage, cpu_usage, gpu_usage, timestamp
        FROM model_performance
        ORDER BY timestamp DESC
        LIMIT 100
    ''')
    
    performance_data = []
    for row in cursor.fetchall():
        performance_data.append({
            "model_id": row[0],
            "version_id": row[1],
            "accuracy": row[2],
            "precision": row[3],
            "recall": row[4],
            "f1_score": row[5],
            "inference_time": row[6],
            "throughput": row[7],
            "memory_usage": row[8],
            "cpu_usage": row[9],
            "gpu_usage": row[10],
            "timestamp": row[11]
        })
    
    conn.close()
    return {"performance_data": performance_data}

@app.post("/api/autopilot/models/performance", summary="Record Model Performance")
async def record_model_performance(performance: ModelPerformance):
    """Record performance metrics for a model"""
    conn = sqlite3.connect('autopilot_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO model_performance (model_id, version_id, accuracy, precision, recall, 
                                         f1_score, inference_time, throughput, memory_usage, 
                                         cpu_usage, gpu_usage, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.model_id,
            performance.version_id,
            performance.accuracy,
            performance.precision,
            performance.recall,
            performance.f1_score,
            performance.inference_time,
            performance.throughput,
            performance.memory_usage,
            performance.cpu_usage,
            performance.gpu_usage,
            performance.timestamp
        ))
        
        conn.commit()
        logger.info(f"Recorded performance for model: {performance.model_id}")
        
        # Notify monitoring connections
        await manager.broadcast_to_monitors(json.dumps({
            "type": "performance_update",
            "model_id": performance.model_id,
            "version_id": performance.version_id,
            "accuracy": performance.accuracy,
            "f1_score": performance.f1_score,
            "inference_time": performance.inference_time
        }))
        
        return {"message": "Performance recorded successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record performance: {str(e)}")
    finally:
        conn.close()

# --- WebSocket Endpoints ---
@app.websocket("/ws/autopilot")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general Autopilot communication"""
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

@app.websocket("/ws/autopilot/monitoring")
async def monitoring_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for ML monitoring"""
    await manager.connect(websocket, "monitoring")
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

# --- Background Tasks ---
async def monitor_experiments():
    """Monitor running experiments and update their status"""
    while True:
        try:
            conn = sqlite3.connect('autopilot_integration.db')
            cursor = conn.cursor()
            
            # Check for experiments that should be completed
            cursor.execute('''
                SELECT experiment_id, started_at FROM ml_experiments 
                WHERE status = 'running' AND started_at IS NOT NULL
            ''')
            
            running_experiments = cursor.fetchall()
            
            for (experiment_id, started_at) in running_experiments:
                # Simulate experiment completion after 5 minutes
                start_time = datetime.fromisoformat(started_at)
                if datetime.now() - start_time > timedelta(minutes=5):
                    # Mark as completed
                    cursor.execute('''
                        UPDATE ml_experiments 
                        SET status = 'completed', completed_at = ? 
                        WHERE experiment_id = ?
                    ''', (datetime.now().isoformat(), experiment_id))
                    
                    logger.info(f"Experiment completed: {experiment_id}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error monitoring experiments: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Autopilot Integration API background tasks...")
    asyncio.create_task(monitor_experiments())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)
