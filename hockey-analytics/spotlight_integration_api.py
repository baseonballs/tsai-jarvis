#!/usr/bin/env python3
"""
TSAI Jarvis - Spotlight Integration API
Video processing and event detection for hockey analytics
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
    title="TSAI Jarvis Spotlight Integration API",
    description="Video processing and event detection for hockey analytics",
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
class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class EventType(str, Enum):
    GOAL = "goal"
    PENALTY = "penalty"
    SAVE = "save"
    HIT = "hit"
    FACEOFF = "faceoff"
    SHOT = "shot"
    PASS = "pass"
    TURNOVER = "turnover"
    ICING = "icing"
    OFFSIDE = "offside"

class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class HighlightType(str, Enum):
    GOAL_HIGHLIGHT = "goal_highlight"
    SAVE_HIGHLIGHT = "save_highlight"
    HIT_HIGHLIGHT = "hit_highlight"
    SKILL_HIGHLIGHT = "skill_highlight"
    TEAM_HIGHLIGHT = "team_highlight"

# --- Pydantic Models ---
class VideoStream(BaseModel):
    stream_id: str
    source_url: str
    stream_type: str  # "live", "recorded", "replay"
    quality: str  # "720p", "1080p", "4k"
    status: VideoStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

class VideoProcessingJob(BaseModel):
    job_id: str
    stream_id: str
    processing_type: str  # "event_detection", "highlight_generation", "analytics"
    status: ProcessingStatus
    progress: float = 0.0
    parameters: Dict[str, Any] = {}
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class HockeyEvent(BaseModel):
    event_id: str
    stream_id: str
    event_type: EventType
    timestamp: datetime
    player_name: Optional[str] = None
    team: Optional[str] = None
    position: Optional[Dict[str, float]] = None  # x, y coordinates
    confidence: float
    description: str
    video_timestamp: float  # seconds from start of video
    metadata: Dict[str, Any] = {}

class Highlight(BaseModel):
    highlight_id: str
    stream_id: str
    highlight_type: HighlightType
    title: str
    description: str
    start_time: float  # seconds from start of video
    end_time: float
    duration: float
    quality_score: float
    events: List[str] = []  # event_ids
    created_at: datetime

class VideoAnalytics(BaseModel):
    analytics_id: str
    stream_id: str
    total_events: int
    events_by_type: Dict[str, int]
    player_activity: Dict[str, int]
    team_activity: Dict[str, int]
    highlight_count: int
    processing_time: float
    created_at: datetime

class RealTimeEvent(BaseModel):
    event_id: str
    event_type: EventType
    timestamp: datetime
    player_name: Optional[str] = None
    team: Optional[str] = None
    confidence: float
    description: str
    position: Optional[Dict[str, float]] = None

# --- Database Setup ---
def init_database():
    """Initialize the Spotlight database"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_streams (
            stream_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            stream_type TEXT NOT NULL,
            quality TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            ended_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_processing_jobs (
            job_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            processing_type TEXT NOT NULL,
            status TEXT DEFAULT 'queued',
            progress REAL DEFAULT 0.0,
            parameters TEXT DEFAULT '{}',
            results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hockey_events (
            event_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            player_name TEXT,
            team TEXT,
            position TEXT,
            confidence REAL NOT NULL,
            description TEXT NOT NULL,
            video_timestamp REAL NOT NULL,
            metadata TEXT DEFAULT '{}'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS highlights (
            highlight_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            highlight_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            duration REAL NOT NULL,
            quality_score REAL NOT NULL,
            events TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_analytics (
            analytics_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            total_events INTEGER NOT NULL,
            events_by_type TEXT NOT NULL,
            player_activity TEXT NOT NULL,
            team_activity TEXT NOT NULL,
            highlight_count INTEGER NOT NULL,
            processing_time REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Spotlight database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.live_stream_connections: List[WebSocket] = []
        self.analytics_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "live_stream":
            self.live_stream_connections.append(websocket)
        elif connection_type == "analytics":
            self.analytics_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {connection_type}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.live_stream_connections:
            self.live_stream_connections.remove(websocket)
        if websocket in self.analytics_connections:
            self.analytics_connections.remove(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_live_streams(self, message: str):
        for connection in self.live_stream_connections:
            await connection.send_text(message)

    async def broadcast_to_analytics(self, message: str):
        for connection in self.analytics_connections:
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Spotlight Integration API"""
    return {"status": "ok", "message": "TSAI Jarvis Spotlight Integration API is running"}

@app.get("/api/spotlight/status", summary="Get Spotlight Status")
async def get_spotlight_status():
    """Get the current status of the Spotlight integration"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    # Get active streams
    cursor.execute("SELECT COUNT(*) FROM video_streams WHERE status IN ('pending', 'processing')")
    active_streams = cursor.fetchone()[0]
    
    # Get processing jobs
    cursor.execute("SELECT COUNT(*) FROM video_processing_jobs WHERE status IN ('queued', 'processing')")
    active_jobs = cursor.fetchone()[0]
    
    # Get recent events
    cursor.execute("SELECT COUNT(*) FROM hockey_events WHERE timestamp > datetime('now', '-1 hour')")
    recent_events = cursor.fetchone()[0]
    
    # Get highlights
    cursor.execute("SELECT COUNT(*) FROM highlights WHERE created_at > datetime('now', '-1 hour')")
    recent_highlights = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "active_streams": active_streams,
        "active_jobs": active_jobs,
        "recent_events": recent_events,
        "recent_highlights": recent_highlights,
        "live_stream_connections": len(manager.live_stream_connections),
        "analytics_connections": len(manager.analytics_connections),
        "total_connections": len(manager.active_connections)
    }

@app.post("/api/spotlight/streams", summary="Create Video Stream")
async def create_video_stream(stream: VideoStream):
    """Create a new video stream for processing"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO video_streams (stream_id, source_url, stream_type, quality, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            stream.stream_id,
            stream.source_url,
            stream.stream_type,
            stream.quality,
            stream.status,
            stream.created_at
        ))
        
        conn.commit()
        logger.info(f"Created video stream: {stream.stream_id}")
        
        # Notify live stream connections about new stream
        await manager.broadcast_to_live_streams(json.dumps({
            "type": "stream_created",
            "stream_id": stream.stream_id,
            "stream_type": stream.stream_type,
            "quality": stream.quality,
            "status": stream.status
        }))
        
        return {"message": "Video stream created successfully", "stream_id": stream.stream_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create video stream: {str(e)}")
    finally:
        conn.close()

@app.get("/api/spotlight/streams", summary="Get All Video Streams")
async def get_video_streams():
    """Get all video streams"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT stream_id, source_url, stream_type, quality, status, created_at, started_at, ended_at
        FROM video_streams
        ORDER BY created_at DESC
    ''')
    
    streams = []
    for row in cursor.fetchall():
        streams.append({
            "stream_id": row[0],
            "source_url": row[1],
            "stream_type": row[2],
            "quality": row[3],
            "status": row[4],
            "created_at": row[5],
            "started_at": row[6],
            "ended_at": row[7]
        })
    
    conn.close()
    return {"streams": streams}

@app.post("/api/spotlight/processing-jobs", summary="Create Processing Job")
async def create_processing_job(job: VideoProcessingJob):
    """Create a new video processing job"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO video_processing_jobs (job_id, stream_id, processing_type, status, 
                                            progress, parameters, results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id,
            job.stream_id,
            job.processing_type,
            job.status,
            job.progress,
            json.dumps(job.parameters),
            json.dumps(job.results) if job.results else None,
            job.created_at
        ))
        
        conn.commit()
        logger.info(f"Created processing job: {job.job_id}")
        
        # Notify analytics connections about new job
        await manager.broadcast_to_analytics(json.dumps({
            "type": "processing_job_created",
            "job_id": job.job_id,
            "stream_id": job.stream_id,
            "processing_type": job.processing_type,
            "status": job.status
        }))
        
        return {"message": "Processing job created successfully", "job_id": job.job_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create processing job: {str(e)}")
    finally:
        conn.close()

@app.post("/api/spotlight/events", summary="Record Hockey Event")
async def record_hockey_event(event: HockeyEvent):
    """Record a hockey event detected in video"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO hockey_events (event_id, stream_id, event_type, timestamp, player_name, 
                                     team, position, confidence, description, video_timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.stream_id,
            event.event_type,
            event.timestamp,
            event.player_name,
            event.team,
            json.dumps(event.position) if event.position else None,
            event.confidence,
            event.description,
            event.video_timestamp,
            json.dumps(event.metadata)
        ))
        
        conn.commit()
        logger.info(f"Recorded hockey event: {event.event_id}")
        
        # Notify all connections about new event
        await manager.broadcast(json.dumps({
            "type": "hockey_event",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "player_name": event.player_name,
            "team": event.team,
            "confidence": event.confidence,
            "timestamp": event.timestamp.isoformat()
        }))
        
        return {"message": "Hockey event recorded successfully", "event_id": event.event_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record event: {str(e)}")
    finally:
        conn.close()

@app.get("/api/spotlight/events", summary="Get Hockey Events")
async def get_hockey_events(stream_id: Optional[str] = None, event_type: Optional[str] = None):
    """Get hockey events with optional filtering"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    query = "SELECT event_id, stream_id, event_type, timestamp, player_name, team, confidence, description, video_timestamp FROM hockey_events"
    params = []
    
    if stream_id or event_type:
        conditions = []
        if stream_id:
            conditions.append("stream_id = ?")
            params.append(stream_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC LIMIT 100"
    
    cursor.execute(query, params)
    
    events = []
    for row in cursor.fetchall():
        events.append({
            "event_id": row[0],
            "stream_id": row[1],
            "event_type": row[2],
            "timestamp": row[3],
            "player_name": row[4],
            "team": row[5],
            "confidence": row[6],
            "description": row[7],
            "video_timestamp": row[8]
        })
    
    conn.close()
    return {"events": events}

@app.post("/api/spotlight/highlights", summary="Create Highlight")
async def create_highlight(highlight: Highlight):
    """Create a new highlight"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO highlights (highlight_id, stream_id, highlight_type, title, description, 
                                 start_time, end_time, duration, quality_score, events, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            highlight.highlight_id,
            highlight.stream_id,
            highlight.highlight_type,
            highlight.title,
            highlight.description,
            highlight.start_time,
            highlight.end_time,
            highlight.duration,
            highlight.quality_score,
            json.dumps(highlight.events),
            highlight.created_at
        ))
        
        conn.commit()
        logger.info(f"Created highlight: {highlight.highlight_id}")
        
        # Notify all connections about new highlight
        await manager.broadcast(json.dumps({
            "type": "highlight_created",
            "highlight_id": highlight.highlight_id,
            "highlight_type": highlight.highlight_type,
            "title": highlight.title,
            "quality_score": highlight.quality_score,
            "duration": highlight.duration
        }))
        
        return {"message": "Highlight created successfully", "highlight_id": highlight.highlight_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create highlight: {str(e)}")
    finally:
        conn.close()

@app.get("/api/spotlight/highlights", summary="Get Highlights")
async def get_highlights(stream_id: Optional[str] = None, highlight_type: Optional[str] = None):
    """Get highlights with optional filtering"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    query = "SELECT highlight_id, stream_id, highlight_type, title, description, start_time, end_time, duration, quality_score, created_at FROM highlights"
    params = []
    
    if stream_id or highlight_type:
        conditions = []
        if stream_id:
            conditions.append("stream_id = ?")
            params.append(stream_id)
        if highlight_type:
            conditions.append("highlight_type = ?")
            params.append(highlight_type)
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY quality_score DESC, created_at DESC LIMIT 50"
    
    cursor.execute(query, params)
    
    highlights = []
    for row in cursor.fetchall():
        highlights.append({
            "highlight_id": row[0],
            "stream_id": row[1],
            "highlight_type": row[2],
            "title": row[3],
            "description": row[4],
            "start_time": row[5],
            "end_time": row[6],
            "duration": row[7],
            "quality_score": row[8],
            "created_at": row[9]
        })
    
    conn.close()
    return {"highlights": highlights}

@app.post("/api/spotlight/analytics", summary="Record Video Analytics")
async def record_video_analytics(analytics: VideoAnalytics):
    """Record video analytics data"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO video_analytics (analytics_id, stream_id, total_events, events_by_type, 
                                       player_activity, team_activity, highlight_count, processing_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analytics.analytics_id,
            analytics.stream_id,
            analytics.total_events,
            json.dumps(analytics.events_by_type),
            json.dumps(analytics.player_activity),
            json.dumps(analytics.team_activity),
            analytics.highlight_count,
            analytics.processing_time,
            analytics.created_at
        ))
        
        conn.commit()
        logger.info(f"Recorded video analytics: {analytics.analytics_id}")
        
        # Notify analytics connections about new analytics
        await manager.broadcast_to_analytics(json.dumps({
            "type": "analytics_update",
            "analytics_id": analytics.analytics_id,
            "stream_id": analytics.stream_id,
            "total_events": analytics.total_events,
            "highlight_count": analytics.highlight_count
        }))
        
        return {"message": "Video analytics recorded successfully", "analytics_id": analytics.analytics_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record analytics: {str(e)}")
    finally:
        conn.close()

@app.get("/api/spotlight/analytics", summary="Get Video Analytics")
async def get_video_analytics(stream_id: Optional[str] = None):
    """Get video analytics data"""
    conn = sqlite3.connect('spotlight_integration.db')
    cursor = conn.cursor()
    
    if stream_id:
        cursor.execute('''
            SELECT analytics_id, stream_id, total_events, events_by_type, player_activity, 
                   team_activity, highlight_count, processing_time, created_at
            FROM video_analytics
            WHERE stream_id = ?
            ORDER BY created_at DESC
        ''', (stream_id,))
    else:
        cursor.execute('''
            SELECT analytics_id, stream_id, total_events, events_by_type, player_activity, 
                   team_activity, highlight_count, processing_time, created_at
            FROM video_analytics
            ORDER BY created_at DESC
            LIMIT 20
        ''')
    
    analytics_data = []
    for row in cursor.fetchall():
        analytics_data.append({
            "analytics_id": row[0],
            "stream_id": row[1],
            "total_events": row[2],
            "events_by_type": json.loads(row[3]),
            "player_activity": json.loads(row[4]),
            "team_activity": json.loads(row[5]),
            "highlight_count": row[6],
            "processing_time": row[7],
            "created_at": row[8]
        })
    
    conn.close()
    return {"analytics": analytics_data}

# --- WebSocket Endpoints ---
@app.websocket("/ws/spotlight")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general Spotlight communication"""
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

@app.websocket("/ws/spotlight/live-stream")
async def live_stream_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live stream processing"""
    await manager.connect(websocket, "live_stream")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from live stream clients
            message = json.loads(data)
            logger.info(f"Received live stream message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Live stream WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/spotlight/analytics")
async def analytics_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for analytics processing"""
    await manager.connect(websocket, "analytics")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from analytics clients
            message = json.loads(data)
            logger.info(f"Received analytics message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Analytics WebSocket error: {e}")
        manager.disconnect(websocket)

# --- Background Tasks ---
async def simulate_live_events():
    """Simulate live hockey events for demonstration"""
    while True:
        try:
            if manager.live_stream_connections:
                # Simulate random hockey events
                event_types = ["goal", "save", "hit", "shot", "pass"]
                players = ["Auston Matthews", "Mitch Marner", "Cole Caufield", "Nick Suzuki", "Connor McDavid"]
                teams = ["Toronto Maple Leafs", "Montreal Canadiens", "Edmonton Oilers"]
                
                import random
                if random.random() < 0.3:  # 30% chance of event every 5 seconds
                    event = RealTimeEvent(
                        event_id=f"evt_{int(time.time())}",
                        event_type=random.choice(event_types),
                        timestamp=datetime.now(),
                        player_name=random.choice(players),
                        team=random.choice(teams),
                        confidence=random.uniform(0.8, 0.99),
                        description=f"{random.choice(event_types).title()} by {random.choice(players)}"
                    )
                    
                    await manager.broadcast_to_live_streams(json.dumps({
                        "type": "live_event",
                        "event": event.dict()
                    }))
            
        except Exception as e:
            logger.error(f"Error simulating live events: {e}")
        
        await asyncio.sleep(5)  # Check every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Spotlight Integration API background tasks...")
    asyncio.create_task(simulate_live_events())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8015)
