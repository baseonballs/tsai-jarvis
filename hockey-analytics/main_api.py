#!/usr/bin/env python3
"""
TSAI Jarvis - Main Hockey Analytics API
Core API service providing game state, player stats, and live events
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TSAI Jarvis Main Hockey Analytics API",
    description="Core API service for hockey analytics dashboard",
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

# Pydantic models
class GameState(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    time_remaining: str
    power_play: Dict[str, Any]

class PlayerStats(BaseModel):
    player_id: int
    name: str
    team: str
    position: str
    goals: int
    assists: int
    points: int
    plus_minus: int
    time_on_ice: str
    speed: float
    distance_covered: float

class LiveEvent(BaseModel):
    event_id: str
    event_type: str
    timestamp: str
    player_name: str
    team: str
    description: str
    confidence: float

class AnalyticsMetrics(BaseModel):
    detection_accuracy: float
    processing_time: float
    fps: float
    active_players: int
    total_events: int
    momentum: Dict[str, float]
    pressure: Dict[str, float]
    enhanced_features: Dict[str, bool]

class VideoProcessingRequest(BaseModel):
    video_source: str
    model_id: str
    confidence_threshold: float
    enable_streaming: bool

# Database initialization
def init_database():
    """Initialize SQLite database with sample data"""
    conn = sqlite3.connect('main_api.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            period INTEGER,
            time_remaining TEXT,
            power_play_home BOOLEAN,
            power_play_away BOOLEAN,
            power_play_time TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            name TEXT,
            team TEXT,
            position TEXT,
            goals INTEGER,
            assists INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            time_on_ice TEXT,
            speed REAL,
            distance_covered REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS live_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            event_type TEXT,
            timestamp TEXT,
            player_name TEXT,
            team TEXT,
            description TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data
    cursor.execute('''
        INSERT OR IGNORE INTO game_states 
        (home_team, away_team, home_score, away_score, period, time_remaining, power_play_home, power_play_away, power_play_time)
        VALUES ('Toronto Maple Leafs', 'Montreal Canadiens', 2, 1, 2, '12:34', 0, 1, '1:45')
    ''')
    
    # Sample player stats
    sample_players = [
        (1, 'Auston Matthews', 'Toronto Maple Leafs', 'C', 1, 1, 2, 1, '18:45', 28.5, 2.3),
        (2, 'Mitch Marner', 'Toronto Maple Leafs', 'RW', 0, 2, 2, 1, '19:12', 26.8, 2.1),
        (3, 'Nick Suzuki', 'Montreal Canadiens', 'C', 1, 0, 1, -1, '17:23', 25.2, 1.9),
        (4, 'Cole Caufield', 'Montreal Canadiens', 'RW', 0, 1, 1, -1, '16:45', 27.1, 2.0),
    ]
    
    for player in sample_players:
        cursor.execute('''
            INSERT OR IGNORE INTO player_stats 
            (player_id, name, team, position, goals, assists, points, plus_minus, time_on_ice, speed, distance_covered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', player)
    
    # Sample live events
    sample_events = [
        ('evt_001', 'goal', '2024-01-15T19:23:45Z', 'Auston Matthews', 'Toronto Maple Leafs', 'Goal scored from the slot', 0.95),
        ('evt_002', 'save', '2024-01-15T19:24:12Z', 'Carey Price', 'Montreal Canadiens', 'Glove save on breakaway', 0.92),
        ('evt_003', 'hit', '2024-01-15T19:25:33Z', 'Zach Hyman', 'Toronto Maple Leafs', 'Clean hit in the corner', 0.88),
    ]
    
    for event in sample_events:
        cursor.execute('''
            INSERT OR IGNORE INTO live_events 
            (event_id, event_type, timestamp, player_name, team, description, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', event)
    
    conn.commit()
    conn.close()
    logger.info("Main API database initialized")

# Initialize database on startup
init_database()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "TSAI Jarvis Main Hockey Analytics API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "game_state": "/api/game/state",
            "player_stats": "/api/players/stats",
            "live_events": "/api/events/live",
            "analytics_metrics": "/api/analytics/metrics",
            "websocket": "/ws/analytics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/game/state", response_model=GameState)
async def get_game_state():
    """Get current game state"""
    try:
        conn = sqlite3.connect('main_api.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT home_team, away_team, home_score, away_score, period, time_remaining,
                   power_play_home, power_play_away, power_play_time
            FROM game_states 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return GameState(
                home_team=result[0],
                away_team=result[1],
                home_score=result[2],
                away_score=result[3],
                period=result[4],
                time_remaining=result[5],
                power_play={
                    "home": bool(result[6]),
                    "away": bool(result[7]),
                    "time_remaining": result[8]
                }
            )
        else:
            # Return default game state
            return GameState(
                home_team="Toronto Maple Leafs",
                away_team="Montreal Canadiens",
                home_score=2,
                away_score=1,
                period=2,
                time_remaining="12:34",
                power_play={
                    "home": False,
                    "away": True,
                    "time_remaining": "1:45"
                }
            )
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/players/stats", response_model=List[PlayerStats])
async def get_player_stats():
    """Get player statistics"""
    try:
        conn = sqlite3.connect('main_api.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT player_id, name, team, position, goals, assists, points, plus_minus,
                   time_on_ice, speed, distance_covered
            FROM player_stats 
            ORDER BY points DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            PlayerStats(
                player_id=row[0],
                name=row[1],
                team=row[2],
                position=row[3],
                goals=row[4],
                assists=row[5],
                points=row[6],
                plus_minus=row[7],
                time_on_ice=row[8],
                speed=row[9],
                distance_covered=row[10]
            )
            for row in results
        ]
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/events/live", response_model=List[LiveEvent])
async def get_live_events():
    """Get live events"""
    try:
        conn = sqlite3.connect('main_api.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT event_id, event_type, timestamp, player_name, team, description, confidence
            FROM live_events 
            ORDER BY created_at DESC 
            LIMIT 10
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            LiveEvent(
                event_id=row[0],
                event_type=row[1],
                timestamp=row[2],
                player_name=row[3],
                team=row[4],
                description=row[5],
                confidence=row[6]
            )
            for row in results
        ]
    except Exception as e:
        logger.error(f"Error getting live events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/metrics", response_model=AnalyticsMetrics)
async def get_analytics_metrics():
    """Get analytics metrics"""
    try:
        # Simulate real-time analytics data
        return AnalyticsMetrics(
            detection_accuracy=0.95,
            processing_time=45.2,
            fps=30.0,
            active_players=4,
            total_events=127,
            momentum={
                "home": 0.65,
                "away": 0.35
            },
            pressure={
                "offensive": 0.72,
                "defensive": 0.28
            },
            enhanced_features={
                "player_tracking": True,
                "shot_analysis": True,
                "formation_detection": True,
                "speed_analysis": True
            }
        )
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/process")
async def start_video_processing(request: VideoProcessingRequest):
    """Start video processing"""
    try:
        logger.info(f"Starting video processing: {request.model_id}")
        return {
            "status": "started",
            "model_id": request.model_id,
            "confidence_threshold": request.confidence_threshold,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/status")
async def get_video_processing_status():
    """Get video processing status"""
    return {
        "status": "processing",
        "fps": 30.0,
        "detected_players": 4,
        "processing_time": 45.2,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/video/stop")
async def stop_video_processing():
    """Stop video processing"""
    return {
        "status": "stopped",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models")
async def get_available_models():
    """Get available AI models"""
    return {
        "models": [
            {
                "id": "yolo_v8",
                "name": "YOLO v8",
                "type": "object_detection",
                "version": "8.0.0",
                "status": "available"
            },
            {
                "id": "pose_estimation",
                "name": "Pose Estimation",
                "type": "pose_detection",
                "version": "1.0.0",
                "status": "available"
            }
        ]
    }

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str):
    """Load specific model"""
    return {
        "status": "loaded",
        "model_id": model_id,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/analytics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics"""
    await manager.connect(websocket)
    try:
        while True:
            # Send real-time data every 2 seconds
            await asyncio.sleep(2)
            
            data = {
                "type": "analytics_update",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "detection_accuracy": 0.95,
                    "processing_time": 45.2,
                    "fps": 30.0,
                    "active_players": 4
                }
            }
            
            await manager.send_personal_message(json.dumps(data), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    logger.info("Starting TSAI Jarvis Main Hockey Analytics API on port 8001...")
    logger.info("API Documentation: http://localhost:8001/docs")
    logger.info("WebSocket: ws://localhost:8001/ws/analytics")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
