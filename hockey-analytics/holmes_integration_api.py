#!/usr/bin/env python3
"""
TSAI Jarvis - Holmes Integration API
Media curation and asset management for hockey analytics
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
    title="TSAI Jarvis Holmes Integration API",
    description="Media curation and asset management for hockey analytics",
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
class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"

class AssetStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ARCHIVED = "archived"
    DELETED = "deleted"

class CurationStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"

class AssetCategory(str, Enum):
    GAME_FOOTAGE = "game_footage"
    PLAYER_INTERVIEWS = "player_interviews"
    COACH_ANALYSIS = "coach_analysis"
    STATISTICS = "statistics"
    HIGHLIGHTS = "highlights"
    TRAINING_MATERIALS = "training_materials"
    MARKETING_CONTENT = "marketing_content"

class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# --- Pydantic Models ---
class MediaAsset(BaseModel):
    asset_id: str
    title: str
    description: str
    media_type: MediaType
    file_path: str
    file_size: int
    duration: Optional[float] = None  # For video/audio
    resolution: Optional[str] = None  # For video/images
    format: str
    category: AssetCategory
    status: AssetStatus
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

class CurationPlaylist(BaseModel):
    playlist_id: str
    name: str
    description: str
    category: AssetCategory
    status: CurationStatus
    assets: List[str] = []  # asset_ids
    curator: str
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

class AssetProcessing(BaseModel):
    processing_id: str
    asset_id: str
    processing_type: str  # "transcode", "compress", "enhance", "extract_metadata"
    status: ProcessingStatus
    progress: float = 0.0
    parameters: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AssetSearch(BaseModel):
    search_id: str
    query: str
    filters: Dict[str, Any] = {}
    results: List[str] = []  # asset_ids
    total_results: int = 0
    search_time: float = 0.0
    created_at: datetime

class AssetAnalytics(BaseModel):
    analytics_id: str
    asset_id: str
    views: int = 0
    downloads: int = 0
    shares: int = 0
    likes: int = 0
    engagement_score: float = 0.0
    time_period: str
    created_at: datetime

class ContentRecommendation(BaseModel):
    recommendation_id: str
    user_id: str
    asset_id: str
    recommendation_type: str  # "similar_content", "trending", "personalized"
    confidence: float = 0.0
    reason: str
    created_at: datetime

class MediaWorkflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = []
    status: ProcessingStatus
    assets: List[str] = []  # asset_ids
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AssetVersion(BaseModel):
    version_id: str
    asset_id: str
    version_number: int
    file_path: str
    changes: str
    created_by: str
    created_at: datetime

# --- Database Setup ---
def init_database():
    """Initialize the Holmes database"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_assets (
            asset_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            media_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            duration REAL,
            resolution TEXT,
            format TEXT NOT NULL,
            category TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS curation_playlists (
            playlist_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            status TEXT DEFAULT 'draft',
            assets TEXT DEFAULT '[]',
            curator TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset_processing (
            processing_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            processing_type TEXT NOT NULL,
            status TEXT DEFAULT 'queued',
            progress REAL DEFAULT 0.0,
            parameters TEXT DEFAULT '{}',
            result TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset_searches (
            search_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            filters TEXT DEFAULT '{}',
            results TEXT DEFAULT '[]',
            total_results INTEGER DEFAULT 0,
            search_time REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset_analytics (
            analytics_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            downloads INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            engagement_score REAL DEFAULT 0.0,
            time_period TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            reason TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_workflows (
            workflow_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            steps TEXT DEFAULT '[]',
            status TEXT DEFAULT 'pending',
            assets TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset_versions (
            version_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            changes TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Holmes database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.curation_connections: List[WebSocket] = []
        self.analytics_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "curation":
            self.curation_connections.append(websocket)
        elif connection_type == "analytics":
            self.analytics_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {connection_type}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.curation_connections:
            self.curation_connections.remove(websocket)
        if websocket in self.analytics_connections:
            self.analytics_connections.remove(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_curators(self, message: str):
        for connection in self.curation_connections:
            await connection.send_text(message)

    async def broadcast_to_analysts(self, message: str):
        for connection in self.analytics_connections:
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Holmes Integration API"""
    return {"status": "ok", "message": "TSAI Jarvis Holmes Integration API is running"}

@app.get("/api/holmes/status", summary="Get Holmes Status")
async def get_holmes_status():
    """Get the current status of the Holmes integration"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    # Get total assets
    cursor.execute("SELECT COUNT(*) FROM media_assets")
    total_assets = cursor.fetchone()[0]
    
    # Get active playlists
    cursor.execute("SELECT COUNT(*) FROM curation_playlists WHERE status IN ('draft', 'review')")
    active_playlists = cursor.fetchone()[0]
    
    # Get processing jobs
    cursor.execute("SELECT COUNT(*) FROM asset_processing WHERE status IN ('queued', 'processing')")
    active_processing = cursor.fetchone()[0]
    
    # Get recent analytics
    cursor.execute("SELECT COUNT(*) FROM asset_analytics WHERE created_at > datetime('now', '-1 hour')")
    recent_analytics = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "total_assets": total_assets,
        "active_playlists": active_playlists,
        "active_processing": active_processing,
        "recent_analytics": recent_analytics,
        "curation_connections": len(manager.curation_connections),
        "analytics_connections": len(manager.analytics_connections),
        "total_connections": len(manager.active_connections)
    }

@app.post("/api/holmes/assets", summary="Create Media Asset")
async def create_media_asset(asset: MediaAsset):
    """Create a new media asset"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO media_assets (asset_id, title, description, media_type, file_path, 
                                    file_size, duration, resolution, format, category, 
                                    status, tags, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            asset.asset_id,
            asset.title,
            asset.description,
            asset.media_type,
            asset.file_path,
            asset.file_size,
            asset.duration,
            asset.resolution,
            asset.format,
            asset.category,
            asset.status,
            json.dumps(asset.tags),
            json.dumps(asset.metadata),
            asset.created_at,
            asset.updated_at
        ))
        
        conn.commit()
        logger.info(f"Created media asset: {asset.asset_id}")
        
        # Notify curation connections about new asset
        await manager.broadcast_to_curators(json.dumps({
            "type": "asset_created",
            "asset_id": asset.asset_id,
            "title": asset.title,
            "media_type": asset.media_type,
            "category": asset.category,
            "status": asset.status
        }))
        
        return {"message": "Media asset created successfully", "asset_id": asset.asset_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create asset: {str(e)}")
    finally:
        conn.close()

@app.get("/api/holmes/assets", summary="Get All Assets")
async def get_assets(category: Optional[str] = None, media_type: Optional[str] = None, limit: int = 50):
    """Get media assets with optional filtering"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    query = "SELECT asset_id, title, media_type, category, status, created_at FROM media_assets"
    params = []
    conditions = []
    
    if category:
        conditions.append("category = ?")
        params.append(category)
    
    if media_type:
        conditions.append("media_type = ?")
        params.append(media_type)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    
    assets = []
    for row in cursor.fetchall():
        assets.append({
            "asset_id": row[0],
            "title": row[1],
            "media_type": row[2],
            "category": row[3],
            "status": row[4],
            "created_at": row[5]
        })
    
    conn.close()
    return {"assets": assets}

@app.post("/api/holmes/playlists", summary="Create Curation Playlist")
async def create_curation_playlist(playlist: CurationPlaylist):
    """Create a new curation playlist"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO curation_playlists (playlist_id, name, description, category, 
                                          status, assets, curator, created_at, updated_at, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            playlist.playlist_id,
            playlist.name,
            playlist.description,
            playlist.category,
            playlist.status,
            json.dumps(playlist.assets),
            playlist.curator,
            playlist.created_at,
            playlist.updated_at,
            playlist.published_at
        ))
        
        conn.commit()
        logger.info(f"Created curation playlist: {playlist.playlist_id}")
        
        # Notify curation connections about new playlist
        await manager.broadcast_to_curators(json.dumps({
            "type": "playlist_created",
            "playlist_id": playlist.playlist_id,
            "name": playlist.name,
            "category": playlist.category,
            "status": playlist.status,
            "asset_count": len(playlist.assets)
        }))
        
        return {"message": "Curation playlist created successfully", "playlist_id": playlist.playlist_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create playlist: {str(e)}")
    finally:
        conn.close()

@app.get("/api/holmes/playlists", summary="Get All Playlists")
async def get_playlists(category: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    """Get curation playlists with optional filtering"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    query = "SELECT playlist_id, name, description, category, status, curator, created_at FROM curation_playlists"
    params = []
    conditions = []
    
    if category:
        conditions.append("category = ?")
        params.append(category)
    
    if status:
        conditions.append("status = ?")
        params.append(status)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    
    playlists = []
    for row in cursor.fetchall():
        playlists.append({
            "playlist_id": row[0],
            "name": row[1],
            "description": row[2],
            "category": row[3],
            "status": row[4],
            "curator": row[5],
            "created_at": row[6]
        })
    
    conn.close()
    return {"playlists": playlists}

@app.post("/api/holmes/processing", summary="Create Asset Processing Job")
async def create_asset_processing(processing: AssetProcessing):
    """Create a new asset processing job"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO asset_processing (processing_id, asset_id, processing_type, 
                                       status, progress, parameters, result, 
                                       error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            processing.processing_id,
            processing.asset_id,
            processing.processing_type,
            processing.status,
            processing.progress,
            json.dumps(processing.parameters),
            json.dumps(processing.result) if processing.result else None,
            processing.error_message,
            processing.created_at
        ))
        
        conn.commit()
        logger.info(f"Created asset processing job: {processing.processing_id}")
        
        # Notify analytics connections about new processing job
        await manager.broadcast_to_analysts(json.dumps({
            "type": "processing_job_created",
            "processing_id": processing.processing_id,
            "asset_id": processing.asset_id,
            "processing_type": processing.processing_type,
            "status": processing.status
        }))
        
        return {"message": "Asset processing job created successfully", "processing_id": processing.processing_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create processing job: {str(e)}")
    finally:
        conn.close()

@app.post("/api/holmes/search", summary="Search Assets")
async def search_assets(search: AssetSearch):
    """Search for assets"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        # Simple text search implementation
        query_conditions = []
        params = []
        
        if search.query:
            query_conditions.append("(title LIKE ? OR description LIKE ?)")
            search_term = f"%{search.query}%"
            params.extend([search_term, search_term])
        
        # Apply filters
        for key, value in search.filters.items():
            if key == "category":
                query_conditions.append("category = ?")
                params.append(value)
            elif key == "media_type":
                query_conditions.append("media_type = ?")
                params.append(value)
            elif key == "status":
                query_conditions.append("status = ?")
                params.append(value)
        
        # Build query
        base_query = "SELECT asset_id FROM media_assets"
        if query_conditions:
            base_query += " WHERE " + " AND ".join(query_conditions)
        
        cursor.execute(base_query, params)
        results = [row[0] for row in cursor.fetchall()]
        
        # Store search results
        cursor.execute('''
            INSERT INTO asset_searches (search_id, query, filters, results, 
                                      total_results, search_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            search.search_id,
            search.query,
            json.dumps(search.filters),
            json.dumps(results),
            len(results),
            search.search_time,
            search.created_at
        ))
        
        conn.commit()
        logger.info(f"Asset search completed: {search.search_id}")
        
        return {
            "message": "Asset search completed successfully",
            "search_id": search.search_id,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to search assets: {str(e)}")
    finally:
        conn.close()

@app.post("/api/holmes/analytics", summary="Record Asset Analytics")
async def record_asset_analytics(analytics: AssetAnalytics):
    """Record analytics for an asset"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO asset_analytics (analytics_id, asset_id, views, downloads, 
                                       shares, likes, engagement_score, time_period, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analytics.analytics_id,
            analytics.asset_id,
            analytics.views,
            analytics.downloads,
            analytics.shares,
            analytics.likes,
            analytics.engagement_score,
            analytics.time_period,
            analytics.created_at
        ))
        
        conn.commit()
        logger.info(f"Recorded asset analytics: {analytics.analytics_id}")
        
        # Notify analytics connections about new analytics
        await manager.broadcast_to_analysts(json.dumps({
            "type": "analytics_update",
            "analytics_id": analytics.analytics_id,
            "asset_id": analytics.asset_id,
            "engagement_score": analytics.engagement_score,
            "views": analytics.views
        }))
        
        return {"message": "Asset analytics recorded successfully", "analytics_id": analytics.analytics_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record analytics: {str(e)}")
    finally:
        conn.close()

@app.post("/api/holmes/recommendations", summary="Create Content Recommendation")
async def create_content_recommendation(recommendation: ContentRecommendation):
    """Create a content recommendation"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO content_recommendations (recommendation_id, user_id, asset_id, 
                                               recommendation_type, confidence, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            recommendation.recommendation_id,
            recommendation.user_id,
            recommendation.asset_id,
            recommendation.recommendation_type,
            recommendation.confidence,
            recommendation.reason,
            recommendation.created_at
        ))
        
        conn.commit()
        logger.info(f"Created content recommendation: {recommendation.recommendation_id}")
        
        # Notify analytics connections about new recommendation
        await manager.broadcast_to_analysts(json.dumps({
            "type": "recommendation_created",
            "recommendation_id": recommendation.recommendation_id,
            "user_id": recommendation.user_id,
            "asset_id": recommendation.asset_id,
            "recommendation_type": recommendation.recommendation_type,
            "confidence": recommendation.confidence
        }))
        
        return {"message": "Content recommendation created successfully", "recommendation_id": recommendation.recommendation_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create recommendation: {str(e)}")
    finally:
        conn.close()

@app.get("/api/holmes/analytics", summary="Get Asset Analytics")
async def get_asset_analytics(asset_id: Optional[str] = None, time_period: str = "24h"):
    """Get analytics for assets"""
    conn = sqlite3.connect('holmes_integration.db')
    cursor = conn.cursor()
    
    # Calculate time filter
    if time_period == "24h":
        time_filter = "datetime('now', '-1 day')"
    elif time_period == "7d":
        time_filter = "datetime('now', '-7 days')"
    elif time_period == "30d":
        time_filter = "datetime('now', '-30 days')"
    else:
        time_filter = "datetime('now', '-1 day')"
    
    if asset_id:
        cursor.execute(f'''
            SELECT analytics_id, asset_id, views, downloads, shares, likes, 
                   engagement_score, time_period, created_at
            FROM asset_analytics 
            WHERE asset_id = ? AND created_at > {time_filter}
            ORDER BY created_at DESC
        ''', (asset_id,))
    else:
        cursor.execute(f'''
            SELECT analytics_id, asset_id, views, downloads, shares, likes, 
                   engagement_score, time_period, created_at
            FROM asset_analytics 
            WHERE created_at > {time_filter}
            ORDER BY created_at DESC
            LIMIT 100
        ''')
    
    analytics_data = []
    for row in cursor.fetchall():
        analytics_data.append({
            "analytics_id": row[0],
            "asset_id": row[1],
            "views": row[2],
            "downloads": row[3],
            "shares": row[4],
            "likes": row[5],
            "engagement_score": row[6],
            "time_period": row[7],
            "created_at": row[8]
        })
    
    conn.close()
    return {"analytics": analytics_data}

# --- WebSocket Endpoints ---
@app.websocket("/ws/holmes")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general Holmes communication"""
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

@app.websocket("/ws/holmes/curation")
async def curation_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for curation"""
    await manager.connect(websocket, "curation")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from curation clients
            message = json.loads(data)
            logger.info(f"Received curation message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Curation WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/holmes/analytics")
async def analytics_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for analytics"""
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
async def simulate_media_curation():
    """Simulate media curation activities for demonstration"""
    while True:
        try:
            if manager.curation_connections:
                # Simulate random curation activities
                import random
                if random.random() < 0.3:  # 30% chance every 15 seconds
                    activities = [
                        "New highlight reel curated for playoff games",
                        "Player interview playlist updated",
                        "Training materials organized by category",
                        "Statistics dashboard refreshed with latest data"
                    ]
                    
                    await manager.broadcast_to_curators(json.dumps({
                        "type": "curation_activity",
                        "activity": random.choice(activities),
                        "timestamp": datetime.now().isoformat()
                    }))
            
        except Exception as e:
            logger.error(f"Error simulating media curation: {e}")
        
        await asyncio.sleep(15)  # Check every 15 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Holmes Integration API background tasks...")
    asyncio.create_task(simulate_media_curation())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8017)
