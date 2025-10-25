"""
TSAI Jarvis - Hockey Analytics API Service
Phase 1.2: API Integration for Real-time Hockey Analytics

This module provides REST API endpoints for the hockey analytics dashboard,
integrating video processing, live analytics, and real-time data streaming.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import our analytics modules
from video_processor import HockeyVideoProcessor, PlayerDetection, GameEvent
from live_analytics_service import LiveAnalyticsService, PlayerMetrics, TeamStats
from yolo_model_manager import YOLOModelManager

logger = logging.getLogger(__name__)

# Pydantic models for API
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
    timestamp: datetime
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

class VideoProcessingRequest(BaseModel):
    video_source: str
    model_id: str = "yolov8n"
    confidence_threshold: float = 0.5
    enable_streaming: bool = True

class HockeyAnalyticsAPI:
    """
    FastAPI service for hockey analytics dashboard integration
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Hockey Analytics API",
            description="Real-time hockey analytics with AI-powered video processing",
            version="1.0.0"
        )
        
        # Initialize services
        self.video_processor = None
        self.live_analytics = LiveAnalyticsService()
        self.model_manager = YOLOModelManager()
        
        # Game state
        self.current_game = None
        self.is_processing = False
        self.websocket_clients = set()
        
        # Setup API routes
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Hockey Analytics API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "health": "/health",
                    "game_state": "/api/game/state",
                    "player_stats": "/api/players/stats",
                    "live_events": "/api/events/live",
                    "analytics": "/api/analytics/metrics",
                    "video_processing": "/api/video/process",
                    "websocket": "/ws/analytics"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "video_processor": self.video_processor is not None,
                    "live_analytics": self.live_analytics.is_running,
                    "model_manager": self.model_manager is not None
                },
                "game_active": self.current_game is not None,
                "processing_active": self.is_processing
            }
        
        @self.app.get("/api/game/state")
        async def get_game_state():
            """Get current game state"""
            if not self.current_game:
                # Return default game state
                return GameState(
                    home_team="Toronto Maple Leafs",
                    away_team="Montreal Canadiens",
                    home_score=2,
                    away_score=1,
                    period=2,
                    time_remaining="12:34",
                    power_play={"home": False, "away": True, "time_remaining": "1:23"}
                )
            
            return self.current_game
        
        @self.app.get("/api/players/stats")
        async def get_player_stats():
            """Get current player statistics"""
            if not self.live_analytics.player_metrics:
                # Return sample data
                return [
                    PlayerStats(
                        player_id=1,
                        name="Auston Matthews",
                        team="home",
                        position="C",
                        goals=1,
                        assists=1,
                        points=2,
                        plus_minus=1,
                        time_on_ice="18:45",
                        speed=28.5,
                        distance_covered=1250.0
                    ),
                    PlayerStats(
                        player_id=2,
                        name="Mitch Marner",
                        team="home",
                        position="RW",
                        goals=0,
                        assists=2,
                        points=2,
                        plus_minus=1,
                        time_on_ice="19:12",
                        speed=26.8,
                        distance_covered=1180.0
                    )
                ]
            
            # Convert live analytics to API format
            stats = []
            for player_id, metrics in self.live_analytics.player_metrics.items():
                stats.append(PlayerStats(
                    player_id=player_id,
                    name=metrics.name,
                    team=metrics.team,
                    position=metrics.position,
                    goals=metrics.goals,
                    assists=metrics.assists,
                    points=metrics.goals + metrics.assists,
                    plus_minus=metrics.plus_minus,
                    time_on_ice=f"{int(metrics.time_on_ice//60):02d}:{int(metrics.time_on_ice%60):02d}",
                    speed=metrics.speed,
                    distance_covered=metrics.distance_covered
                ))
            
            return stats
        
        @self.app.get("/api/events/live")
        async def get_live_events():
            """Get recent live events"""
            if not self.live_analytics.game_events:
                # Return sample events
                return [
                    LiveEvent(
                        event_id="1",
                        event_type="goal",
                        timestamp=datetime.now() - timedelta(minutes=5),
                        player_name="Auston Matthews",
                        team="home",
                        description="Goal scored by Auston Matthews",
                        confidence=0.95
                    ),
                    LiveEvent(
                        event_id="2",
                        event_type="save",
                        timestamp=datetime.now() - timedelta(minutes=3),
                        player_name="Carey Price",
                        team="away",
                        description="Save by Carey Price",
                        confidence=0.87
                    )
                ]
            
            # Convert live analytics events to API format
            events = []
            for event in self.live_analytics.game_events[-10:]:  # Last 10 events
                events.append(LiveEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    player_name=event.player_name,
                    team=event.team,
                    description=event.description,
                    confidence=event.confidence
                ))
            
            return events
        
        @self.app.get("/api/analytics/metrics")
        async def get_analytics_metrics():
            """Get current analytics metrics"""
            analytics_summary = self.live_analytics.get_analytics_summary()
            
            return AnalyticsMetrics(
                detection_accuracy=analytics_summary.get("analytics_cache", {}).get("detection_accuracy", 94.2),
                processing_time=analytics_summary.get("analytics_cache", {}).get("processing_time", 0.045),
                fps=analytics_summary.get("analytics_cache", {}).get("performance_metrics", {}).get("fps", 30.0),
                active_players=analytics_summary.get("active_players", 0),
                total_events=analytics_summary.get("total_events", 0)
            )
        
        @self.app.post("/api/video/process")
        async def start_video_processing(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
            """Start video processing with YOLO model"""
            try:
                # Initialize video processor
                self.video_processor = HockeyVideoProcessor(
                    model_path=f"{request.model_id}.pt",
                    confidence_threshold=request.confidence_threshold
                )
                
                # Start processing in background
                background_tasks.add_task(
                    self._process_video_background,
                    request.video_source,
                    request.enable_streaming
                )
                
                return {
                    "status": "started",
                    "message": "Video processing started",
                    "model_id": request.model_id,
                    "video_source": request.video_source
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/video/status")
        async def get_video_processing_status():
            """Get video processing status"""
            if not self.video_processor:
                return {"status": "not_started", "message": "No video processing active"}
            
            return {
                "status": "processing" if self.is_processing else "stopped",
                "is_processing": self.is_processing,
                "analytics_summary": self.video_processor.get_analytics_summary()
            }
        
        @self.app.post("/api/video/stop")
        async def stop_video_processing():
            """Stop video processing"""
            if self.video_processor:
                await self.video_processor.stop_processing()
                self.is_processing = False
                return {"status": "stopped", "message": "Video processing stopped"}
            
            return {"status": "not_running", "message": "No video processing to stop"}
        
        @self.app.get("/api/models")
        async def get_available_models():
            """Get available YOLO models"""
            models = await self.model_manager.list_models()
            return {"models": models}
        
        @self.app.post("/api/models/{model_id}/load")
        async def load_model(model_id: str):
            """Load specific YOLO model"""
            success = await self.model_manager.load_model(model_id)
            if success:
                return {"status": "loaded", "model_id": model_id}
            else:
                raise HTTPException(status_code=400, detail=f"Failed to load model {model_id}")
        
        @self.app.websocket("/ws/analytics")
        async def websocket_analytics(websocket: WebSocket):
            """WebSocket endpoint for real-time analytics"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send analytics updates
                    if self.live_analytics.analytics_cache:
                        await websocket.send_json({
                            "type": "analytics_update",
                            "data": self.live_analytics.analytics_cache
                        })
                    
                    await asyncio.sleep(0.1)  # 10 FPS updates
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
    
    async def _process_video_background(self, video_source: str, enable_streaming: bool):
        """Background task for video processing"""
        try:
            self.is_processing = True
            logger.info(f"🎬 Starting video processing: {video_source}")
            
            # Start live analytics service
            if not self.live_analytics.is_running:
                await self.live_analytics.start_service()
            
            # Process video
            results = await self.video_processor.process_video_stream(video_source)
            
            logger.info(f"✅ Video processing completed: {results}")
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
        finally:
            self.is_processing = False
    
    async def start_service(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the hockey analytics API service"""
        logger.info(f"🚀 Starting Hockey Analytics API on {host}:{port}")
        
        # Start live analytics service
        await self.live_analytics.start_service()
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_service(self):
        """Stop the hockey analytics API service"""
        logger.info("🛑 Stopping Hockey Analytics API...")
        
        # Stop live analytics
        await self.live_analytics.stop_service()
        
        # Stop video processing
        if self.video_processor:
            await self.video_processor.stop_processing()

# Create FastAPI app instance
hockey_api = HockeyAnalyticsAPI()
app = hockey_api.app

# Example usage and testing
async def test_hockey_api():
    """Test hockey analytics API"""
    logger.info("🧪 Testing Hockey Analytics API...")
    
    # Start service
    await hockey_api.start_service(host="localhost", port=8001)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run service
    asyncio.run(test_hockey_api())
