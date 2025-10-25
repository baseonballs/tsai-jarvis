"""
TSAI Jarvis - Enhanced Hockey Analytics API Service
Phase 1.2: Enhanced API Integration for Live Analytics

This module provides enhanced REST API endpoints for the hockey analytics dashboard,
integrating advanced video processing, live analytics, and real-time data streaming.
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

# Import our enhanced analytics modules
from live_analytics_implementation import (
    LiveAnalyticsEngine, 
    EnhancedPlayerMetrics, 
    EnhancedGameEvent, 
    TeamAnalytics
)
from video_processor import HockeyVideoProcessor, PlayerDetection, GameEvent
from yolo_model_manager import YOLOModelManager

logger = logging.getLogger(__name__)

# Enhanced Pydantic models for API
class EnhancedGameState(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    time_remaining: str
    power_play: Dict[str, Any]
    momentum: Dict[str, float]
    pressure: Dict[str, float]

class EnhancedPlayerStats(BaseModel):
    player_id: int
    name: str
    team: str
    position: str
    jersey_number: int
    speed: float
    acceleration: float
    distance_covered: float
    time_on_ice: str
    goals: int
    assists: int
    points: int
    plus_minus: int
    shots: int
    shots_on_goal: int
    hits: int
    blocked_shots: int
    faceoff_wins: int
    faceoff_attempts: int
    power_play_time: float
    penalty_kill_time: float
    offensive_zone_time: float
    defensive_zone_time: float
    neutral_zone_time: float
    current_position: List[float]
    movement_direction: float

class EnhancedLiveEvent(BaseModel):
    event_id: str
    event_type: str
    timestamp: datetime
    period: int
    time_remaining: str
    primary_player_id: int
    primary_player_name: str
    secondary_player_id: Optional[int] = None
    secondary_player_name: Optional[str] = None
    team: str
    location: List[float]
    confidence: float
    description: str
    event_velocity: Optional[float] = None
    event_angle: Optional[float] = None
    event_distance: Optional[float] = None
    event_context: Optional[str] = None
    video_timestamp: float
    frame_number: int
    detection_confidence: float

class EnhancedAnalyticsMetrics(BaseModel):
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
    model_id: str = "yolov8n"
    confidence_threshold: float = 0.5
    enable_streaming: bool = True
    enable_enhanced_analytics: bool = True

class EnhancedHockeyAnalyticsAPI:
    """
    Enhanced FastAPI service for hockey analytics dashboard integration
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Enhanced Hockey Analytics API",
            description="Enhanced real-time hockey analytics with advanced AI-powered video processing",
            version="1.2.0"
        )
        
        # Initialize enhanced services
        self.video_processor = None
        self.live_analytics_engine = LiveAnalyticsEngine()
        self.model_manager = YOLOModelManager()
        
        # Enhanced game state
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
        """Setup enhanced API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Enhanced Hockey Analytics API",
                "version": "1.2.0",
                "status": "running",
                "phase": "1.2 - Live Analytics",
                "features": {
                    "enhanced_analytics": True,
                    "momentum_tracking": True,
                    "pressure_analysis": True,
                    "performance_prediction": True,
                    "advanced_event_detection": True
                },
                "endpoints": {
                    "health": "/health",
                    "game_state": "/api/game/state",
                    "player_stats": "/api/players/stats",
                    "live_events": "/api/events/live",
                    "analytics": "/api/analytics/metrics",
                    "momentum": "/api/analytics/momentum",
                    "pressure": "/api/analytics/pressure",
                    "video_processing": "/api/video/process",
                    "websocket": "/ws/analytics"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Enhanced health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.2.0",
                "phase": "Live Analytics",
                "services": {
                    "video_processor": self.video_processor is not None,
                    "live_analytics_engine": self.live_analytics_engine.is_running,
                    "model_manager": self.model_manager is not None
                },
                "game_active": self.current_game is not None,
                "processing_active": self.is_processing,
                "enhanced_features": {
                    "momentum_tracking": True,
                    "pressure_analysis": True,
                    "performance_prediction": True,
                    "advanced_event_detection": True
                }
            }
        
        @self.app.get("/api/game/state")
        async def get_enhanced_game_state():
            """Get enhanced game state with momentum and pressure"""
            if not self.current_game:
                # Return enhanced default game state
                return EnhancedGameState(
                    home_team="Toronto Maple Leafs",
                    away_team="Montreal Canadiens",
                    home_score=2,
                    away_score=1,
                    period=2,
                    time_remaining="12:34",
                    power_play={"home": False, "away": True, "time_remaining": "1:23"},
                    momentum={"home": 0.6, "away": 0.4},
                    pressure={"home": 0.7, "away": 0.8}
                )
            
            return self.current_game
        
        @self.app.get("/api/players/stats")
        async def get_enhanced_player_stats():
            """Get enhanced player statistics"""
            if not self.live_analytics_engine.player_metrics:
                # Return enhanced sample data
                return [
                    EnhancedPlayerStats(
                        player_id=1,
                        name="Auston Matthews",
                        team="home",
                        position="C",
                        jersey_number=34,
                        speed=28.5,
                        acceleration=2.1,
                        distance_covered=1250.0,
                        time_on_ice="18:45",
                        goals=1, assists=1, points=2, plus_minus=1,
                        shots=3, shots_on_goal=2, hits=1, blocked_shots=0,
                        faceoff_wins=8, faceoff_attempts=12,
                        power_play_time=120.0, penalty_kill_time=60.0,
                        offensive_zone_time=0.4, defensive_zone_time=0.3, neutral_zone_time=0.3,
                        current_position=[100.0, 200.0],
                        movement_direction=45.0
                    ),
                    EnhancedPlayerStats(
                        player_id=2,
                        name="Mitch Marner",
                        team="home",
                        position="RW",
                        jersey_number=16,
                        speed=26.8,
                        acceleration=1.8,
                        distance_covered=1180.0,
                        time_on_ice="19:12",
                        goals=0, assists=2, points=2, plus_minus=1,
                        shots=2, shots_on_goal=1, hits=0, blocked_shots=1,
                        faceoff_wins=0, faceoff_attempts=0,
                        power_play_time=150.0, penalty_kill_time=30.0,
                        offensive_zone_time=0.5, defensive_zone_time=0.2, neutral_zone_time=0.3,
                        current_position=[150.0, 180.0],
                        movement_direction=30.0
                    )
                ]
            
            # Convert enhanced live analytics to API format
            stats = []
            for player_id, metrics in self.live_analytics_engine.player_metrics.items():
                stats.append(EnhancedPlayerStats(
                    player_id=metrics.player_id,
                    name=metrics.name,
                    team=metrics.team,
                    position=metrics.position,
                    jersey_number=metrics.jersey_number,
                    speed=metrics.speed,
                    acceleration=metrics.acceleration,
                    distance_covered=metrics.distance_covered,
                    time_on_ice=f"{int(metrics.time_on_ice//60):02d}:{int(metrics.time_on_ice%60):02d}",
                    goals=metrics.goals,
                    assists=metrics.assists,
                    points=metrics.goals + metrics.assists,
                    plus_minus=metrics.plus_minus,
                    shots=metrics.shots,
                    shots_on_goal=metrics.shots_on_goal,
                    hits=metrics.hits,
                    blocked_shots=metrics.blocked_shots,
                    faceoff_wins=metrics.faceoff_wins,
                    faceoff_attempts=metrics.faceoff_attempts,
                    power_play_time=metrics.power_play_time,
                    penalty_kill_time=metrics.penalty_kill_time,
                    offensive_zone_time=metrics.offensive_zone_time,
                    defensive_zone_time=metrics.defensive_zone_time,
                    neutral_zone_time=metrics.neutral_zone_time,
                    current_position=list(metrics.current_position),
                    movement_direction=metrics.movement_direction
                ))
            
            return stats
        
        @self.app.get("/api/events/live")
        async def get_enhanced_live_events():
            """Get enhanced live events"""
            if not self.live_analytics_engine.game_events:
                # Return enhanced sample events
                return [
                    EnhancedLiveEvent(
                        event_id="1",
                        event_type="goal",
                        timestamp=datetime.now() - timedelta(minutes=5),
                        period=2,
                        time_remaining="15:00",
                        primary_player_id=1,
                        primary_player_name="Auston Matthews",
                        team="home",
                        location=[100.0, 200.0],
                        confidence=0.95,
                        description="Goal scored by Auston Matthews",
                        event_velocity=85.2,
                        event_angle=15.0,
                        event_distance=25.0,
                        event_context="even_strength",
                        video_timestamp=time.time(),
                        frame_number=1500,
                        detection_confidence=0.95
                    ),
                    EnhancedLiveEvent(
                        event_id="2",
                        event_type="save",
                        timestamp=datetime.now() - timedelta(minutes=3),
                        period=2,
                        time_remaining="17:00",
                        primary_player_id=30,
                        primary_player_name="Carey Price",
                        team="away",
                        location=[200.0, 100.0],
                        confidence=0.87,
                        description="Save by Carey Price",
                        event_velocity=0.0,
                        event_angle=0.0,
                        event_distance=0.0,
                        event_context="even_strength",
                        video_timestamp=time.time(),
                        frame_number=1800,
                        detection_confidence=0.87
                    )
                ]
            
            # Convert enhanced live analytics events to API format
            events = []
            for event in list(self.live_analytics_engine.game_events)[-10:]:  # Last 10 events
                events.append(EnhancedLiveEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    period=event.period,
                    time_remaining=event.time_remaining,
                    primary_player_id=event.primary_player_id,
                    primary_player_name=event.primary_player_name,
                    secondary_player_id=event.secondary_player_id,
                    secondary_player_name=event.secondary_player_name,
                    team=event.team,
                    location=list(event.location),
                    confidence=event.confidence,
                    description=event.description,
                    event_velocity=event.event_velocity,
                    event_angle=event.event_angle,
                    event_distance=event.event_distance,
                    event_context=event.event_context,
                    video_timestamp=event.video_timestamp,
                    frame_number=event.frame_number,
                    detection_confidence=event.detection_confidence
                ))
            
            return events
        
        @self.app.get("/api/analytics/metrics")
        async def get_enhanced_analytics_metrics():
            """Get enhanced analytics metrics"""
            analytics_summary = self.live_analytics_engine.get_analytics_summary()
            
            return EnhancedAnalyticsMetrics(
                detection_accuracy=analytics_summary.get("analytics_cache", {}).get("detection_accuracy", 94.2),
                processing_time=analytics_summary.get("analytics_cache", {}).get("processing_time", 0.045),
                fps=analytics_summary.get("analytics_cache", {}).get("performance_metrics", {}).get("fps", 30.0),
                active_players=analytics_summary.get("active_players", 0),
                total_events=analytics_summary.get("total_events", 0),
                momentum=analytics_summary.get("analytics_cache", {}).get("momentum", {"home": 0.5, "away": 0.5}),
                pressure=analytics_summary.get("analytics_cache", {}).get("pressure", {"home": 0.5, "away": 0.5}),
                enhanced_features={
                    "momentum_tracking": True,
                    "pressure_analysis": True,
                    "performance_prediction": True,
                    "advanced_event_detection": True
                }
            )
        
        @self.app.get("/api/analytics/momentum")
        async def get_momentum_analytics():
            """Get momentum analytics"""
            return {
                "current_momentum": self.live_analytics_engine.momentum_tracker.get_current_momentum(),
                "team_momentum": {
                    "home": self.live_analytics_engine.momentum_tracker.get_team_momentum("home"),
                    "away": self.live_analytics_engine.momentum_tracker.get_team_momentum("away")
                },
                "momentum_history": list(self.live_analytics_engine.momentum_tracker.momentum_history)
            }
        
        @self.app.get("/api/analytics/pressure")
        async def get_pressure_analytics():
            """Get pressure analytics"""
            return {
                "current_pressure": self.live_analytics_engine.pressure_analyzer.get_current_pressure(),
                "team_pressure": {
                    "home": self.live_analytics_engine.pressure_analyzer.get_team_pressure("home"),
                    "away": self.live_analytics_engine.pressure_analyzer.get_team_pressure("away")
                },
                "pressure_history": list(self.live_analytics_engine.pressure_analyzer.pressure_history)
            }
        
        @self.app.post("/api/video/process")
        async def start_enhanced_video_processing(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
            """Start enhanced video processing with advanced analytics"""
            try:
                # Initialize enhanced video processor
                self.video_processor = HockeyVideoProcessor(
                    model_path=f"{request.model_id}.pt",
                    confidence_threshold=request.confidence_threshold
                )
                
                # Start enhanced processing in background
                background_tasks.add_task(
                    self._process_enhanced_video_background,
                    request.video_source,
                    request.enable_streaming,
                    request.enable_enhanced_analytics
                )
                
                return {
                    "status": "started",
                    "message": "Enhanced video processing started",
                    "model_id": request.model_id,
                    "video_source": request.video_source,
                    "enhanced_analytics": request.enable_enhanced_analytics,
                    "features": {
                        "momentum_tracking": True,
                        "pressure_analysis": True,
                        "performance_prediction": True,
                        "advanced_event_detection": True
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/video/status")
        async def get_enhanced_video_processing_status():
            """Get enhanced video processing status"""
            if not self.video_processor:
                return {"status": "not_started", "message": "No video processing active"}
            
            return {
                "status": "processing" if self.is_processing else "stopped",
                "is_processing": self.is_processing,
                "enhanced_analytics": True,
                "analytics_summary": self.video_processor.get_analytics_summary(),
                "live_analytics": self.live_analytics_engine.get_analytics_summary()
            }
        
        @self.app.post("/api/video/stop")
        async def stop_enhanced_video_processing():
            """Stop enhanced video processing"""
            if self.video_processor:
                await self.video_processor.stop_processing()
                self.is_processing = False
                return {"status": "stopped", "message": "Enhanced video processing stopped"}
            
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
        async def websocket_enhanced_analytics(websocket: WebSocket):
            """Enhanced WebSocket endpoint for real-time analytics"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send enhanced analytics updates
                    if self.live_analytics_engine.analytics_cache:
                        await websocket.send_json({
                            "type": "enhanced_analytics_update",
                            "data": self.live_analytics_engine.analytics_cache,
                            "momentum": self.live_analytics_engine.momentum_tracker.get_current_momentum(),
                            "pressure": self.live_analytics_engine.pressure_analyzer.get_current_pressure()
                        })
                    
                    await asyncio.sleep(0.1)  # 10 FPS updates
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
    
    async def _process_enhanced_video_background(self, video_source: str, enable_streaming: bool, enable_enhanced_analytics: bool):
        """Enhanced background task for video processing"""
        try:
            self.is_processing = True
            logger.info(f"🎬 Starting enhanced video processing: {video_source}")
            
            # Start enhanced live analytics engine
            if not self.live_analytics_engine.is_running:
                await self.live_analytics_engine.start_engine()
            
            # Process video with enhanced analytics
            results = await self.video_processor.process_video_stream(video_source)
            
            logger.info(f"✅ Enhanced video processing completed: {results}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced video processing failed: {e}")
        finally:
            self.is_processing = False
    
    async def start_service(self, host: str = "0.0.0.0", port: int = 8002):
        """Start the enhanced hockey analytics API service"""
        logger.info(f"🚀 Starting Enhanced Hockey Analytics API on {host}:{port}")
        
        # Start enhanced live analytics engine
        await self.live_analytics_engine.start_engine()
        
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
        """Stop the enhanced hockey analytics API service"""
        logger.info("🛑 Stopping Enhanced Hockey Analytics API...")
        
        # Stop enhanced live analytics
        await self.live_analytics_engine.stop_engine()
        
        # Stop video processing
        if self.video_processor:
            await self.video_processor.stop_processing()

# Create enhanced FastAPI app instance
enhanced_hockey_api = EnhancedHockeyAnalyticsAPI()
app = enhanced_hockey_api.app

# Example usage and testing
async def test_enhanced_hockey_api():
    """Test enhanced hockey analytics API"""
    logger.info("🧪 Testing Enhanced Hockey Analytics API...")
    
    # Start service
    await enhanced_hockey_api.start_service(host="localhost", port=8002)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run service
    asyncio.run(test_enhanced_hockey_api())
