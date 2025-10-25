"""
TSAI Jarvis - Advanced Hockey Analytics API Service
Phase 1.3: Advanced Analytics API Integration

This module provides advanced REST API endpoints for comprehensive hockey analytics,
including player speed tracking, shot analysis, team formation analysis, and strategy insights.
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

# Import our advanced analytics modules
from advanced_analytics_engine import (
    AdvancedAnalyticsEngine,
    PlayerSpeedMetrics,
    ShotAnalysis,
    TeamFormation,
    GameStrategyInsights
)
from video_processor import HockeyVideoProcessor, PlayerDetection, GameEvent
from yolo_model_manager import YOLOModelManager

logger = logging.getLogger(__name__)

# Advanced Pydantic models for API
class AdvancedGameState(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    time_remaining: str
    power_play: Dict[str, Any]
    advanced_analytics: Dict[str, Any]

class AdvancedPlayerStats(BaseModel):
    player_id: int
    name: str
    team: str
    position: str
    
    # Speed analytics
    current_speed: float
    max_speed: float
    avg_speed: float
    speed_acceleration: float
    speed_deceleration: float
    
    # Movement analytics
    total_distance: float
    movement_efficiency: float
    direction_changes: int
    movement_smoothness: float
    
    # Performance metrics
    speed_consistency: float
    acceleration_consistency: float
    fatigue_level: float
    energy_usage: float
    
    # Zone-specific speed
    offensive_zone_speed: float
    defensive_zone_speed: float
    neutral_zone_speed: float

class AdvancedShotAnalysis(BaseModel):
    shot_id: str
    player_id: int
    player_name: str
    team: str
    shot_type: str
    shot_location: List[float]
    goal_location: List[float]
    shot_distance: float
    shot_angle: float
    shot_velocity: float
    shot_accuracy: float
    shot_power: float
    shot_quality: float
    game_situation: str
    pressure_level: float
    goal_probability: float
    save_probability: float
    miss_probability: float
    timestamp: datetime
    confidence: float

class AdvancedTeamFormation(BaseModel):
    team: str
    formation_type: str
    forward_positions: List[List[float]]
    defense_positions: List[List[float]]
    goalie_position: List[float]
    formation_width: float
    formation_depth: float
    formation_compactness: float
    formation_balance: float
    avg_player_distance: float
    min_player_distance: float
    max_player_distance: float
    formation_stability: float
    formation_flexibility: float
    formation_coverage: float
    timestamp: datetime
    confidence: float

class AdvancedStrategyInsights(BaseModel):
    team: str
    period: int
    time_remaining: str
    possession_time: float
    possession_percentage: float
    possession_quality: float
    possession_efficiency: float
    offensive_zone_time: float
    defensive_zone_time: float
    neutral_zone_time: float
    zone_control_balance: float
    shots_generated: int
    shot_quality_avg: float
    shot_velocity_avg: float
    shot_accuracy_avg: float
    shots_allowed: int
    shot_quality_allowed: float
    defensive_pressure: float
    defensive_efficiency: float
    breakouts_successful: int
    breakouts_attempted: int
    breakout_success_rate: float
    forecheck_pressure: float
    power_play_efficiency: float
    penalty_kill_efficiency: float
    special_teams_advantage: float
    momentum_swings: int
    pressure_peaks: int
    strategic_advantages: List[str]
    tactical_recommendations: List[str]
    timestamp: datetime

class AdvancedAnalyticsMetrics(BaseModel):
    processing_time: float
    total_speed_metrics: int
    total_shots_analyzed: int
    total_formations: int
    strategy_insights: int
    advanced_features: Dict[str, bool]
    performance_metrics: Dict[str, float]

class VideoProcessingRequest(BaseModel):
    video_source: str
    model_id: str = "yolov8n"
    confidence_threshold: float = 0.5
    enable_streaming: bool = True
    enable_advanced_analytics: bool = True

class AdvancedHockeyAnalyticsAPI:
    """
    Advanced FastAPI service for comprehensive hockey analytics
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Advanced Hockey Analytics API",
            description="Advanced real-time hockey analytics with comprehensive AI-powered analysis",
            version="1.3.0"
        )
        
        # Initialize advanced services
        self.video_processor = None
        self.advanced_analytics_engine = AdvancedAnalyticsEngine()
        self.model_manager = YOLOModelManager()
        
        # Advanced game state
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
        """Setup advanced API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Advanced Hockey Analytics API",
                "version": "1.3.0",
                "status": "running",
                "phase": "1.3 - Advanced Analytics",
                "features": {
                    "player_speed_tracking": True,
                    "shot_analysis": True,
                    "formation_analysis": True,
                    "strategy_insights": True,
                    "advanced_analytics": True
                },
                "endpoints": {
                    "health": "/health",
                    "game_state": "/api/game/state",
                    "player_stats": "/api/players/stats",
                    "speed_analytics": "/api/analytics/speed",
                    "shot_analytics": "/api/analytics/shots",
                    "formation_analytics": "/api/analytics/formations",
                    "strategy_analytics": "/api/analytics/strategy",
                    "video_processing": "/api/video/process",
                    "websocket": "/ws/analytics"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Advanced health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.3.0",
                "phase": "Advanced Analytics",
                "services": {
                    "video_processor": self.video_processor is not None,
                    "advanced_analytics_engine": self.advanced_analytics_engine.is_running,
                    "model_manager": self.model_manager is not None
                },
                "game_active": self.current_game is not None,
                "processing_active": self.is_processing,
                "advanced_features": {
                    "player_speed_tracking": True,
                    "shot_analysis": True,
                    "formation_analysis": True,
                    "strategy_insights": True,
                    "advanced_analytics": True
                }
            }
        
        @self.app.get("/api/game/state")
        async def get_advanced_game_state():
            """Get advanced game state with comprehensive analytics"""
            if not self.current_game:
                # Return advanced default game state
                return AdvancedGameState(
                    home_team="Toronto Maple Leafs",
                    away_team="Montreal Canadiens",
                    home_score=2,
                    away_score=1,
                    period=2,
                    time_remaining="12:34",
                    power_play={"home": False, "away": True, "time_remaining": "1:23"},
                    advanced_analytics={
                        "momentum": {"home": 0.6, "away": 0.4},
                        "pressure": {"home": 0.7, "away": 0.8},
                        "possession": {"home": 0.55, "away": 0.45},
                        "shot_quality": {"home": 0.7, "away": 0.6}
                    }
                )
            
            return self.current_game
        
        @self.app.get("/api/players/stats")
        async def get_advanced_player_stats():
            """Get advanced player statistics with speed analytics"""
            if not self.advanced_analytics_engine.player_speed_metrics:
                # Return advanced sample data
                return [
                    AdvancedPlayerStats(
                        player_id=1,
                        name="Auston Matthews",
                        team="home",
                        position="C",
                        current_speed=28.5,
                        max_speed=35.2,
                        avg_speed=24.8,
                        speed_acceleration=2.1,
                        speed_deceleration=1.8,
                        total_distance=1250.0,
                        movement_efficiency=0.85,
                        direction_changes=12,
                        movement_smoothness=0.78,
                        speed_consistency=0.82,
                        acceleration_consistency=0.75,
                        fatigue_level=0.3,
                        energy_usage=0.65,
                        offensive_zone_speed=26.5,
                        defensive_zone_speed=22.1,
                        neutral_zone_speed=28.8
                    ),
                    AdvancedPlayerStats(
                        player_id=2,
                        name="Mitch Marner",
                        team="home",
                        position="RW",
                        current_speed=26.8,
                        max_speed=32.1,
                        avg_speed=23.2,
                        speed_acceleration=1.9,
                        speed_deceleration=1.6,
                        total_distance=1180.0,
                        movement_efficiency=0.88,
                        direction_changes=15,
                        movement_smoothness=0.82,
                        speed_consistency=0.85,
                        acceleration_consistency=0.78,
                        fatigue_level=0.25,
                        energy_usage=0.58,
                        offensive_zone_speed=25.8,
                        defensive_zone_speed=21.5,
                        neutral_zone_speed=27.2
                    )
                ]
            
            # Convert advanced analytics to API format
            stats = []
            for player_id, metrics in self.advanced_analytics_engine.player_speed_metrics.items():
                stats.append(AdvancedPlayerStats(
                    player_id=metrics.player_id,
                    name=metrics.name,
                    team=metrics.team,
                    position=metrics.position,
                    current_speed=metrics.current_speed,
                    max_speed=metrics.max_speed,
                    avg_speed=metrics.avg_speed,
                    speed_acceleration=metrics.speed_acceleration,
                    speed_deceleration=metrics.speed_deceleration,
                    total_distance=metrics.total_distance,
                    movement_efficiency=metrics.movement_efficiency,
                    direction_changes=metrics.direction_changes,
                    movement_smoothness=metrics.movement_smoothness,
                    speed_consistency=metrics.speed_consistency,
                    acceleration_consistency=metrics.acceleration_consistency,
                    fatigue_level=metrics.fatigue_level,
                    energy_usage=metrics.energy_usage,
                    offensive_zone_speed=metrics.offensive_zone_speed,
                    defensive_zone_speed=metrics.defensive_zone_speed,
                    neutral_zone_speed=metrics.neutral_zone_speed
                ))
            
            return stats
        
        @self.app.get("/api/analytics/speed")
        async def get_speed_analytics():
            """Get player speed analytics"""
            return {
                "total_players": len(self.advanced_analytics_engine.player_speed_metrics),
                "speed_metrics": {str(k): asdict(v) for k, v in self.advanced_analytics_engine.player_speed_metrics.items()},
                "analytics_summary": {
                    "avg_speed": np.mean([m.avg_speed for m in self.advanced_analytics_engine.player_speed_metrics.values()]) if self.advanced_analytics_engine.player_speed_metrics else 0,
                    "max_speed": max([m.max_speed for m in self.advanced_analytics_engine.player_speed_metrics.values()]) if self.advanced_analytics_engine.player_speed_metrics else 0,
                    "speed_consistency": np.mean([m.speed_consistency for m in self.advanced_analytics_engine.player_speed_metrics.values()]) if self.advanced_analytics_engine.player_speed_metrics else 0
                }
            }
        
        @self.app.get("/api/analytics/shots")
        async def get_shot_analytics():
            """Get shot analysis data"""
            recent_shots = list(self.advanced_analytics_engine.shot_analyses)[-10:]
            
            return {
                "total_shots": len(self.advanced_analytics_engine.shot_analyses),
                "recent_shots": [asdict(shot) for shot in recent_shots],
                "analytics_summary": {
                    "avg_shot_velocity": np.mean([s.shot_velocity for s in recent_shots]) if recent_shots else 0,
                    "avg_shot_quality": np.mean([s.shot_quality for s in recent_shots]) if recent_shots else 0,
                    "avg_goal_probability": np.mean([s.goal_probability for s in recent_shots]) if recent_shots else 0
                }
            }
        
        @self.app.get("/api/analytics/formations")
        async def get_formation_analytics():
            """Get team formation analytics"""
            recent_formations = list(self.advanced_analytics_engine.team_formations)[-5:]
            
            return {
                "total_formations": len(self.advanced_analytics_engine.team_formations),
                "recent_formations": [asdict(formation) for formation in recent_formations],
                "analytics_summary": {
                    "avg_formation_stability": np.mean([f.formation_stability for f in recent_formations]) if recent_formations else 0,
                    "avg_formation_compactness": np.mean([f.formation_compactness for f in recent_formations]) if recent_formations else 0,
                    "avg_formation_balance": np.mean([f.formation_balance for f in recent_formations]) if recent_formations else 0
                }
            }
        
        @self.app.get("/api/analytics/strategy")
        async def get_strategy_analytics():
            """Get game strategy insights"""
            return {
                "strategy_insights": {k: asdict(v) for k, v in self.advanced_analytics_engine.game_strategy_insights.items()},
                "analytics_summary": {
                    "total_insights": len(self.advanced_analytics_engine.game_strategy_insights),
                    "possession_balance": 0.5,  # TODO: Calculate actual possession balance
                    "strategic_advantages": []  # TODO: Calculate actual advantages
                }
            }
        
        @self.app.get("/api/analytics/metrics")
        async def get_advanced_analytics_metrics():
            """Get advanced analytics metrics"""
            analytics_summary = self.advanced_analytics_engine.get_analytics_summary()
            
            return AdvancedAnalyticsMetrics(
                processing_time=analytics_summary.get("analytics_cache", {}).get("processing_time", 0.045),
                total_speed_metrics=analytics_summary.get("total_speed_metrics", 0),
                total_shots_analyzed=analytics_summary.get("total_shots_analyzed", 0),
                total_formations=analytics_summary.get("total_formations", 0),
                strategy_insights=analytics_summary.get("strategy_insights", 0),
                advanced_features={
                    "player_speed_tracking": True,
                    "shot_analysis": True,
                    "formation_analysis": True,
                    "strategy_insights": True,
                    "advanced_analytics": True
                },
                performance_metrics={
                    "fps": 10.0,
                    "latency": analytics_summary.get("analytics_cache", {}).get("processing_time", 0.045) * 1000,
                    "throughput": analytics_summary.get("total_shots_analyzed", 0) / 60.0
                }
            )
        
        @self.app.post("/api/video/process")
        async def start_advanced_video_processing(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
            """Start advanced video processing with comprehensive analytics"""
            try:
                # Initialize advanced video processor
                self.video_processor = HockeyVideoProcessor(
                    model_path=f"{request.model_id}.pt",
                    confidence_threshold=request.confidence_threshold
                )
                
                # Start advanced processing in background
                background_tasks.add_task(
                    self._process_advanced_video_background,
                    request.video_source,
                    request.enable_streaming,
                    request.enable_advanced_analytics
                )
                
                return {
                    "status": "started",
                    "message": "Advanced video processing started",
                    "model_id": request.model_id,
                    "video_source": request.video_source,
                    "advanced_analytics": request.enable_advanced_analytics,
                    "features": {
                        "player_speed_tracking": True,
                        "shot_analysis": True,
                        "formation_analysis": True,
                        "strategy_insights": True,
                        "advanced_analytics": True
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/video/status")
        async def get_advanced_video_processing_status():
            """Get advanced video processing status"""
            if not self.video_processor:
                return {"status": "not_started", "message": "No video processing active"}
            
            return {
                "status": "processing" if self.is_processing else "stopped",
                "is_processing": self.is_processing,
                "advanced_analytics": True,
                "analytics_summary": self.video_processor.get_analytics_summary(),
                "advanced_analytics": self.advanced_analytics_engine.get_analytics_summary()
            }
        
        @self.app.post("/api/video/stop")
        async def stop_advanced_video_processing():
            """Stop advanced video processing"""
            if self.video_processor:
                await self.video_processor.stop_processing()
                self.is_processing = False
                return {"status": "stopped", "message": "Advanced video processing stopped"}
            
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
        async def websocket_advanced_analytics(websocket: WebSocket):
            """Advanced WebSocket endpoint for real-time analytics"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send advanced analytics updates
                    if self.advanced_analytics_engine.analytics_cache:
                        await websocket.send_json({
                            "type": "advanced_analytics_update",
                            "data": self.advanced_analytics_engine.analytics_cache,
                            "speed_metrics": len(self.advanced_analytics_engine.player_speed_metrics),
                            "shot_analyses": len(self.advanced_analytics_engine.shot_analyses),
                            "formations": len(self.advanced_analytics_engine.team_formations),
                            "strategy_insights": len(self.advanced_analytics_engine.game_strategy_insights)
                        })
                    
                    await asyncio.sleep(0.1)  # 10 FPS updates
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
    
    async def _process_advanced_video_background(self, video_source: str, enable_streaming: bool, enable_advanced_analytics: bool):
        """Advanced background task for video processing"""
        try:
            self.is_processing = True
            logger.info(f"🎬 Starting advanced video processing: {video_source}")
            
            # Start advanced analytics engine
            if not self.advanced_analytics_engine.is_running:
                await self.advanced_analytics_engine.start_engine()
            
            # Process video with advanced analytics
            results = await self.video_processor.process_video_stream(video_source)
            
            logger.info(f"✅ Advanced video processing completed: {results}")
            
        except Exception as e:
            logger.error(f"❌ Advanced video processing failed: {e}")
        finally:
            self.is_processing = False
    
    async def start_service(self, host: str = "0.0.0.0", port: int = 8003):
        """Start the advanced hockey analytics API service"""
        logger.info(f"🚀 Starting Advanced Hockey Analytics API on {host}:{port}")
        
        # Start advanced analytics engine
        await self.advanced_analytics_engine.start_engine()
        
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
        """Stop the advanced hockey analytics API service"""
        logger.info("🛑 Stopping Advanced Hockey Analytics API...")
        
        # Stop advanced analytics
        await self.advanced_analytics_engine.stop_engine()
        
        # Stop video processing
        if self.video_processor:
            await self.video_processor.stop_processing()

# Create advanced FastAPI app instance
advanced_hockey_api = AdvancedHockeyAnalyticsAPI()
app = advanced_hockey_api.app

# Example usage and testing
async def test_advanced_hockey_api():
    """Test advanced hockey analytics API"""
    logger.info("🧪 Testing Advanced Hockey Analytics API...")
    
    # Start service
    await advanced_hockey_api.start_service(host="localhost", port=8003)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run service
    asyncio.run(test_advanced_hockey_api())
