"""
TSAI Jarvis - Enterprise AI API Service
Phase 2.1: Advanced AI Models API Integration

This module provides enterprise-grade REST API endpoints for advanced AI models,
including player performance prediction, team strategy optimization, game outcome prediction,
and injury risk assessment.
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

# Import our enterprise AI modules
from enterprise_ai_engine import (
    EnterpriseAIEngine,
    PlayerPerformancePrediction,
    TeamStrategyOptimization,
    GameOutcomePrediction,
    InjuryRiskAssessment
)
from video_processor import HockeyVideoProcessor, PlayerDetection, GameEvent
from yolo_model_manager import YOLOModelManager

logger = logging.getLogger(__name__)

# Enterprise AI Pydantic models for API
class PlayerPerformanceRequest(BaseModel):
    player_id: int
    player_name: str
    team: str
    position: str
    age: int
    recent_form: float
    fatigue_level: float
    home_advantage: float
    matchup_advantage: float
    season_average: float
    career_average: float
    improvement_trend: float
    team_strength: float

class TeamStrategyRequest(BaseModel):
    team: str
    opponent: str
    game_context: str = "regular"
    team_form: float = 0.5
    opponent_form: float = 0.5
    home_advantage: float = 0.1
    historical_advantage: float = 0.0
    team_chemistry: float = 0.5
    opponent_weakness: float = 0.5
    tactical_advantage: float = 0.5

class GameOutcomeRequest(BaseModel):
    home_team: str
    away_team: str
    game_date: datetime
    home_team_form: float = 0.5
    away_team_form: float = 0.5
    home_advantage: float = 0.1
    head_to_head_history: float = 0.0
    injury_impact: float = 0.0
    rest_advantage: float = 0.0
    team_strength_difference: float = 0.0
    motivation_factor: float = 0.5

class InjuryRiskRequest(BaseModel):
    player_id: int
    player_name: str
    team: str
    position: str
    age: int
    fatigue_level: float
    recent_workload: float
    injury_history: float
    position_risk: float
    recent_performance: float
    training_intensity: float
    recovery_time: float
    stress_level: float
    sleep_quality: float

class EnterpriseAIMetrics(BaseModel):
    active_models: int
    total_predictions: int
    model_accuracy: Dict[str, float]
    ai_features: Dict[str, bool]
    performance_metrics: Dict[str, float]

class EnterpriseHockeyAIAPI:
    """
    Enterprise FastAPI service for advanced AI models
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Enterprise AI API",
            description="Enterprise-grade AI models for comprehensive hockey analysis",
            version="2.1.0"
        )
        
        # Initialize enterprise services
        self.video_processor = None
        self.enterprise_ai_engine = EnterpriseAIEngine()
        self.model_manager = YOLOModelManager()
        
        # Enterprise game state
        self.current_predictions = {}
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
        """Setup enterprise AI API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Enterprise AI API",
                "version": "2.1.0",
                "status": "running",
                "phase": "2.1 - Advanced AI Models",
                "features": {
                    "player_performance_prediction": True,
                    "team_strategy_optimization": True,
                    "game_outcome_prediction": True,
                    "injury_risk_assessment": True,
                    "real_time_inference": True,
                    "model_auto_training": True
                },
                "endpoints": {
                    "health": "/health",
                    "player_performance": "/api/ai/player/performance",
                    "team_strategy": "/api/ai/team/strategy",
                    "game_outcome": "/api/ai/game/outcome",
                    "injury_risk": "/api/ai/player/injury",
                    "ai_metrics": "/api/ai/metrics",
                    "websocket": "/ws/ai"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Enterprise AI health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.1.0",
                "phase": "Advanced AI Models",
                "services": {
                    "video_processor": self.video_processor is not None,
                    "enterprise_ai_engine": self.enterprise_ai_engine.is_running,
                    "model_manager": self.model_manager is not None
                },
                "ai_models": {
                    "player_performance_model": self.enterprise_ai_engine.player_performance_model is not None,
                    "team_strategy_model": self.enterprise_ai_engine.team_strategy_model is not None,
                    "game_outcome_model": self.enterprise_ai_engine.game_outcome_model is not None,
                    "injury_risk_model": self.enterprise_ai_engine.injury_risk_model is not None
                },
                "ai_features": {
                    "player_performance_prediction": True,
                    "team_strategy_optimization": True,
                    "game_outcome_prediction": True,
                    "injury_risk_assessment": True,
                    "real_time_inference": True,
                    "model_auto_training": True
                }
            }
        
        @self.app.post("/api/ai/player/performance")
        async def predict_player_performance(request: PlayerPerformanceRequest):
            """Predict player performance using AI models"""
            try:
                # Prepare player data
                player_data = {
                    'name': request.player_name,
                    'team': request.team,
                    'position': request.position,
                    'age': request.age,
                    'recent_form': request.recent_form,
                    'fatigue_level': request.fatigue_level,
                    'home_advantage': request.home_advantage,
                    'matchup_advantage': request.matchup_advantage,
                    'season_average': request.season_average,
                    'career_average': request.career_average,
                    'improvement_trend': request.improvement_trend,
                    'team_strength': request.team_strength
                }
                
                # Get AI prediction
                prediction = await self.enterprise_ai_engine.predict_player_performance(
                    request.player_id, player_data
                )
                
                return {
                    "status": "success",
                    "prediction": asdict(prediction),
                    "message": f"Player performance predicted for {request.player_name}",
                    "confidence": prediction.overall_confidence,
                    "model_version": prediction.model_version
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/ai/team/strategy")
        async def optimize_team_strategy(request: TeamStrategyRequest):
            """Optimize team strategy using AI models"""
            try:
                # Get AI strategy optimization
                strategy = await self.enterprise_ai_engine.optimize_team_strategy(
                    request.team, request.opponent, request.game_context
                )
                
                return {
                    "status": "success",
                    "strategy": asdict(strategy),
                    "message": f"Team strategy optimized for {request.team} vs {request.opponent}",
                    "confidence": strategy.strategy_confidence,
                    "model_version": strategy.model_version
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/ai/game/outcome")
        async def predict_game_outcome(request: GameOutcomeRequest):
            """Predict game outcome using AI models"""
            try:
                # Get AI game outcome prediction
                prediction = await self.enterprise_ai_engine.predict_game_outcome(
                    request.home_team, request.away_team, request.game_date
                )
                
                return {
                    "status": "success",
                    "prediction": asdict(prediction),
                    "message": f"Game outcome predicted for {request.home_team} vs {request.away_team}",
                    "confidence": prediction.overall_confidence,
                    "model_version": prediction.model_version
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/ai/player/injury")
        async def assess_injury_risk(request: InjuryRiskRequest):
            """Assess player injury risk using AI models"""
            try:
                # Prepare injury risk data
                injury_data = {
                    'name': request.player_name,
                    'team': request.team,
                    'position': request.position,
                    'age': request.age,
                    'fatigue_level': request.fatigue_level,
                    'recent_workload': request.recent_workload,
                    'injury_history': request.injury_history,
                    'position_risk': request.position_risk,
                    'recent_performance': request.recent_performance,
                    'training_intensity': request.training_intensity,
                    'recovery_time': request.recovery_time,
                    'stress_level': request.stress_level,
                    'sleep_quality': request.sleep_quality
                }
                
                # Get AI injury risk assessment
                assessment = await self.enterprise_ai_engine.assess_injury_risk(
                    request.player_id, injury_data
                )
                
                return {
                    "status": "success",
                    "assessment": asdict(assessment),
                    "message": f"Injury risk assessed for {request.player_name}",
                    "confidence": assessment.risk_confidence,
                    "model_version": assessment.model_version
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/ai/player/predictions")
        async def get_player_predictions():
            """Get all player performance predictions"""
            return {
                "player_predictions": {str(k): asdict(v) for k, v in self.enterprise_ai_engine.player_predictions.items()},
                "total_predictions": len(self.enterprise_ai_engine.player_predictions),
                "analytics_summary": {
                    "avg_confidence": np.mean([p.overall_confidence for p in self.enterprise_ai_engine.player_predictions.values()]) if self.enterprise_ai_engine.player_predictions else 0,
                    "avg_predicted_points": np.mean([p.predicted_points for p in self.enterprise_ai_engine.player_predictions.values()]) if self.enterprise_ai_engine.player_predictions else 0,
                    "top_performers": sorted(
                        [(p.player_name, p.predicted_points) for p in self.enterprise_ai_engine.player_predictions.values()],
                        key=lambda x: x[1], reverse=True
                    )[:5]
                }
            }
        
        @self.app.get("/api/ai/team/strategies")
        async def get_team_strategies():
            """Get all team strategy optimizations"""
            return {
                "team_strategies": {str(k): asdict(v) for k, v in self.enterprise_ai_engine.team_predictions.items()},
                "total_strategies": len(self.enterprise_ai_engine.team_predictions),
                "analytics_summary": {
                    "avg_confidence": np.mean([s.strategy_confidence for s in self.enterprise_ai_engine.team_predictions.values()]) if self.enterprise_ai_engine.team_predictions else 0,
                    "avg_win_probability": np.mean([s.win_probability for s in self.enterprise_ai_engine.team_predictions.values()]) if self.enterprise_ai_engine.team_predictions else 0,
                    "strategy_distribution": {
                        "aggressive": sum(1 for s in self.enterprise_ai_engine.team_predictions.values() if s.offensive_strategy == "aggressive"),
                        "balanced": sum(1 for s in self.enterprise_ai_engine.team_predictions.values() if s.offensive_strategy == "balanced"),
                        "defensive": sum(1 for s in self.enterprise_ai_engine.team_predictions.values() if s.offensive_strategy == "defensive")
                    }
                }
            }
        
        @self.app.get("/api/ai/game/predictions")
        async def get_game_predictions():
            """Get all game outcome predictions"""
            return {
                "game_predictions": {str(k): asdict(v) for k, v in self.enterprise_ai_engine.game_predictions.items()},
                "total_predictions": len(self.enterprise_ai_engine.game_predictions),
                "analytics_summary": {
                    "avg_confidence": np.mean([g.overall_confidence for g in self.enterprise_ai_engine.game_predictions.values()]) if self.enterprise_ai_engine.game_predictions else 0,
                    "avg_home_win_probability": np.mean([g.home_win_probability for g in self.enterprise_ai_engine.game_predictions.values()]) if self.enterprise_ai_engine.game_predictions else 0,
                    "avg_predicted_goals": np.mean([g.predicted_total_goals for g in self.enterprise_ai_engine.game_predictions.values()]) if self.enterprise_ai_engine.game_predictions else 0
                }
            }
        
        @self.app.get("/api/ai/player/injury-assessments")
        async def get_injury_assessments():
            """Get all injury risk assessments"""
            return {
                "injury_assessments": {str(k): asdict(v) for k, v in self.enterprise_ai_engine.injury_assessments.items()},
                "total_assessments": len(self.enterprise_ai_engine.injury_assessments),
                "analytics_summary": {
                    "avg_risk": np.mean([i.overall_injury_risk for i in self.enterprise_ai_engine.injury_assessments.values()]) if self.enterprise_ai_engine.injury_assessments else 0,
                    "high_risk_players": [i.player_name for i in self.enterprise_ai_engine.injury_assessments.values() if i.overall_injury_risk > 0.7],
                    "risk_distribution": {
                        "low": sum(1 for i in self.enterprise_ai_engine.injury_assessments.values() if i.overall_injury_risk < 0.3),
                        "medium": sum(1 for i in self.enterprise_ai_engine.injury_assessments.values() if 0.3 <= i.overall_injury_risk < 0.7),
                        "high": sum(1 for i in self.enterprise_ai_engine.injury_assessments.values() if i.overall_injury_risk >= 0.7)
                    }
                }
            }
        
        @self.app.get("/api/ai/metrics")
        async def get_ai_metrics():
            """Get enterprise AI metrics"""
            analytics_summary = self.enterprise_ai_engine.get_analytics_summary()
            
            return EnterpriseAIMetrics(
                active_models=analytics_summary.get("active_models", 0),
                total_predictions=analytics_summary.get("total_predictions", 0),
                model_accuracy=analytics_summary.get("model_accuracy", {}),
                ai_features=analytics_summary.get("ai_features", {}),
                performance_metrics=analytics_summary.get("performance_metrics", {})
            )
        
        @self.app.get("/api/ai/models/status")
        async def get_ai_models_status():
            """Get AI models status and performance"""
            return {
                "models": {
                    "player_performance": {
                        "status": "active" if self.enterprise_ai_engine.player_performance_model else "inactive",
                        "accuracy": self.enterprise_ai_engine.model_accuracy.get("player_performance", 0.0),
                        "predictions": len(self.enterprise_ai_engine.player_predictions)
                    },
                    "team_strategy": {
                        "status": "active" if self.enterprise_ai_engine.team_strategy_model else "inactive",
                        "accuracy": self.enterprise_ai_engine.model_accuracy.get("team_strategy", 0.0),
                        "predictions": len(self.enterprise_ai_engine.team_predictions)
                    },
                    "game_outcome": {
                        "status": "active" if self.enterprise_ai_engine.game_outcome_model else "inactive",
                        "accuracy": self.enterprise_ai_engine.model_accuracy.get("game_outcome", 0.0),
                        "predictions": len(self.enterprise_ai_engine.game_predictions)
                    },
                    "injury_risk": {
                        "status": "active" if self.enterprise_ai_engine.injury_risk_model else "inactive",
                        "accuracy": self.enterprise_ai_engine.model_accuracy.get("injury_risk", 0.0),
                        "predictions": len(self.enterprise_ai_engine.injury_assessments)
                    }
                },
                "overall_performance": {
                    "avg_accuracy": np.mean(list(self.enterprise_ai_engine.model_accuracy.values())) if self.enterprise_ai_engine.model_accuracy else 0.0,
                    "total_predictions": analytics_summary.get("total_predictions", 0),
                    "active_models": analytics_summary.get("active_models", 0)
                }
            }
        
        @self.app.websocket("/ws/ai")
        async def websocket_enterprise_ai(websocket: WebSocket):
            """Enterprise AI WebSocket endpoint for real-time AI analytics"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send enterprise AI updates
                    if self.enterprise_ai_engine.ai_analytics_cache:
                        await websocket.send_json({
                            "type": "enterprise_ai_update",
                            "data": self.enterprise_ai_engine.ai_analytics_cache,
                            "player_predictions": len(self.enterprise_ai_engine.player_predictions),
                            "team_strategies": len(self.enterprise_ai_engine.team_predictions),
                            "game_predictions": len(self.enterprise_ai_engine.game_predictions),
                            "injury_assessments": len(self.enterprise_ai_engine.injury_assessments)
                        })
                    
                    await asyncio.sleep(0.1)  # 10 FPS updates
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
    
    async def start_service(self, host: str = "0.0.0.0", port: int = 8005):
        """Start the enterprise AI API service"""
        logger.info(f"🚀 Starting Enterprise AI API on {host}:{port}")
        
        # Start enterprise AI engine
        await self.enterprise_ai_engine.start_engine()
        
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
        """Stop the enterprise AI API service"""
        logger.info("🛑 Stopping Enterprise AI API...")
        
        # Stop enterprise AI
        await self.enterprise_ai_engine.stop_engine()
        
        # Stop video processing
        if self.video_processor:
            await self.video_processor.stop_processing()

# Create enterprise AI FastAPI app instance
enterprise_ai_api = EnterpriseHockeyAIAPI()
app = enterprise_ai_api.app

# Example usage and testing
async def test_enterprise_ai_api():
    """Test enterprise AI API"""
    logger.info("🧪 Testing Enterprise AI API...")
    
    # Start service
    await enterprise_ai_api.start_service(host="localhost", port=8005)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run service
    asyncio.run(test_enterprise_ai_api())
