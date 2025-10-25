"""
TSAI Jarvis - Advanced Analytics Engine
Phase 1.3: Advanced Analytics Implementation

This module implements advanced analytics for hockey game analysis,
including player speed tracking, shot analysis, team formation analysis,
and game strategy insights.
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import aiohttp
from aiofiles import open as aio_open

logger = logging.getLogger(__name__)

@dataclass
class PlayerSpeedMetrics:
    """Advanced player speed and movement analytics"""
    player_id: int
    name: str
    team: str
    position: str
    
    # Speed analytics
    current_speed: float  # km/h
    max_speed: float  # km/h
    avg_speed: float  # km/h
    speed_acceleration: float  # m/s²
    speed_deceleration: float  # m/s²
    
    # Movement analytics
    total_distance: float  # meters
    distance_per_period: Dict[int, float]  # distance by period
    movement_efficiency: float  # 0-1 score
    direction_changes: int  # number of direction changes
    movement_smoothness: float  # 0-1 score
    
    # Performance metrics
    speed_consistency: float  # 0-1 score
    acceleration_consistency: float  # 0-1 score
    fatigue_level: float  # 0-1 score
    energy_usage: float  # 0-1 score
    
    # Zone-specific speed
    offensive_zone_speed: float
    defensive_zone_speed: float
    neutral_zone_speed: float
    
    last_updated: datetime

@dataclass
class ShotAnalysis:
    """Advanced shot analysis and quality metrics"""
    shot_id: str
    player_id: int
    player_name: str
    team: str
    
    # Shot details
    shot_type: str  # "wrist", "slap", "snap", "backhand", "deflection"
    shot_location: Tuple[float, float]  # x, y coordinates
    goal_location: Tuple[float, float]  # goal coordinates
    shot_distance: float  # meters from goal
    shot_angle: float  # degrees from goal line
    
    # Shot quality metrics
    shot_velocity: float  # km/h
    shot_accuracy: float  # 0-1 score
    shot_power: float  # 0-1 score
    shot_quality: float  # 0-1 overall quality score
    
    # Context analysis
    game_situation: str  # "even_strength", "power_play", "penalty_kill"
    pressure_level: float  # 0-1 score
    time_on_ice: float  # seconds before shot
    player_fatigue: float  # 0-1 score
    
    # Outcome prediction
    goal_probability: float  # 0-1 probability
    save_probability: float  # 0-1 probability
    miss_probability: float  # 0-1 probability
    
    # Advanced metrics
    shot_trajectory: List[Tuple[float, float]]  # trajectory points
    shot_spin: float  # spin rate
    shot_release_time: float  # seconds to release
    
    timestamp: datetime
    confidence: float

@dataclass
class TeamFormation:
    """Team formation analysis and tracking"""
    team: str
    formation_type: str  # "offensive", "defensive", "neutral", "power_play", "penalty_kill"
    
    # Formation structure
    forward_positions: List[Tuple[float, float]]  # forward positions
    defense_positions: List[Tuple[float, float]]  # defense positions
    goalie_position: Tuple[float, float]  # goalie position
    
    # Formation metrics
    formation_width: float  # meters
    formation_depth: float  # meters
    formation_compactness: float  # 0-1 score
    formation_balance: float  # 0-1 score
    
    # Player spacing
    avg_player_distance: float  # meters
    min_player_distance: float  # meters
    max_player_distance: float  # meters
    
    # Formation effectiveness
    formation_stability: float  # 0-1 score
    formation_flexibility: float  # 0-1 score
    formation_coverage: float  # 0-1 score
    
    timestamp: datetime
    confidence: float

@dataclass
class GameStrategyInsights:
    """Advanced game strategy and possession analytics"""
    team: str
    period: int
    time_remaining: str
    
    # Possession analytics
    possession_time: float  # seconds
    possession_percentage: float  # 0-100%
    possession_quality: float  # 0-1 score
    possession_efficiency: float  # 0-1 score
    
    # Zone control
    offensive_zone_time: float  # seconds
    defensive_zone_time: float  # seconds
    neutral_zone_time: float  # seconds
    zone_control_balance: float  # 0-1 score
    
    # Shot generation
    shots_generated: int
    shot_quality_avg: float  # 0-1 score
    shot_velocity_avg: float  # km/h
    shot_accuracy_avg: float  # 0-1 score
    
    # Defensive metrics
    shots_allowed: int
    shot_quality_allowed: float  # 0-1 score
    defensive_pressure: float  # 0-1 score
    defensive_efficiency: float  # 0-1 score
    
    # Transition analytics
    breakouts_successful: int
    breakouts_attempted: int
    breakout_success_rate: float  # 0-1 score
    forecheck_pressure: float  # 0-1 score
    
    # Special teams
    power_play_efficiency: float  # 0-1 score
    penalty_kill_efficiency: float  # 0-1 score
    special_teams_advantage: float  # 0-1 score
    
    # Advanced insights
    momentum_swings: int
    pressure_peaks: int
    strategic_advantages: List[str]
    tactical_recommendations: List[str]
    
    timestamp: datetime

class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine for comprehensive hockey game analysis
    """
    
    def __init__(self, websocket_port: int = 8766):
        self.websocket_port = websocket_port
        self.is_running = False
        self.clients = set()
        
        # Advanced analytics data
        self.player_speed_metrics = {}
        self.shot_analyses = deque(maxlen=1000)
        self.team_formations = deque(maxlen=500)
        self.game_strategy_insights = {}
        
        # Advanced tracking
        self.player_speed_history = defaultdict(list)
        self.shot_trajectories = defaultdict(list)
        self.formation_history = defaultdict(list)
        self.possession_tracking = defaultdict(list)
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.analytics_cache = {}
        
        # Advanced analyzers
        self.speed_analyzer = SpeedAnalyzer()
        self.shot_analyzer = ShotAnalyzer()
        self.formation_analyzer = FormationAnalyzer()
        self.strategy_analyzer = StrategyAnalyzer()
        
    async def start_engine(self):
        """Start the advanced analytics engine"""
        logger.info("🚀 Starting Advanced Analytics Engine...")
        
        self.is_running = True
        
        # Start WebSocket server
        start_server = websockets.serve(
            self.handle_client, 
            "localhost", 
            self.websocket_port
        )
        
        await start_server
        logger.info(f"📡 Advanced Analytics Engine running on port {self.websocket_port}")
        
        # Start background analytics processing
        asyncio.create_task(self._advanced_analytics_loop())
        asyncio.create_task(self._speed_analysis_loop())
        asyncio.create_task(self._shot_analysis_loop())
        asyncio.create_task(self._formation_analysis_loop())
        asyncio.create_task(self._strategy_analysis_loop())
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"📱 Client connected: {client_addr}")
        
        try:
            # Send initial advanced analytics state
            await self._send_advanced_analytics_state(websocket)
            
            # Keep connection alive and send updates
            while self.is_running:
                await asyncio.sleep(0.1)  # 10 FPS updates
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📱 Client disconnected: {client_addr}")
        finally:
            self.clients.remove(websocket)
    
    async def _send_advanced_analytics_state(self, websocket):
        """Send advanced analytics state to client"""
        try:
            advanced_state = {
                "type": "advanced_analytics_state",
                "timestamp": datetime.now().isoformat(),
                "player_speed_metrics": {str(k): asdict(v) for k, v in self.player_speed_metrics.items()},
                "shot_analyses": [asdict(shot) for shot in list(self.shot_analyses)[-10:]],
                "team_formations": [asdict(formation) for formation in list(self.team_formations)[-5:]],
                "game_strategy_insights": {k: asdict(v) for k, v in self.game_strategy_insights.items()},
                "analytics": self.analytics_cache
            }
            
            await websocket.send(json.dumps(advanced_state))
            
        except Exception as e:
            logger.error(f"❌ Failed to send advanced analytics state: {e}")
    
    async def _advanced_analytics_loop(self):
        """Main advanced analytics processing loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process advanced analytics
                await self._update_analytics_cache()
                
                # Broadcast advanced updates
                await self._broadcast_advanced_updates()
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Sleep for next iteration
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Advanced analytics processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _speed_analysis_loop(self):
        """Player speed analysis loop"""
        while self.is_running:
            try:
                # Analyze player speeds
                await self.speed_analyzer.analyze_player_speeds(self.player_speed_history)
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"❌ Speed analysis error: {e}")
                await asyncio.sleep(5.0)
    
    async def _shot_analysis_loop(self):
        """Shot analysis loop"""
        while self.is_running:
            try:
                # Analyze shots
                await self.shot_analyzer.analyze_shots(self.shot_trajectories)
                await asyncio.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Shot analysis error: {e}")
                await asyncio.sleep(5.0)
    
    async def _formation_analysis_loop(self):
        """Formation analysis loop"""
        while self.is_running:
            try:
                # Analyze team formations
                await self.formation_analyzer.analyze_formations(self.formation_history)
                await asyncio.sleep(3.0)  # Update every 3 seconds
                
            except Exception as e:
                logger.error(f"❌ Formation analysis error: {e}")
                await asyncio.sleep(5.0)
    
    async def _strategy_analysis_loop(self):
        """Strategy analysis loop"""
        while self.is_running:
            try:
                # Analyze game strategy
                await self.strategy_analyzer.analyze_strategy(self.possession_tracking)
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Strategy analysis error: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_analytics_cache(self):
        """Update advanced analytics cache"""
        current_time = datetime.now()
        
        # Calculate advanced metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        self.analytics_cache = {
            "timestamp": current_time.isoformat(),
            "processing_time": avg_processing_time,
            "total_speed_metrics": len(self.player_speed_metrics),
            "total_shots_analyzed": len(self.shot_analyses),
            "total_formations": len(self.team_formations),
            "strategy_insights": len(self.game_strategy_insights),
            "advanced_features": {
                "player_speed_tracking": True,
                "shot_analysis": True,
                "formation_analysis": True,
                "strategy_insights": True,
                "advanced_analytics": True
            },
            "performance_metrics": {
                "fps": 10.0,
                "latency": avg_processing_time * 1000,
                "throughput": len(self.shot_analyses) / 60.0
            }
        }
    
    async def _broadcast_advanced_updates(self):
        """Broadcast advanced updates to all clients"""
        if not self.clients:
            return
        
        try:
            # Prepare advanced update data
            update_data = {
                "type": "advanced_analytics_update",
                "timestamp": datetime.now().isoformat(),
                "player_speed_metrics": {str(k): asdict(v) for k, v in self.player_speed_metrics.items()},
                "shot_analyses": [asdict(shot) for shot in list(self.shot_analyses)[-5:]],
                "team_formations": [asdict(formation) for formation in list(self.team_formations)[-3:]],
                "game_strategy_insights": {k: asdict(v) for k, v in self.game_strategy_insights.items()},
                "analytics": self.analytics_cache
            }
            
            # Broadcast to all clients
            message = json.dumps(update_data)
            disconnected_clients = set()
            
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast advanced updates: {e}")
    
    async def add_player_speed_data(self, player_id: int, speed: float, acceleration: float, 
                                  position: Tuple[float, float], timestamp: datetime = None):
        """Add player speed data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.player_speed_history[player_id].append({
            'speed': speed,
            'acceleration': acceleration,
            'position': position,
            'timestamp': timestamp
        })
        
        # Keep only recent data
        if len(self.player_speed_history[player_id]) > 100:
            self.player_speed_history[player_id] = self.player_speed_history[player_id][-100:]
    
    async def add_shot_data(self, player_id: int, shot_location: Tuple[float, float], 
                          shot_velocity: float, shot_angle: float, timestamp: datetime = None):
        """Add shot data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        shot_id = f"shot_{int(time.time())}_{player_id}"
        
        shot_analysis = ShotAnalysis(
            shot_id=shot_id,
            player_id=player_id,
            player_name=f"Player {player_id}",
            team="home",  # TODO: Determine actual team
            shot_type="wrist",  # TODO: Determine shot type
            shot_location=shot_location,
            goal_location=(200.0, 100.0),  # TODO: Get actual goal location
            shot_distance=self._calculate_distance(shot_location, (200.0, 100.0)),
            shot_angle=shot_angle,
            shot_velocity=shot_velocity,
            shot_accuracy=0.8,  # TODO: Calculate actual accuracy
            shot_power=min(shot_velocity / 100.0, 1.0),
            shot_quality=0.7,  # TODO: Calculate actual quality
            game_situation="even_strength",
            pressure_level=0.5,
            time_on_ice=0.0,
            player_fatigue=0.0,
            goal_probability=0.3,
            save_probability=0.6,
            miss_probability=0.1,
            shot_trajectory=[shot_location],
            shot_spin=0.0,
            shot_release_time=0.5,
            timestamp=timestamp,
            confidence=0.8
        )
        
        self.shot_analyses.append(shot_analysis)
        logger.info(f"🎯 Shot analysis added: {shot_analysis.description}")
    
    async def add_formation_data(self, team: str, player_positions: List[Tuple[float, float]], 
                               formation_type: str, timestamp: datetime = None):
        """Add team formation data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        formation = TeamFormation(
            team=team,
            formation_type=formation_type,
            forward_positions=player_positions[:3] if len(player_positions) >= 3 else player_positions,
            defense_positions=player_positions[3:5] if len(player_positions) >= 5 else [],
            goalie_position=player_positions[5] if len(player_positions) >= 6 else (0, 0),
            formation_width=self._calculate_formation_width(player_positions),
            formation_depth=self._calculate_formation_depth(player_positions),
            formation_compactness=self._calculate_formation_compactness(player_positions),
            formation_balance=self._calculate_formation_balance(player_positions),
            avg_player_distance=self._calculate_avg_player_distance(player_positions),
            min_player_distance=self._calculate_min_player_distance(player_positions),
            max_player_distance=self._calculate_max_player_distance(player_positions),
            formation_stability=0.8,  # TODO: Calculate actual stability
            formation_flexibility=0.7,  # TODO: Calculate actual flexibility
            formation_coverage=0.9,  # TODO: Calculate actual coverage
            timestamp=timestamp,
            confidence=0.8
        )
        
        self.team_formations.append(formation)
        logger.info(f"🏒 Formation analysis added: {team} {formation_type}")
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions"""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _calculate_formation_width(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate formation width"""
        if len(positions) < 2:
            return 0.0
        
        x_coords = [pos[0] for pos in positions]
        return max(x_coords) - min(x_coords)
    
    def _calculate_formation_depth(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate formation depth"""
        if len(positions) < 2:
            return 0.0
        
        y_coords = [pos[1] for pos in positions]
        return max(y_coords) - min(y_coords)
    
    def _calculate_formation_compactness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate formation compactness (0-1 score)"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance from center
        center_x = np.mean([pos[0] for pos in positions])
        center_y = np.mean([pos[1] for pos in positions])
        
        distances = [self._calculate_distance(pos, (center_x, center_y)) for pos in positions]
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 (closer to 0 = more compact)
        return max(0, 1 - (avg_distance / 50.0))  # Assuming max distance of 50m
    
    def _calculate_formation_balance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate formation balance (0-1 score)"""
        if len(positions) < 3:
            return 0.0
        
        # Calculate balance based on distribution
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_balance = 1 - (np.std(x_coords) / np.mean(x_coords)) if np.mean(x_coords) > 0 else 0
        y_balance = 1 - (np.std(y_coords) / np.mean(y_coords)) if np.mean(y_coords) > 0 else 0
        
        return (x_balance + y_balance) / 2
    
    def _calculate_avg_player_distance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate average distance between players"""
        if len(positions) < 2:
            return 0.0
        
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self._calculate_distance(positions[i], positions[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_min_player_distance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance between players"""
        if len(positions) < 2:
            return 0.0
        
        min_distance = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self._calculate_distance(positions[i], positions[j])
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_max_player_distance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate maximum distance between players"""
        if len(positions) < 2:
            return 0.0
        
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self._calculate_distance(positions[i], positions[j])
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    async def stop_engine(self):
        """Stop the advanced analytics engine"""
        self.is_running = False
        logger.info("🛑 Advanced Analytics Engine stopped")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get advanced analytics summary"""
        return {
            "engine_status": "running" if self.is_running else "stopped",
            "connected_clients": len(self.clients),
            "total_speed_metrics": len(self.player_speed_metrics),
            "total_shots_analyzed": len(self.shot_analyses),
            "total_formations": len(self.team_formations),
            "strategy_insights": len(self.game_strategy_insights),
            "analytics_cache": self.analytics_cache,
            "performance": {
                "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0.0
            }
        }

# Supporting analyzer classes
class SpeedAnalyzer:
    """Analyzes player speed and movement patterns"""
    
    def __init__(self):
        self.speed_history = defaultdict(list)
    
    async def analyze_player_speeds(self, player_speed_history: Dict[int, List[Dict]]):
        """Analyze player speed patterns"""
        for player_id, speed_data in player_speed_history.items():
            if len(speed_data) < 3:
                continue
            
            # Analyze speed patterns
            speeds = [data['speed'] for data in speed_data]
            accelerations = [data['acceleration'] for data in speed_data]
            
            # Calculate metrics
            max_speed = max(speeds)
            avg_speed = np.mean(speeds)
            speed_consistency = 1 - (np.std(speeds) / avg_speed) if avg_speed > 0 else 0
            
            logger.info(f"🏃 Player {player_id}: Max Speed {max_speed:.1f} km/h, Avg {avg_speed:.1f} km/h, Consistency {speed_consistency:.2f}")

class ShotAnalyzer:
    """Analyzes shot quality and patterns"""
    
    def __init__(self):
        self.shot_history = defaultdict(list)
    
    async def analyze_shots(self, shot_trajectories: Dict[str, List[Tuple[float, float]]]):
        """Analyze shot patterns and quality"""
        for shot_id, trajectory in shot_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Analyze shot trajectory
            start_pos = trajectory[0]
            end_pos = trajectory[-1]
            
            # Calculate shot metrics
            distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            
            logger.info(f"🎯 Shot {shot_id}: Distance {distance:.1f}m, Angle {np.degrees(angle):.1f}°")

class FormationAnalyzer:
    """Analyzes team formations and positioning"""
    
    def __init__(self):
        self.formation_history = defaultdict(list)
    
    async def analyze_formations(self, formation_history: Dict[str, List[TeamFormation]]):
        """Analyze team formation patterns"""
        for team, formations in formation_history.items():
            if not formations:
                continue
            
            # Analyze formation patterns
            formation_types = [f.formation_type for f in formations]
            formation_stability = np.mean([f.formation_stability for f in formations])
            
            logger.info(f"🏒 Team {team}: Formation Stability {formation_stability:.2f}")

class StrategyAnalyzer:
    """Analyzes game strategy and possession patterns"""
    
    def __init__(self):
        self.strategy_history = defaultdict(list)
    
    async def analyze_strategy(self, possession_tracking: Dict[str, List[Dict]]):
        """Analyze game strategy and possession"""
        for team, possession_data in possession_tracking.items():
            if not possession_data:
                continue
            
            # Analyze possession patterns
            possession_times = [data.get('possession_time', 0) for data in possession_data]
            avg_possession = np.mean(possession_times)
            
            logger.info(f"📊 Team {team}: Avg Possession {avg_possession:.1f}s")

# Example usage and testing
async def test_advanced_analytics_engine():
    """Test advanced analytics engine"""
    logger.info("🧪 Testing Advanced Analytics Engine...")
    
    # Initialize engine
    engine = AdvancedAnalyticsEngine()
    
    # Start engine
    await engine.start_engine()
    
    # Simulate advanced analytics data
    for i in range(20):
        await engine.add_player_speed_data(1, 25.0 + i, 2.0, (100 + i*2, 200 + i*1.5))
        await engine.add_shot_data(1, (150 + i*3, 180 + i*2), 80.0 + i*2, 15.0 + i)
        await engine.add_formation_data("home", [(100, 200), (120, 180), (140, 160)], "offensive")
        await asyncio.sleep(0.1)
    
    # Get analytics summary
    summary = engine.get_analytics_summary()
    logger.info(f"📊 Advanced analytics summary: {summary}")
    
    # Stop engine
    await engine.stop_engine()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_advanced_analytics_engine())
