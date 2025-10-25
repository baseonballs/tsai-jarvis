"""
TSAI Jarvis - Live Analytics Service for Hockey Analytics
Phase 1.2: Real-time Analytics and Event Detection

This module provides real-time analytics processing for live hockey games,
including player tracking, event detection, and performance metrics.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import websockets
import aiohttp
from aiofiles import open as aio_open

logger = logging.getLogger(__name__)

@dataclass
class PlayerMetrics:
    """Real-time player performance metrics"""
    player_id: int
    name: str
    team: str
    position: str
    speed: float  # km/h
    distance_covered: float  # meters
    time_on_ice: float  # seconds
    shots: int
    goals: int
    assists: int
    plus_minus: int
    faceoff_wins: int
    faceoff_attempts: int
    hits: int
    blocked_shots: int
    last_updated: datetime

@dataclass
class GameEvent:
    """Live game event with real-time detection"""
    event_id: str
    event_type: str  # "goal", "save", "penalty", "faceoff", "hit", "shot"
    timestamp: datetime
    period: int
    time_remaining: str
    player_id: int
    player_name: str
    team: str
    location: Tuple[float, float]  # x, y coordinates on ice
    confidence: float
    description: str
    video_timestamp: float  # seconds into video

@dataclass
class TeamStats:
    """Real-time team statistics"""
    team_name: str
    goals: int
    shots_on_goal: int
    shots_blocked: int
    hits: int
    faceoff_wins: int
    faceoff_attempts: int
    power_play_goals: int
    power_play_attempts: int
    penalty_kill_goals: int
    penalty_kill_attempts: int
    possession_time: float  # seconds
    zone_time: Dict[str, float]  # offensive, defensive, neutral

class LiveAnalyticsService:
    """
    Real-time analytics service for live hockey games
    """
    
    def __init__(self, websocket_port: int = 8765):
        self.websocket_port = websocket_port
        self.is_running = False
        self.clients = set()
        
        # Game state
        self.current_game = None
        self.game_events = []
        self.player_metrics = {}
        self.team_stats = {}
        
        # Real-time tracking
        self.player_positions = defaultdict(list)
        self.speed_calculations = defaultdict(list)
        self.event_detection_buffer = deque(maxlen=100)
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.detection_accuracy = 0.0
        
        # Analytics cache
        self.analytics_cache = {}
        self.cache_ttl = 1.0  # 1 second cache TTL
        
    async def start_service(self):
        """Start the live analytics service"""
        logger.info("🚀 Starting Live Analytics Service...")
        
        self.is_running = True
        
        # Start WebSocket server
        start_server = websockets.serve(
            self.handle_client, 
            "localhost", 
            self.websocket_port
        )
        
        await start_server
        logger.info(f"📡 Live Analytics Service running on port {self.websocket_port}")
        
        # Start background analytics processing
        asyncio.create_task(self._process_analytics_loop())
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"📱 Client connected: {client_addr}")
        
        try:
            # Send initial game state
            await self._send_game_state(websocket)
            
            # Keep connection alive and send updates
            while self.is_running:
                await asyncio.sleep(0.1)  # 10 FPS updates
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📱 Client disconnected: {client_addr}")
        finally:
            self.clients.remove(websocket)
    
    async def _send_game_state(self, websocket):
        """Send current game state to client"""
        try:
            game_state = {
                "type": "game_state",
                "timestamp": datetime.now().isoformat(),
                "game": self.current_game,
                "events": [asdict(event) for event in self.game_events[-10:]],  # Last 10 events
                "player_metrics": {str(k): asdict(v) for k, v in self.player_metrics.items()},
                "team_stats": {k: asdict(v) for k, v in self.team_stats.items()},
                "analytics": self.analytics_cache
            }
            
            await websocket.send(json.dumps(game_state))
            
        except Exception as e:
            logger.error(f"❌ Failed to send game state: {e}")
    
    async def _process_analytics_loop(self):
        """Main analytics processing loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process real-time analytics
                await self._update_player_metrics()
                await self._detect_live_events()
                await self._calculate_team_stats()
                await self._update_analytics_cache()
                
                # Broadcast updates to all clients
                await self._broadcast_updates()
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Sleep for next iteration (target 10 FPS)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Analytics processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_player_metrics(self):
        """Update real-time player metrics"""
        current_time = datetime.now()
        
        for player_id, positions in self.player_positions.items():
            if len(positions) < 2:
                continue
                
            # Calculate speed from recent positions
            recent_positions = positions[-10:]  # Last 10 positions
            if len(recent_positions) >= 2:
                speed = self._calculate_speed(recent_positions)
                
                # Update player metrics
                if player_id not in self.player_metrics:
                    self.player_metrics[player_id] = PlayerMetrics(
                        player_id=player_id,
                        name=f"Player {player_id}",
                        team="unknown",
                        position="unknown",
                        speed=0.0,
                        distance_covered=0.0,
                        time_on_ice=0.0,
                        shots=0,
                        goals=0,
                        assists=0,
                        plus_minus=0,
                        faceoff_wins=0,
                        faceoff_attempts=0,
                        hits=0,
                        blocked_shots=0,
                        last_updated=current_time
                    )
                
                # Update speed and distance
                self.player_metrics[player_id].speed = speed
                self.player_metrics[player_id].distance_covered += self._calculate_distance(
                    recent_positions[-2], recent_positions[-1]
                )
                self.player_metrics[player_id].time_on_ice += 0.1  # 0.1 second intervals
                self.player_metrics[player_id].last_updated = current_time
    
    def _calculate_speed(self, positions: List[Tuple[float, float, datetime]]) -> float:
        """Calculate player speed from position history"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate total distance and time
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            distance = self._calculate_distance(prev_pos, curr_pos)
            time_diff = (curr_pos[2] - prev_pos[2]).total_seconds()
            
            total_distance += distance
            total_time += time_diff
        
        if total_time > 0:
            # Convert to km/h (assuming positions are in meters)
            speed_mps = total_distance / total_time
            speed_kmh = speed_mps * 3.6
            return speed_kmh
        
        return 0.0
    
    def _calculate_distance(self, pos1: Tuple[float, float, datetime], pos2: Tuple[float, float, datetime]) -> float:
        """Calculate distance between two positions"""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    async def _detect_live_events(self):
        """Detect live game events from player positions and movements"""
        current_time = datetime.now()
        
        # Analyze recent player movements for events
        for player_id, positions in self.player_positions.items():
            if len(positions) < 5:  # Need enough history
                continue
            
            recent_positions = positions[-5:]
            
            # Detect potential goal (sudden movement toward goal)
            if self._detect_goal_attempt(recent_positions):
                event = GameEvent(
                    event_id=f"goal_attempt_{int(time.time())}",
                    event_type="shot",
                    timestamp=current_time,
                    period=1,  # TODO: Get actual period
                    time_remaining="15:00",  # TODO: Get actual time
                    player_id=player_id,
                    player_name=f"Player {player_id}",
                    team="home",  # TODO: Determine team
                    location=recent_positions[-1][:2],
                    confidence=0.8,
                    description=f"Shot attempt by Player {player_id}",
                    video_timestamp=time.time()
                )
                
                self.game_events.append(event)
                logger.info(f"🎯 Shot detected: {event.description}")
            
            # Detect potential hit (sudden direction change)
            if self._detect_hit(recent_positions):
                event = GameEvent(
                    event_id=f"hit_{int(time.time())}",
                    event_type="hit",
                    timestamp=current_time,
                    period=1,
                    time_remaining="15:00",
                    player_id=player_id,
                    player_name=f"Player {player_id}",
                    team="home",
                    location=recent_positions[-1][:2],
                    confidence=0.7,
                    description=f"Hit by Player {player_id}",
                    video_timestamp=time.time()
                )
                
                self.game_events.append(event)
                logger.info(f"💥 Hit detected: {event.description}")
    
    def _detect_goal_attempt(self, positions: List[Tuple[float, float, datetime]]) -> bool:
        """Detect potential goal attempt from player movement"""
        if len(positions) < 3:
            return False
        
        # Check for movement toward goal area
        # Simple heuristic: rapid movement in one direction
        recent_positions = positions[-3:]
        
        # Calculate movement vector
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        
        movement_x = end_pos[0] - start_pos[0]
        movement_y = end_pos[1] - start_pos[1]
        movement_magnitude = np.sqrt(movement_x**2 + movement_y**2)
        
        # Check if movement is significant and in goal direction
        # (This is a simplified heuristic - real implementation would be more sophisticated)
        return movement_magnitude > 20  # Threshold for significant movement
    
    def _detect_hit(self, positions: List[Tuple[float, float, datetime]]) -> bool:
        """Detect potential hit from player movement"""
        if len(positions) < 3:
            return False
        
        # Check for sudden direction change
        recent_positions = positions[-3:]
        
        # Calculate direction changes
        for i in range(1, len(recent_positions) - 1):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            next_pos = recent_positions[i+1]
            
            # Calculate direction vectors
            dir1 = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            dir2 = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
            
            # Calculate angle between directions
            if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                # Sudden direction change (angle > 90 degrees)
                if angle > np.pi / 2:
                    return True
        
        return False
    
    async def _calculate_team_stats(self):
        """Calculate real-time team statistics"""
        # Initialize team stats if not exists
        if not self.team_stats:
            self.team_stats = {
                "home": TeamStats(
                    team_name="Home Team",
                    goals=0, shots_on_goal=0, shots_blocked=0,
                    hits=0, faceoff_wins=0, faceoff_attempts=0,
                    power_play_goals=0, power_play_attempts=0,
                    penalty_kill_goals=0, penalty_kill_attempts=0,
                    possession_time=0.0,
                    zone_time={"offensive": 0.0, "defensive": 0.0, "neutral": 0.0}
                ),
                "away": TeamStats(
                    team_name="Away Team",
                    goals=0, shots_on_goal=0, shots_blocked=0,
                    hits=0, faceoff_wins=0, faceoff_attempts=0,
                    power_play_goals=0, power_play_attempts=0,
                    penalty_kill_goals=0, penalty_kill_attempts=0,
                    possession_time=0.0,
                    zone_time={"offensive": 0.0, "defensive": 0.0, "neutral": 0.0}
                )
            }
        
        # Update team stats based on recent events
        for event in self.game_events[-10:]:  # Last 10 events
            team = event.team
            if team in self.team_stats:
                if event.event_type == "goal":
                    self.team_stats[team].goals += 1
                elif event.event_type == "shot":
                    self.team_stats[team].shots_on_goal += 1
                elif event.event_type == "hit":
                    self.team_stats[team].hits += 1
    
    async def _update_analytics_cache(self):
        """Update analytics cache with current metrics"""
        current_time = datetime.now()
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        # Calculate detection accuracy (simplified)
        total_events = len(self.game_events)
        recent_events = len([e for e in self.game_events if (current_time - e.timestamp).total_seconds() < 60])
        
        self.analytics_cache = {
            "timestamp": current_time.isoformat(),
            "processing_time": avg_processing_time,
            "detection_accuracy": self.detection_accuracy,
            "total_events": total_events,
            "recent_events": recent_events,
            "active_players": len(self.player_metrics),
            "game_duration": "15:00",  # TODO: Calculate actual game duration
            "performance_metrics": {
                "fps": 10.0,  # Target FPS
                "latency": avg_processing_time * 1000,  # ms
                "throughput": recent_events / 60.0  # events per second
            }
        }
    
    async def _broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        if not self.clients:
            return
        
        try:
            # Prepare update data
            update_data = {
                "type": "analytics_update",
                "timestamp": datetime.now().isoformat(),
                "events": [asdict(event) for event in self.game_events[-5:]],  # Last 5 events
                "player_metrics": {str(k): asdict(v) for k, v in self.player_metrics.items()},
                "team_stats": {k: asdict(v) for k, v in self.team_stats.items()},
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
            logger.error(f"❌ Failed to broadcast updates: {e}")
    
    async def add_player_position(self, player_id: int, x: float, y: float, timestamp: datetime = None):
        """Add player position for tracking"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.player_positions[player_id].append((x, y, timestamp))
        
        # Keep only recent positions (last 100)
        if len(self.player_positions[player_id]) > 100:
            self.player_positions[player_id] = self.player_positions[player_id][-100:]
    
    async def add_game_event(self, event: GameEvent):
        """Add game event to analytics"""
        self.game_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.game_events) > 1000:
            self.game_events = self.game_events[-1000:]
    
    async def stop_service(self):
        """Stop the live analytics service"""
        self.is_running = False
        logger.info("🛑 Live Analytics Service stopped")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get current analytics summary"""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "connected_clients": len(self.clients),
            "total_events": len(self.game_events),
            "active_players": len(self.player_metrics),
            "analytics_cache": self.analytics_cache,
            "performance": {
                "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0.0,
                "detection_accuracy": self.detection_accuracy
            }
        }

# Example usage and testing
async def test_live_analytics_service():
    """Test live analytics service"""
    logger.info("🧪 Testing Live Analytics Service...")
    
    # Initialize service
    service = LiveAnalyticsService()
    
    # Start service
    await service.start_service()
    
    # Simulate some player movements
    for i in range(10):
        await service.add_player_position(1, 100 + i*5, 200 + i*3)
        await service.add_player_position(2, 200 + i*3, 150 + i*5)
        await asyncio.sleep(0.1)
    
    # Get analytics summary
    summary = service.get_analytics_summary()
    logger.info(f"📊 Analytics summary: {summary}")
    
    # Stop service
    await service.stop_service()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_live_analytics_service())
