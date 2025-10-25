"""
TSAI Jarvis - Advanced Visualization API
Phase 2.4: Advanced Visualization Implementation

This module implements advanced visualization features for hockey analytics,
including 3D ice rink visualization, player movement tracking, heat maps,
VR/AR experiences, and interactive data exploration.
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import aiohttp
from aiofiles import open as aio_open
import sqlite3
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import math

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available. Basic visualizations will be used.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Basic visualizations will be used.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for advanced visualization
class IceRink3D(BaseModel):
    """3D Ice Rink visualization model"""
    rink_id: str
    dimensions: Dict[str, float]  # length, width, radius
    zones: List[Dict[str, Any]]  # offensive, defensive, neutral zones
    features: List[Dict[str, Any]]  # goals, faceoff circles, blue lines
    created_at: datetime = Field(default_factory=datetime.now)

class PlayerMovement3D(BaseModel):
    """3D Player Movement tracking model"""
    player_id: str
    player_name: str
    position: str
    movement_data: List[Dict[str, Any]]  # x, y, z, timestamp
    trajectory: List[Dict[str, Any]]  # path, speed, acceleration
    heat_map_data: List[Dict[str, Any]]  # position density
    created_at: datetime = Field(default_factory=datetime.now)

class HeatMapVisualization(BaseModel):
    """Heat Map visualization model"""
    heatmap_id: str
    map_type: str  # player_positions, shot_locations, event_density
    data_points: List[Dict[str, Any]]  # x, y, intensity
    color_scheme: str
    intensity_range: Tuple[float, float]
    resolution: int
    created_at: datetime = Field(default_factory=datetime.now)

class VRAnalytics(BaseModel):
    """VR Analytics experience model"""
    vr_session_id: str
    user_id: str
    analytics_type: str  # game_analysis, player_tracking, strategy_review
    vr_data: Dict[str, Any]  # headset data, interactions
    immersion_level: str  # basic, intermediate, advanced
    session_duration: float
    created_at: datetime = Field(default_factory=datetime.now)

class ARFeatures(BaseModel):
    """AR Features for mobile model"""
    ar_feature_id: str
    feature_type: str  # player_overlay, stats_display, replay_analysis
    mobile_platform: str  # ios, android
    ar_data: Dict[str, Any]  # overlay data, tracking info
    user_interactions: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.now)

class InteractiveExploration(BaseModel):
    """Interactive data exploration model"""
    exploration_id: str
    data_type: str  # game_data, player_stats, team_analytics
    exploration_filters: Dict[str, Any]
    visualization_options: List[str]
    user_preferences: Dict[str, Any]
    exploration_results: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

# Advanced Visualization Engine
class AdvancedVisualizationEngine:
    """Advanced Visualization Engine for hockey analytics"""
    
    def __init__(self):
        self.app = FastAPI(
            title="TSAI Jarvis Advanced Visualization API",
            description="Advanced visualization features for hockey analytics",
            version="2.4.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.ice_rink_3d = IceRink3DService()
        self.player_movement_3d = PlayerMovement3DService()
        self.heat_map_viz = HeatMapVisualizationService()
        self.vr_analytics = VRAnalyticsService()
        self.ar_features = ARFeaturesService()
        self.interactive_exploration = InteractiveExplorationService()
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self.setup_routes()
        
        # Initialize database
        self.init_database()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "TSAI Jarvis Advanced Visualization API",
                "version": "2.4.0",
                "status": "operational",
                "phase": "2.4 - Advanced Visualization"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "ice_rink_3d": "operational",
                    "player_movement_3d": "operational",
                    "heat_map_viz": "operational",
                    "vr_analytics": "operational",
                    "ar_features": "operational",
                    "interactive_exploration": "operational"
                }
            }
        
        # 3D Ice Rink Visualization
        @self.app.post("/api/visualization/ice-rink-3d")
        async def create_ice_rink_3d(rink: IceRink3D):
            """Create 3D ice rink visualization"""
            try:
                result = await self.ice_rink_3d.create_ice_rink(rink)
                return {"status": "success", "rink_id": result["rink_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/ice-rink-3d/{rink_id}")
        async def get_ice_rink_3d(rink_id: str):
            """Get 3D ice rink visualization"""
            try:
                rink = await self.ice_rink_3d.get_ice_rink(rink_id)
                return rink
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/api/visualization/ice-rink-3d/{rink_id}/render")
        async def render_ice_rink_3d(rink_id: str, format: str = "json"):
            """Render 3D ice rink visualization"""
            try:
                render_data = await self.ice_rink_3d.render_ice_rink(rink_id, format)
                return render_data
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 3D Player Movement Tracking
        @self.app.post("/api/visualization/player-movement-3d")
        async def create_player_movement_3d(movement: PlayerMovement3D):
            """Create 3D player movement tracking"""
            try:
                result = await self.player_movement_3d.create_movement(movement)
                return {"status": "success", "movement_id": result["movement_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/player-movement-3d/{player_id}")
        async def get_player_movement_3d(player_id: str):
            """Get 3D player movement data"""
            try:
                movement = await self.player_movement_3d.get_movement(player_id)
                return movement
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/api/visualization/player-movement-3d/{player_id}/trajectory")
        async def get_player_trajectory(player_id: str):
            """Get player trajectory visualization"""
            try:
                trajectory = await self.player_movement_3d.get_trajectory(player_id)
                return trajectory
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Heat Map Visualization
        @self.app.post("/api/visualization/heat-map")
        async def create_heat_map(heatmap: HeatMapVisualization):
            """Create heat map visualization"""
            try:
                result = await self.heat_map_viz.create_heat_map(heatmap)
                return {"status": "success", "heatmap_id": result["heatmap_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/heat-map/{heatmap_id}")
        async def get_heat_map(heatmap_id: str):
            """Get heat map visualization"""
            try:
                heatmap = await self.heat_map_viz.get_heat_map(heatmap_id)
                return heatmap
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/api/visualization/heat-map/{heatmap_id}/render")
        async def render_heat_map(heatmap_id: str, format: str = "json"):
            """Render heat map visualization"""
            try:
                render_data = await self.heat_map_viz.render_heat_map(heatmap_id, format)
                return render_data
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # VR Analytics
        @self.app.post("/api/visualization/vr-analytics")
        async def create_vr_session(vr_session: VRAnalytics):
            """Create VR analytics session"""
            try:
                result = await self.vr_analytics.create_vr_session(vr_session)
                return {"status": "success", "session_id": result["session_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/vr-analytics/{session_id}")
        async def get_vr_session(session_id: str):
            """Get VR analytics session"""
            try:
                session = await self.vr_analytics.get_vr_session(session_id)
                return session
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.post("/api/visualization/vr-analytics/{session_id}/interact")
        async def vr_interaction(session_id: str, interaction: Dict[str, Any]):
            """Handle VR interaction"""
            try:
                result = await self.vr_analytics.handle_interaction(session_id, interaction)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # AR Features
        @self.app.post("/api/visualization/ar-features")
        async def create_ar_feature(ar_feature: ARFeatures):
            """Create AR feature"""
            try:
                result = await self.ar_features.create_ar_feature(ar_feature)
                return {"status": "success", "feature_id": result["feature_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/ar-features/{feature_id}")
        async def get_ar_feature(feature_id: str):
            """Get AR feature"""
            try:
                feature = await self.ar_features.get_ar_feature(feature_id)
                return feature
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.post("/api/visualization/ar-features/{feature_id}/track")
        async def ar_tracking(feature_id: str, tracking_data: Dict[str, Any]):
            """Handle AR tracking"""
            try:
                result = await self.ar_features.handle_tracking(feature_id, tracking_data)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Interactive Exploration
        @self.app.post("/api/visualization/interactive-exploration")
        async def create_exploration(exploration: InteractiveExploration):
            """Create interactive exploration session"""
            try:
                result = await self.interactive_exploration.create_exploration(exploration)
                return {"status": "success", "exploration_id": result["exploration_id"]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualization/interactive-exploration/{exploration_id}")
        async def get_exploration(exploration_id: str):
            """Get interactive exploration session"""
            try:
                exploration = await self.interactive_exploration.get_exploration(exploration_id)
                return exploration
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.post("/api/visualization/interactive-exploration/{exploration_id}/filter")
        async def apply_exploration_filter(exploration_id: str, filters: Dict[str, Any]):
            """Apply exploration filters"""
            try:
                result = await self.interactive_exploration.apply_filters(exploration_id, filters)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint
        @self.app.websocket("/ws/visualization")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_connection(websocket)
    
    async def websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send visualization updates
                await self.send_visualization_updates(websocket)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def send_visualization_updates(self, websocket: WebSocket):
        """Send visualization updates"""
        try:
            # Get real-time visualization data
            visualization_data = await self.get_visualization_data()
            
            await websocket.send_json({
                "type": "visualization_update",
                "timestamp": datetime.now().isoformat(),
                "data": visualization_data
            })
        except Exception as e:
            logger.error(f"Error sending visualization updates: {e}")
    
    async def get_visualization_data(self):
        """Get real-time visualization data"""
        return {
            "ice_rink_3d": await self.ice_rink_3d.get_status(),
            "player_movement_3d": await self.player_movement_3d.get_status(),
            "heat_map_viz": await self.heat_map_viz.get_status(),
            "vr_analytics": await self.vr_analytics.get_status(),
            "ar_features": await self.ar_features.get_status(),
            "interactive_exploration": await self.interactive_exploration.get_status()
        }
    
    def init_database(self):
        """Initialize database for advanced visualization"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ice_rinks_3d (
                    rink_id TEXT PRIMARY KEY,
                    dimensions TEXT,
                    zones TEXT,
                    features TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_movements_3d (
                    movement_id TEXT PRIMARY KEY,
                    player_id TEXT,
                    player_name TEXT,
                    position TEXT,
                    movement_data TEXT,
                    trajectory TEXT,
                    heat_map_data TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS heat_maps (
                    heatmap_id TEXT PRIMARY KEY,
                    map_type TEXT,
                    data_points TEXT,
                    color_scheme TEXT,
                    intensity_range TEXT,
                    resolution INTEGER,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vr_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    analytics_type TEXT,
                    vr_data TEXT,
                    immersion_level TEXT,
                    session_duration REAL,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ar_features (
                    feature_id TEXT PRIMARY KEY,
                    feature_type TEXT,
                    mobile_platform TEXT,
                    ar_data TEXT,
                    user_interactions TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactive_explorations (
                    exploration_id TEXT PRIMARY KEY,
                    data_type TEXT,
                    exploration_filters TEXT,
                    visualization_options TEXT,
                    user_preferences TEXT,
                    exploration_results TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Advanced visualization database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

# 3D Ice Rink Service
class IceRink3DService:
    """3D Ice Rink Visualization Service"""
    
    def __init__(self):
        self.status = "operational"
        self.rink_count = 0
    
    async def create_ice_rink(self, rink: IceRink3D) -> Dict[str, Any]:
        """Create 3D ice rink visualization"""
        try:
            rink_id = f"rink_{int(time.time())}"
            
            # Store ice rink data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ice_rinks_3d 
                (rink_id, dimensions, zones, features, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                rink_id, json.dumps(rink.dimensions), json.dumps(rink.zones),
                json.dumps(rink.features), datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.rink_count += 1
            return {"rink_id": rink_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating ice rink: {e}")
            raise e
    
    async def get_ice_rink(self, rink_id: str) -> Dict[str, Any]:
        """Get 3D ice rink visualization"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ice_rinks_3d WHERE rink_id = ?', (rink_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("Ice rink not found")
            
            conn.close()
            
            return {
                "rink_id": row[0],
                "dimensions": json.loads(row[1]),
                "zones": json.loads(row[2]),
                "features": json.loads(row[3]),
                "created_at": row[4]
            }
            
        except Exception as e:
            logger.error(f"Error getting ice rink: {e}")
            raise e
    
    async def render_ice_rink(self, rink_id: str, format: str = "json") -> Dict[str, Any]:
        """Render 3D ice rink visualization"""
        try:
            # Get ice rink data
            rink_data = await self.get_ice_rink(rink_id)
            
            # Generate 3D visualization data
            render_data = {
                "rink_id": rink_id,
                "format": format,
                "3d_model": {
                    "vertices": self.generate_ice_rink_vertices(rink_data["dimensions"]),
                    "faces": self.generate_ice_rink_faces(),
                    "textures": self.generate_ice_rink_textures(),
                    "materials": self.generate_ice_rink_materials()
                },
                "zones": rink_data["zones"],
                "features": rink_data["features"],
                "rendering_options": {
                    "lighting": "realistic",
                    "shadows": True,
                    "reflections": True,
                    "quality": "high"
                }
            }
            
            return render_data
            
        except Exception as e:
            logger.error(f"Error rendering ice rink: {e}")
            raise e
    
    def generate_ice_rink_vertices(self, dimensions: Dict[str, float]) -> List[List[float]]:
        """Generate ice rink vertices for 3D model"""
        length = dimensions.get("length", 200)
        width = dimensions.get("width", 85)
        
        vertices = [
            [-length/2, -width/2, 0],  # Bottom left
            [length/2, -width/2, 0],   # Bottom right
            [length/2, width/2, 0],    # Top right
            [-length/2, width/2, 0],   # Top left
            [-length/2, -width/2, 0.1], # Bottom left elevated
            [length/2, -width/2, 0.1],  # Bottom right elevated
            [length/2, width/2, 0.1],  # Top right elevated
            [-length/2, width/2, 0.1]  # Top left elevated
        ]
        
        return vertices
    
    def generate_ice_rink_faces(self) -> List[List[int]]:
        """Generate ice rink faces for 3D model"""
        faces = [
            [0, 1, 2, 3],  # Bottom face
            [4, 7, 6, 5],   # Top face
            [0, 4, 5, 1],   # Front face
            [2, 6, 7, 3],   # Back face
            [0, 3, 7, 4],   # Left face
            [1, 5, 6, 2]    # Right face
        ]
        
        return faces
    
    def generate_ice_rink_textures(self) -> Dict[str, Any]:
        """Generate ice rink textures"""
        return {
            "ice_surface": {
                "type": "procedural",
                "color": [0.9, 0.95, 1.0],
                "roughness": 0.1,
                "metallic": 0.0
            },
            "boards": {
                "type": "image",
                "texture_url": "/textures/hockey_boards.jpg",
                "color": [0.8, 0.6, 0.4],
                "roughness": 0.3,
                "metallic": 0.0
            }
        }
    
    def generate_ice_rink_materials(self) -> Dict[str, Any]:
        """Generate ice rink materials"""
        return {
            "ice": {
                "diffuse": [0.9, 0.95, 1.0],
                "specular": [0.1, 0.1, 0.1],
                "shininess": 100.0,
                "transparency": 0.1
            },
            "boards": {
                "diffuse": [0.8, 0.6, 0.4],
                "specular": [0.2, 0.2, 0.2],
                "shininess": 50.0,
                "transparency": 0.0
            }
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get 3D ice rink service status"""
        return {
            "status": self.status,
            "rink_count": self.rink_count,
            "last_updated": datetime.now().isoformat()
        }

# 3D Player Movement Service
class PlayerMovement3DService:
    """3D Player Movement Tracking Service"""
    
    def __init__(self):
        self.status = "operational"
        self.movement_count = 0
    
    async def create_movement(self, movement: PlayerMovement3D) -> Dict[str, Any]:
        """Create 3D player movement tracking"""
        try:
            movement_id = f"movement_{int(time.time())}"
            
            # Store movement data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO player_movements_3d 
                (movement_id, player_id, player_name, position, movement_data, 
                 trajectory, heat_map_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                movement_id, movement.player_id, movement.player_name, movement.position,
                json.dumps(movement.movement_data), json.dumps(movement.trajectory),
                json.dumps(movement.heat_map_data), datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.movement_count += 1
            return {"movement_id": movement_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating player movement: {e}")
            raise e
    
    async def get_movement(self, player_id: str) -> Dict[str, Any]:
        """Get 3D player movement data"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM player_movements_3d WHERE player_id = ?', (player_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("Player movement not found")
            
            conn.close()
            
            return {
                "movement_id": row[0],
                "player_id": row[1],
                "player_name": row[2],
                "position": row[3],
                "movement_data": json.loads(row[4]),
                "trajectory": json.loads(row[5]),
                "heat_map_data": json.loads(row[6]),
                "created_at": row[7]
            }
            
        except Exception as e:
            logger.error(f"Error getting player movement: {e}")
            raise e
    
    async def get_trajectory(self, player_id: str) -> Dict[str, Any]:
        """Get player trajectory visualization"""
        try:
            movement_data = await self.get_movement(player_id)
            
            trajectory = {
                "player_id": player_id,
                "trajectory_data": movement_data["trajectory"],
                "movement_analysis": {
                    "total_distance": self.calculate_total_distance(movement_data["movement_data"]),
                    "average_speed": self.calculate_average_speed(movement_data["movement_data"]),
                    "max_speed": self.calculate_max_speed(movement_data["movement_data"]),
                    "acceleration_patterns": self.analyze_acceleration_patterns(movement_data["movement_data"])
                },
                "visualization_data": {
                    "path_points": self.generate_path_points(movement_data["movement_data"]),
                    "speed_colors": self.generate_speed_colors(movement_data["movement_data"]),
                    "acceleration_vectors": self.generate_acceleration_vectors(movement_data["movement_data"])
                }
            }
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error getting player trajectory: {e}")
            raise e
    
    def calculate_total_distance(self, movement_data: List[Dict[str, Any]]) -> float:
        """Calculate total distance traveled"""
        total_distance = 0.0
        for i in range(1, len(movement_data)):
            prev = movement_data[i-1]
            curr = movement_data[i]
            distance = math.sqrt(
                (curr["x"] - prev["x"])**2 + 
                (curr["y"] - prev["y"])**2 + 
                (curr["z"] - prev["z"])**2
            )
            total_distance += distance
        return total_distance
    
    def calculate_average_speed(self, movement_data: List[Dict[str, Any]]) -> float:
        """Calculate average speed"""
        if len(movement_data) < 2:
            return 0.0
        
        total_distance = self.calculate_total_distance(movement_data)
        total_time = movement_data[-1]["timestamp"] - movement_data[0]["timestamp"]
        
        if total_time == 0:
            return 0.0
        
        return total_distance / total_time
    
    def calculate_max_speed(self, movement_data: List[Dict[str, Any]]) -> float:
        """Calculate maximum speed"""
        max_speed = 0.0
        for i in range(1, len(movement_data)):
            prev = movement_data[i-1]
            curr = movement_data[i]
            time_diff = curr["timestamp"] - prev["timestamp"]
            if time_diff > 0:
                distance = math.sqrt(
                    (curr["x"] - prev["x"])**2 + 
                    (curr["y"] - prev["y"])**2 + 
                    (curr["z"] - prev["z"])**2
                )
                speed = distance / time_diff
                max_speed = max(max_speed, speed)
        return max_speed
    
    def analyze_acceleration_patterns(self, movement_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze acceleration patterns"""
        accelerations = []
        for i in range(2, len(movement_data)):
            # Calculate acceleration
            prev_prev = movement_data[i-2]
            prev = movement_data[i-1]
            curr = movement_data[i]
            
            # This is a simplified acceleration calculation
            # In reality, you'd want more sophisticated analysis
            accel = (curr["timestamp"] - prev["timestamp"]) - (prev["timestamp"] - prev_prev["timestamp"])
            accelerations.append(accel)
        
        return {
            "average_acceleration": np.mean(accelerations) if accelerations else 0.0,
            "max_acceleration": np.max(accelerations) if accelerations else 0.0,
            "acceleration_variance": np.var(accelerations) if accelerations else 0.0
        }
    
    def generate_path_points(self, movement_data: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate path points for visualization"""
        return [[point["x"], point["y"], point["z"]] for point in movement_data]
    
    def generate_speed_colors(self, movement_data: List[Dict[str, Any]]) -> List[str]:
        """Generate colors based on speed"""
        colors = []
        for i in range(1, len(movement_data)):
            prev = movement_data[i-1]
            curr = movement_data[i]
            time_diff = curr["timestamp"] - prev["timestamp"]
            if time_diff > 0:
                distance = math.sqrt(
                    (curr["x"] - prev["x"])**2 + 
                    (curr["y"] - prev["y"])**2 + 
                    (curr["z"] - prev["z"])**2
                )
                speed = distance / time_diff
                # Color based on speed (red = fast, blue = slow)
                if speed > 10:
                    colors.append("red")
                elif speed > 5:
                    colors.append("orange")
                elif speed > 2:
                    colors.append("yellow")
                else:
                    colors.append("blue")
            else:
                colors.append("gray")
        return colors
    
    def generate_acceleration_vectors(self, movement_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate acceleration vectors for visualization"""
        vectors = []
        for i in range(2, len(movement_data)):
            prev_prev = movement_data[i-2]
            prev = movement_data[i-1]
            curr = movement_data[i]
            
            # Calculate acceleration vector
            accel_x = (curr["x"] - 2*prev["x"] + prev_prev["x"])
            accel_y = (curr["y"] - 2*prev["y"] + prev_prev["y"])
            accel_z = (curr["z"] - 2*prev["z"] + prev_prev["z"])
            
            vectors.append({
                "position": [curr["x"], curr["y"], curr["z"]],
                "vector": [accel_x, accel_y, accel_z],
                "magnitude": math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            })
        
        return vectors
    
    async def get_status(self) -> Dict[str, Any]:
        """Get 3D player movement service status"""
        return {
            "status": self.status,
            "movement_count": self.movement_count,
            "last_updated": datetime.now().isoformat()
        }

# Heat Map Visualization Service
class HeatMapVisualizationService:
    """Heat Map Visualization Service"""
    
    def __init__(self):
        self.status = "operational"
        self.heatmap_count = 0
    
    async def create_heat_map(self, heatmap: HeatMapVisualization) -> Dict[str, Any]:
        """Create heat map visualization"""
        try:
            heatmap_id = f"heatmap_{int(time.time())}"
            
            # Store heat map data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO heat_maps 
                (heatmap_id, map_type, data_points, color_scheme, intensity_range, 
                 resolution, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                heatmap_id, heatmap.map_type, json.dumps(heatmap.data_points),
                heatmap.color_scheme, json.dumps(heatmap.intensity_range),
                heatmap.resolution, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.heatmap_count += 1
            return {"heatmap_id": heatmap_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating heat map: {e}")
            raise e
    
    async def get_heat_map(self, heatmap_id: str) -> Dict[str, Any]:
        """Get heat map visualization"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM heat_maps WHERE heatmap_id = ?', (heatmap_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("Heat map not found")
            
            conn.close()
            
            return {
                "heatmap_id": row[0],
                "map_type": row[1],
                "data_points": json.loads(row[2]),
                "color_scheme": row[3],
                "intensity_range": json.loads(row[4]),
                "resolution": row[5],
                "created_at": row[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting heat map: {e}")
            raise e
    
    async def render_heat_map(self, heatmap_id: str, format: str = "json") -> Dict[str, Any]:
        """Render heat map visualization"""
        try:
            heatmap_data = await self.get_heat_map(heatmap_id)
            
            # Generate heat map visualization data
            render_data = {
                "heatmap_id": heatmap_id,
                "format": format,
                "visualization_data": {
                    "grid_data": self.generate_heat_map_grid(heatmap_data["data_points"], heatmap_data["resolution"]),
                    "color_mapping": self.generate_color_mapping(heatmap_data["color_scheme"], heatmap_data["intensity_range"]),
                    "contour_lines": self.generate_contour_lines(heatmap_data["data_points"]),
                    "density_analysis": self.analyze_density_patterns(heatmap_data["data_points"])
                },
                "rendering_options": {
                    "smoothing": True,
                    "interpolation": "bicubic",
                    "opacity": 0.7,
                    "blend_mode": "multiply"
                }
            }
            
            return render_data
            
        except Exception as e:
            logger.error(f"Error rendering heat map: {e}")
            raise e
    
    def generate_heat_map_grid(self, data_points: List[Dict[str, Any]], resolution: int) -> List[List[float]]:
        """Generate heat map grid data"""
        # Create a grid based on data points
        grid = []
        for y in range(resolution):
            row = []
            for x in range(resolution):
                # Calculate intensity at this grid point
                intensity = self.calculate_intensity_at_point(x, y, data_points, resolution)
                row.append(intensity)
            grid.append(row)
        return grid
    
    def calculate_intensity_at_point(self, x: int, y: int, data_points: List[Dict[str, Any]], resolution: int) -> float:
        """Calculate intensity at a specific grid point"""
        total_intensity = 0.0
        total_weight = 0.0
        
        for point in data_points:
            # Calculate distance from grid point to data point
            distance = math.sqrt(
                (x/resolution - point["x"])**2 + 
                (y/resolution - point["y"])**2
            )
            
            # Use Gaussian kernel for intensity calculation
            weight = math.exp(-distance**2 / 0.1)  # Adjust sigma for smoothing
            total_intensity += point["intensity"] * weight
            total_weight += weight
        
        return total_intensity / total_weight if total_weight > 0 else 0.0
    
    def generate_color_mapping(self, color_scheme: str, intensity_range: Tuple[float, float]) -> Dict[str, Any]:
        """Generate color mapping for heat map"""
        min_intensity, max_intensity = intensity_range
        
        if color_scheme == "hot":
            return {
                "type": "hot",
                "colors": ["black", "red", "orange", "yellow", "white"],
                "intensity_range": [min_intensity, max_intensity]
            }
        elif color_scheme == "cool":
            return {
                "type": "cool",
                "colors": ["blue", "cyan", "green", "yellow"],
                "intensity_range": [min_intensity, max_intensity]
            }
        else:  # default
            return {
                "type": "default",
                "colors": ["blue", "green", "yellow", "red"],
                "intensity_range": [min_intensity, max_intensity]
            }
    
    def generate_contour_lines(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate contour lines for heat map"""
        # This is a simplified contour line generation
        # In reality, you'd use more sophisticated algorithms
        contour_lines = []
        
        # Generate contour levels
        intensities = [point["intensity"] for point in data_points]
        min_intensity = min(intensities)
        max_intensity = max(intensities)
        
        num_levels = 5
        for i in range(num_levels + 1):
            level = min_intensity + (max_intensity - min_intensity) * i / num_levels
            contour_lines.append({
                "level": level,
                "color": self.get_contour_color(level, min_intensity, max_intensity),
                "points": self.generate_contour_points(level, data_points)
            })
        
        return contour_lines
    
    def get_contour_color(self, level: float, min_intensity: float, max_intensity: float) -> str:
        """Get color for contour line based on level"""
        normalized = (level - min_intensity) / (max_intensity - min_intensity)
        if normalized < 0.2:
            return "blue"
        elif normalized < 0.4:
            return "green"
        elif normalized < 0.6:
            return "yellow"
        elif normalized < 0.8:
            return "orange"
        else:
            return "red"
    
    def generate_contour_points(self, level: float, data_points: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate contour points for a specific level"""
        # This is a simplified implementation
        # In reality, you'd use marching squares or similar algorithms
        points = []
        for point in data_points:
            if abs(point["intensity"] - level) < 0.1:  # Close to contour level
                points.append([point["x"], point["y"]])
        return points
    
    def analyze_density_patterns(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze density patterns in heat map"""
        intensities = [point["intensity"] for point in data_points]
        
        return {
            "density_centers": self.find_density_centers(data_points),
            "density_variance": np.var(intensities) if intensities else 0.0,
            "density_skewness": self.calculate_skewness(intensities),
            "hot_spots": self.find_hot_spots(data_points)
        }
    
    def find_density_centers(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find density centers in the data"""
        # Simple clustering to find density centers
        centers = []
        
        # Group nearby points
        clusters = []
        for point in data_points:
            added_to_cluster = False
            for cluster in clusters:
                if any(math.sqrt((point["x"] - p["x"])**2 + (point["y"] - p["y"])**2) < 0.1 for p in cluster):
                    cluster.append(point)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([point])
        
        # Calculate center for each cluster
        for cluster in clusters:
            if len(cluster) > 1:  # Only consider clusters with multiple points
                center_x = sum(p["x"] for p in cluster) / len(cluster)
                center_y = sum(p["y"] for p in cluster) / len(cluster)
                center_intensity = sum(p["intensity"] for p in cluster) / len(cluster)
                centers.append({
                    "x": center_x,
                    "y": center_y,
                    "intensity": center_intensity,
                    "point_count": len(cluster)
                })
        
        return centers
    
    def calculate_skewness(self, intensities: List[float]) -> float:
        """Calculate skewness of intensity distribution"""
        if len(intensities) < 3:
            return 0.0
        
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        if std_intensity == 0:
            return 0.0
        
        skewness = np.mean([(x - mean_intensity)**3 for x in intensities]) / (std_intensity**3)
        return skewness
    
    def find_hot_spots(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find hot spots in the data"""
        hot_spots = []
        intensities = [point["intensity"] for point in data_points]
        
        if not intensities:
            return hot_spots
        
        threshold = np.percentile(intensities, 90)  # Top 10% of intensities
        
        for point in data_points:
            if point["intensity"] >= threshold:
                hot_spots.append({
                    "x": point["x"],
                    "y": point["y"],
                    "intensity": point["intensity"],
                    "significance": point["intensity"] / max(intensities)
                })
        
        return hot_spots
    
    async def get_status(self) -> Dict[str, Any]:
        """Get heat map visualization service status"""
        return {
            "status": self.status,
            "heatmap_count": self.heatmap_count,
            "last_updated": datetime.now().isoformat()
        }

# VR Analytics Service
class VRAnalyticsService:
    """VR Analytics Service"""
    
    def __init__(self):
        self.status = "operational"
        self.session_count = 0
    
    async def create_vr_session(self, vr_session: VRAnalytics) -> Dict[str, Any]:
        """Create VR analytics session"""
        try:
            session_id = f"vr_{int(time.time())}"
            
            # Store VR session data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO vr_sessions 
                (session_id, user_id, analytics_type, vr_data, immersion_level, 
                 session_duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, vr_session.user_id, vr_session.analytics_type,
                json.dumps(vr_session.vr_data), vr_session.immersion_level,
                vr_session.session_duration, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.session_count += 1
            return {"session_id": session_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating VR session: {e}")
            raise e
    
    async def get_vr_session(self, session_id: str) -> Dict[str, Any]:
        """Get VR analytics session"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM vr_sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("VR session not found")
            
            conn.close()
            
            return {
                "session_id": row[0],
                "user_id": row[1],
                "analytics_type": row[2],
                "vr_data": json.loads(row[3]),
                "immersion_level": row[4],
                "session_duration": row[5],
                "created_at": row[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting VR session: {e}")
            raise e
    
    async def handle_interaction(self, session_id: str, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VR interaction"""
        try:
            # Process VR interaction
            interaction_result = {
                "session_id": session_id,
                "interaction_type": interaction.get("type", "unknown"),
                "response": self.process_vr_interaction(interaction),
                "timestamp": datetime.now().isoformat()
            }
            
            return interaction_result
            
        except Exception as e:
            logger.error(f"Error handling VR interaction: {e}")
            raise e
    
    def process_vr_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process VR interaction"""
        interaction_type = interaction.get("type", "unknown")
        
        if interaction_type == "gaze_tracking":
            return {
                "action": "update_gaze_visualization",
                "gaze_point": interaction.get("gaze_point", [0, 0, 0]),
                "attention_focus": self.analyze_attention_focus(interaction)
            }
        elif interaction_type == "hand_tracking":
            return {
                "action": "update_hand_visualization",
                "hand_position": interaction.get("hand_position", [0, 0, 0]),
                "gesture": self.recognize_gesture(interaction)
            }
        elif interaction_type == "voice_command":
            return {
                "action": "process_voice_command",
                "command": interaction.get("command", ""),
                "response": self.process_voice_command(interaction.get("command", ""))
            }
        else:
            return {
                "action": "unknown_interaction",
                "message": "Interaction type not recognized"
            }
    
    def analyze_attention_focus(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention focus from gaze tracking"""
        gaze_point = interaction.get("gaze_point", [0, 0, 0])
        
        return {
            "focus_area": self.determine_focus_area(gaze_point),
            "attention_duration": interaction.get("duration", 0),
            "focus_intensity": self.calculate_focus_intensity(interaction)
        }
    
    def determine_focus_area(self, gaze_point: List[float]) -> str:
        """Determine focus area from gaze point"""
        x, y, z = gaze_point
        
        # Simple area determination based on coordinates
        if z > 0.5:
            return "forward_zone"
        elif z < -0.5:
            return "defensive_zone"
        else:
            return "neutral_zone"
    
    def calculate_focus_intensity(self, interaction: Dict[str, Any]) -> float:
        """Calculate focus intensity"""
        duration = interaction.get("duration", 0)
        stability = interaction.get("stability", 1.0)
        
        return min(duration * stability, 1.0)
    
    def recognize_gesture(self, interaction: Dict[str, Any]) -> str:
        """Recognize hand gesture"""
        hand_data = interaction.get("hand_data", {})
        
        # Simple gesture recognition
        if hand_data.get("fingers_extended", 0) == 5:
            return "open_hand"
        elif hand_data.get("fingers_extended", 0) == 0:
            return "fist"
        elif hand_data.get("fingers_extended", 0) == 1:
            return "point"
        else:
            return "unknown"
    
    def process_voice_command(self, command: str) -> str:
        """Process voice command"""
        command_lower = command.lower()
        
        if "show" in command_lower and "stats" in command_lower:
            return "Displaying player statistics"
        elif "analyze" in command_lower and "play" in command_lower:
            return "Analyzing play sequence"
        elif "zoom" in command_lower:
            return "Adjusting zoom level"
        else:
            return "Command not recognized"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get VR analytics service status"""
        return {
            "status": self.status,
            "session_count": self.session_count,
            "last_updated": datetime.now().isoformat()
        }

# AR Features Service
class ARFeaturesService:
    """AR Features Service"""
    
    def __init__(self):
        self.status = "operational"
        self.feature_count = 0
    
    async def create_ar_feature(self, ar_feature: ARFeatures) -> Dict[str, Any]:
        """Create AR feature"""
        try:
            feature_id = f"ar_{int(time.time())}"
            
            # Store AR feature data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ar_features 
                (feature_id, feature_type, mobile_platform, ar_data, user_interactions, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                feature_id, ar_feature.feature_type, ar_feature.mobile_platform,
                json.dumps(ar_feature.ar_data), json.dumps(ar_feature.user_interactions),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.feature_count += 1
            return {"feature_id": feature_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating AR feature: {e}")
            raise e
    
    async def get_ar_feature(self, feature_id: str) -> Dict[str, Any]:
        """Get AR feature"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ar_features WHERE feature_id = ?', (feature_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("AR feature not found")
            
            conn.close()
            
            return {
                "feature_id": row[0],
                "feature_type": row[1],
                "mobile_platform": row[2],
                "ar_data": json.loads(row[3]),
                "user_interactions": json.loads(row[4]),
                "created_at": row[5]
            }
            
        except Exception as e:
            logger.error(f"Error getting AR feature: {e}")
            raise e
    
    async def handle_tracking(self, feature_id: str, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AR tracking"""
        try:
            # Process AR tracking data
            tracking_result = {
                "feature_id": feature_id,
                "tracking_type": tracking_data.get("type", "unknown"),
                "response": self.process_ar_tracking(tracking_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return tracking_result
            
        except Exception as e:
            logger.error(f"Error handling AR tracking: {e}")
            raise e
    
    def process_ar_tracking(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process AR tracking data"""
        tracking_type = tracking_data.get("type", "unknown")
        
        if tracking_type == "player_tracking":
            return {
                "action": "update_player_overlay",
                "player_position": tracking_data.get("player_position", [0, 0, 0]),
                "overlay_data": self.generate_player_overlay(tracking_data)
            }
        elif tracking_type == "object_tracking":
            return {
                "action": "update_object_overlay",
                "object_position": tracking_data.get("object_position", [0, 0, 0]),
                "overlay_data": self.generate_object_overlay(tracking_data)
            }
        else:
            return {
                "action": "unknown_tracking",
                "message": "Tracking type not recognized"
            }
    
    def generate_player_overlay(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate player overlay data"""
        player_id = tracking_data.get("player_id", "unknown")
        position = tracking_data.get("player_position", [0, 0, 0])
        
        return {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "stats": {
                "speed": tracking_data.get("speed", 0),
                "distance": tracking_data.get("distance", 0),
                "time_on_ice": tracking_data.get("time_on_ice", 0)
            },
            "overlay_position": position,
            "overlay_style": {
                "color": "blue",
                "opacity": 0.8,
                "size": "medium"
            }
        }
    
    def generate_object_overlay(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate object overlay data"""
        object_type = tracking_data.get("object_type", "unknown")
        position = tracking_data.get("object_position", [0, 0, 0])
        
        return {
            "object_type": object_type,
            "object_name": f"{object_type.title()} Object",
            "overlay_position": position,
            "overlay_style": {
                "color": "red" if object_type == "puck" else "green",
                "opacity": 0.9,
                "size": "large" if object_type == "puck" else "small"
            }
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get AR features service status"""
        return {
            "status": self.status,
            "feature_count": self.feature_count,
            "last_updated": datetime.now().isoformat()
        }

# Interactive Exploration Service
class InteractiveExplorationService:
    """Interactive Exploration Service"""
    
    def __init__(self):
        self.status = "operational"
        self.exploration_count = 0
    
    async def create_exploration(self, exploration: InteractiveExploration) -> Dict[str, Any]:
        """Create interactive exploration session"""
        try:
            exploration_id = f"explore_{int(time.time())}"
            
            # Store exploration data
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO interactive_explorations 
                (exploration_id, data_type, exploration_filters, visualization_options, 
                 user_preferences, exploration_results, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                exploration_id, exploration.data_type, json.dumps(exploration.exploration_filters),
                json.dumps(exploration.visualization_options), json.dumps(exploration.user_preferences),
                json.dumps(exploration.exploration_results), datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.exploration_count += 1
            return {"exploration_id": exploration_id, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating exploration: {e}")
            raise e
    
    async def get_exploration(self, exploration_id: str) -> Dict[str, Any]:
        """Get interactive exploration session"""
        try:
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM interactive_explorations WHERE exploration_id = ?', (exploration_id,))
            row = cursor.fetchone()
            
            if not row:
                raise Exception("Exploration not found")
            
            conn.close()
            
            return {
                "exploration_id": row[0],
                "data_type": row[1],
                "exploration_filters": json.loads(row[2]),
                "visualization_options": json.loads(row[3]),
                "user_preferences": json.loads(row[4]),
                "exploration_results": json.loads(row[5]),
                "created_at": row[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting exploration: {e}")
            raise e
    
    async def apply_filters(self, exploration_id: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exploration filters"""
        try:
            # Get current exploration
            exploration = await self.get_exploration(exploration_id)
            
            # Apply new filters
            updated_filters = {**exploration["exploration_filters"], **filters}
            
            # Generate filtered results
            filtered_results = self.generate_filtered_results(exploration["data_type"], updated_filters)
            
            # Update exploration with new results
            conn = sqlite3.connect('advanced_visualization.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE interactive_explorations 
                SET exploration_filters = ?, exploration_results = ?
                WHERE exploration_id = ?
            ''', (
                json.dumps(updated_filters), json.dumps(filtered_results), exploration_id
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "exploration_id": exploration_id,
                "filters_applied": updated_filters,
                "filtered_results": filtered_results,
                "result_count": len(filtered_results.get("data", []))
            }
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            raise e
    
    def generate_filtered_results(self, data_type: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate filtered results based on data type and filters"""
        if data_type == "game_data":
            return self.filter_game_data(filters)
        elif data_type == "player_stats":
            return self.filter_player_stats(filters)
        elif data_type == "team_analytics":
            return self.filter_team_analytics(filters)
        else:
            return {"data": [], "summary": "Unknown data type"}
    
    def filter_game_data(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter game data"""
        # Simulate game data filtering
        filtered_data = []
        
        # Apply filters
        if "team" in filters:
            filtered_data.append({
                "game_id": "game_001",
                "team": filters["team"],
                "opponent": "Opponent Team",
                "date": "2024-10-25",
                "score": "3-2",
                "events": [
                    {"type": "goal", "player": "Player A", "time": "15:30"},
                    {"type": "assist", "player": "Player B", "time": "15:30"}
                ]
            })
        
        return {
            "data": filtered_data,
            "summary": f"Found {len(filtered_data)} games matching filters",
            "filters_applied": filters
        }
    
    def filter_player_stats(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter player statistics"""
        # Simulate player stats filtering
        filtered_data = []
        
        if "position" in filters:
            filtered_data.append({
                "player_id": "player_001",
                "player_name": "Connor McDavid",
                "position": filters["position"],
                "stats": {
                    "goals": 15,
                    "assists": 25,
                    "points": 40,
                    "plus_minus": 12
                }
            })
        
        return {
            "data": filtered_data,
            "summary": f"Found {len(filtered_data)} players matching filters",
            "filters_applied": filters
        }
    
    def filter_team_analytics(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter team analytics"""
        # Simulate team analytics filtering
        filtered_data = []
        
        if "season" in filters:
            filtered_data.append({
                "team_id": "team_001",
                "team_name": "Boston Bruins",
                "season": filters["season"],
                "analytics": {
                    "wins": 25,
                    "losses": 15,
                    "points": 55,
                    "goals_for": 120,
                    "goals_against": 95
                }
            })
        
        return {
            "data": filtered_data,
            "summary": f"Found {len(filtered_data)} teams matching filters",
            "filters_applied": filters
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get interactive exploration service status"""
        return {
            "status": self.status,
            "exploration_count": self.exploration_count,
            "last_updated": datetime.now().isoformat()
        }

# Main execution
if __name__ == "__main__":
    # Initialize advanced visualization engine
    engine = AdvancedVisualizationEngine()
    
    # Run the application
    uvicorn.run(
        engine.app,
        host="0.0.0.0",
        port=8008,
        log_level="info"
    )
