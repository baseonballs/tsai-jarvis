"""
TSAI Jarvis - Real Hockey Analytics Video Processor
Phase 1.1: Video Processing Pipeline with YOLO Model Integration

This module implements real-time hockey video processing with YOLO v8 model
for player detection, tracking, and analytics.
"""

import cv2
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import base64
from pathlib import Path

# YOLO and Computer Vision imports
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install with: pip install ultralytics")

# Video processing imports
import ffmpeg
import websockets
import aiohttp
from aiofiles import open as aio_open

logger = logging.getLogger(__name__)

@dataclass
class PlayerDetection:
    """Represents a detected hockey player"""
    player_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str  # "player", "goalie", "referee"
    jersey_number: Optional[int] = None
    team: Optional[str] = None  # "home", "away"
    position: Optional[Tuple[float, float]] = None  # x, y coordinates on ice

@dataclass
class GameEvent:
    """Represents a detected game event"""
    event_type: str  # "goal", "save", "penalty", "faceoff"
    timestamp: datetime
    confidence: float
    players_involved: List[int]
    location: Tuple[float, float]
    description: str

class HockeyVideoProcessor:
    """
    Real-time hockey video processor with YOLO model integration
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_processing = False
        self.detections_history = []
        self.game_events = []
        
        # Video processing settings
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        
        # Player tracking
        self.player_tracker = {}
        self.next_player_id = 1
        
        # Initialize YOLO model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model for hockey player detection"""
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available. Please install ultralytics")
            return
        
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            logger.info(f"✅ YOLO model loaded: {self.model_path}")
            
            # Check if CUDA is available for GPU acceleration
            if torch.cuda.is_available():
                logger.info(f"🚀 GPU acceleration available: {torch.cuda.get_device_name()}")
                self.device = 'cuda'
            else:
                logger.info("💻 Using CPU for inference")
                self.device = 'cpu'
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize YOLO model: {e}")
            self.model = None
    
    async def process_video_stream(self, video_source: str) -> Dict[str, Any]:
        """
        Process real-time hockey video stream
        
        Args:
            video_source: Video source (file path, URL, or camera index)
            
        Returns:
            Processing results with detections and analytics
        """
        logger.info(f"🎬 Starting video processing: {video_source}")
        
        try:
            # Open video source
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {video_source}")
            
            # Set video properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_processing = True
            frame_count = 0
            start_time = datetime.now()
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with YOLO
                detections = await self._process_frame(frame, frame_count)
                
                # Track players across frames
                tracked_players = await self._track_players(detections, frame_count)
                
                # Detect game events
                events = await self._detect_game_events(tracked_players, frame_count)
                
                # Store results
                self.detections_history.append({
                    'frame': frame_count,
                    'timestamp': datetime.now(),
                    'detections': detections,
                    'tracked_players': tracked_players,
                    'events': events
                })
                
                frame_count += 1
                
                # Yield control for async processing
                await asyncio.sleep(0.033)  # ~30 FPS
            
            cap.release()
            
            # Calculate processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            fps_actual = frame_count / processing_time if processing_time > 0 else 0
            
            return {
                'status': 'completed',
                'frames_processed': frame_count,
                'processing_time': processing_time,
                'fps_actual': fps_actual,
                'total_detections': len(self.detections_history),
                'total_events': len(self.game_events)
            }
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _process_frame(self, frame: np.ndarray, frame_number: int) -> List[PlayerDetection]:
        """
        Process single frame with YOLO model
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number for tracking
            
        Returns:
            List of player detections
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Create player detection
                        detection = PlayerDetection(
                            player_id=0,  # Will be assigned during tracking
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(confidence),
                            class_name=class_name
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Frame processing failed: {e}")
            return []
    
    async def _track_players(self, detections: List[PlayerDetection], frame_number: int) -> List[PlayerDetection]:
        """
        Track players across frames using simple tracking algorithm
        
        Args:
            detections: Current frame detections
            frame_number: Current frame number
            
        Returns:
            List of tracked players with IDs
        """
        tracked_players = []
        
        for detection in detections:
            # Simple tracking: find closest previous detection
            best_match = None
            best_distance = float('inf')
            
            for player_id, player_data in self.player_tracker.items():
                if player_data['last_seen'] == frame_number - 1:
                    # Calculate distance between detections
                    prev_bbox = player_data['last_bbox']
                    curr_bbox = detection.bbox
                    
                    # Calculate center points
                    prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                    curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
                    
                    distance = np.sqrt((prev_center[0] - curr_center[0])**2 + (prev_center[1] - curr_center[1])**2)
                    
                    if distance < best_distance and distance < 100:  # Max tracking distance
                        best_distance = distance
                        best_match = player_id
            
            if best_match:
                # Update existing player
                detection.player_id = best_match
                self.player_tracker[best_match]['last_seen'] = frame_number
                self.player_tracker[best_match]['last_bbox'] = detection.bbox
            else:
                # Create new player
                detection.player_id = self.next_player_id
                self.player_tracker[self.next_player_id] = {
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'last_bbox': detection.bbox
                }
                self.next_player_id += 1
            
            tracked_players.append(detection)
        
        return tracked_players
    
    async def _detect_game_events(self, players: List[PlayerDetection], frame_number: int) -> List[GameEvent]:
        """
        Detect game events based on player positions and movements
        
        Args:
            players: List of tracked players
            frame_number: Current frame number
            
        Returns:
            List of detected game events
        """
        events = []
        
        # Simple event detection logic
        if len(players) >= 2:
            # Detect potential faceoff (players close together)
            for i, player1 in enumerate(players):
                for j, player2 in enumerate(players[i+1:], i+1):
                    distance = self._calculate_distance(player1.bbox, player2.bbox)
                    if distance < 50:  # Close proximity threshold
                        event = GameEvent(
                            event_type="faceoff",
                            timestamp=datetime.now(),
                            confidence=0.8,
                            players_involved=[player1.player_id, player2.player_id],
                            location=self._get_center_position(player1.bbox),
                            description=f"Potential faceoff between players {player1.player_id} and {player2.player_id}"
                        )
                        events.append(event)
        
        # Store events
        self.game_events.extend(events)
        
        return events
    
    def _calculate_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_center_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center position of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    async def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        logger.info("🛑 Video processing stopped")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary of processed video"""
        if not self.detections_history:
            return {"message": "No video processed yet"}
        
        total_frames = len(self.detections_history)
        total_players = len(self.player_tracker)
        total_events = len(self.game_events)
        
        # Calculate average detections per frame
        avg_detections = sum(len(d['detections']) for d in self.detections_history) / total_frames
        
        return {
            "total_frames": total_frames,
            "total_players_detected": total_players,
            "total_events": total_events,
            "average_detections_per_frame": avg_detections,
            "processing_summary": {
                "player_tracking": f"{total_players} unique players tracked",
                "event_detection": f"{total_events} game events detected",
                "detection_accuracy": f"{avg_detections:.1f} detections per frame"
            }
        }

class HockeyVideoStreamer:
    """
    Real-time hockey video streaming with WebSocket support
    """
    
    def __init__(self, processor: HockeyVideoProcessor):
        self.processor = processor
        self.clients = set()
        self.is_streaming = False
    
    async def start_streaming(self, video_source: str, websocket_port: int = 8765):
        """Start real-time video streaming with WebSocket"""
        logger.info(f"📡 Starting video streaming on port {websocket_port}")
        
        async def handle_client(websocket, path):
            self.clients.add(websocket)
            logger.info(f"📱 Client connected: {websocket.remote_address}")
            
            try:
                while True:
                    # Send analytics data to client
                    if self.processor.detections_history:
                        latest_data = self.processor.detections_history[-1]
                        await websocket.send(json.dumps({
                            "type": "analytics_update",
                            "frame": latest_data['frame'],
                            "timestamp": latest_data['timestamp'].isoformat(),
                            "detections": len(latest_data['detections']),
                            "tracked_players": len(latest_data['tracked_players']),
                            "events": len(latest_data['events'])
                        }))
                    
                    await asyncio.sleep(0.1)  # 10 FPS updates
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"📱 Client disconnected: {websocket.remote_address}")
            finally:
                self.clients.remove(websocket)
        
        # Start WebSocket server
        start_server = websockets.serve(handle_client, "localhost", websocket_port)
        await start_server
        
        # Start video processing in background
        asyncio.create_task(self.processor.process_video_stream(video_source))
        
        logger.info(f"🎬 Video streaming started: {video_source}")

# Example usage and testing
async def test_hockey_video_processing():
    """Test hockey video processing with sample video"""
    logger.info("🧪 Testing hockey video processing...")
    
    # Initialize processor
    processor = HockeyVideoProcessor()
    
    if processor.model is None:
        logger.error("❌ YOLO model not available. Cannot test.")
        return
    
    # Test with sample video (you can replace with actual hockey video)
    video_source = "sample_hockey_video.mp4"  # Replace with actual video path
    
    try:
        # Process video
        results = await processor.process_video_stream(video_source)
        logger.info(f"✅ Video processing completed: {results}")
        
        # Get analytics summary
        analytics = processor.get_analytics_summary()
        logger.info(f"📊 Analytics summary: {analytics}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_hockey_video_processing())
