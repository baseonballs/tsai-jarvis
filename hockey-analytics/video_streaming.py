"""
TSAI Jarvis - Real-time Hockey Video Streaming
Phase 1.1: Video Streaming Infrastructure for Live Analytics

This module implements real-time video streaming with WebSocket support
for live hockey analytics and player detection.
"""

import asyncio
import logging
import json
import base64
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
import aiohttp
from aiofiles import open as aio_open
import ffmpeg
from pathlib import Path

logger = logging.getLogger(__name__)

class HockeyVideoStreamer:
    """
    Real-time hockey video streaming with WebSocket support for live analytics
    """
    
    def __init__(self, 
                 stream_port: int = 8765,
                 video_port: int = 8080,
                 max_clients: int = 100):
        self.stream_port = stream_port
        self.video_port = video_port
        self.max_clients = max_clients
        
        # Client management
        self.clients = set()
        self.analytics_clients = set()
        self.video_clients = set()
        
        # Streaming state
        self.is_streaming = False
        self.current_video_source = None
        self.streaming_task = None
        
        # Analytics callback
        self.analytics_callback = None
        
        # Video processing settings
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.quality = 85  # JPEG quality
        
    async def start_streaming(self, video_source: str, analytics_callback: Optional[Callable] = None):
        """
        Start real-time video streaming
        
        Args:
            video_source: Video source (file path, URL, or camera index)
            analytics_callback: Callback function for analytics data
        """
        logger.info(f"🎬 Starting video streaming: {video_source}")
        
        self.current_video_source = video_source
        self.analytics_callback = analytics_callback
        self.is_streaming = True
        
        # Start WebSocket servers
        await asyncio.gather(
            self._start_analytics_server(),
            self._start_video_server(),
            self._process_video_stream()
        )
    
    async def _start_analytics_server(self):
        """Start WebSocket server for analytics data"""
        logger.info(f"📊 Starting analytics server on port {self.stream_port}")
        
        async def handle_analytics_client(websocket: WebSocketServerProtocol, path: str):
            self.analytics_clients.add(websocket)
            client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"📱 Analytics client connected: {client_addr}")
            
            try:
                # Send welcome message
                await websocket.send(json.dumps({
                    "type": "connection",
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Connected to TSAI Jarvis Hockey Analytics"
                }))
                
                # Keep connection alive
                async for message in websocket:
                    if message == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"📱 Analytics client disconnected: {client_addr}")
            finally:
                self.analytics_clients.discard(websocket)
        
        # Start WebSocket server
        server = await websockets.serve(
            handle_analytics_client, 
            "localhost", 
            self.stream_port,
            max_size=2**20,  # 1MB max message size
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✅ Analytics server running on ws://localhost:{self.stream_port}")
        await server.wait_closed()
    
    async def _start_video_server(self):
        """Start WebSocket server for video frames"""
        logger.info(f"🎥 Starting video server on port {self.video_port}")
        
        async def handle_video_client(websocket: WebSocketServerProtocol, path: str):
            self.video_clients.add(websocket)
            client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"📱 Video client connected: {client_addr}")
            
            try:
                # Send welcome message
                await websocket.send(json.dumps({
                    "type": "video_connection",
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Connected to TSAI Jarvis Video Stream"
                }))
                
                # Keep connection alive
                async for message in websocket:
                    if message == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"📱 Video client disconnected: {client_addr}")
            finally:
                self.video_clients.discard(websocket)
        
        # Start WebSocket server
        server = await websockets.serve(
            handle_video_client,
            "localhost",
            self.video_port,
            max_size=10**7,  # 10MB max message size for video frames
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✅ Video server running on ws://localhost:{self.video_port}")
        await server.wait_closed()
    
    async def _process_video_stream(self):
        """Process video stream and send to clients"""
        logger.info("🎬 Starting video stream processing...")
        
        try:
            # Open video source
            cap = cv2.VideoCapture(self.current_video_source)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.current_video_source}")
            
            # Set video properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            frame_count = 0
            start_time = datetime.now()
            
            while self.is_streaming:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📹 End of video stream")
                    break
                
                # Process frame
                processed_frame = await self._process_frame(frame, frame_count)
                
                # Send to video clients
                await self._send_video_frame(processed_frame, frame_count)
                
                # Send analytics data
                await self._send_analytics_data(frame_count, processed_frame)
                
                frame_count += 1
                
                # Control frame rate
                await asyncio.sleep(1.0 / self.fps)
            
            cap.release()
            
            # Calculate streaming statistics
            streaming_time = (datetime.now() - start_time).total_seconds()
            actual_fps = frame_count / streaming_time if streaming_time > 0 else 0
            
            logger.info(f"📊 Streaming completed: {frame_count} frames, {actual_fps:.1f} FPS")
            
        except Exception as e:
            logger.error(f"❌ Video streaming failed: {e}")
            self.is_streaming = False
    
    async def _process_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Process video frame for analytics
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number
            
        Returns:
            Processed frame data
        """
        # Resize frame for streaming
        resized_frame = cv2.resize(frame, (640, 360))  # 16:9 aspect ratio
        
        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Create frame metadata
        frame_info = {
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat(),
            "width": resized_frame.shape[1],
            "height": resized_frame.shape[0],
            "quality": self.quality,
            "data": frame_data
        }
        
        return frame_info
    
    async def _send_video_frame(self, frame_data: Dict[str, Any], frame_number: int):
        """Send video frame to connected clients"""
        if not self.video_clients:
            return
        
        message = json.dumps({
            "type": "video_frame",
            "frame_number": frame_number,
            "timestamp": frame_data["timestamp"],
            "width": frame_data["width"],
            "height": frame_data["height"],
            "data": frame_data["data"]
        })
        
        # Send to all video clients
        disconnected_clients = set()
        for client in self.video_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"❌ Failed to send video frame to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.video_clients -= disconnected_clients
    
    async def _send_analytics_data(self, frame_number: int, frame_data: Dict[str, Any]):
        """Send analytics data to connected clients"""
        if not self.analytics_clients:
            return
        
        # Get analytics data from callback
        analytics_data = {}
        if self.analytics_callback:
            try:
                analytics_data = await self.analytics_callback(frame_number, frame_data)
            except Exception as e:
                logger.error(f"❌ Analytics callback failed: {e}")
                analytics_data = {"error": str(e)}
        
        # Create analytics message
        message = json.dumps({
            "type": "analytics_update",
            "frame_number": frame_number,
            "timestamp": frame_data["timestamp"],
            "analytics": analytics_data,
            "clients_connected": len(self.analytics_clients)
        })
        
        # Send to all analytics clients
        disconnected_clients = set()
        for client in self.analytics_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"❌ Failed to send analytics to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.analytics_clients -= disconnected_clients
    
    async def stop_streaming(self):
        """Stop video streaming"""
        logger.info("🛑 Stopping video streaming...")
        self.is_streaming = False
        
        # Close all client connections
        for client in self.analytics_clients.copy():
            await client.close()
        
        for client in self.video_clients.copy():
            await client.close()
        
        self.analytics_clients.clear()
        self.video_clients.clear()
        
        logger.info("✅ Video streaming stopped")
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            "is_streaming": self.is_streaming,
            "video_source": self.current_video_source,
            "analytics_clients": len(self.analytics_clients),
            "video_clients": len(self.video_clients),
            "total_clients": len(self.analytics_clients) + len(self.video_clients),
            "stream_port": self.stream_port,
            "video_port": self.video_port,
            "frame_settings": {
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": self.fps,
                "quality": self.quality
            }
        }

class HockeyVideoRecorder:
    """
    Record hockey video streams with analytics overlay
    """
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.is_recording = False
        self.recording_task = None
        
    async def start_recording(self, video_source: str, duration: Optional[int] = None):
        """
        Start recording video stream
        
        Args:
            video_source: Video source to record
            duration: Recording duration in seconds (None for continuous)
        """
        logger.info(f"📹 Starting video recording: {video_source}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"hockey_recording_{timestamp}.mp4"
        
        self.is_recording = True
        
        # Start recording task
        self.recording_task = asyncio.create_task(
            self._record_video(video_source, str(output_path), duration)
        )
        
        logger.info(f"✅ Recording started: {output_path}")
    
    async def _record_video(self, video_source: str, output_path: str, duration: Optional[int]):
        """Record video using FFmpeg"""
        try:
            # FFmpeg command for recording
            input_stream = ffmpeg.input(video_source)
            
            # Add video filters for quality optimization
            video_stream = input_stream.video.filter('scale', 1920, 1080)
            audio_stream = input_stream.audio
            
            # Output stream
            output_stream = ffmpeg.output(
                video_stream, 
                audio_stream, 
                output_path,
                vcodec='libx264',
                acodec='aac',
                preset='fast',
                crf=23
            )
            
            # Add duration limit if specified
            if duration:
                output_stream = ffmpeg.output(
                    video_stream,
                    audio_stream,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    preset='fast',
                    crf=23,
                    t=duration
                )
            
            # Run FFmpeg
            await asyncio.create_subprocess_exec(
                *ffmpeg.compile(output_stream, overwrite_output=True),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"✅ Recording completed: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Recording failed: {e}")
        finally:
            self.is_recording = False
    
    async def stop_recording(self):
        """Stop video recording"""
        if self.recording_task:
            self.recording_task.cancel()
            self.is_recording = False
            logger.info("🛑 Recording stopped")

# Example usage and testing
async def test_hockey_video_streaming():
    """Test hockey video streaming functionality"""
    logger.info("🧪 Testing Hockey Video Streaming...")
    
    # Initialize streamer
    streamer = HockeyVideoStreamer()
    
    # Mock analytics callback
    async def mock_analytics_callback(frame_number: int, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "players_detected": np.random.randint(0, 10),
            "events": np.random.randint(0, 3),
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": np.random.uniform(0.01, 0.05)
        }
    
    # Test video source (replace with actual hockey video)
    video_source = "sample_hockey_video.mp4"  # Replace with actual video path
    
    try:
        # Start streaming
        await streamer.start_streaming(video_source, mock_analytics_callback)
        
    except KeyboardInterrupt:
        logger.info("🛑 Stopping streaming...")
        await streamer.stop_streaming()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_hockey_video_streaming())
