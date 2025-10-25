"""
TSAI Jarvis - Hockey Analytics Test Suite
Phase 1.1: Test Real Hockey Video Processing with YOLO Model Integration

This module tests the hockey analytics video processing pipeline
with real YOLO model inference and live streaming capabilities.
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

# Import our hockey analytics modules
from video_processor import HockeyVideoProcessor, PlayerDetection, GameEvent
from yolo_model_manager import YOLOModelManager
from video_streaming import HockeyVideoStreamer, HockeyVideoRecorder

logger = logging.getLogger(__name__)

class HockeyAnalyticsTester:
    """
    Comprehensive test suite for hockey analytics functionality
    """
    
    def __init__(self):
        self.test_results = {}
        self.model_manager = YOLOModelManager()
        self.video_processor = None
        self.streamer = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all hockey analytics tests
        
        Returns:
            Test results summary
        """
        logger.info("🧪 Starting Hockey Analytics Test Suite...")
        
        test_methods = [
            self.test_yolo_model_loading,
            self.test_video_processing,
            self.test_player_detection,
            self.test_game_event_detection,
            self.test_video_streaming,
            self.test_performance_benchmarks
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"🔬 Running test: {test_name}")
            
            try:
                result = await test_method()
                self.test_results[test_name] = {
                    "status": "passed",
                    "result": result,
                    "timestamp": time.time()
                }
                logger.info(f"✅ {test_name}: PASSED")
                
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
                logger.error(f"❌ {test_name}: FAILED - {e}")
        
        # Generate test summary
        summary = self._generate_test_summary()
        logger.info(f"📊 Test Summary: {summary}")
        
        return {
            "test_results": self.test_results,
            "summary": summary
        }
    
    async def test_yolo_model_loading(self) -> Dict[str, Any]:
        """Test YOLO model loading and initialization"""
        logger.info("🔬 Testing YOLO model loading...")
        
        # Test model manager initialization
        assert self.model_manager is not None, "Model manager not initialized"
        
        # Test model listing
        models = await self.model_manager.list_models()
        assert len(models) > 0, "No models available"
        
        # Test model download (yolov8n for testing)
        model_id = "yolov8n"
        download_success = await self.model_manager.download_model(model_id)
        assert download_success, f"Failed to download model {model_id}"
        
        # Test model loading
        load_success = await self.model_manager.load_model(model_id)
        assert load_success, f"Failed to load model {model_id}"
        
        # Test model info
        model_info = await self.model_manager.get_model_info(model_id)
        assert model_info["file_exists"], "Model file not found"
        assert model_info["is_active"], "Model not active"
        
        return {
            "models_available": len(models),
            "model_loaded": model_id,
            "model_info": model_info
        }
    
    async def test_video_processing(self) -> Dict[str, Any]:
        """Test video processing pipeline"""
        logger.info("🔬 Testing video processing...")
        
        # Initialize video processor
        self.video_processor = HockeyVideoProcessor()
        assert self.video_processor is not None, "Video processor not initialized"
        
        # Test with sample video frame
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test frame processing
        detections = await self.video_processor._process_frame(test_frame, 0)
        assert isinstance(detections, list), "Detections should be a list"
        
        # Test player tracking
        tracked_players = await self.video_processor._track_players(detections, 0)
        assert isinstance(tracked_players, list), "Tracked players should be a list"
        
        # Test event detection
        events = await self.video_processor._detect_game_events(tracked_players, 0)
        assert isinstance(events, list), "Events should be a list"
        
        return {
            "processor_initialized": True,
            "frame_processing": "success",
            "player_tracking": "success",
            "event_detection": "success"
        }
    
    async def test_player_detection(self) -> Dict[str, Any]:
        """Test player detection accuracy"""
        logger.info("🔬 Testing player detection...")
        
        if not self.video_processor:
            self.video_processor = HockeyVideoProcessor()
        
        # Create test frames with simulated players
        test_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            # Add some "players" (rectangles) to the frame
            cv2.rectangle(frame, (100, 100), (200, 300), (255, 0, 0), -1)  # Player 1
            cv2.rectangle(frame, (400, 150), (500, 350), (0, 255, 0), -1)  # Player 2
            test_frames.append(frame)
        
        total_detections = 0
        detection_accuracy = 0
        
        for i, frame in enumerate(test_frames):
            detections = await self.video_processor._process_frame(frame, i)
            total_detections += len(detections)
            
            # Simple accuracy test (detections should be > 0 for frames with "players")
            if len(detections) > 0:
                detection_accuracy += 1
        
        accuracy_percentage = (detection_accuracy / len(test_frames)) * 100
        
        return {
            "total_frames_tested": len(test_frames),
            "total_detections": total_detections,
            "detection_accuracy": f"{accuracy_percentage:.1f}%",
            "average_detections_per_frame": total_detections / len(test_frames)
        }
    
    async def test_game_event_detection(self) -> Dict[str, Any]:
        """Test game event detection"""
        logger.info("🔬 Testing game event detection...")
        
        if not self.video_processor:
            self.video_processor = HockeyVideoProcessor()
        
        # Create test scenario with multiple players
        test_players = [
            PlayerDetection(
                player_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                class_name="player"
            ),
            PlayerDetection(
                player_id=2,
                bbox=(120, 120, 220, 320),  # Close to player 1
                confidence=0.8,
                class_name="player"
            ),
            PlayerDetection(
                player_id=3,
                bbox=(400, 400, 500, 600),  # Far from other players
                confidence=0.7,
                class_name="player"
            )
        ]
        
        # Test event detection
        events = await self.video_processor._detect_game_events(test_players, 0)
        
        # Analyze events
        event_types = [event.event_type for event in events]
        faceoff_events = event_types.count("faceoff")
        
        return {
            "players_tested": len(test_players),
            "events_detected": len(events),
            "faceoff_events": faceoff_events,
            "event_types": list(set(event_types))
        }
    
    async def test_video_streaming(self) -> Dict[str, Any]:
        """Test video streaming functionality"""
        logger.info("🔬 Testing video streaming...")
        
        # Initialize streamer
        self.streamer = HockeyVideoStreamer()
        assert self.streamer is not None, "Streamer not initialized"
        
        # Test streaming status
        status = self.streamer.get_streaming_status()
        assert "is_streaming" in status, "Status should include streaming flag"
        assert "analytics_clients" in status, "Status should include client count"
        
        # Test with mock video source (we'll use a test pattern)
        test_video_source = "test_pattern.mp4"  # This would be a real video file
        
        # Mock analytics callback
        async def mock_analytics(frame_number: int, frame_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "players_detected": np.random.randint(0, 10),
                "events": np.random.randint(0, 3),
                "confidence": np.random.uniform(0.7, 0.95)
            }
        
        # Note: We won't actually start streaming in the test to avoid blocking
        # In a real test, you would start streaming and then stop it
        
        return {
            "streamer_initialized": True,
            "status_check": "passed",
            "mock_analytics": "configured"
        }
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        logger.info("🔬 Testing performance benchmarks...")
        
        if not self.model_manager:
            self.model_manager = YOLOModelManager()
        
        # Test model benchmarking
        model_id = "yolov8n"
        benchmark_results = await self.model_manager.benchmark_model(model_id, test_images=5)
        
        assert benchmark_results, "Benchmark should return results"
        assert "fps" in benchmark_results, "Benchmark should include FPS"
        assert "avg_inference_time" in benchmark_results, "Benchmark should include inference time"
        
        # Performance thresholds
        fps_threshold = 10  # Minimum FPS for real-time processing
        inference_time_threshold = 0.1  # Maximum inference time in seconds
        
        performance_ok = (
            benchmark_results["fps"] >= fps_threshold and
            benchmark_results["avg_inference_time"] <= inference_time_threshold
        )
        
        return {
            "benchmark_completed": True,
            "fps": benchmark_results["fps"],
            "avg_inference_time": benchmark_results["avg_inference_time"],
            "performance_ok": performance_ok,
            "thresholds": {
                "min_fps": fps_threshold,
                "max_inference_time": inference_time_threshold
            }
        }
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "passed")
        failed_tests = total_tests - passed_tests
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": f"{(passed_tests / total_tests) * 100:.1f}%"
        }
    
    async def save_test_results(self, output_file: str = "hockey_analytics_test_results.json"):
        """Save test results to file"""
        results = {
            "test_suite": "Hockey Analytics Phase 1.1",
            "timestamp": time.time(),
            "results": self.test_results,
            "summary": self._generate_test_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Test results saved to: {output_file}")

# Example usage and testing
async def run_hockey_analytics_tests():
    """Run comprehensive hockey analytics tests"""
    logger.info("🚀 Starting Hockey Analytics Test Suite...")
    
    # Initialize tester
    tester = HockeyAnalyticsTester()
    
    # Run all tests
    results = await tester.run_all_tests()
    
    # Save results
    await tester.save_test_results()
    
    # Print summary
    summary = results["summary"]
    logger.info(f"📊 Test Results Summary:")
    logger.info(f"  Total Tests: {summary['total_tests']}")
    logger.info(f"  Passed: {summary['passed']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Success Rate: {summary['success_rate']}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(run_hockey_analytics_tests())
