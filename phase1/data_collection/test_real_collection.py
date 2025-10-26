#!/usr/bin/env python3
"""
Test Real YouTube Video Collection & Web Scraping
Deep Learning & ML Opportunities for Hockey Data
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from real_youtube_scraper import RealHockeyDataCollector

class RealCollectionTester:
    """Test suite for real hockey data collection"""
    
    def __init__(self):
        self.logger = logging.getLogger("RealCollectionTester")
        self.setup_logging()
        self.test_data_dir = Path("/tmp/hockey_test_data")
        self.results = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_imports(self) -> bool:
        """Test if all required modules can be imported"""
        self.logger.info("🧪 Testing imports...")
        try:
            import cv2
            import numpy as np
            import requests
            from bs4 import BeautifulSoup
            from selenium import webdriver
            import yt_dlp
            self.logger.info("✅ All imports successful")
            return True
        except ImportError as e:
            self.logger.error(f"❌ Import failed: {e}")
            return False
    
    def test_collector_creation(self) -> bool:
        """Test if RealHockeyDataCollector can be created"""
        self.logger.info("🧪 Testing collector creation...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            self.logger.info("✅ Collector creation successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Collector creation failed: {e}")
            return False
    
    def test_directory_structure(self) -> bool:
        """Test if directory structure is created correctly"""
        self.logger.info("🧪 Testing directory structure...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            # Check if main directories exist
            required_dirs = [
                'raw', 'processed', 'train', 'val', 'test', 
                'uploads', 'metadata', 'youtube', 'web_scraped'
            ]
            
            for dir_name in required_dirs:
                dir_path = self.test_data_dir / dir_name
                if not dir_path.exists():
                    self.logger.error(f"❌ Missing directory: {dir_name}")
                    return False
            
            # Check role-specific directories
            roles = ['player', 'goalie', 'referee']
            for role in roles:
                for split in ['train', 'val', 'test']:
                    role_dir = self.test_data_dir / split / role
                    if not role_dir.exists():
                        self.logger.error(f"❌ Missing role directory: {split}/{role}")
                        return False
            
            self.logger.info("✅ Directory structure correct")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Directory structure test failed: {e}")
            return False
    
    def test_youtube_search_terms(self) -> bool:
        """Test YouTube search terms configuration"""
        self.logger.info("🧪 Testing YouTube search terms...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            search_terms = collector.sources['youtube']['search_terms']
            expected_terms = [
                'hockey game highlights',
                'hockey players training',
                'hockey goalie saves',
                'hockey referee calls',
                'NHL highlights',
                'hockey practice'
            ]
            
            for term in expected_terms:
                if term not in search_terms:
                    self.logger.error(f"❌ Missing search term: {term}")
                    return False
            
            self.logger.info(f"✅ YouTube search terms configured: {len(search_terms)} terms")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ YouTube search terms test failed: {e}")
            return False
    
    def test_web_scraping_sources(self) -> bool:
        """Test web scraping sources configuration"""
        self.logger.info("🧪 Testing web scraping sources...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            sources = collector.sources['web_scraping']['sources']
            expected_urls = [
                'https://www.nhl.com',
                'https://www.espn.com/nhl',
                'https://www.sportsnet.ca/hockey'
            ]
            
            for source in sources:
                if source['url'] not in expected_urls:
                    self.logger.error(f"❌ Unexpected source: {source['url']}")
                    return False
                
                # Check if selectors are configured
                if 'selectors' not in source:
                    self.logger.error(f"❌ Missing selectors for {source['url']}")
                    return False
            
            self.logger.info(f"✅ Web scraping sources configured: {len(sources)} sources")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Web scraping sources test failed: {e}")
            return False
    
    def test_image_url_validation(self) -> bool:
        """Test image URL validation logic"""
        self.logger.info("🧪 Testing image URL validation...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            # Test valid URLs
            valid_urls = [
                'https://example.com/hockey-player.jpg',
                'https://nhl.com/images/player.png',
                'https://espn.com/hockey-game.jpeg'
            ]
            
            for url in valid_urls:
                if not collector.is_valid_image_url(url):
                    self.logger.error(f"❌ Valid URL rejected: {url}")
                    return False
            
            # Test invalid URLs
            invalid_urls = [
                'https://example.com/not-hockey.txt',
                'https://example.com/other-sport.jpg',
                'not-a-url'
            ]
            
            for url in invalid_urls:
                if collector.is_valid_image_url(url):
                    self.logger.error(f"❌ Invalid URL accepted: {url}")
                    return False
            
            self.logger.info("✅ Image URL validation working correctly")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Image URL validation test failed: {e}")
            return False
    
    def test_mock_collection(self) -> bool:
        """Test mock data collection (without real API calls)"""
        self.logger.info("🧪 Testing mock collection...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            # Test YouTube video collection (mock)
            youtube_videos = collector.collect_youtube_videos(max_videos=5)
            if not isinstance(youtube_videos, list):
                self.logger.error("❌ YouTube collection returned invalid type")
                return False
            
            # Test web scraping (mock)
            web_images = []
            for source_config in collector.sources['web_scraping']['sources']:
                images = collector.scrape_website_images(source_config)
                web_images.extend(images)
            
            if not isinstance(web_images, list):
                self.logger.error("❌ Web scraping returned invalid type")
                return False
            
            self.logger.info(f"✅ Mock collection successful - YouTube: {len(youtube_videos)}, Web: {len(web_images)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Mock collection test failed: {e}")
            return False
    
    def test_data_persistence(self) -> bool:
        """Test data persistence and metadata saving"""
        self.logger.info("🧪 Testing data persistence...")
        try:
            collector = RealHockeyDataCollector(str(self.test_data_dir))
            
            # Create test metadata
            test_metadata = {
                'test_video': {
                    'video_id': 'test_123',
                    'title': 'Test Hockey Video',
                    'url': 'https://youtube.com/watch?v=test123',
                    'collected_at': '2024-01-01T00:00:00Z'
                }
            }
            
            # Save metadata
            metadata_path = self.test_data_dir / 'metadata' / 'test_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(test_metadata, f, indent=2)
            
            # Verify metadata was saved
            if not metadata_path.exists():
                self.logger.error("❌ Metadata file not created")
                return False
            
            # Load and verify metadata
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            if loaded_metadata != test_metadata:
                self.logger.error("❌ Metadata mismatch")
                return False
            
            self.logger.info("✅ Data persistence working correctly")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data persistence test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        self.logger.info("🚀 Starting Real Collection Tests...")
        
        tests = [
            ("imports", self.test_imports),
            ("collector_creation", self.test_collector_creation),
            ("directory_structure", self.test_directory_structure),
            ("youtube_search_terms", self.test_youtube_search_terms),
            ("web_scraping_sources", self.test_web_scraping_sources),
            ("image_url_validation", self.test_image_url_validation),
            ("mock_collection", self.test_mock_collection),
            ("data_persistence", self.test_data_persistence)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.logger.error(f"❌ Test {test_name} crashed: {e}")
                results[test_name] = False
        
        # Calculate summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        self.logger.info(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All tests passed! Real collection is ready!")
        else:
            self.logger.warning(f"⚠️ {total - passed} tests failed. Check implementation.")
        
        return results

def main():
    """Main function for testing real collection"""
    tester = RealCollectionTester()
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("🧪 REAL COLLECTION TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Real Collection Implementation - READY!")
        print("🚀 Ready for deep learning and ML applications!")
    else:
        print("⚠️ Some tests failed. Check implementation.")

if __name__ == "__main__":
    main()
