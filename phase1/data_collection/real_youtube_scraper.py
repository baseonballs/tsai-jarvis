#!/usr/bin/env python3
"""
Real YouTube Video Collection & Web Scraping Implementation
Deep Learning & ML Opportunities for Hockey Data
"""

import os
import sys
import json
import time
import logging
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import shutil
import yt_dlp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re

class RealHockeyDataCollector:
    """Real implementation of hockey data collection from YouTube and web sources"""
    
    def __init__(self, data_dir: str = "/data/hockey_players"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("RealHockeyDataCollector")
        self.setup_logging()
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize data sources
        self.setup_data_sources()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_directories(self):
        """Setup directory structure for hockey data"""
        try:
            # Create main directories
            directories = [
                self.data_dir,
                self.data_dir / 'raw',
                self.data_dir / 'processed',
                self.data_dir / 'train',
                self.data_dir / 'val',
                self.data_dir / 'test',
                self.data_dir / 'uploads',
                self.data_dir / 'metadata',
                self.data_dir / 'youtube',
                self.data_dir / 'web_scraped'
            ]
            
            # Create role-specific directories
            roles = ['player', 'goalie', 'referee']
            for role in roles:
                for split in ['train', 'val', 'test']:
                    directories.append(self.data_dir / split / role)
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"✅ Directory structure created at {self.data_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup directories: {e}")
            raise
    
    def setup_data_sources(self):
        """Setup data source configurations"""
        self.sources = {
            'youtube': {
                'enabled': True,
                'api_key': os.getenv('YOUTUBE_API_KEY'),
                'search_terms': [
                    'hockey game highlights',
                    'hockey players training',
                    'hockey goalie saves',
                    'hockey referee calls',
                    'NHL highlights',
                    'hockey practice',
                    'hockey skills',
                    'hockey drills',
                    'hockey coaching',
                    'hockey equipment'
                ],
                'max_videos_per_term': 10,
                'video_quality': '720p',
                'frame_interval': 30  # Extract frame every 30 seconds
            },
            'web_scraping': {
                'enabled': True,
                'sources': [
                    {
                        'url': 'https://www.nhl.com',
                        'selectors': {
                            'images': 'img[src*="hockey"], img[alt*="hockey"], img[alt*="player"]',
                            'articles': 'article, .news-item, .story'
                        }
                    },
                    {
                        'url': 'https://www.espn.com/nhl',
                        'selectors': {
                            'images': 'img[src*="hockey"], img[alt*="hockey"], img[alt*="player"]',
                            'articles': 'article, .news-item, .story'
                        }
                    },
                    {
                        'url': 'https://www.sportsnet.ca/hockey',
                        'selectors': {
                            'images': 'img[src*="hockey"], img[alt*="hockey"], img[alt*="player"]',
                            'articles': 'article, .news-item, .story'
                        }
                    }
                ],
                'max_images_per_source': 50,
                'min_image_size': (224, 224),
                'quality_threshold': 0.7
            }
        }
    
    def collect_youtube_videos(self, max_videos: int = 50) -> List[Dict[str, Any]]:
        """Collect hockey videos from YouTube using yt-dlp"""
        try:
            self.logger.info("📺 Collecting hockey videos from YouTube...")
            
            if not self.sources['youtube']['enabled']:
                self.logger.warning("⚠️ YouTube collection disabled")
                return []
            
            collected_videos = []
            
            for search_term in self.sources['youtube']['search_terms']:
                self.logger.info(f"🔍 Searching for: {search_term}")
                
                # Use yt-dlp to search and collect videos
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'default_search': 'ytsearch',
                    'max_downloads': self.sources['youtube']['max_videos_per_term']
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        # Search for videos
                        search_query = f"ytsearch{self.sources['youtube']['max_videos_per_term']}:{search_term}"
                        results = ydl.extract_info(search_query, download=False)
                        
                        if 'entries' in results:
                            for video in results['entries']:
                                if video:
                                    video_info = {
                                        'video_id': video.get('id', ''),
                                        'title': video.get('title', ''),
                                        'url': video.get('url', ''),
                                        'thumbnail': video.get('thumbnail', ''),
                                        'duration': video.get('duration', 0),
                                        'view_count': video.get('view_count', 0),
                                        'published_at': video.get('upload_date', ''),
                                        'search_term': search_term,
                                        'collected_at': datetime.now().isoformat()
                                    }
                                    collected_videos.append(video_info)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to search for '{search_term}': {e}")
                        continue
            
            self.logger.info(f"✅ Collected {len(collected_videos)} videos from YouTube")
            return collected_videos
            
        except Exception as e:
            self.logger.error(f"❌ YouTube video collection failed: {e}")
            return []
    
    def download_youtube_video(self, video_info: Dict[str, Any], output_dir: Path) -> bool:
        """Download a YouTube video using yt-dlp"""
        try:
            video_id = video_info['video_id']
            video_url = video_info['url']
            
            self.logger.info(f"📥 Downloading video: {video_info['title']}")
            
            # Create output directory for this video
            video_dir = output_dir / f"video_{video_id}"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': str(video_dir / '%(title)s.%(ext)s'),
                'format': 'best[height<=720]',  # Limit to 720p for efficiency
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Save video metadata
            metadata_path = video_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(video_info, f, indent=2)
            
            self.logger.info(f"✅ Video downloaded: {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Video download failed: {e}")
            return False
    
    def extract_video_frames(self, video_path: Path, output_dir: Path, 
                           frame_interval: int = 30) -> List[str]:
        """Extract frames from a video file"""
        try:
            self.logger.info(f"🎬 Extracting frames from: {video_path}")
            
            # Create output directory for frames
            frames_dir = output_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"❌ Could not open video: {video_path}")
                return []
            
            frame_count = 0
            extracted_frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % (frame_interval * fps) == 0:
                    frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"✅ Extracted {len(extracted_frames)} frames")
            return extracted_frames
            
        except Exception as e:
            self.logger.error(f"❌ Frame extraction failed: {e}")
            return []
    
    def setup_selenium_driver(self) -> webdriver.Chrome:
        """Setup Selenium WebDriver for web scraping"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=chrome_options)
            return driver
            
        except Exception as e:
            self.logger.error(f"❌ Selenium driver setup failed: {e}")
            raise
    
    def scrape_website_images(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape images from a website using Selenium"""
        try:
            self.logger.info(f"🌐 Scraping images from: {source_config['url']}")
            
            driver = self.setup_selenium_driver()
            collected_images = []
            
            try:
                # Navigate to the website
                driver.get(source_config['url'])
                time.sleep(3)  # Wait for page to load
                
                # Find images using CSS selectors
                image_elements = driver.find_elements(By.CSS_SELECTOR, source_config['selectors']['images'])
                
                for i, img_element in enumerate(image_elements[:source_config.get('max_images_per_source', 50)]):
                    try:
                        # Get image attributes
                        img_src = img_element.get_attribute('src')
                        img_alt = img_element.get_attribute('alt')
                        img_title = img_element.get_attribute('title')
                        
                        if img_src and self.is_valid_image_url(img_src):
                            image_info = {
                                'url': img_src,
                                'alt_text': img_alt or '',
                                'title': img_title or '',
                                'source': source_config['url'],
                                'collected_at': datetime.now().isoformat()
                            }
                            collected_images.append(image_info)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to process image {i}: {e}")
                        continue
            
            finally:
                driver.quit()
            
            self.logger.info(f"✅ Collected {len(collected_images)} images from {source_config['url']}")
            return collected_images
            
        except Exception as e:
            self.logger.error(f"❌ Web scraping failed: {e}")
            return []
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if image URL is valid and relevant"""
        try:
            # Check if URL is valid
            if not url or not url.startswith('http'):
                return False
            
            # Check if it's an image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            if not any(ext in url.lower() for ext in image_extensions):
                return False
            
            # Check if it's a hockey-related image
            hockey_keywords = ['hockey', 'player', 'game', 'ice', 'rink', 'stick', 'puck']
            url_lower = url.lower()
            if not any(keyword in url_lower for keyword in hockey_keywords):
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def download_image(self, image_info: Dict[str, Any], output_dir: Path) -> bool:
        """Download an image from URL"""
        try:
            image_url = image_info['url']
            
            # Create filename from URL
            image_filename = hashlib.md5(image_url.encode()).hexdigest()[:8] + '.jpg'
            image_path = output_dir / image_filename
            
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Save metadata
            metadata_path = output_dir / f"{image_filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(image_info, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Image download failed: {e}")
            return False
    
    def run_complete_collection(self) -> Dict[str, Any]:
        """Run complete data collection from all sources"""
        try:
            self.logger.info("🚀 Starting complete hockey data collection...")
            
            collection_results = {
                'youtube_videos': [],
                'web_images': [],
                'total_collected': 0,
                'collection_time': datetime.now().isoformat()
            }
            
            # Step 1: Collect YouTube videos
            if self.sources['youtube']['enabled']:
                self.logger.info("📺 Step 1: Collecting YouTube videos...")
                youtube_videos = self.collect_youtube_videos()
                collection_results['youtube_videos'] = youtube_videos
                
                # Download videos and extract frames
                for video_info in youtube_videos[:5]:  # Limit to 5 videos for demo
                    video_dir = self.data_dir / 'youtube' / f"video_{video_info['video_id']}"
                    if self.download_youtube_video(video_info, video_dir):
                        # Extract frames from downloaded video
                        video_files = list(video_dir.glob('*.mp4'))
                        if video_files:
                            frames = self.extract_video_frames(
                                video_files[0], 
                                video_dir, 
                                self.sources['youtube']['frame_interval']
                            )
                            video_info['frames_extracted'] = len(frames)
            
            # Step 2: Scrape web images
            if self.sources['web_scraping']['enabled']:
                self.logger.info("🌐 Step 2: Scraping web images...")
                for source_config in self.sources['web_scraping']['sources']:
                    web_images = self.scrape_website_images(source_config)
                    collection_results['web_images'].extend(web_images)
                    
                    # Download images
                    for image_info in web_images[:10]:  # Limit to 10 images per source
                        image_dir = self.data_dir / 'web_scraped' / source_config['url'].split('//')[1]
                        image_dir.mkdir(parents=True, exist_ok=True)
                        self.download_image(image_info, image_dir)
            
            # Calculate totals
            collection_results['total_collected'] = len(collection_results['youtube_videos']) + len(collection_results['web_images'])
            
            # Save collection report
            report_path = self.data_dir / 'metadata' / 'collection_report.json'
            with open(report_path, 'w') as f:
                json.dump(collection_results, f, indent=2)
            
            self.logger.info("🎉 Complete hockey data collection finished!")
            self.logger.info(f"📊 Total collected: {collection_results['total_collected']}")
            self.logger.info(f"📺 YouTube videos: {len(collection_results['youtube_videos'])}")
            self.logger.info(f"🌐 Web images: {len(collection_results['web_images'])}")
            
            return collection_results
            
        except Exception as e:
            self.logger.error(f"❌ Complete collection failed: {e}")
            return {}

def main():
    """Main function for real hockey data collection"""
    collector = RealHockeyDataCollector()
    results = collector.run_complete_collection()
    
    if results:
        print("🎉 Real Hockey Data Collection - SUCCESS!")
        print(f"📊 Total collected: {results['total_collected']}")
        print(f"📺 YouTube videos: {len(results['youtube_videos'])}")
        print(f"🌐 Web images: {len(results['web_images'])}")
        print("\n🚀 Ready for deep learning and ML applications!")
    else:
        print("❌ Real Hockey Data Collection - FAILED!")

if __name__ == "__main__":
    main()
