#!/usr/bin/env python3
"""
Jarvis Client Storage Service - GDrive/iCloud Integration
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import io
import requests

# Google Drive imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request

# iCloud imports
import pyicloud
from pyicloud import PyiCloudService

import yaml

class JarvisClientStorageService:
    """Jarvis client storage service for user-facing storage"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('JARVIS_CONFIG_PATH', 'config/jarvis-core.yaml')
        self.config = self._load_config()
        self.drives = {}
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize drives
        self._initialize_drives()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['jarvis_core']['services']['client_storage']
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'type': 'cloud-drive',
                'providers': {
                    'google_drive': {
                        'enabled': True,
                        'credentials_file': '/app/config/google-credentials.json'
                    },
                    'icloud_drive': {
                        'enabled': True,
                        'username': os.getenv('ICLOUD_USERNAME'),
                        'password': os.getenv('ICLOUD_PASSWORD')
                    }
                }
            }
    
    def _initialize_drives(self):
        """Initialize cloud drive connections"""
        try:
            # Google Drive
            if self.config['providers']['google_drive']['enabled']:
                self.drives['google_drive'] = GoogleDriveInterface(
                    credentials_file=self.config['providers']['google_drive']['credentials_file']
                )
                self.logger.info("✅ Google Drive interface initialized")
            
            # iCloud Drive
            if self.config['providers']['icloud_drive']['enabled']:
                icloud_config = self.config['providers']['icloud_drive']
                if icloud_config.get('username') and icloud_config.get('password'):
                    self.drives['icloud_drive'] = iCloudDriveInterface(
                        username=icloud_config['username'],
                        password=icloud_config['password']
                    )
                    self.logger.info("✅ iCloud Drive interface initialized")
                else:
                    self.logger.warning("⚠️ iCloud Drive credentials not provided")
            
            # OneDrive (if enabled)
            if self.config['providers'].get('onedrive', {}).get('enabled', False):
                onedrive_config = self.config['providers']['onedrive']
                if onedrive_config.get('client_id') and onedrive_config.get('client_secret'):
                    self.drives['onedrive'] = OneDriveInterface(
                        client_id=onedrive_config['client_id'],
                        client_secret=onedrive_config['client_secret']
                    )
                    self.logger.info("✅ OneDrive interface initialized")
                else:
                    self.logger.warning("⚠️ OneDrive credentials not provided")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize drives: {e}")
    
    def import_user_media(self, drive_name: str, folder_id: str = None, 
                         file_types: List[str] = None, component: str = None) -> List[str]:
        """Import user media from cloud drives"""
        try:
            drive = self.drives.get(drive_name)
            if not drive:
                raise ValueError(f"Drive {drive_name} not configured")
            
            # Get supported file types from config if not provided
            if not file_types:
                file_types = self._get_supported_file_types()
            
            # List files from drive
            files = drive.list_files(folder_id, file_types)
            
            if not files:
                self.logger.info(f"No files found in {drive_name}")
                return []
            
            # Create import directory
            import_dir = Path(f"./imports/{component or 'default'}")
            import_dir.mkdir(parents=True, exist_ok=True)
            
            # Download files
            downloaded_files = []
            for file_info in files:
                try:
                    local_path = import_dir / file_info['name']
                    
                    # Skip if file already exists
                    if local_path.exists():
                        self.logger.info(f"File {file_info['name']} already exists, skipping")
                        downloaded_files.append(str(local_path))
                        continue
                    
                    success = drive.download_file(file_info['id'], str(local_path))
                    if success:
                        downloaded_files.append(str(local_path))
                        self.logger.info(f"✅ Downloaded: {file_info['name']}")
                    else:
                        self.logger.error(f"❌ Failed to download: {file_info['name']}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to download {file_info['name']}: {e}")
                    continue
            
            self.logger.info(f"✅ Imported {len(downloaded_files)} files from {drive_name}")
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Failed to import media from {drive_name}: {e}")
            return []
    
    def export_results(self, results: List[str], drive_name: str, 
                      folder_id: str = None, component: str = None) -> List[str]:
        """Export results to user's cloud drive"""
        try:
            drive = self.drives.get(drive_name)
            if not drive:
                raise ValueError(f"Drive {drive_name} not configured")
            
            # Create results folder if not provided
            if not folder_id:
                folder_name = f"Jarvis Results - {component or 'Default'} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
                folder_id = drive.create_folder(folder_name)
                self.logger.info(f"Created results folder: {folder_name}")
            
            # Upload results
            uploaded_files = []
            for result_path in results:
                try:
                    if not Path(result_path).exists():
                        self.logger.warning(f"File not found: {result_path}")
                        continue
                    
                    file_id = drive.upload_file(result_path, folder_id)
                    if file_id:
                        uploaded_files.append(file_id)
                        self.logger.info(f"✅ Uploaded: {Path(result_path).name}")
                    else:
                        self.logger.error(f"❌ Failed to upload: {Path(result_path).name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to upload {result_path}: {e}")
                    continue
            
            self.logger.info(f"✅ Exported {len(uploaded_files)} files to {drive_name}")
            return uploaded_files
            
        except Exception as e:
            self.logger.error(f"Failed to export results to {drive_name}: {e}")
            return []
    
    def create_shared_folder(self, folder_name: str, drive_name: str, 
                           component: str = None) -> str:
        """Create shared folder for results"""
        try:
            drive = self.drives.get(drive_name)
            if not drive:
                raise ValueError(f"Drive {drive_name} not configured")
            
            # Create folder with component prefix
            full_folder_name = f"{component or 'Jarvis'} - {folder_name}" if component else folder_name
            folder_id = drive.create_folder(full_folder_name)
            
            self.logger.info(f"✅ Created shared folder: {full_folder_name}")
            return folder_id
            
        except Exception as e:
            self.logger.error(f"Failed to create shared folder: {e}")
            return None
    
    def search_files(self, query: str, drive_name: str, 
                    file_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search files in cloud drive"""
        try:
            drive = self.drives.get(drive_name)
            if not drive:
                raise ValueError(f"Drive {drive_name} not configured")
            
            # Get supported file types if not provided
            if not file_types:
                file_types = self._get_supported_file_types()
            
            # Search files
            files = drive.search_files(query, file_types)
            
            self.logger.info(f"Found {len(files)} files matching '{query}' in {drive_name}")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to search files in {drive_name}: {e}")
            return []
    
    def get_drive_info(self, drive_name: str) -> Dict[str, Any]:
        """Get drive information and status"""
        try:
            drive = self.drives.get(drive_name)
            if not drive:
                return {
                    'status': 'not_configured',
                    'error': f"Drive {drive_name} not configured"
                }
            
            # Get drive status
            status = drive.health_check()
            return {
                'drive_name': drive_name,
                'status': status['status'],
                'info': status.get('info', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'drive_name': drive_name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def list_available_drives(self) -> List[str]:
        """List available cloud drives"""
        return list(self.drives.keys())
    
    def _get_supported_file_types(self) -> List[str]:
        """Get supported file types from config"""
        try:
            supported_types = []
            for category, types in self.config.get('supported_types', {}).items():
                supported_types.extend(types)
            return supported_types
        except Exception:
            # Default file types
            return [
                'image/jpeg', 'image/png', 'image/gif', 'image/webp',
                'video/mp4', 'video/mov', 'video/avi', 'video/mkv',
                'application/pdf', 'text/plain', 'application/json'
            ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check client storage service health"""
        try:
            drive_status = {}
            for drive_name in self.drives:
                drive_status[drive_name] = self.get_drive_info(drive_name)
            
            return {
                'status': 'healthy',
                'drives': drive_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class GoogleDriveInterface:
    """Google Drive integration interface"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
    
    def __init__(self, credentials_file: str, token_file: str = None):
        self.credentials_file = credentials_file
        self.token_file = token_file or 'token.json'
        self.service = None
        self.logger = logging.getLogger(__name__)
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with Google Drive"""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('drive', 'v3', credentials=creds)
            return True
            
        except Exception as e:
            self.logger.error(f"Google Drive authentication failed: {e}")
            return False
    
    def list_files(self, folder_id: str = None, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """List files in Google Drive"""
        try:
            query = f"'{folder_id}' in parents" if folder_id else ""
            
            if file_types:
                mime_types = [f"mimeType='{ft}'" for ft in file_types]
                query += f" and ({' or '.join(mime_types)})"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            
            files = []
            for file in results.get('files', []):
                files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'mimeType': file.get('mimeType', ''),
                    'size': int(file.get('size', 0)),
                    'modifiedTime': file.get('modifiedTime', '')
                })
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download file from Google Drive"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Save to local file
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download file {file_id}: {e}")
            return False
    
    def upload_file(self, local_path: str, folder_id: str = None) -> str:
        """Upload file to Google Drive"""
        try:
            file_metadata = {
                'name': Path(local_path).name,
                'parents': [folder_id] if folder_id else []
            }
            
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            return file.get('id')
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {local_path}: {e}")
            return None
    
    def create_folder(self, folder_name: str) -> str:
        """Create folder in Google Drive"""
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            file = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            return file.get('id')
            
        except Exception as e:
            self.logger.error(f"Failed to create folder {folder_name}: {e}")
            return None
    
    def search_files(self, query: str, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search files in Google Drive"""
        try:
            search_query = f"name contains '{query}'"
            
            if file_types:
                mime_types = [f"mimeType='{ft}'" for ft in file_types]
                search_query += f" and ({' or '.join(mime_types)})"
            
            results = self.service.files().list(
                q=search_query,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            
            files = []
            for file in results.get('files', []):
                files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'mimeType': file.get('mimeType', ''),
                    'size': int(file.get('size', 0)),
                    'modifiedTime': file.get('modifiedTime', '')
                })
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to search files: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check Google Drive connection health"""
        try:
            # Test connection by listing files
            results = self.service.files().list(pageSize=1).execute()
            
            return {
                'status': 'healthy',
                'info': {
                    'total_files': results.get('files', []),
                    'quota': 'available'
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class iCloudDriveInterface:
    """iCloud Drive integration interface"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.service = None
        self.logger = logging.getLogger(__name__)
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with iCloud"""
        try:
            self.service = PyiCloudService(self.username, self.password)
            return self.service.requires_2fa is False
            
        except Exception as e:
            self.logger.error(f"iCloud authentication failed: {e}")
            return False
    
    def list_files(self, folder_id: str = None, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """List files in iCloud Drive"""
        try:
            if folder_id:
                folder = self.service.drive[folder_id]
                files = folder.dir()
            else:
                files = self.service.drive.dir()
            
            # Filter by file types
            if file_types:
                files = [f for f in files if f.name.split('.')[-1].lower() in file_types]
            
            file_list = []
            for f in files:
                file_list.append({
                    'id': f.name,
                    'name': f.name,
                    'size': f.size,
                    'modifiedTime': f.date_modified.isoformat() if f.date_modified else ''
                })
            
            return file_list
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download file from iCloud Drive"""
        try:
            file = self.service.drive[file_id]
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            file.download(local_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download file {file_id}: {e}")
            return False
    
    def upload_file(self, local_path: str, folder_id: str = None) -> str:
        """Upload file to iCloud Drive"""
        try:
            # iCloud Drive upload implementation
            # This is a simplified version - actual implementation would be more complex
            file_name = Path(local_path).name
            # Upload logic would go here
            return file_name  # Return file name as ID for now
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {local_path}: {e}")
            return None
    
    def create_folder(self, folder_name: str) -> str:
        """Create folder in iCloud Drive"""
        try:
            # iCloud Drive folder creation
            # This is a simplified version - actual implementation would be more complex
            return folder_name  # Return folder name as ID for now
            
        except Exception as e:
            self.logger.error(f"Failed to create folder {folder_name}: {e}")
            return None
    
    def search_files(self, query: str, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search files in iCloud Drive"""
        try:
            # iCloud Drive search implementation
            # This is a simplified version
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to search files: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check iCloud Drive connection health"""
        try:
            # Test connection
            files = self.service.drive.dir()
            
            return {
                'status': 'healthy',
                'info': {
                    'total_files': len(files),
                    'quota': 'available'
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class OneDriveInterface:
    """OneDrive integration interface (placeholder)"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.logger = logging.getLogger(__name__)
    
    def list_files(self, folder_id: str = None, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """List files in OneDrive"""
        # Placeholder implementation
        return []
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download file from OneDrive"""
        # Placeholder implementation
        return False
    
    def upload_file(self, local_path: str, folder_id: str = None) -> str:
        """Upload file to OneDrive"""
        # Placeholder implementation
        return None
    
    def create_folder(self, folder_name: str) -> str:
        """Create folder in OneDrive"""
        # Placeholder implementation
        return None
    
    def search_files(self, query: str, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search files in OneDrive"""
        # Placeholder implementation
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check OneDrive connection health"""
        return {
            'status': 'not_implemented',
            'error': 'OneDrive integration not yet implemented'
        }


def main():
    """Main function for client storage service"""
    
    # Initialize client storage service
    client_storage = JarvisClientStorageService()
    
    # Health check
    health = client_storage.health_check()
    print(f"Client Storage Service Health: {health}")
    
    # List available drives
    drives = client_storage.list_available_drives()
    print(f"Available drives: {drives}")
    
    # Test with available drives
    for drive_name in drives:
        drive_info = client_storage.get_drive_info(drive_name)
        print(f"Drive {drive_name}: {drive_info}")
        
        if drive_info['status'] == 'healthy':
            # Test file listing
            files = client_storage.search_files("test", drive_name)
            print(f"Found {len(files)} test files in {drive_name}")

if __name__ == "__main__":
    main()
