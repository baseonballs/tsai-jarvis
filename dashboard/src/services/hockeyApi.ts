/**
 * TSAI Jarvis - Hockey Analytics API Service
 * Client-side API service for connecting to the hockey analytics backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8002';

export interface GameState {
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  period: number;
  time_remaining: string;
  power_play: {
    home: boolean;
    away: boolean;
    time_remaining: string;
  };
}

export interface PlayerStats {
  player_id: number;
  name: string;
  team: string;
  position: string;
  goals: number;
  assists: number;
  points: number;
  plus_minus: number;
  time_on_ice: string;
  speed: number;
  distance_covered: number;
}

export interface LiveEvent {
  event_id: string;
  event_type: 'goal' | 'save' | 'penalty' | 'faceoff' | 'hit' | 'shot';
  timestamp: string;
  player_name: string;
  team: string;
  description: string;
  confidence: number;
}

export interface AnalyticsMetrics {
  detection_accuracy: number;
  processing_time: number;
  fps: number;
  active_players: number;
  total_events: number;
  momentum: { [key: string]: number };
  pressure: { [key: string]: number };
  enhanced_features: { [key: string]: boolean };
}

export interface VideoProcessingRequest {
  video_source: string;
  model_id: string;
  confidence_threshold: number;
  enable_streaming: boolean;
}

class HockeyApiService {
  private baseUrl: string;
  private wsConnection: WebSocket | null = null;
  private wsReconnectInterval: NodeJS.Timeout | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  /**
   * Get current game state
   */
  async getGameState(): Promise<GameState> {
    const response = await fetch(`${this.baseUrl}/api/game/state`);
    return response.json();
  }

  /**
   * Get player statistics
   */
  async getPlayerStats(): Promise<PlayerStats[]> {
    const response = await fetch(`${this.baseUrl}/api/players/stats`);
    return response.json();
  }

  /**
   * Get live events
   */
  async getLiveEvents(): Promise<LiveEvent[]> {
    const response = await fetch(`${this.baseUrl}/api/events/live`);
    return response.json();
  }

  /**
   * Get analytics metrics
   */
  async getAnalyticsMetrics(): Promise<AnalyticsMetrics> {
    const response = await fetch(`${this.baseUrl}/api/analytics/metrics`);
    return response.json();
  }

  /**
   * Start video processing
   */
  async startVideoProcessing(request: VideoProcessingRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/video/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return response.json();
  }

  /**
   * Get video processing status
   */
  async getVideoProcessingStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/video/status`);
    return response.json();
  }

  /**
   * Stop video processing
   */
  async stopVideoProcessing(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/video/stop`, {
      method: 'POST',
    });
    return response.json();
  }

  /**
   * Get available models
   */
  async getAvailableModels(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/models`);
    return response.json();
  }

  /**
   * Load specific model
   */
  async loadModel(modelId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/models/${modelId}/load`, {
      method: 'POST',
    });
    return response.json();
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  connectWebSocket(onMessage: (data: any) => void, onError?: (error: Event) => void): void {
    if (this.wsConnection) {
      this.wsConnection.close();
    }

    const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws/analytics';
    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onopen = () => {
      console.log('🔌 Connected to hockey analytics WebSocket');
      if (this.wsReconnectInterval) {
        clearInterval(this.wsReconnectInterval);
        this.wsReconnectInterval = null;
      }
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('❌ Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };

    this.wsConnection.onclose = () => {
      console.log('🔌 WebSocket connection closed');
      // Attempt to reconnect after 5 seconds
      this.wsReconnectInterval = setTimeout(() => {
        console.log('🔄 Attempting to reconnect WebSocket...');
        this.connectWebSocket(onMessage, onError);
      }, 5000);
    };
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    if (this.wsReconnectInterval) {
      clearInterval(this.wsReconnectInterval);
      this.wsReconnectInterval = null;
    }
  }

  /**
   * Send message via WebSocket
   */
  sendWebSocketMessage(message: any): void {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify(message));
    }
  }

  /**
   * Get momentum analytics
   */
  async getMomentumAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/momentum`);
    return response.json();
  }

  /**
   * Get pressure analytics
   */
  async getPressureAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/pressure`);
    return response.json();
  }

  /**
   * Get speed analytics
   */
  async getSpeedAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/speed`);
    return response.json();
  }

  /**
   * Get shot analytics
   */
  async getShotAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/shots`);
    return response.json();
  }

  /**
   * Get formation analytics
   */
  async getFormationAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/formations`);
    return response.json();
  }

  /**
   * Get strategy analytics
   */
  async getStrategyAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/strategy`);
    return response.json();
  }

  /**
   * Get game sessions
   */
  async getGameSessions(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/games/sessions`);
    return response.json();
  }

  /**
   * Get historical data
   */
  async getHistoricalData(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/historical`);
    return response.json();
  }

  /**
   * Get team analyses
   */
  async getTeamAnalyses(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/teams`);
    return response.json();
  }

  /**
   * Get production metrics
   */
  async getProductionMetrics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analytics/production`);
    return response.json();
  }
}

// Export singleton instance
export const hockeyApi = new HockeyApiService();
export default hockeyApi;
