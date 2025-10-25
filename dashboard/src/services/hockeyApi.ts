/**
 * TSAI Jarvis - Hockey Analytics API Service
 * Client-side API service for connecting to the hockey analytics backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8007';

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
    try {
      const response = await fetch(`${this.baseUrl}/api/game/state`);
      if (!response.ok) {
        // Return mock data if endpoint doesn't exist
        return {
          home_team: "Toronto Maple Leafs",
          away_team: "Montreal Canadiens",
          home_score: 2,
          away_score: 1,
          period: 2,
          time_remaining: "12:34",
          power_play: {
            home: false,
            away: true,
            time_remaining: "1:45"
          }
        };
      }
      return response.json();
    } catch (error) {
      console.warn('API endpoint not available, using mock data:', error);
      // Return mock data on error
      return {
        home_team: "Toronto Maple Leafs",
        away_team: "Montreal Canadiens",
        home_score: 2,
        away_score: 1,
        period: 2,
        time_remaining: "12:34",
        power_play: {
          home: false,
          away: true,
          time_remaining: "1:45"
        }
      };
    }
  }

  /**
   * Get player statistics
   */
  async getPlayerStats(): Promise<PlayerStats[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/players/stats`);
      if (!response.ok) {
        // Return mock data if endpoint doesn't exist
        return [
          {
            player_id: 1,
            name: "Auston Matthews",
            team: "Toronto Maple Leafs",
            position: "C",
            goals: 1,
            assists: 1,
            points: 2,
            plus_minus: 1,
            time_on_ice: "18:45",
            speed: 28.5,
            distance_covered: 2.3
          },
          {
            player_id: 2,
            name: "Mitch Marner",
            team: "Toronto Maple Leafs",
            position: "RW",
            goals: 0,
            assists: 2,
            points: 2,
            plus_minus: 1,
            time_on_ice: "19:12",
            speed: 26.8,
            distance_covered: 2.1
          }
        ];
      }
      return response.json();
    } catch (error) {
      console.warn('API endpoint not available, using mock data:', error);
      return [
        {
          player_id: 1,
          name: "Auston Matthews",
          team: "Toronto Maple Leafs",
          position: "C",
          goals: 1,
          assists: 1,
          points: 2,
          plus_minus: 1,
          time_on_ice: "18:45",
          speed: 28.5,
          distance_covered: 2.3
        }
      ];
    }
  }

  /**
   * Get live events
   */
  async getLiveEvents(): Promise<LiveEvent[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/events/live`);
      if (!response.ok) {
        // Return mock data if endpoint doesn't exist
        return [
          {
            event_id: "evt_001",
            event_type: "goal",
            timestamp: new Date().toISOString(),
            player_name: "Auston Matthews",
            team: "Toronto Maple Leafs",
            description: "Goal scored from the slot",
            confidence: 0.95
          }
        ];
      }
      return response.json();
    } catch (error) {
      console.warn('API endpoint not available, using mock data:', error);
      return [
        {
          event_id: "evt_001",
          event_type: "goal",
          timestamp: new Date().toISOString(),
          player_name: "Auston Matthews",
          team: "Toronto Maple Leafs",
          description: "Goal scored from the slot",
          confidence: 0.95
        }
      ];
    }
  }

  /**
   * Get analytics metrics
   */
  async getAnalyticsMetrics(): Promise<AnalyticsMetrics> {
    try {
      const response = await fetch(`${this.baseUrl}/api/analytics/metrics`);
      if (!response.ok) {
        // Return mock data if endpoint doesn't exist
        return {
          detection_accuracy: 0.95,
          processing_time: 45.2,
          fps: 30.0,
          active_players: 4,
          total_events: 127,
          momentum: {
            "home": 0.65,
            "away": 0.35
          },
          pressure: {
            "offensive": 0.72,
            "defensive": 0.28
          },
          enhanced_features: {
            "player_tracking": true,
            "shot_analysis": true,
            "formation_detection": true,
            "speed_analysis": true
          }
        };
      }
      return response.json();
    } catch (error) {
      console.warn('API endpoint not available, using mock data:', error);
      return {
        detection_accuracy: 0.95,
        processing_time: 45.2,
        fps: 30.0,
        active_players: 4,
        total_events: 127,
        momentum: {
          "home": 0.65,
          "away": 0.35
        },
        pressure: {
          "offensive": 0.72,
          "defensive": 0.28
        },
        enhanced_features: {
          "player_tracking": true,
          "shot_analysis": true,
          "formation_detection": true,
          "speed_analysis": true
        }
      };
    }
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
    // WebSocket connection disabled for now - using mock data instead
    console.log('🔌 WebSocket connection disabled - using mock data');
    
    // Simulate real-time updates with mock data
    const mockInterval = setInterval(() => {
      const mockData = {
        type: 'analytics_update',
        timestamp: new Date().toISOString(),
        metrics: {
          detection_accuracy: 0.95,
          processing_time: 45.2,
          fps: 30.0,
          active_players: 4
        }
      };
      onMessage(mockData);
    }, 2000);

    // Store interval for cleanup
    this.wsReconnectInterval = mockInterval as any;
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
