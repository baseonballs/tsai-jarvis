"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import VideoProcessing from "../components/VideoProcessing";
import RealTimeAnalytics from "../components/RealTimeAnalytics";
import EnhancedAnalytics from "../components/EnhancedAnalytics";
import AdvancedAnalytics from "../components/AdvancedAnalytics";
import ProductionAnalytics from "../components/ProductionAnalytics";
import hockeyApi, { GameState, PlayerStats, LiveEvent, AnalyticsMetrics } from "../services/hockeyApi";

interface GameStats {
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  period: number;
  timeRemaining: string;
  shotsOnGoal: {
    home: number;
    away: number;
  };
  powerPlay: {
    home: boolean;
    away: boolean;
    timeRemaining: string;
  };
}

interface PlayerStats {
  name: string;
  number: number;
  position: string;
  goals: number;
  assists: number;
  points: number;
  plusMinus: number;
  timeOnIce: string;
}

interface LiveEvent {
  id: string;
  time: string;
  type: 'goal' | 'save' | 'penalty' | 'faceoff';
  description: string;
  player: string;
  team: 'home' | 'away';
}

export default function HockeyAnalyticsDashboard() {
  const [gameStats, setGameStats] = useState<GameState | null>(null);
  const [playerStats, setPlayerStats] = useState<PlayerStats[]>([]);
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);
  const [analyticsMetrics, setAnalyticsMetrics] = useState<AnalyticsMetrics | null>(null);
  const [momentumData, setMomentumData] = useState<any>(null);
  const [pressureData, setPressureData] = useState<any>(null);
  const [speedMetrics, setSpeedMetrics] = useState<any[]>([]);
  const [shotAnalyses, setShotAnalyses] = useState<any[]>([]);
  const [formations, setFormations] = useState<any[]>([]);
  const [strategyInsights, setStrategyInsights] = useState<any[]>([]);
  const [gameSessions, setGameSessions] = useState<any[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [teamAnalyses, setTeamAnalyses] = useState<any[]>([]);
  const [productionMetrics, setProductionMetrics] = useState<any>(null);
  const [isLive, setIsLive] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  // Setup WebSocket connection for real-time updates
  useEffect(() => {
    const handleWebSocketMessage = (data: any) => {
      if (data.type === 'analytics_update') {
        setAnalyticsMetrics(data.data);
      }
    };

    const handleWebSocketError = (error: Event) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    hockeyApi.connectWebSocket(handleWebSocketMessage, handleWebSocketError);
    setIsConnected(true);

    return () => {
      hockeyApi.disconnectWebSocket();
      setIsConnected(false);
    };
  }, []);

  // Poll for updates every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadGameData();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      await loadGameData();
    } catch (error) {
      console.error('Failed to load initial data:', error);
      setError('Failed to connect to analytics service');
    }
  };

  const loadGameData = async () => {
    try {
      const [gameState, players, events, metrics, momentum, pressure, speed, shots, formations, strategy, sessions, historical, teams, production] = await Promise.all([
        hockeyApi.getGameState(),
        hockeyApi.getPlayerStats(),
        hockeyApi.getLiveEvents(),
        hockeyApi.getAnalyticsMetrics(),
        hockeyApi.getMomentumAnalytics(),
        hockeyApi.getPressureAnalytics(),
        hockeyApi.getSpeedAnalytics(),
        hockeyApi.getShotAnalytics(),
        hockeyApi.getFormationAnalytics(),
        hockeyApi.getStrategyAnalytics(),
        hockeyApi.getGameSessions(),
        hockeyApi.getHistoricalData(),
        hockeyApi.getTeamAnalyses(),
        hockeyApi.getProductionMetrics()
      ]);

      setGameStats(gameState);
      setPlayerStats(players);
      setLiveEvents(events);
      setAnalyticsMetrics(metrics);
      setMomentumData(momentum);
      setPressureData(pressure);
      setSpeedMetrics(speed.speed_metrics || []);
      setShotAnalyses(shots.recent_shots || []);
      setFormations(formations.recent_formations || []);
      setStrategyInsights(strategy.strategy_insights || []);
      setGameSessions(sessions.active_sessions || []);
      setHistoricalData(historical.historical_data || []);
      setTeamAnalyses(teams.team_analyses || []);
      setProductionMetrics(production);
      setError(null);
    } catch (error) {
      console.error('Failed to load game data:', error);
      setError('Failed to load game data');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white">
      {/* Header */}
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">TSAI</span>
              </div>
              <h1 className="text-xl font-bold">Jarvis Hockey Analytics</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                <span className="text-sm">{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
              </div>
              {analyticsMetrics && (
                <div className="text-sm text-slate-300">
                  Detection Accuracy: {analyticsMetrics.detection_accuracy.toFixed(1)}%
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="bg-red-800/50 rounded-lg p-4 mb-8 border border-red-700">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span className="text-red-200">{error}</span>
            </div>
          </div>
        )}

        {/* Game Score */}
        {gameStats && (
          <div className="bg-slate-800/50 rounded-lg p-6 mb-8 border border-slate-700">
            <div className="flex items-center justify-between">
              <div className="text-center">
                <h2 className="text-2xl font-bold">{gameStats.home_team}</h2>
                <div className="text-4xl font-bold text-blue-400">{gameStats.home_score}</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold">Period {gameStats.period}</div>
                <div className="text-2xl font-bold">{gameStats.time_remaining}</div>
                {gameStats.power_play.home && (
                  <div className="text-sm text-yellow-400">PP: {gameStats.power_play.time_remaining}</div>
                )}
              </div>
              <div className="text-center">
                <h2 className="text-2xl font-bold">{gameStats.away_team}</h2>
                <div className="text-4xl font-bold text-red-400">{gameStats.away_score}</div>
              </div>
            </div>
            
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-700/50 rounded-lg p-4 text-center">
                <div className="text-sm text-slate-300">Shots on Goal</div>
                <div className="text-2xl font-bold">18 - 15</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-4 text-center">
                <div className="text-sm text-slate-300">Power Play</div>
                <div className="text-lg font-semibold">
                  {gameStats.power_play.home ? 'Home' : gameStats.power_play.away ? 'Away' : 'Even Strength'}
                </div>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-4 text-center">
                <div className="text-sm text-slate-300">AI Analysis</div>
                <div className="text-lg font-semibold text-green-400">Active</div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Player Statistics */}
          <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4">Top Performers</h3>
            <div className="space-y-3">
              {playerStats.length > 0 ? (
                playerStats.map((player, index) => (
                  <div key={player.player_id} className="bg-slate-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                          {player.player_id}
                        </div>
                        <div>
                          <div className="font-semibold">{player.name}</div>
                          <div className="text-sm text-slate-300">{player.position} • {player.time_on_ice}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold">{player.points}P</div>
                        <div className="text-sm text-slate-300">{player.goals}G {player.assists}A</div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-400 py-8">
                  <div className="text-lg">No player data available</div>
                  <div className="text-sm">Waiting for analytics service...</div>
                </div>
              )}
            </div>
          </div>

          {/* Live Events */}
          <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4">Live Events</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {liveEvents.length > 0 ? (
                liveEvents.map((event) => (
                  <div key={event.event_id} className="bg-slate-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          event.event_type === 'goal' ? 'bg-green-500' :
                          event.event_type === 'save' ? 'bg-blue-500' :
                          event.event_type === 'penalty' ? 'bg-yellow-500' : 'bg-gray-500'
                        }`}></div>
                        <div>
                          <div className="font-semibold">{event.description}</div>
                          <div className="text-sm text-slate-300">{event.player_name}</div>
                        </div>
                      </div>
                      <div className="text-sm text-slate-400">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-400 py-8">
                  <div className="text-lg">No events detected</div>
                  <div className="text-sm">Waiting for live game data...</div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Video Processing Section */}
        <div className="mt-8">
          <VideoProcessing 
            isProcessing={true}
            frameRate={30}
            detectionCount={4}
            processingLatency={45}
          />
        </div>

        {/* Real-Time Analytics Section */}
        <div className="mt-8">
          <RealTimeAnalytics />
        </div>

        {/* Enhanced Analytics Section */}
        <div className="mt-8">
          <EnhancedAnalytics 
            momentumData={momentumData}
            pressureData={pressureData}
            isLoading={!momentumData && !pressureData}
          />
        </div>

        {/* Advanced Analytics Section */}
        <div className="mt-8">
          <AdvancedAnalytics 
            speedMetrics={speedMetrics}
            shotAnalyses={shotAnalyses}
            formations={formations}
            strategyInsights={strategyInsights}
            isLoading={speedMetrics.length === 0 && shotAnalyses.length === 0}
          />
        </div>

        {/* Production Analytics Section */}
        <div className="mt-8">
          <ProductionAnalytics 
            gameSessions={gameSessions}
            historicalData={historicalData}
            teamAnalyses={teamAnalyses}
            productionMetrics={productionMetrics}
            isLoading={gameSessions.length === 0 && historicalData.length === 0}
          />
        </div>

        {/* AI Analytics Section */}
        <div className="mt-8 bg-slate-800/50 rounded-lg p-6 border border-slate-700">
          <h3 className="text-xl font-bold mb-4">AI-Powered Analytics</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-400">
                {analyticsMetrics ? `${analyticsMetrics.detection_accuracy.toFixed(1)}%` : '--'}
              </div>
              <div className="text-sm text-slate-300">Player Detection</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-400">
                {analyticsMetrics ? `${(analyticsMetrics.processing_time * 1000).toFixed(1)}ms` : '--'}
              </div>
              <div className="text-sm text-slate-300">Processing Time</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {analyticsMetrics ? `${analyticsMetrics.fps.toFixed(1)}` : '--'}
              </div>
              <div className="text-sm text-slate-300">FPS</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-purple-400">
                {analyticsMetrics ? analyticsMetrics.active_players : '--'}
              </div>
              <div className="text-sm text-slate-300">Active Players</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
