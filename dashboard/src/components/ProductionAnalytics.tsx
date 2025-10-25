"use client";

import { useState, useEffect } from "react";

interface GameSession {
  session_id: string;
  game_id: string;
  home_team: string;
  away_team: string;
  start_time: string;
  end_time?: string;
  status: string;
  season: string;
  game_type: string;
  venue: string;
  attendance: number;
  total_events: number;
  total_shots: number;
  total_goals: number;
  processing_time: number;
  data_quality: number;
  avg_fps: number;
  max_latency: number;
  error_count: number;
  recovery_count: number;
}

interface HistoricalData {
  player_id: number;
  player_name: string;
  team: string;
  season: string;
  games_played: number;
  total_ice_time: number;
  avg_speed: number;
  max_speed: number;
  total_distance: number;
  total_shots: number;
  total_goals: number;
  total_assists: number;
  total_points: number;
  consistency_score: number;
  improvement_rate: number;
  peak_performance: number;
  decline_rate: number;
  league_rank: number;
  team_rank: number;
  position_rank: number;
  percentile_scores: { [key: string]: number };
}

interface TeamAnalysis {
  team: string;
  season: string;
  games_played: number;
  wins: number;
  losses: number;
  overtime_losses: number;
  points: number;
  goals_for: number;
  goals_against: number;
  goal_differential: number;
  possession_percentage: number;
  shot_quality_avg: number;
  defensive_efficiency: number;
  power_play_percentage: number;
  penalty_kill_percentage: number;
  win_streak: number;
  loss_streak: number;
  home_record: string;
  away_record: string;
  offensive_style: string;
  defensive_style: string;
  special_teams_style: string;
}

interface ProductionMetrics {
  active_sessions: number;
  historical_players: number;
  team_analyses: number;
  api_integrations: number;
  mobile_optimizations: number;
  production_features: { [key: string]: boolean };
  performance_metrics: { [key: string]: number };
}

interface ProductionAnalyticsProps {
  gameSessions?: GameSession[];
  historicalData?: HistoricalData[];
  teamAnalyses?: TeamAnalysis[];
  productionMetrics?: ProductionMetrics;
  isLoading?: boolean;
}

export default function ProductionAnalytics({ 
  gameSessions = [], 
  historicalData = [], 
  teamAnalyses = [], 
  productionMetrics,
  isLoading = false 
}: ProductionAnalyticsProps) {
  const [selectedTab, setSelectedTab] = useState<'sessions' | 'historical' | 'teams' | 'production'>('sessions');
  const [sessionHistory, setSessionHistory] = useState<number[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<number[]>([]);

  useEffect(() => {
    // Simulate session history data
    const history = Array.from({ length: 20 }, (_, i) => 1 + Math.random() * 4);
    setSessionHistory(history);
  }, []);

  useEffect(() => {
    // Simulate performance history data
    const history = Array.from({ length: 15 }, (_, i) => 0.8 + Math.random() * 0.2);
    setPerformanceHistory(history);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'paused': return 'text-yellow-400';
      case 'completed': return 'text-blue-400';
      case 'archived': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  };

  const getDataQualityColor = (quality: number) => {
    if (quality > 0.9) return 'text-green-400';
    if (quality > 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getPerformanceColor = (performance: number) => {
    if (performance > 0.8) return 'text-green-400';
    if (performance > 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Production Analytics</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-slate-300">Loading production analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Production Analytics</h3>
        
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedTab('sessions')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'sessions'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Game Sessions
          </button>
          <button
            onClick={() => setSelectedTab('historical')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'historical'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Historical Data
          </button>
          <button
            onClick={() => setSelectedTab('teams')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'teams'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Team Analysis
          </button>
          <button
            onClick={() => setSelectedTab('production')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'production'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Production Metrics
          </button>
        </div>

        {/* Game Sessions Tab */}
        {selectedTab === 'sessions' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Session Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Session Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Active Sessions:</span>
                    <span className="text-green-400 font-bold">{gameSessions.filter(s => s.status === 'active').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Sessions:</span>
                    <span className="text-blue-400 font-bold">{gameSessions.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Data Quality:</span>
                    <span className="text-yellow-400 font-bold">
                      {gameSessions.length > 0 ? (gameSessions.reduce((sum, s) => sum + s.data_quality, 0) / gameSessions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg FPS:</span>
                    <span className="text-green-400 font-bold">
                      {gameSessions.length > 0 ? (gameSessions.reduce((sum, s) => sum + s.avg_fps, 0) / gameSessions.length).toFixed(1) : '0'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Max Latency:</span>
                    <span className="text-red-400 font-bold">
                      {gameSessions.length > 0 ? Math.max(...gameSessions.map(s => s.max_latency)).toFixed(1) : '0'}ms
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Error Rate:</span>
                    <span className="text-yellow-400 font-bold">
                      {gameSessions.length > 0 ? (gameSessions.reduce((sum, s) => sum + s.error_count, 0) / gameSessions.length).toFixed(1) : '0'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Session Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Session Activity</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {sessionHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-blue-500 rounded-sm"
                      style={{ height: `${Math.max(value * 20, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Active Sessions */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Active Game Sessions</h4>
              <div className="space-y-3">
                {gameSessions.slice(0, 5).map((session) => (
                  <div key={session.session_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🎮
                      </div>
                      <div>
                        <div className="font-semibold">{session.home_team} vs {session.away_team}</div>
                        <div className="text-sm text-slate-300">
                          {session.season} • {session.game_type} • {session.venue}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getStatusColor(session.status)}`}>
                        {session.status.toUpperCase()}
                      </div>
                      <div className="text-sm text-slate-300">
                        Quality: <span className={getDataQualityColor(session.data_quality)}>
                          {(session.data_quality * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Historical Data Tab */}
        {selectedTab === 'historical' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Historical Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Historical Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Players:</span>
                    <span className="text-blue-400 font-bold">{historicalData.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Consistency:</span>
                    <span className="text-green-400 font-bold">
                      {historicalData.length > 0 ? (historicalData.reduce((sum, h) => sum + h.consistency_score, 0) / historicalData.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Improvement:</span>
                    <span className="text-yellow-400 font-bold">
                      {historicalData.length > 0 ? (historicalData.reduce((sum, h) => sum + h.improvement_rate, 0) / historicalData.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Performance Trends */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance Trends</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Peak Performance:</span>
                    <span className="text-green-400 font-bold">
                      {historicalData.length > 0 ? Math.max(...historicalData.map(h => h.peak_performance) * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Decline Rate:</span>
                    <span className="text-red-400 font-bold">
                      {historicalData.length > 0 ? (historicalData.reduce((sum, h) => sum + h.decline_rate, 0) / historicalData.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Performance Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {performanceHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-green-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Top Performers */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Top Performers</h4>
              <div className="space-y-3">
                {historicalData
                  .sort((a, b) => b.peak_performance - a.peak_performance)
                  .slice(0, 5)
                  .map((player) => (
                    <div key={player.player_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-sm font-bold">
                          {player.player_id}
                        </div>
                        <div>
                          <div className="font-semibold">{player.player_name}</div>
                          <div className="text-sm text-slate-300">{player.team} • {player.season}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-lg font-bold ${getPerformanceColor(player.peak_performance)}`}>
                          {(player.peak_performance * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-slate-300">
                          Rank: #{player.league_rank}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Team Analysis Tab */}
        {selectedTab === 'teams' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Team Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Team Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Teams:</span>
                    <span className="text-blue-400 font-bold">{teamAnalyses.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Possession:</span>
                    <span className="text-green-400 font-bold">
                      {teamAnalyses.length > 0 ? (teamAnalyses.reduce((sum, t) => sum + t.possession_percentage, 0) / teamAnalyses.length).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Shot Quality:</span>
                    <span className="text-yellow-400 font-bold">
                      {teamAnalyses.length > 0 ? (teamAnalyses.reduce((sum, t) => sum + t.shot_quality_avg, 0) / teamAnalyses.length).toFixed(2) : '0'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Team Styles */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Team Styles</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Aggressive:</span>
                    <span className="text-red-400 font-bold">
                      {teamAnalyses.filter(t => t.offensive_style === 'aggressive').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Balanced:</span>
                    <span className="text-blue-400 font-bold">
                      {teamAnalyses.filter(t => t.offensive_style === 'balanced').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Defensive:</span>
                    <span className="text-green-400 font-bold">
                      {teamAnalyses.filter(t => t.offensive_style === 'defensive').length}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Team Rankings */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Team Rankings</h4>
              <div className="space-y-3">
                {teamAnalyses
                  .sort((a, b) => b.points - a.points)
                  .slice(0, 5)
                  .map((team, index) => (
                    <div key={team.team} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-sm font-bold">
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-semibold">{team.team}</div>
                          <div className="text-sm text-slate-300">{team.season} • {team.offensive_style} style</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-blue-400">{team.points} pts</div>
                        <div className="text-sm text-slate-300">
                          {team.wins}W-{team.losses}L-{team.overtime_losses}OTL
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Production Metrics Tab */}
        {selectedTab === 'production' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {/* System Health */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">System Health</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Active Sessions:</span>
                    <span className="text-green-400 font-bold">{productionMetrics?.active_sessions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Historical Players:</span>
                    <span className="text-blue-400 font-bold">{productionMetrics?.historical_players || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">API Integrations:</span>
                    <span className="text-yellow-400 font-bold">{productionMetrics?.api_integrations || 0}</span>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Processing Efficiency:</span>
                    <span className="text-green-400 font-bold">95%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Data Quality:</span>
                    <span className="text-blue-400 font-bold">98%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Uptime:</span>
                    <span className="text-yellow-400 font-bold">99.9%</span>
                  </div>
                </div>
              </div>

              {/* Scalability */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Scalability</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Max Games:</span>
                    <span className="text-green-400 font-bold">50</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Max Players:</span>
                    <span className="text-blue-400 font-bold">10K</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Max APIs:</span>
                    <span className="text-yellow-400 font-bold">20</span>
                  </div>
                </div>
              </div>

              {/* Production Features */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Features</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Multi-Game:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Historical:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">API Integration:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Mobile:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
