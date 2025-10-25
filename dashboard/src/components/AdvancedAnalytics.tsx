"use client";

import { useState, useEffect } from "react";

interface PlayerSpeedMetrics {
  player_id: number;
  name: string;
  team: string;
  position: string;
  current_speed: number;
  max_speed: number;
  avg_speed: number;
  speed_acceleration: number;
  speed_deceleration: number;
  total_distance: number;
  movement_efficiency: number;
  direction_changes: number;
  movement_smoothness: number;
  speed_consistency: number;
  acceleration_consistency: number;
  fatigue_level: number;
  energy_usage: number;
  offensive_zone_speed: number;
  defensive_zone_speed: number;
  neutral_zone_speed: number;
}

interface ShotAnalysis {
  shot_id: string;
  player_id: number;
  player_name: string;
  team: string;
  shot_type: string;
  shot_location: number[];
  goal_location: number[];
  shot_distance: number;
  shot_angle: number;
  shot_velocity: number;
  shot_accuracy: number;
  shot_power: number;
  shot_quality: number;
  game_situation: string;
  pressure_level: number;
  goal_probability: number;
  save_probability: number;
  miss_probability: number;
  timestamp: string;
  confidence: number;
}

interface TeamFormation {
  team: string;
  formation_type: string;
  forward_positions: number[][];
  defense_positions: number[][];
  goalie_position: number[];
  formation_width: number;
  formation_depth: number;
  formation_compactness: number;
  formation_balance: number;
  avg_player_distance: number;
  min_player_distance: number;
  max_player_distance: number;
  formation_stability: number;
  formation_flexibility: number;
  formation_coverage: number;
  timestamp: string;
  confidence: number;
}

interface StrategyInsights {
  team: string;
  period: number;
  time_remaining: string;
  possession_time: number;
  possession_percentage: number;
  possession_quality: number;
  possession_efficiency: number;
  offensive_zone_time: number;
  defensive_zone_time: number;
  neutral_zone_time: number;
  zone_control_balance: number;
  shots_generated: number;
  shot_quality_avg: number;
  shot_velocity_avg: number;
  shot_accuracy_avg: number;
  shots_allowed: number;
  shot_quality_allowed: number;
  defensive_pressure: number;
  defensive_efficiency: number;
  breakouts_successful: number;
  breakouts_attempted: number;
  breakout_success_rate: number;
  forecheck_pressure: number;
  power_play_efficiency: number;
  penalty_kill_efficiency: number;
  special_teams_advantage: number;
  momentum_swings: number;
  pressure_peaks: number;
  strategic_advantages: string[];
  tactical_recommendations: string[];
  timestamp: string;
}

interface AdvancedAnalyticsProps {
  speedMetrics?: PlayerSpeedMetrics[];
  shotAnalyses?: ShotAnalysis[];
  formations?: TeamFormation[];
  strategyInsights?: StrategyInsights[];
  isLoading?: boolean;
}

export default function AdvancedAnalytics({ 
  speedMetrics = [], 
  shotAnalyses = [], 
  formations = [], 
  strategyInsights = [], 
  isLoading = false 
}: AdvancedAnalyticsProps) {
  const [selectedTab, setSelectedTab] = useState<'speed' | 'shots' | 'formations' | 'strategy'>('speed');
  const [speedHistory, setSpeedHistory] = useState<number[]>([]);
  const [shotHistory, setShotHistory] = useState<number[]>([]);

  useEffect(() => {
    // Simulate speed history data
    const history = Array.from({ length: 20 }, (_, i) => 20 + Math.random() * 15);
    setSpeedHistory(history);
  }, []);

  useEffect(() => {
    // Simulate shot history data
    const history = Array.from({ length: 15 }, (_, i) => 0.3 + Math.random() * 0.4);
    setShotHistory(history);
  }, []);

  const getSpeedColor = (speed: number) => {
    if (speed > 30) return "text-green-400";
    if (speed > 25) return "text-yellow-400";
    return "text-red-400";
  };

  const getShotQualityColor = (quality: number) => {
    if (quality > 0.8) return "text-green-400";
    if (quality > 0.6) return "text-yellow-400";
    return "text-red-400";
  };

  const getFormationStabilityColor = (stability: number) => {
    if (stability > 0.8) return "text-green-400";
    if (stability > 0.6) return "text-yellow-400";
    return "text-red-400";
  };

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Advanced Analytics</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-slate-300">Loading advanced analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Advanced Analytics</h3>
        
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedTab('speed')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'speed'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Speed Analytics
          </button>
          <button
            onClick={() => setSelectedTab('shots')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'shots'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Shot Analysis
          </button>
          <button
            onClick={() => setSelectedTab('formations')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'formations'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Formations
          </button>
          <button
            onClick={() => setSelectedTab('strategy')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'strategy'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Strategy
          </button>
        </div>

        {/* Speed Analytics Tab */}
        {selectedTab === 'speed' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Speed Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Speed Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Speed:</span>
                    <span className="text-blue-400 font-bold">24.8 km/h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Max Speed:</span>
                    <span className="text-green-400 font-bold">35.2 km/h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Consistency:</span>
                    <span className="text-yellow-400 font-bold">82%</span>
                  </div>
                </div>
              </div>

              {/* Zone Speed Analysis */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Zone Speed Analysis</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Offensive:</span>
                    <span className="text-green-400 font-bold">26.5 km/h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Defensive:</span>
                    <span className="text-blue-400 font-bold">22.1 km/h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Neutral:</span>
                    <span className="text-purple-400 font-bold">28.8 km/h</span>
                  </div>
                </div>
              </div>

              {/* Speed Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Speed Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {speedHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-blue-500 rounded-sm"
                      style={{ height: `${Math.max((value / 35) * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Player Speed Metrics */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Player Speed Metrics</h4>
              <div className="space-y-3">
                {speedMetrics.slice(0, 5).map((player) => (
                  <div key={player.player_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                        {player.player.player_id}
                      </div>
                      <div>
                        <div className="font-semibold">{player.name}</div>
                        <div className="text-sm text-slate-300">{player.position} • {player.team}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getSpeedColor(player.current_speed)}`}>
                        {player.current_speed.toFixed(1)} km/h
                      </div>
                      <div className="text-sm text-slate-300">
                        Max: {player.max_speed.toFixed(1)} km/h
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Shot Analysis Tab */}
        {selectedTab === 'shots' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Shot Quality Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Shot Quality</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Quality:</span>
                    <span className="text-green-400 font-bold">0.72</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Velocity:</span>
                    <span className="text-blue-400 font-bold">85.2 km/h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Goal Probability:</span>
                    <span className="text-yellow-400 font-bold">32%</span>
                  </div>
                </div>
              </div>

              {/* Shot Types */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Shot Types</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Wrist Shots:</span>
                    <span className="text-blue-400 font-bold">45%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Slap Shots:</span>
                    <span className="text-green-400 font-bold">25%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Snap Shots:</span>
                    <span className="text-purple-400 font-bold">30%</span>
                  </div>
                </div>
              </div>

              {/* Shot Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Shot Quality Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {shotHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-green-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Shots */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent Shots</h4>
              <div className="space-y-3">
                {shotAnalyses.slice(0, 5).map((shot) => (
                  <div key={shot.shot_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🎯
                      </div>
                      <div>
                        <div className="font-semibold">{shot.player_name}</div>
                        <div className="text-sm text-slate-300">{shot.shot_type} • {shot.team}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getShotQualityColor(shot.shot_quality)}`}>
                        {(shot.shot_quality * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-300">
                        {shot.shot_velocity.toFixed(1)} km/h
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Formations Tab */}
        {selectedTab === 'formations' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Formation Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Formation Analysis</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Stability:</span>
                    <span className="text-green-400 font-bold">0.82</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Compactness:</span>
                    <span className="text-blue-400 font-bold">0.75</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Balance:</span>
                    <span className="text-yellow-400 font-bold">0.68</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Coverage:</span>
                    <span className="text-purple-400 font-bold">0.91</span>
                  </div>
                </div>
              </div>

              {/* Formation Types */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Formation Types</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Offensive:</span>
                    <span className="text-green-400 font-bold">35%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Defensive:</span>
                    <span className="text-blue-400 font-bold">40%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Neutral:</span>
                    <span className="text-yellow-400 font-bold">25%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Team Formations */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Team Formations</h4>
              <div className="space-y-3">
                {formations.slice(0, 3).map((formation, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🏒
                      </div>
                      <div>
                        <div className="font-semibold">{formation.team} - {formation.formation_type}</div>
                        <div className="text-sm text-slate-300">
                          Width: {formation.formation_width.toFixed(1)}m • Depth: {formation.formation_depth.toFixed(1)}m
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getFormationStabilityColor(formation.formation_stability)}`}>
                        {(formation.formation_stability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Balance: {(formation.formation_balance * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Strategy Tab */}
        {selectedTab === 'strategy' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Possession Analysis */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Possession Analysis</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Home Team:</span>
                    <span className="text-blue-400 font-bold">55%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Away Team:</span>
                    <span className="text-red-400 font-bold">45%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Quality:</span>
                    <span className="text-green-400 font-bold">0.72</span>
                  </div>
                </div>
              </div>

              {/* Zone Control */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Zone Control</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Offensive Zone:</span>
                    <span className="text-green-400 font-bold">40%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Defensive Zone:</span>
                    <span className="text-blue-400 font-bold">35%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Neutral Zone:</span>
                    <span className="text-yellow-400 font-bold">25%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Strategy Insights */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Strategy Insights</h4>
              <div className="space-y-3">
                {strategyInsights.slice(0, 3).map((insight, index) => (
                  <div key={index} className="p-3 bg-slate-600/50 rounded-lg">
                    <div className="font-semibold mb-2">{insight.team} - Period {insight.period}</div>
                    <div className="text-sm text-slate-300 space-y-1">
                      <div>Possession: {insight.possession_percentage.toFixed(1)}%</div>
                      <div>Shots Generated: {insight.shots_generated}</div>
                      <div>Defensive Efficiency: {(insight.defensive_efficiency * 100).toFixed(1)}%</div>
                      {insight.strategic_advantages.length > 0 && (
                        <div>Advantages: {insight.strategic_advantages.join(', ')}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
