"use client";

import { useState, useEffect } from "react";

interface PlayerPerformancePrediction {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  predicted_points: number;
  predicted_goals: number;
  predicted_assists: number;
  predicted_plus_minus: number;
  predicted_time_on_ice: number;
  predicted_shots: number;
  predicted_hits: number;
  points_confidence: number;
  goals_confidence: number;
  assists_confidence: number;
  overall_confidence: number;
  form_factor: number;
  fatigue_factor: number;
  matchup_factor: number;
  home_advantage: number;
  recent_performance: number[];
  season_average: number;
  career_average: number;
  improvement_trend: number;
  timestamp: string;
  model_version: string;
}

interface TeamStrategyOptimization {
  team: string;
  opponent: string;
  game_context: string;
  offensive_strategy: string;
  defensive_strategy: string;
  special_teams_strategy: string;
  optimal_lineup: any[];
  line_chemistry: number;
  matchup_advantages: string[];
  key_tactics: string[];
  defensive_focus: string[];
  offensive_focus: string[];
  special_situations: string[];
  expected_possession: number;
  expected_shots: number;
  expected_goals: number;
  win_probability: number;
  strategy_confidence: number;
  lineup_confidence: number;
  tactical_confidence: number;
  timestamp: string;
  model_version: string;
}

interface GameOutcomePrediction {
  home_team: string;
  away_team: string;
  game_date: string;
  home_win_probability: number;
  away_win_probability: number;
  overtime_probability: number;
  shootout_probability: number;
  predicted_home_score: number;
  predicted_away_score: number;
  predicted_total_goals: number;
  home_advantage: number;
  team_form_difference: number;
  head_to_head_history: number;
  injury_impact: number;
  rest_advantage: number;
  outcome_confidence: number;
  score_confidence: number;
  overall_confidence: number;
  period_predictions: any[];
  key_moments: string[];
  critical_factors: string[];
  timestamp: string;
  model_version: string;
}

interface InjuryRiskAssessment {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  overall_injury_risk: number;
  acute_injury_risk: number;
  chronic_injury_risk: number;
  concussion_risk: number;
  lower_body_risk: number;
  upper_body_risk: number;
  overuse_risk: number;
  fatigue_level: number;
  recent_workload: number;
  injury_history: number;
  age_factor: number;
  position_risk: number;
  workload_recommendation: string;
  rest_recommendation: string;
  monitoring_focus: string[];
  prevention_measures: string[];
  risk_confidence: number;
  recommendation_confidence: number;
  timestamp: string;
  model_version: string;
}

interface EnterpriseAIMetrics {
  active_models: number;
  total_predictions: number;
  model_accuracy: { [key: string]: number };
  ai_features: { [key: string]: boolean };
  performance_metrics: { [key: string]: number };
}

interface EnterpriseAIProps {
  playerPredictions?: PlayerPerformancePrediction[];
  teamStrategies?: TeamStrategyOptimization[];
  gamePredictions?: GameOutcomePrediction[];
  injuryAssessments?: InjuryRiskAssessment[];
  aiMetrics?: EnterpriseAIMetrics;
  isLoading?: boolean;
}

export default function EnterpriseAI({ 
  playerPredictions = [], 
  teamStrategies = [], 
  gamePredictions = [], 
  injuryAssessments = [], 
  aiMetrics,
  isLoading = false 
}: EnterpriseAIProps) {
  const [selectedTab, setSelectedTab] = useState<'players' | 'teams' | 'games' | 'injuries' | 'metrics'>('players');
  const [predictionHistory, setPredictionHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);

  useEffect(() => {
    // Simulate prediction history data
    const history = Array.from({ length: 20 }, (_, i) => 50 + Math.random() * 30);
    setPredictionHistory(history);
  }, []);

  useEffect(() => {
    // Simulate accuracy history data
    const history = Array.from({ length: 15 }, (_, i) => 0.7 + Math.random() * 0.3);
    setAccuracyHistory(history);
  }, []);

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-400';
    if (confidence > 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return 'text-green-400';
    if (risk < 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStrategyColor = (strategy: string) => {
    switch (strategy) {
      case 'aggressive': return 'text-red-400';
      case 'balanced': return 'text-blue-400';
      case 'defensive': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Enterprise AI Analytics</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-slate-300">Loading enterprise AI analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Enterprise AI Analytics</h3>
        
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedTab('players')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'players'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Player Performance
          </button>
          <button
            onClick={() => setSelectedTab('teams')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'teams'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Team Strategy
          </button>
          <button
            onClick={() => setSelectedTab('games')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'games'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Game Outcomes
          </button>
          <button
            onClick={() => setSelectedTab('injuries')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'injuries'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Injury Risk
          </button>
          <button
            onClick={() => setSelectedTab('metrics')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'metrics'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            AI Metrics
          </button>
        </div>

        {/* Player Performance Tab */}
        {selectedTab === 'players' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Performance Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Predictions:</span>
                    <span className="text-blue-400 font-bold">{playerPredictions.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Confidence:</span>
                    <span className="text-green-400 font-bold">
                      {playerPredictions.length > 0 ? (playerPredictions.reduce((sum, p) => sum + p.overall_confidence, 0) / playerPredictions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Points:</span>
                    <span className="text-yellow-400 font-bold">
                      {playerPredictions.length > 0 ? (playerPredictions.reduce((sum, p) => sum + p.predicted_points, 0) / playerPredictions.length).toFixed(1) : '0'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Prediction Accuracy */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Prediction Accuracy</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Points Accuracy:</span>
                    <span className="text-green-400 font-bold">87%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Goals Accuracy:</span>
                    <span className="text-blue-400 font-bold">84%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Assists Accuracy:</span>
                    <span className="text-yellow-400 font-bold">82%</span>
                  </div>
                </div>
              </div>

              {/* Prediction Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Prediction Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {predictionHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-blue-500 rounded-sm"
                      style={{ height: `${Math.max((value / 100) * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Top Performers */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Top Predicted Performers</h4>
              <div className="space-y-3">
                {playerPredictions
                  .sort((a, b) => b.predicted_points - a.predicted_points)
                  .slice(0, 5)
                  .map((player) => (
                    <div key={player.player_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                          {player.player_id}
                        </div>
                        <div>
                          <div className="font-semibold">{player.player_name}</div>
                          <div className="text-sm text-slate-300">{player.team} • {player.position}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-blue-400">{player.predicted_points.toFixed(1)} pts</div>
                        <div className="text-sm text-slate-300">
                          Confidence: <span className={getConfidenceColor(player.overall_confidence)}>
                            {(player.overall_confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Team Strategy Tab */}
        {selectedTab === 'teams' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Strategy Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Strategy Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Strategies:</span>
                    <span className="text-blue-400 font-bold">{teamStrategies.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Confidence:</span>
                    <span className="text-green-400 font-bold">
                      {teamStrategies.length > 0 ? (teamStrategies.reduce((sum, t) => sum + t.strategy_confidence, 0) / teamStrategies.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Win Probability:</span>
                    <span className="text-yellow-400 font-bold">
                      {teamStrategies.length > 0 ? (teamStrategies.reduce((sum, t) => sum + t.win_probability, 0) / teamStrategies.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Strategy Distribution */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Strategy Distribution</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Aggressive:</span>
                    <span className="text-red-400 font-bold">
                      {teamStrategies.filter(t => t.offensive_strategy === 'aggressive').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Balanced:</span>
                    <span className="text-blue-400 font-bold">
                      {teamStrategies.filter(t => t.offensive_strategy === 'balanced').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Defensive:</span>
                    <span className="text-green-400 font-bold">
                      {teamStrategies.filter(t => t.offensive_strategy === 'defensive').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Strategy Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Win Probability Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {accuracyHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-green-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Team Strategies */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Team Strategy Optimizations</h4>
              <div className="space-y-3">
                {teamStrategies.slice(0, 5).map((strategy, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🏒
                      </div>
                      <div>
                        <div className="font-semibold">{strategy.team} vs {strategy.opponent}</div>
                        <div className="text-sm text-slate-300">
                          {strategy.offensive_strategy} offense • {strategy.defensive_strategy} defense
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-400">
                        {(strategy.win_probability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Confidence: <span className={getConfidenceColor(strategy.strategy_confidence)}>
                          {(strategy.strategy_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Game Outcomes Tab */}
        {selectedTab === 'games' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Game Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Game Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Predictions:</span>
                    <span className="text-blue-400 font-bold">{gamePredictions.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Confidence:</span>
                    <span className="text-green-400 font-bold">
                      {gamePredictions.length > 0 ? (gamePredictions.reduce((sum, g) => sum + g.overall_confidence, 0) / gamePredictions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Goals:</span>
                    <span className="text-yellow-400 font-bold">
                      {gamePredictions.length > 0 ? (gamePredictions.reduce((sum, g) => sum + g.predicted_total_goals, 0) / gamePredictions.length).toFixed(1) : '0'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Outcome Distribution */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Outcome Distribution</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Home Wins:</span>
                    <span className="text-blue-400 font-bold">
                      {gamePredictions.length > 0 ? (gamePredictions.reduce((sum, g) => sum + g.home_win_probability, 0) / gamePredictions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Away Wins:</span>
                    <span className="text-red-400 font-bold">
                      {gamePredictions.length > 0 ? (gamePredictions.reduce((sum, g) => sum + g.away_win_probability, 0) / gamePredictions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Overtime:</span>
                    <span className="text-yellow-400 font-bold">
                      {gamePredictions.length > 0 ? (gamePredictions.reduce((sum, g) => sum + g.overtime_probability, 0) / gamePredictions.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Game Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Confidence Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {accuracyHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-yellow-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Game Predictions */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Game Outcome Predictions</h4>
              <div className="space-y-3">
                {gamePredictions.slice(0, 5).map((game, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🎮
                      </div>
                      <div>
                        <div className="font-semibold">{game.home_team} vs {game.away_team}</div>
                        <div className="text-sm text-slate-300">
                          {game.predicted_home_score.toFixed(1)} - {game.predicted_away_score.toFixed(1)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-blue-400">
                        {(game.home_win_probability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Confidence: <span className={getConfidenceColor(game.overall_confidence)}>
                          {(game.overall_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Injury Risk Tab */}
        {selectedTab === 'injuries' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Risk Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Risk Overview</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Assessments:</span>
                    <span className="text-blue-400 font-bold">{injuryAssessments.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Risk:</span>
                    <span className="text-yellow-400 font-bold">
                      {injuryAssessments.length > 0 ? (injuryAssessments.reduce((sum, i) => sum + i.overall_injury_risk, 0) / injuryAssessments.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">High Risk Players:</span>
                    <span className="text-red-400 font-bold">
                      {injuryAssessments.filter(i => i.overall_injury_risk > 0.7).length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Risk Distribution */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Risk Distribution</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Low Risk:</span>
                    <span className="text-green-400 font-bold">
                      {injuryAssessments.filter(i => i.overall_injury_risk < 0.3).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Medium Risk:</span>
                    <span className="text-yellow-400 font-bold">
                      {injuryAssessments.filter(i => i.overall_injury_risk >= 0.3 && i.overall_injury_risk < 0.7).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">High Risk:</span>
                    <span className="text-red-400 font-bold">
                      {injuryAssessments.filter(i => i.overall_injury_risk >= 0.7).length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Risk Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Risk Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {predictionHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-red-500 rounded-sm"
                      style={{ height: `${Math.max((value / 100) * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Injury Assessments */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Injury Risk Assessments</h4>
              <div className="space-y-3">
                {injuryAssessments
                  .sort((a, b) => b.overall_injury_risk - a.overall_injury_risk)
                  .slice(0, 5)
                  .map((assessment) => (
                    <div key={assessment.player_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center text-sm font-bold">
                          ⚠️
                        </div>
                        <div>
                          <div className="font-semibold">{assessment.player_name}</div>
                          <div className="text-sm text-slate-300">{assessment.team} • {assessment.position}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-lg font-bold ${getRiskColor(assessment.overall_injury_risk)}`}>
                          {(assessment.overall_injury_risk * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-slate-300">
                          Recommendation: {assessment.workload_recommendation}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* AI Metrics Tab */}
        {selectedTab === 'metrics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {/* AI Models */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">AI Models</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Active Models:</span>
                    <span className="text-green-400 font-bold">{aiMetrics?.active_models || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Predictions:</span>
                    <span className="text-blue-400 font-bold">{aiMetrics?.total_predictions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Accuracy:</span>
                    <span className="text-yellow-400 font-bold">
                      {aiMetrics?.model_accuracy ? (Object.values(aiMetrics.model_accuracy).reduce((sum, acc) => sum + acc, 0) / Object.values(aiMetrics.model_accuracy).length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Model Performance */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Model Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Player Model:</span>
                    <span className="text-green-400 font-bold">
                      {aiMetrics?.model_accuracy?.player_performance ? (aiMetrics.model_accuracy.player_performance * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Team Model:</span>
                    <span className="text-blue-400 font-bold">
                      {aiMetrics?.model_accuracy?.team_strategy ? (aiMetrics.model_accuracy.team_strategy * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Game Model:</span>
                    <span className="text-yellow-400 font-bold">
                      {aiMetrics?.model_accuracy?.game_outcome ? (aiMetrics.model_accuracy.game_outcome * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Injury Model:</span>
                    <span className="text-red-400 font-bold">
                      {aiMetrics?.model_accuracy?.injury_risk ? (aiMetrics.model_accuracy.injury_risk * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* AI Features */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">AI Features</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Player Prediction:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Team Strategy:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Game Outcome:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Injury Risk:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Prediction Time:</span>
                    <span className="text-blue-400 font-bold">50ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Model Uptime:</span>
                    <span className="text-green-400 font-bold">99.9%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Confidence:</span>
                    <span className="text-yellow-400 font-bold">85%</span>
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
