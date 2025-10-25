"use client";

import { useState, useEffect } from "react";

interface LiveVideoAnalysis {
  frame_id: number;
  timestamp: string;
  frame_rate: number;
  resolution: number[];
  players_detected: number;
  player_positions: any[];
  player_tracking: any[];
  events_detected: any[];
  event_confidence: number[];
  brightness: number;
  contrast: number;
  sharpness: number;
  noise_level: number;
  processing_time: number;
  detection_accuracy: number;
  tracking_consistency: number;
  game_momentum: number;
  team_pressure: { [key: string]: number };
  key_moments: string[];
  stream_quality: string;
  bandwidth_usage: number;
  latency: number;
}

interface AICommentary {
  commentary_id: string;
  timestamp: string;
  game_time: string;
  main_commentary: string;
  technical_analysis: string;
  player_insights: string;
  tactical_analysis: string;
  game_situation: string;
  key_players: string[];
  important_events: string[];
  commentary_confidence: number;
  technical_accuracy: number;
  engagement_score: number;
  voice_style: string;
  speaking_pace: number;
  emphasis_points: string[];
  language_complexity: string;
  target_audience: string;
  broadcast_ready: boolean;
  censor_rating: string;
  commercial_breaks: string[];
}

interface InstantReplayAnalysis {
  replay_id: string;
  timestamp: string;
  game_time: string;
  replay_type: string;
  replay_duration: number;
  key_frames: number[];
  event_analysis: string;
  player_analysis: string;
  referee_analysis: string;
  rule_interpretation: string;
  slow_motion_frames: any[];
  zoom_focus: any[];
  angle_analysis: any[];
  decision_confidence: number;
  controversy_level: number;
  impact_assessment: string;
  replay_quality: string;
  broadcast_ready: boolean;
  highlight_potential: number;
}

interface BroadcastQuality {
  stream_id: string;
  timestamp: string;
  resolution: number[];
  bitrate: number;
  frame_rate: number;
  color_depth: number;
  audio_bitrate: number;
  audio_channels: number;
  audio_sample_rate: number;
  buffer_health: number;
  packet_loss: number;
  jitter: number;
  latency: number;
  video_quality_score: number;
  audio_quality_score: number;
  overall_quality_score: number;
  broadcast_standard: string;
  color_grading: string;
  aspect_ratio: string;
  adaptive_bitrate: boolean;
  quality_scaling: boolean;
  error_correction: boolean;
}

interface MultiCameraAnalysis {
  analysis_id: string;
  timestamp: string;
  active_cameras: number;
  camera_positions: any[];
  camera_angles: string[];
  synchronized_frames: any[];
  cross_camera_tracking: any[];
  perspective_analysis: any[];
  player_3d_positions: any[];
  ice_surface_mapping: any;
  depth_analysis: any[];
  optimal_angle: string;
  best_camera: number;
  coverage_analysis: { [key: string]: number };
  director_recommendations: string[];
  cut_timing: number[];
  transition_smoothness: number;
}

interface RealtimeStreamingMetrics {
  active_streams: number;
  total_commentary: number;
  total_replays: number;
  broadcast_quality: number;
  multi_camera_analysis: number;
  streaming_features: { [key: string]: boolean };
  performance_metrics: { [key: string]: number };
}

interface RealtimeStreamingProps {
  liveAnalysis?: LiveVideoAnalysis[];
  aiCommentary?: AICommentary[];
  instantReplays?: InstantReplayAnalysis[];
  broadcastQuality?: BroadcastQuality[];
  multiCameraAnalysis?: MultiCameraAnalysis[];
  streamingMetrics?: RealtimeStreamingMetrics;
  isLoading?: boolean;
}

export default function RealtimeStreaming({ 
  liveAnalysis = [], 
  aiCommentary = [], 
  instantReplays = [], 
  broadcastQuality = [], 
  multiCameraAnalysis = [], 
  streamingMetrics,
  isLoading = false 
}: RealtimeStreamingProps) {
  const [selectedTab, setSelectedTab] = useState<'live' | 'commentary' | 'replays' | 'broadcast' | 'multi-camera' | 'metrics'>('live');
  const [streamingHistory, setStreamingHistory] = useState<number[]>([]);
  const [qualityHistory, setQualityHistory] = useState<number[]>([]);

  useEffect(() => {
    // Simulate streaming history data
    const history = Array.from({ length: 20 }, (_, i) => 20 + Math.random() * 30);
    setStreamingHistory(history);
  }, []);

  useEffect(() => {
    // Simulate quality history data
    const history = Array.from({ length: 15 }, (_, i) => 0.7 + Math.random() * 0.3);
    setQualityHistory(history);
  }, []);

  const getQualityColor = (quality: number) => {
    if (quality > 0.9) return 'text-green-400';
    if (quality > 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-400';
    if (confidence > 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStreamQualityColor = (quality: string) => {
    switch (quality) {
      case '4K': return 'text-purple-400';
      case 'HD': return 'text-blue-400';
      case 'SD': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Real-time Streaming</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-slate-300">Loading real-time streaming...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Real-time Streaming</h3>
        
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedTab('live')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'live'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Live Video
          </button>
          <button
            onClick={() => setSelectedTab('commentary')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'commentary'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            AI Commentary
          </button>
          <button
            onClick={() => setSelectedTab('replays')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'replays'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Instant Replays
          </button>
          <button
            onClick={() => setSelectedTab('broadcast')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'broadcast'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Broadcast Quality
          </button>
          <button
            onClick={() => setSelectedTab('multi-camera')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'multi-camera'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Multi-Camera
          </button>
          <button
            onClick={() => setSelectedTab('metrics')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedTab === 'metrics'
                ? 'bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Metrics
          </button>
        </div>

        {/* Live Video Tab */}
        {selectedTab === 'live' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Live Analysis Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Live Analysis</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Active Streams:</span>
                    <span className="text-blue-400 font-bold">{liveAnalysis.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Players:</span>
                    <span className="text-green-400 font-bold">
                      {liveAnalysis.length > 0 ? (liveAnalysis.reduce((sum, l) => sum + l.players_detected, 0) / liveAnalysis.length).toFixed(1) : '0'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Detection Accuracy:</span>
                    <span className="text-yellow-400 font-bold">
                      {liveAnalysis.length > 0 ? (liveAnalysis.reduce((sum, l) => sum + l.detection_accuracy, 0) / liveAnalysis.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Video Quality */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Video Quality</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Brightness:</span>
                    <span className="text-blue-400 font-bold">
                      {liveAnalysis.length > 0 ? (liveAnalysis.reduce((sum, l) => sum + l.brightness, 0) / liveAnalysis.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Contrast:</span>
                    <span className="text-green-400 font-bold">
                      {liveAnalysis.length > 0 ? (liveAnalysis.reduce((sum, l) => sum + l.contrast, 0) / liveAnalysis.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Sharpness:</span>
                    <span className="text-yellow-400 font-bold">
                      {liveAnalysis.length > 0 ? (liveAnalysis.reduce((sum, l) => sum + l.sharpness, 0) / liveAnalysis.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Streaming Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Streaming Activity</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {streamingHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-blue-500 rounded-sm"
                      style={{ height: `${Math.max((value / 50) * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Live Analysis */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent Live Analysis</h4>
              <div className="space-y-3">
                {liveAnalysis.slice(0, 5).map((analysis) => (
                  <div key={analysis.frame_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                        📺
                      </div>
                      <div>
                        <div className="font-semibold">Frame {analysis.frame_id}</div>
                        <div className="text-sm text-slate-300">
                          {analysis.players_detected} players • {analysis.events_detected.length} events
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-400">
                        {analysis.detection_accuracy * 100}%
                      </div>
                      <div className="text-sm text-slate-300">
                        {analysis.processing_time.toFixed(1)}ms
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* AI Commentary Tab */}
        {selectedTab === 'commentary' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Commentary Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">AI Commentary</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Commentary:</span>
                    <span className="text-blue-400 font-bold">{aiCommentary.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Confidence:</span>
                    <span className="text-green-400 font-bold">
                      {aiCommentary.length > 0 ? (aiCommentary.reduce((sum, c) => sum + c.commentary_confidence, 0) / aiCommentary.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Engagement:</span>
                    <span className="text-yellow-400 font-bold">
                      {aiCommentary.length > 0 ? (aiCommentary.reduce((sum, c) => sum + c.engagement_score, 0) / aiCommentary.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Commentary Styles */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Commentary Styles</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Excited:</span>
                    <span className="text-red-400 font-bold">
                      {aiCommentary.filter(c => c.voice_style === 'excited').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Analytical:</span>
                    <span className="text-blue-400 font-bold">
                      {aiCommentary.filter(c => c.voice_style === 'analytical').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Dramatic:</span>
                    <span className="text-purple-400 font-bold">
                      {aiCommentary.filter(c => c.voice_style === 'dramatic').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Commentary Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Engagement Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {qualityHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-green-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Commentary */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent AI Commentary</h4>
              <div className="space-y-3">
                {aiCommentary.slice(0, 5).map((commentary) => (
                  <div key={commentary.commentary_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🎙️
                      </div>
                      <div>
                        <div className="font-semibold">{commentary.main_commentary}</div>
                        <div className="text-sm text-slate-300">
                          {commentary.voice_style} • {commentary.target_audience}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getConfidenceColor(commentary.commentary_confidence)}`}>
                        {commentary.commentary_confidence * 100}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Engagement: {commentary.engagement_score * 100}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Instant Replays Tab */}
        {selectedTab === 'replays' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Replay Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Instant Replays</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Replays:</span>
                    <span className="text-blue-400 font-bold">{instantReplays.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Confidence:</span>
                    <span className="text-green-400 font-bold">
                      {instantReplays.length > 0 ? (instantReplays.reduce((sum, r) => sum + r.decision_confidence, 0) / instantReplays.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Highlight Potential:</span>
                    <span className="text-yellow-400 font-bold">
                      {instantReplays.length > 0 ? (instantReplays.reduce((sum, r) => sum + r.highlight_potential, 0) / instantReplays.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Replay Types */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Replay Types</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Goals:</span>
                    <span className="text-green-400 font-bold">
                      {instantReplays.filter(r => r.replay_type === 'goal').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Penalties:</span>
                    <span className="text-red-400 font-bold">
                      {instantReplays.filter(r => r.replay_type === 'penalty').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Saves:</span>
                    <span className="text-blue-400 font-bold">
                      {instantReplays.filter(r => r.replay_type === 'save').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Replay Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Highlight Potential</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {qualityHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-yellow-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Replays */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent Instant Replays</h4>
              <div className="space-y-3">
                {instantReplays.slice(0, 5).map((replay) => (
                  <div key={replay.replay_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center text-sm font-bold">
                        🔄
                      </div>
                      <div>
                        <div className="font-semibold">{replay.replay_type.toUpperCase()} Replay</div>
                        <div className="text-sm text-slate-300">
                          {replay.replay_duration}s • {replay.replay_quality}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getConfidenceColor(replay.decision_confidence)}`}>
                        {replay.decision_confidence * 100}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Highlight: {replay.highlight_potential * 100}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Broadcast Quality Tab */}
        {selectedTab === 'broadcast' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Quality Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Broadcast Quality</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Assessments:</span>
                    <span className="text-blue-400 font-bold">{broadcastQuality.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Video Quality:</span>
                    <span className="text-green-400 font-bold">
                      {broadcastQuality.length > 0 ? (broadcastQuality.reduce((sum, b) => sum + b.video_quality_score, 0) / broadcastQuality.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Audio Quality:</span>
                    <span className="text-yellow-400 font-bold">
                      {broadcastQuality.length > 0 ? (broadcastQuality.reduce((sum, b) => sum + b.audio_quality_score, 0) / broadcastQuality.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Quality Standards */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Quality Standards</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">4K:</span>
                    <span className="text-purple-400 font-bold">
                      {broadcastQuality.filter(b => b.broadcast_standard === '4K').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">HD:</span>
                    <span className="text-blue-400 font-bold">
                      {broadcastQuality.filter(b => b.broadcast_standard === 'HD').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">SD:</span>
                    <span className="text-yellow-400 font-bold">
                      {broadcastQuality.filter(b => b.broadcast_standard === 'SD').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quality Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Quality Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {qualityHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-purple-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Quality Assessments */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent Quality Assessments</h4>
              <div className="space-y-3">
                {broadcastQuality.slice(0, 5).map((quality) => (
                  <div key={quality.stream_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-sm font-bold">
                        📺
                      </div>
                      <div>
                        <div className="font-semibold">{quality.broadcast_standard} Stream</div>
                        <div className="text-sm text-slate-300">
                          {quality.resolution[0]}x{quality.resolution[1]} • {quality.bitrate}kbps
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getQualityColor(quality.overall_quality_score)}`}>
                        {quality.overall_quality_score * 100}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Latency: {quality.latency.toFixed(0)}ms
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Multi-Camera Tab */}
        {selectedTab === 'multi-camera' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Multi-Camera Overview */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Multi-Camera Analysis</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Analysis:</span>
                    <span className="text-blue-400 font-bold">{multiCameraAnalysis.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Cameras:</span>
                    <span className="text-green-400 font-bold">
                      {multiCameraAnalysis.length > 0 ? (multiCameraAnalysis.reduce((sum, m) => sum + m.active_cameras, 0) / multiCameraAnalysis.length).toFixed(1) : '0'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Coverage:</span>
                    <span className="text-yellow-400 font-bold">
                      {multiCameraAnalysis.length > 0 ? (multiCameraAnalysis.reduce((sum, m) => sum + m.coverage_analysis.ice_coverage, 0) / multiCameraAnalysis.length * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Camera Angles */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Camera Angles</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Overhead:</span>
                    <span className="text-blue-400 font-bold">
                      {multiCameraAnalysis.filter(m => m.optimal_angle === 'overhead').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Low:</span>
                    <span className="text-green-400 font-bold">
                      {multiCameraAnalysis.filter(m => m.optimal_angle === 'low').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Wide:</span>
                    <span className="text-yellow-400 font-bold">
                      {multiCameraAnalysis.filter(m => m.optimal_angle === 'wide').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Coverage Chart */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Coverage Trend</h4>
                <div className="h-24 bg-slate-600/50 rounded-lg p-2 flex items-end space-x-1">
                  {qualityHistory.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-cyan-500 rounded-sm"
                      style={{ height: `${Math.max(value * 100, 5)}%` }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Multi-Camera Analysis */}
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3">Recent Multi-Camera Analysis</h4>
              <div className="space-y-3">
                {multiCameraAnalysis.slice(0, 5).map((analysis) => (
                  <div key={analysis.analysis_id} className="flex items-center justify-between p-3 bg-slate-600/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-cyan-500 rounded-full flex items-center justify-center text-sm font-bold">
                        📹
                      </div>
                      <div>
                        <div className="font-semibold">{analysis.active_cameras} Cameras</div>
                        <div className="text-sm text-slate-300">
                          {analysis.optimal_angle} • Camera {analysis.best_camera}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-cyan-400">
                        {analysis.coverage_analysis.ice_coverage * 100}%
                      </div>
                      <div className="text-sm text-slate-300">
                        Smoothness: {analysis.transition_smoothness * 100}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Metrics Tab */}
        {selectedTab === 'metrics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {/* Streaming Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Streaming Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Active Streams:</span>
                    <span className="text-green-400 font-bold">{streamingMetrics?.active_streams || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Commentary:</span>
                    <span className="text-blue-400 font-bold">{streamingMetrics?.total_commentary || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Total Replays:</span>
                    <span className="text-yellow-400 font-bold">{streamingMetrics?.total_replays || 0}</span>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Avg Processing:</span>
                    <span className="text-green-400 font-bold">25ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Detection Accuracy:</span>
                    <span className="text-blue-400 font-bold">87%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Stream Quality:</span>
                    <span className="text-yellow-400 font-bold">HD</span>
                  </div>
                </div>
              </div>

              {/* Quality Metrics */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Quality</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Video Quality:</span>
                    <span className="text-green-400 font-bold">92%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Audio Quality:</span>
                    <span className="text-blue-400 font-bold">88%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Overall Quality:</span>
                    <span className="text-yellow-400 font-bold">90%</span>
                  </div>
                </div>
              </div>

              {/* Streaming Features */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold mb-3">Features</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Live Video:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">AI Commentary:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Instant Replay:</span>
                    <span className="text-green-400 font-bold">✓</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Multi-Camera:</span>
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
