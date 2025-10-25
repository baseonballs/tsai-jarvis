"use client";

import { useState, useEffect } from "react";

interface AnalyticsData {
  playerSpeed: number;
  shotVelocity: number;
  possessionTime: number;
  zoneTime: number;
  faceoffWinRate: number;
}

interface HeatMapData {
  x: number;
  y: number;
  intensity: number;
  team: 'home' | 'away';
}

export default function RealTimeAnalytics() {
  const [analytics, setAnalytics] = useState<AnalyticsData>({
    playerSpeed: 28.5,
    shotVelocity: 95.2,
    possessionTime: 45.3,
    zoneTime: 32.1,
    faceoffWinRate: 58.7
  });

  const [heatMapData] = useState<HeatMapData[]>([
    { x: 20, y: 30, intensity: 0.8, team: 'home' },
    { x: 40, y: 50, intensity: 0.6, team: 'home' },
    { x: 60, y: 70, intensity: 0.9, team: 'away' },
    { x: 80, y: 20, intensity: 0.7, team: 'away' },
    { x: 30, y: 80, intensity: 0.5, team: 'home' }
  ]);

  const [isLive, setIsLive] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setAnalytics(prev => ({
        playerSpeed: prev.playerSpeed + (Math.random() - 0.5) * 2,
        shotVelocity: prev.shotVelocity + (Math.random() - 0.5) * 5,
        possessionTime: Math.max(0, Math.min(100, prev.possessionTime + (Math.random() - 0.5) * 3)),
        zoneTime: Math.max(0, Math.min(100, prev.zoneTime + (Math.random() - 0.5) * 2)),
        faceoffWinRate: Math.max(0, Math.min(100, prev.faceoffWinRate + (Math.random() - 0.5) * 1))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Live Analytics Header */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold">Real-Time Analytics</h3>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm">{isLive ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-400">{analytics.playerSpeed.toFixed(1)} km/h</div>
            <div className="text-sm text-slate-300">Avg Speed</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400">{analytics.shotVelocity.toFixed(1)} mph</div>
            <div className="text-sm text-slate-300">Shot Velocity</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-yellow-400">{analytics.possessionTime.toFixed(1)}%</div>
            <div className="text-sm text-slate-300">Possession</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-400">{analytics.zoneTime.toFixed(1)}%</div>
            <div className="text-sm text-slate-300">Zone Time</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-red-400">{analytics.faceoffWinRate.toFixed(1)}%</div>
            <div className="text-sm text-slate-300">Faceoff Win</div>
          </div>
        </div>
      </div>

      {/* Heat Map */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Player Position Heat Map</h3>
        <div className="relative bg-slate-900 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
          <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900">
            {/* Ice Rink Lines */}
            <div className="absolute inset-0">
              <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-white/20"></div>
              <div className="absolute top-1/4 left-0 right-0 h-0.5 bg-white/10"></div>
              <div className="absolute top-3/4 left-0 right-0 h-0.5 bg-white/10"></div>
              <div className="absolute top-1/2 left-1/4 w-0.5 h-1/2 bg-white/10"></div>
              <div className="absolute top-1/2 right-1/4 w-0.5 h-1/2 bg-white/10"></div>
            </div>
            
            {/* Heat Map Points */}
            {heatMapData.map((point, index) => (
              <div
                key={index}
                className="absolute w-4 h-4 rounded-full opacity-70"
                style={{
                  left: `${point.x}%`,
                  top: `${point.y}%`,
                  backgroundColor: point.team === 'home' ? '#3b82f6' : '#ef4444',
                  opacity: point.intensity * 0.8 + 0.2,
                  transform: 'translate(-50%, -50%)'
                }}
              />
            ))}
          </div>
          
          {/* Legend */}
          <div className="absolute top-4 right-4 bg-slate-800/80 rounded-lg p-3">
            <div className="text-sm font-semibold mb-2">Team Positions</div>
            <div className="flex items-center space-x-2 mb-1">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="text-xs">Home Team</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="text-xs">Away Team</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Trends */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Performance Trends</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-lg font-semibold mb-3">Speed Distribution</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">0-10 km/h</span>
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '15%' }}></div>
                </div>
                <span className="text-sm">15%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">10-20 km/h</span>
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '35%' }}></div>
                </div>
                <span className="text-sm">35%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">20-30 km/h</span>
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '40%' }}></div>
                </div>
                <span className="text-sm">40%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">30+ km/h</span>
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div className="bg-red-500 h-2 rounded-full" style={{ width: '10%' }}></div>
                </div>
                <span className="text-sm">10%</span>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="text-lg font-semibold mb-3">Zone Occupancy</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Defensive Zone</span>
                <span className="text-sm font-semibold">32%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Neutral Zone</span>
                <span className="text-sm font-semibold">28%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Offensive Zone</span>
                <span className="text-sm font-semibold">40%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
