"use client";

import { useState, useEffect } from "react";

interface MomentumData {
  current_momentum: number;
  team_momentum: {
    home: number;
    away: number;
  };
  momentum_history: number[];
}

interface PressureData {
  current_pressure: number;
  team_pressure: {
    home: number;
    away: number;
  };
  pressure_history: number[];
}

interface EnhancedAnalyticsProps {
  momentumData?: MomentumData;
  pressureData?: PressureData;
  isLoading?: boolean;
}

export default function EnhancedAnalytics({ 
  momentumData, 
  pressureData, 
  isLoading = false 
}: EnhancedAnalyticsProps) {
  const [momentumHistory, setMomentumHistory] = useState<number[]>([]);
  const [pressureHistory, setPressureHistory] = useState<number[]>([]);

  useEffect(() => {
    if (momentumData?.momentum_history) {
      setMomentumHistory(momentumData.momentum_history.slice(-20)); // Last 20 data points
    }
  }, [momentumData]);

  useEffect(() => {
    if (pressureData?.pressure_history) {
      setPressureHistory(pressureData.pressure_history.slice(-20)); // Last 20 data points
    }
  }, [pressureData]);

  const getMomentumColor = (momentum: number) => {
    if (momentum > 0.6) return "text-green-400";
    if (momentum > 0.4) return "text-yellow-400";
    return "text-red-400";
  };

  const getPressureColor = (pressure: number) => {
    if (pressure > 0.7) return "text-red-400";
    if (pressure > 0.4) return "text-yellow-400";
    return "text-green-400";
  };

  const getMomentumLabel = (momentum: number) => {
    if (momentum > 0.6) return "High Momentum";
    if (momentum > 0.4) return "Balanced";
    return "Low Momentum";
  };

  const getPressureLabel = (pressure: number) => {
    if (pressure > 0.7) return "High Pressure";
    if (pressure > 0.4) return "Moderate Pressure";
    return "Low Pressure";
  };

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Enhanced Analytics</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-slate-300">Loading enhanced analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Momentum Analytics */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Game Momentum</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Overall Momentum */}
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {momentumData?.current_momentum ? (momentumData.current_momentum * 100).toFixed(1) : '--'}%
            </div>
            <div className="text-sm text-slate-300">Overall Momentum</div>
            <div className={`text-xs mt-1 ${getMomentumColor(momentumData?.current_momentum || 0)}`}>
              {getMomentumLabel(momentumData?.current_momentum || 0)}
            </div>
          </div>

          {/* Team Momentum */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">Home Team</span>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold text-blue-400">
                  {momentumData?.team_momentum?.home ? (momentumData.team_momentum.home * 100).toFixed(1) : '--'}%
                </div>
                <div className={`w-3 h-3 rounded-full ${getMomentumColor(momentumData?.team_momentum?.home || 0)}`}></div>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">Away Team</span>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold text-red-400">
                  {momentumData?.team_momentum?.away ? (momentumData.team_momentum.away * 100).toFixed(1) : '--'}%
                </div>
                <div className={`w-3 h-3 rounded-full ${getMomentumColor(momentumData?.team_momentum?.away || 0)}`}></div>
              </div>
            </div>
          </div>

          {/* Momentum Chart */}
          <div>
            <div className="text-sm text-slate-300 mb-2">Momentum Trend</div>
            <div className="h-16 bg-slate-700/50 rounded-lg p-2 flex items-end space-x-1">
              {momentumHistory.map((value, index) => (
                <div
                  key={index}
                  className="flex-1 bg-blue-500 rounded-sm"
                  style={{ height: `${Math.max(value * 100, 5)}%` }}
                ></div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Pressure Analytics */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Game Pressure</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Overall Pressure */}
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">
              {pressureData?.current_pressure ? (pressureData.current_pressure * 100).toFixed(1) : '--'}%
            </div>
            <div className="text-sm text-slate-300">Overall Pressure</div>
            <div className={`text-xs mt-1 ${getPressureColor(pressureData?.current_pressure || 0)}`}>
              {getPressureLabel(pressureData?.current_pressure || 0)}
            </div>
          </div>

          {/* Team Pressure */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">Home Team</span>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold text-blue-400">
                  {pressureData?.team_pressure?.home ? (pressureData.team_pressure.home * 100).toFixed(1) : '--'}%
                </div>
                <div className={`w-3 h-3 rounded-full ${getPressureColor(pressureData?.team_pressure?.home || 0)}`}></div>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">Away Team</span>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold text-red-400">
                  {pressureData?.team_pressure?.away ? (pressureData.team_pressure.away * 100).toFixed(1) : '--'}%
                </div>
                <div className={`w-3 h-3 rounded-full ${getPressureColor(pressureData?.team_pressure?.away || 0)}`}></div>
              </div>
            </div>
          </div>

          {/* Pressure Chart */}
          <div>
            <div className="text-sm text-slate-300 mb-2">Pressure Trend</div>
            <div className="h-16 bg-slate-700/50 rounded-lg p-2 flex items-end space-x-1">
              {pressureHistory.map((value, index) => (
                <div
                  key={index}
                  className="flex-1 bg-purple-500 rounded-sm"
                  style={{ height: `${Math.max(value * 100, 5)}%` }}
                ></div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Features Status */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-xl font-bold mb-4">Enhanced Features</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400">✓</div>
            <div className="text-sm text-slate-300">Momentum Tracking</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400">✓</div>
            <div className="text-sm text-slate-300">Pressure Analysis</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400">✓</div>
            <div className="text-sm text-slate-300">Performance Prediction</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400">✓</div>
            <div className="text-sm text-slate-300">Advanced Events</div>
          </div>
        </div>
      </div>
    </div>
  );
}
