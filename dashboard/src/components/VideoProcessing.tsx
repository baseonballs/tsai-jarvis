"use client";

import { useState, useEffect } from "react";

interface VideoProcessingProps {
  isProcessing: boolean;
  frameRate: number;
  detectionCount: number;
  processingLatency: number;
}

export default function VideoProcessing({ 
  isProcessing, 
  frameRate, 
  detectionCount, 
  processingLatency 
}: VideoProcessingProps) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [detectedPlayers, setDetectedPlayers] = useState([
    { id: 1, x: 120, y: 80, confidence: 0.95, team: 'home' },
    { id: 2, x: 200, y: 150, confidence: 0.87, team: 'away' },
    { id: 3, x: 300, y: 200, confidence: 0.92, team: 'home' },
    { id: 4, x: 150, y: 250, confidence: 0.89, team: 'away' }
  ]);

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        setCurrentFrame(prev => prev + 1);
        // Simulate player movement
        setDetectedPlayers(prev => prev.map(player => ({
          ...player,
          x: player.x + (Math.random() - 0.5) * 10,
          y: player.y + (Math.random() - 0.5) * 10,
          confidence: Math.max(0.7, player.confidence + (Math.random() - 0.5) * 0.1)
        })));
      }, 1000 / frameRate);

      return () => clearInterval(interval);
    }
  }, [isProcessing, frameRate]);

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold">Live Video Processing</h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm">{isProcessing ? 'Processing' : 'Stopped'}</span>
          </div>
          <div className="text-sm text-slate-300">
            {frameRate} FPS
          </div>
        </div>
      </div>

      {/* Video Feed Simulation */}
      <div className="relative bg-slate-900 rounded-lg overflow-hidden mb-4" style={{ aspectRatio: '16/9' }}>
        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
          <div className="text-center">
            <div className="text-4xl mb-2">🏒</div>
            <div className="text-lg font-semibold">Live Hockey Feed</div>
            <div className="text-sm text-slate-400">Frame #{currentFrame}</div>
          </div>
        </div>
        
        {/* Player Detection Overlays */}
        {detectedPlayers.map((player) => (
          <div
            key={player.id}
            className="absolute w-6 h-6 border-2 rounded-full flex items-center justify-center text-xs font-bold"
            style={{
              left: `${player.x}px`,
              top: `${player.y}px`,
              borderColor: player.team === 'home' ? '#3b82f6' : '#ef4444',
              backgroundColor: player.team === 'home' ? '#3b82f6' : '#ef4444',
              color: 'white'
            }}
          >
            {player.id}
          </div>
        ))}
      </div>

      {/* Processing Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-700/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-green-400">{detectionCount}</div>
          <div className="text-xs text-slate-300">Players Detected</div>
        </div>
        <div className="bg-slate-700/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-blue-400">{processingLatency}ms</div>
          <div className="text-xs text-slate-300">Processing Time</div>
        </div>
        <div className="bg-slate-700/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-yellow-400">{frameRate}</div>
          <div className="text-xs text-slate-300">Frames/Second</div>
        </div>
        <div className="bg-slate-700/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-purple-400">YOLO v8</div>
          <div className="text-xs text-slate-300">AI Model</div>
        </div>
      </div>

      {/* Detection Details */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold mb-2">Player Detection Details</h4>
        <div className="space-y-2">
          {detectedPlayers.map((player) => (
            <div key={player.id} className="flex items-center justify-between bg-slate-700/30 rounded p-2">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${player.team === 'home' ? 'bg-blue-500' : 'bg-red-500'}`}></div>
                <span className="text-sm">Player {player.id}</span>
              </div>
              <div className="text-sm text-slate-300">
                {(player.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
