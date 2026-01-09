/**
 * GraphVisualizer component for BFS/DFS visualization - Enhanced with modern design
 */

import { useCallback, useEffect, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  Gauge,
  Code2,
  Layers,
  GitBranch,
} from 'lucide-react';
import type { GraphVisualization, GraphStep } from '../../api/visualization';

interface GraphVisualizerProps {
  visualization: GraphVisualization;
  autoPlay?: boolean;
  speed?: number;
}

const NODE_COLORS: Record<string, { fill: string; glow: string }> = {
  default: { fill: '#94a3b8', glow: 'transparent' },
  current: { fill: '#f97316', glow: '#f97316' },
  active: { fill: '#3b82f6', glow: '#3b82f6' },
  visited: { fill: '#22c55e', glow: '#22c55e' },
  found: { fill: '#22c55e', glow: '#22c55e' },
};

const EDGE_COLORS: Record<string, string> = {
  default: '#e2e8f0',
  active: '#3b82f6',
  visited: '#22c55e',
};

const LEGEND_ITEMS = [
  { color: 'bg-slate-400', label: '미방문' },
  { color: 'bg-orange-500', label: '현재 노드' },
  { color: 'bg-blue-500', label: '탐색 중' },
  { color: 'bg-emerald-500', label: '방문 완료' },
];

export default function GraphVisualizer({
  visualization,
  autoPlay = false,
  speed = 800,
}: GraphVisualizerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playSpeed, setPlaySpeed] = useState(speed);

  const steps = visualization.steps;
  const maxStep = steps.length - 1;
  const step: GraphStep | undefined = steps[currentStep];

  // Auto-play logic
  useEffect(() => {
    if (!isPlaying || currentStep >= maxStep) {
      if (currentStep >= maxStep) {
        setIsPlaying(false);
      }
      return;
    }

    const timer = setTimeout(() => {
      setCurrentStep((prev) => Math.min(prev + 1, maxStep));
    }, playSpeed);

    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, maxStep, playSpeed]);

  const handlePlay = useCallback(() => {
    if (currentStep >= maxStep) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  }, [currentStep, maxStep]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep(0);
  }, []);

  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep((prev) => Math.min(prev + 1, maxStep));
  }, [maxStep]);

  const handleStepBackward = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const getSpeedLabel = () => {
    if (playSpeed < 500) return '빠름';
    if (playSpeed < 1000) return '보통';
    return '느림';
  };

  if (!step) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 dark:text-slate-400">
        시각화 데이터가 없습니다
      </div>
    );
  }

  const getNodeColor = (nodeId: string) => {
    const state = step.node_states[nodeId] || 'default';
    return NODE_COLORS[state] || NODE_COLORS.default;
  };

  const getEdgeColor = (source: string, target: string) => {
    const key1 = `${source}-${target}`;
    const key2 = `${target}-${source}`;
    const state = step.edge_states[key1] || step.edge_states[key2] || 'default';
    return EDGE_COLORS[state] || EDGE_COLORS.default;
  };

  const progressPercent = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="w-full space-y-6">
      {/* Graph Visualization */}
      <div className="bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 rounded-2xl p-6">
        <svg width="100%" height="350" viewBox="0 0 400 350" className="mx-auto">
          <defs>
            {/* Glow filters for nodes */}
            {Object.entries(NODE_COLORS).map(([state, colors]) => (
              colors.glow !== 'transparent' && (
                <filter key={state} id={`glow-${state}`} x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              )
            ))}
            {/* Arrow marker */}
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Edges */}
          {visualization.edges.map((edge, idx) => {
            const sourceNode = visualization.nodes.find((n) => n.id === edge.source);
            const targetNode = visualization.nodes.find((n) => n.id === edge.target);
            if (!sourceNode || !targetNode) return null;

            const edgeColor = getEdgeColor(edge.source, edge.target);
            const isActive = edgeColor !== EDGE_COLORS.default;

            return (
              <line
                key={idx}
                x1={sourceNode.x}
                y1={sourceNode.y}
                x2={targetNode.x}
                y2={targetNode.y}
                stroke={edgeColor}
                strokeWidth={isActive ? 4 : 3}
                className="transition-all duration-300"
                strokeLinecap="round"
              />
            );
          })}

          {/* Nodes */}
          {visualization.nodes.map((node) => {
            const nodeColor = getNodeColor(node.id);
            const state = step.node_states[node.id] || 'default';
            const isCurrent = step.current_node === node.id;

            return (
              <g key={node.id}>
                {/* Pulse animation for current node */}
                {isCurrent && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={35}
                    fill="none"
                    stroke={nodeColor.fill}
                    strokeWidth={2}
                    className="animate-ping opacity-75"
                  />
                )}

                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={28}
                  fill={nodeColor.fill}
                  className="transition-all duration-300"
                  filter={state !== 'default' ? `url(#glow-${state})` : undefined}
                  stroke={isCurrent ? '#fff' : 'transparent'}
                  strokeWidth={3}
                />

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="white"
                  fontWeight="bold"
                  fontSize="16"
                  className="select-none"
                >
                  {node.value}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Step Description */}
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 border border-violet-200 dark:border-violet-800 rounded-xl p-4">
        <p className="text-violet-800 dark:text-violet-200 font-medium">{step.description}</p>
        <div className="mt-3 flex flex-wrap gap-4">
          {step.queue && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Layers className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm text-blue-700 dark:text-blue-300">
                큐: [{step.queue.join(', ')}]
              </span>
            </div>
          )}
          {step.stack && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <GitBranch className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm text-purple-700 dark:text-purple-300">
                스택: [{step.stack.join(', ')}]
              </span>
            </div>
          )}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
            <span className="text-sm text-emerald-700 dark:text-emerald-300">
              방문: [{step.visited.join(', ')}]
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={handleReset}
          className="p-3 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-xl transition-colors"
          title="처음으로"
        >
          <RotateCcw className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        </button>
        <button
          onClick={handleStepBackward}
          disabled={currentStep === 0}
          className="p-3 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 rounded-xl transition-colors"
          title="이전"
        >
          <SkipBack className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        </button>
        {isPlaying ? (
          <button
            onClick={handlePause}
            className="p-4 bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 text-white rounded-xl shadow-lg transition-all"
          >
            <Pause className="w-6 h-6" />
          </button>
        ) : (
          <button
            onClick={handlePlay}
            className="p-4 bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600 text-white rounded-xl shadow-lg transition-all"
          >
            <Play className="w-6 h-6" />
          </button>
        )}
        <button
          onClick={handleStepForward}
          disabled={currentStep >= maxStep}
          className="p-3 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 rounded-xl transition-colors"
          title="다음"
        >
          <SkipForward className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        </button>
      </div>

      {/* Progress */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-600 dark:text-slate-400">
            Step {currentStep + 1} / {steps.length}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="relative h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-violet-500 to-purple-500 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* Step Slider */}
        <input
          type="range"
          min={0}
          max={maxStep}
          value={currentStep}
          onChange={(e) => {
            setIsPlaying(false);
            setCurrentStep(Number(e.target.value));
          }}
          className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
        />
      </div>

      {/* Speed Control */}
      <div className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
        <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
          <Gauge className="w-5 h-5" />
          <span className="text-sm">속도:</span>
        </div>
        <input
          type="range"
          min={200}
          max={2000}
          step={100}
          value={2200 - playSpeed}
          onChange={(e) => setPlaySpeed(2200 - Number(e.target.value))}
          className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
        />
        <span className="text-sm font-medium text-violet-600 dark:text-violet-400 min-w-[40px]">
          {getSpeedLabel()}
        </span>
      </div>

      {/* Code Display */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Code2 className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300">알고리즘 코드</h4>
        </div>
        <div className="bg-slate-900 rounded-xl overflow-hidden">
          <div className="px-4 py-2 bg-slate-800 border-b border-slate-700 flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
            </div>
            <span className="text-xs text-slate-400 ml-2">python</span>
          </div>
          <pre className="p-4 text-sm overflow-x-auto max-h-64 text-slate-100">
            {visualization.code}
          </pre>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
        {LEGEND_ITEMS.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <div className={`w-4 h-4 ${item.color} rounded-full`} />
            <span className="text-sm text-slate-600 dark:text-slate-400">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
