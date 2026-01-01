/**
 * GraphVisualizer component for BFS/DFS visualization
 */

import { useCallback, useEffect, useState } from 'react';
import type { GraphVisualization, GraphStep } from '../../api/visualization';

interface GraphVisualizerProps {
  visualization: GraphVisualization;
  autoPlay?: boolean;
  speed?: number;
}

const NODE_COLORS: Record<string, string> = {
  default: '#94a3b8', // gray
  current: '#f97316', // orange
  active: '#3b82f6', // blue
  visited: '#22c55e', // green
  found: '#22c55e', // green
};

const EDGE_COLORS: Record<string, string> = {
  default: '#cbd5e1', // gray
  active: '#3b82f6', // blue
  visited: '#22c55e', // green
};

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

  if (!step) {
    return <div>No visualization data</div>;
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

  return (
    <div className="w-full">
      {/* Graph Visualization */}
      <div className="bg-gray-50 rounded-lg p-6 mb-4">
        <svg width="400" height="400" viewBox="0 0 400 400" className="mx-auto">
          {/* Edges */}
          {visualization.edges.map((edge, idx) => {
            const sourceNode = visualization.nodes.find((n) => n.id === edge.source);
            const targetNode = visualization.nodes.find((n) => n.id === edge.target);
            if (!sourceNode || !targetNode) return null;

            return (
              <line
                key={idx}
                x1={sourceNode.x}
                y1={sourceNode.y}
                x2={targetNode.x}
                y2={targetNode.y}
                stroke={getEdgeColor(edge.source, edge.target)}
                strokeWidth={3}
                className="transition-colors duration-300"
              />
            );
          })}

          {/* Nodes */}
          {visualization.nodes.map((node) => (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={25}
                fill={getNodeColor(node.id)}
                className="transition-colors duration-300"
                stroke={step.current_node === node.id ? '#000' : 'transparent'}
                strokeWidth={3}
              />
              <text
                x={node.x}
                y={node.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontWeight="bold"
                fontSize="14"
              >
                {node.value}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Step Description */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="text-blue-800">{step.description}</p>
        <div className="mt-2 flex flex-wrap gap-4 text-sm">
          {step.queue && (
            <span className="text-gray-600">
              큐: [{step.queue.join(', ')}]
            </span>
          )}
          {step.stack && (
            <span className="text-gray-600">
              스택: [{step.stack.join(', ')}]
            </span>
          )}
          <span className="text-green-600">
            방문: [{step.visited.join(', ')}]
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4 mb-4">
        <button
          onClick={handleReset}
          className="px-3 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg"
          title="처음으로"
        >
          ⏮️
        </button>
        <button
          onClick={handleStepBackward}
          disabled={currentStep === 0}
          className="px-3 py-2 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 rounded-lg"
          title="이전"
        >
          ⏪
        </button>
        {isPlaying ? (
          <button
            onClick={handlePause}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
          >
            ⏸️ 일시정지
          </button>
        ) : (
          <button
            onClick={handlePlay}
            className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg"
          >
            ▶️ 재생
          </button>
        )}
        <button
          onClick={handleStepForward}
          disabled={currentStep >= maxStep}
          className="px-3 py-2 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 rounded-lg"
          title="다음"
        >
          ⏩
        </button>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>Step {currentStep + 1} / {steps.length}</span>
        </div>
        <input
          type="range"
          min={0}
          max={maxStep}
          value={currentStep}
          onChange={(e) => {
            setIsPlaying(false);
            setCurrentStep(Number(e.target.value));
          }}
          className="w-full"
        />
      </div>

      {/* Speed Control */}
      <div className="flex items-center gap-4">
        <label className="text-sm text-gray-600">속도:</label>
        <input
          type="range"
          min={200}
          max={2000}
          step={100}
          value={2200 - playSpeed}
          onChange={(e) => setPlaySpeed(2200 - Number(e.target.value))}
          className="w-32"
        />
        <span className="text-sm text-gray-600">
          {playSpeed < 500 ? '빠름' : playSpeed < 1000 ? '보통' : '느림'}
        </span>
      </div>

      {/* Code Display */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-700 mb-2">알고리즘 코드</h4>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
          {visualization.code}
        </pre>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.default }} />
          <span>미방문</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.current }} />
          <span>현재 노드</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.active }} />
          <span>탐색 중</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.visited }} />
          <span>방문 완료</span>
        </div>
      </div>
    </div>
  );
}
