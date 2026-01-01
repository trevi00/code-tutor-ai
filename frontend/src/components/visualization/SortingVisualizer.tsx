/**
 * SortingVisualizer component
 */

import { useCallback, useEffect, useState } from 'react';
import ArrayBar from './ArrayBar';
import type { SortingVisualization, VisualizationStep } from '../../api/visualization';

interface SortingVisualizerProps {
  visualization: SortingVisualization;
  autoPlay?: boolean;
  speed?: number; // ms per step
}

export default function SortingVisualizer({
  visualization,
  autoPlay = false,
  speed = 500,
}: SortingVisualizerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playSpeed, setPlaySpeed] = useState(speed);

  const steps = visualization.steps;
  const maxStep = steps.length - 1;
  const step: VisualizationStep | undefined = steps[currentStep];

  const maxValue = Math.max(...visualization.initial_data);

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

  return (
    <div className="w-full">
      {/* Array Visualization */}
      <div className="bg-gray-50 rounded-lg p-6 mb-4">
        <div
          className="flex items-end justify-center gap-1"
          style={{ height: '250px' }}
        >
          {step.array_state.map((value, index) => (
            <ArrayBar
              key={index}
              value={value}
              maxValue={maxValue}
              state={step.element_states[index]}
              index={index}
            />
          ))}
        </div>
      </div>

      {/* Step Description */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="text-blue-800">{step.description}</p>
        {step.code_line && (
          <p className="text-sm text-blue-600 mt-1">코드 라인: {step.code_line}</p>
        )}
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
          <span>
            비교: {visualization.total_comparisons} | 교환: {visualization.total_swaps}
          </span>
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
          min={100}
          max={2000}
          step={100}
          value={2100 - playSpeed}
          onChange={(e) => setPlaySpeed(2100 - Number(e.target.value))}
          className="w-32"
        />
        <span className="text-sm text-gray-600">
          {playSpeed < 300 ? '빠름' : playSpeed < 700 ? '보통' : '느림'}
        </span>
      </div>

      {/* Code Display */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-700 mb-2">알고리즘 코드</h4>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
          {visualization.code.split('\n').map((line, idx) => (
            <div
              key={idx}
              className={`${
                step.code_line === idx + 1
                  ? 'bg-yellow-500 bg-opacity-30'
                  : ''
              }`}
            >
              <span className="text-gray-500 mr-4">{idx + 1}</span>
              {line}
            </div>
          ))}
        </pre>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-400 rounded" />
          <span>기본</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-400 rounded" />
          <span>비교 중</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-400 rounded" />
          <span>교환 중</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span>정렬됨</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-purple-500 rounded" />
          <span>피벗</span>
        </div>
      </div>
    </div>
  );
}
