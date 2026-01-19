/**
 * SortingVisualizer component - Enhanced with modern design
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
  RefreshCw,
  ArrowLeftRight,
} from 'lucide-react';
import ArrayBar from './ArrayBar';
import type { SortingVisualization, VisualizationStep } from '../../api/visualization';

interface SortingVisualizerProps {
  visualization: SortingVisualization;
  autoPlay?: boolean;
  speed?: number; // ms per step
}

const LEGEND_ITEMS = [
  { color: 'bg-gradient-to-t from-blue-500 to-blue-400', label: '기본' },
  { color: 'bg-gradient-to-t from-amber-500 to-yellow-400', label: '비교 중' },
  { color: 'bg-gradient-to-t from-red-500 to-rose-400', label: '교환 중' },
  { color: 'bg-gradient-to-t from-emerald-500 to-green-400', label: '정렬됨' },
  { color: 'bg-gradient-to-t from-purple-500 to-violet-400', label: '피벗' },
];

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
      if (currentStep >= maxStep && isPlaying) {
        // Use requestAnimationFrame to avoid synchronous setState in effect
        const frame = requestAnimationFrame(() => setIsPlaying(false));
        return () => cancelAnimationFrame(frame);
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
    if (playSpeed < 300) return '빠름';
    if (playSpeed < 700) return '보통';
    return '느림';
  };

  if (!step) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 dark:text-slate-400">
        시각화 데이터가 없습니다
      </div>
    );
  }

  const progressPercent = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="w-full space-y-6">
      {/* Array Visualization */}
      <div className="bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 rounded-2xl p-6">
        <div
          className="flex items-end justify-center gap-1"
          style={{ height: '220px' }}
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
      <div className="bg-gradient-to-r from-cyan-50 to-teal-50 dark:from-cyan-900/20 dark:to-teal-900/20 border border-cyan-200 dark:border-cyan-800 rounded-xl p-4">
        <p className="text-cyan-800 dark:text-cyan-200 font-medium">{step.description}</p>
        {step.code_line && (
          <p className="text-sm text-cyan-600 dark:text-cyan-400 mt-1 flex items-center gap-1">
            <Code2 className="w-4 h-4" />
            코드 라인: {step.code_line}
          </p>
        )}
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

      {/* Progress & Stats */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-600 dark:text-slate-400">
            Step {currentStep + 1} / {steps.length}
          </span>
          <div className="flex items-center gap-4 text-slate-500 dark:text-slate-400">
            <span className="flex items-center gap-1">
              <RefreshCw className="w-4 h-4 text-amber-500" />
              비교: <span className="font-bold text-amber-600 dark:text-amber-400">{visualization.total_comparisons}</span>
            </span>
            <span className="flex items-center gap-1">
              <ArrowLeftRight className="w-4 h-4 text-rose-500" />
              교환: <span className="font-bold text-rose-600 dark:text-rose-400">{visualization.total_swaps}</span>
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="relative h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500 to-teal-500 rounded-full transition-all duration-300"
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
          className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
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
          min={100}
          max={2000}
          step={100}
          value={2100 - playSpeed}
          onChange={(e) => setPlaySpeed(2100 - Number(e.target.value))}
          className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
        />
        <span className="text-sm font-medium text-cyan-600 dark:text-cyan-400 min-w-[40px]">
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
          <pre className="p-4 text-sm overflow-x-auto max-h-64">
            {visualization.code.split('\n').map((line, idx) => (
              <div
                key={idx}
                className={`flex ${
                  step.code_line === idx + 1
                    ? 'bg-yellow-500/20 -mx-4 px-4'
                    : ''
                }`}
              >
                <span className="text-slate-500 w-8 text-right mr-4 select-none">
                  {idx + 1}
                </span>
                <span className="text-slate-100">{line}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
        {LEGEND_ITEMS.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <div className={`w-4 h-4 ${item.color} rounded`} />
            <span className="text-sm text-slate-600 dark:text-slate-400">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
