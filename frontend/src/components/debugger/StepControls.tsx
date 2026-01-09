/**
 * Debugger Step Controls Component - Enhanced with modern design
 */

import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  FastForward,
  Rewind,
  Square,
  Gauge,
} from 'lucide-react';

interface StepControlsProps {
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  onStepForward: () => void;
  onStepBack: () => void;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onGoToStart: () => void;
  onGoToEnd: () => void;
  playSpeed: number;
  onSpeedChange: (speed: number) => void;
  disabled?: boolean;
}

export default function StepControls({
  currentStep,
  totalSteps,
  isPlaying,
  onStepForward,
  onStepBack,
  onPlay,
  onPause,
  onReset,
  onGoToStart,
  onGoToEnd,
  playSpeed,
  onSpeedChange,
  disabled = false,
}: StepControlsProps) {
  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
      {/* Progress Bar */}
      <div className="mb-5">
        <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
          <span className="font-medium">
            스텝 <span className="text-purple-600 dark:text-purple-400 font-bold">{currentStep}</span> / {totalSteps}
          </span>
          <span className="font-mono text-purple-600 dark:text-purple-400">{progress.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5 overflow-hidden">
          <div
            className="bg-gradient-to-r from-purple-600 to-violet-600 h-2.5 rounded-full transition-all duration-200"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-center gap-2">
        {/* Go to Start */}
        <ControlButton
          onClick={onGoToStart}
          disabled={disabled || currentStep === 0}
          title="처음으로"
        >
          <Rewind className="w-4 h-4" />
        </ControlButton>

        {/* Step Back */}
        <ControlButton
          onClick={onStepBack}
          disabled={disabled || currentStep === 0}
          title="이전 스텝"
        >
          <SkipBack className="w-4 h-4" />
        </ControlButton>

        {/* Play/Pause */}
        <ControlButton
          onClick={isPlaying ? onPause : onPlay}
          disabled={disabled || currentStep >= totalSteps}
          primary
          title={isPlaying ? '일시정지' : '재생'}
        >
          {isPlaying ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5" />
          )}
        </ControlButton>

        {/* Step Forward */}
        <ControlButton
          onClick={onStepForward}
          disabled={disabled || currentStep >= totalSteps}
          title="다음 스텝"
        >
          <SkipForward className="w-4 h-4" />
        </ControlButton>

        {/* Go to End */}
        <ControlButton
          onClick={onGoToEnd}
          disabled={disabled || currentStep >= totalSteps}
          title="끝으로"
        >
          <FastForward className="w-4 h-4" />
        </ControlButton>

        {/* Divider */}
        <div className="w-px h-8 bg-slate-200 dark:bg-slate-600 mx-2" />

        {/* Reset */}
        <ControlButton onClick={onReset} disabled={disabled} title="리셋" danger>
          <Square className="w-4 h-4" />
        </ControlButton>
      </div>

      {/* Speed Control */}
      <div className="mt-5 flex items-center justify-center gap-3">
        <div className="flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-400">
          <Gauge className="w-4 h-4" />
          <span>속도:</span>
        </div>
        <div className="flex gap-1.5">
          {[0.5, 1, 2, 4].map((speed) => (
            <button
              key={speed}
              onClick={() => onSpeedChange(speed)}
              disabled={disabled}
              className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-all ${
                playSpeed === speed
                  ? 'bg-gradient-to-r from-purple-600 to-violet-600 text-white shadow-md'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
              } disabled:opacity-50`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

interface ControlButtonProps {
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  danger?: boolean;
  title?: string;
  children: React.ReactNode;
}

function ControlButton({
  onClick,
  disabled,
  primary,
  danger,
  title,
  children,
}: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-2.5 rounded-xl transition-all ${
        primary
          ? 'bg-gradient-to-r from-purple-600 to-violet-600 text-white hover:from-purple-700 hover:to-violet-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 shadow-lg scale-110'
          : danger
          ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 disabled:bg-slate-100 dark:disabled:bg-slate-700 disabled:text-slate-400 dark:disabled:text-slate-500'
          : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:text-slate-400 dark:disabled:text-slate-600'
      } disabled:cursor-not-allowed`}
    >
      {children}
    </button>
  );
}

// Compact controls for inline use
interface CompactControlsProps {
  currentStep: number;
  totalSteps: number;
  onStepForward: () => void;
  onStepBack: () => void;
  disabled?: boolean;
}

export function CompactControls({
  currentStep,
  totalSteps,
  onStepForward,
  onStepBack,
  disabled,
}: CompactControlsProps) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onStepBack}
        disabled={disabled || currentStep === 0}
        className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 transition-colors"
      >
        <SkipBack className="w-4 h-4 text-slate-600 dark:text-slate-400" />
      </button>
      <span className="text-sm text-slate-600 dark:text-slate-400 font-mono px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded-lg">
        {currentStep}/{totalSteps}
      </span>
      <button
        onClick={onStepForward}
        disabled={disabled || currentStep >= totalSteps}
        className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 transition-colors"
      >
        <SkipForward className="w-4 h-4 text-slate-600 dark:text-slate-400" />
      </button>
    </div>
  );
}
