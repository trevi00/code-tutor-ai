/**
 * Debugger Step Controls Component
 */

import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  FastForward,
  Rewind,
  Square,
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
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>
            스텝 {currentStep} / {totalSteps}
          </span>
          <span>{progress.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-100"
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

        {/* Reset */}
        <div className="w-px h-6 bg-gray-300 mx-2" />
        <ControlButton onClick={onReset} disabled={disabled} title="리셋">
          <Square className="w-4 h-4" />
        </ControlButton>
      </div>

      {/* Speed Control */}
      <div className="mt-4 flex items-center justify-center gap-2">
        <span className="text-sm text-gray-600">속도:</span>
        <div className="flex gap-1">
          {[0.5, 1, 2, 4].map((speed) => (
            <button
              key={speed}
              onClick={() => onSpeedChange(speed)}
              disabled={disabled}
              className={`px-2 py-1 text-xs rounded ${
                playSpeed === speed
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
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
  title?: string;
  children: React.ReactNode;
}

function ControlButton({
  onClick,
  disabled,
  primary,
  title,
  children,
}: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-2 rounded-full transition-colors ${
        primary
          ? 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-300'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:bg-gray-100 disabled:text-gray-400'
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
        className="p-1 rounded hover:bg-gray-100 disabled:opacity-50"
      >
        <SkipBack className="w-4 h-4" />
      </button>
      <span className="text-sm text-gray-600">
        {currentStep}/{totalSteps}
      </span>
      <button
        onClick={onStepForward}
        disabled={disabled || currentStep >= totalSteps}
        className="p-1 rounded hover:bg-gray-100 disabled:opacity-50"
      >
        <SkipForward className="w-4 h-4" />
      </button>
    </div>
  );
}
