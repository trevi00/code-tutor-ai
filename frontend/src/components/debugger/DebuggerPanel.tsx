/**
 * Main Debugger Panel Component
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { Bug, Play, AlertCircle, Terminal, Code2 } from 'lucide-react';
import { debugCode, quickDebug } from '../../api/debugger';
import type { DebugResponse, QuickDebugResponse, ExecutionStep, QuickDebugStep } from '../../api/debugger';
import { STEP_TYPE_INFO } from '../../api/debugger';
import StepControls from './StepControls';
import CallStack from './CallStack';
import VariableInspector from './VariableInspector';

interface DebuggerPanelProps {
  code: string;
  input?: string;
  onLineHighlight?: (lineNumber: number) => void;
}

export default function DebuggerPanel({ code, input, onLineHighlight }: DebuggerPanelProps) {
  // Debug state
  const [isDebugging, setIsDebugging] = useState(false);
  const [debugResult, setDebugResult] = useState<DebugResponse | null>(null);
  const [quickResult, setQuickResult] = useState<QuickDebugResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Playback state
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);

  // Breakpoints
  const [breakpoints] = useState<number[]>([]);

  // Timer ref for auto-play
  const playTimerRef = useRef<number | null>(null);

  // Get current step data
  const getCurrentStepData = (): ExecutionStep | null => {
    if (debugResult && debugResult.steps[currentStep]) {
      return debugResult.steps[currentStep];
    }
    return null;
  };

  // Convert quick step to execution step (for unified display)
  const quickStepToExecutionStep = (step: QuickDebugStep): ExecutionStep => ({
    step_number: step.step,
    step_type: step.type,
    line_number: step.line,
    line_content: step.code,
    function_name: '<module>',
    variables: Object.entries(step.vars).map(([name, value]) => ({
      name,
      value,
      type: 'string' as const,
    })),
    call_stack: [],
    output: '',
    return_value: null,
    exception: null,
  });

  const totalSteps = debugResult?.total_steps ?? quickResult?.total_steps ?? 0;

  // Start debugging
  const handleStartDebug = async () => {
    if (!code.trim()) {
      setError('코드를 입력해주세요.');
      return;
    }

    setIsDebugging(true);
    setError(null);
    setCurrentStep(0);
    setIsPlaying(false);

    try {
      const result = await debugCode({
        code,
        input_data: input,
        breakpoints: breakpoints.length > 0 ? breakpoints : undefined,
      });
      setDebugResult(result);
      setQuickResult(null);

      if (result.error) {
        setError(result.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '디버깅 중 오류가 발생했습니다.');
    } finally {
      setIsDebugging(false);
    }
  };

  // Quick debug (lighter weight)
  const handleQuickDebug = async () => {
    if (!code.trim()) {
      setError('코드를 입력해주세요.');
      return;
    }

    setIsDebugging(true);
    setError(null);
    setCurrentStep(0);
    setIsPlaying(false);

    try {
      const result = await quickDebug({
        code,
        input_data: input,
      });
      setQuickResult(result);
      setDebugResult(null);

      if (result.error) {
        setError(result.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '디버깅 중 오류가 발생했습니다.');
    } finally {
      setIsDebugging(false);
    }
  };

  // Step controls
  const handleStepForward = useCallback(() => {
    if (currentStep < totalSteps) {
      setCurrentStep((prev) => prev + 1);
    }
  }, [currentStep, totalSteps]);

  const handleStepBack = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  }, [currentStep]);

  const handleGoToStart = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const handleGoToEnd = () => {
    setCurrentStep(totalSteps);
    setIsPlaying(false);
  };

  const handleReset = () => {
    setDebugResult(null);
    setQuickResult(null);
    setCurrentStep(0);
    setIsPlaying(false);
    setError(null);
  };

  const handlePlay = () => {
    if (currentStep < totalSteps) {
      setIsPlaying(true);
    }
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  // Auto-play effect
  useEffect(() => {
    if (isPlaying && currentStep < totalSteps) {
      const interval = 1000 / playSpeed;
      playTimerRef.current = window.setTimeout(() => {
        setCurrentStep((prev) => {
          const next = prev + 1;
          if (next >= totalSteps) {
            setIsPlaying(false);
          }
          return next;
        });
      }, interval);
    }

    return () => {
      if (playTimerRef.current) {
        clearTimeout(playTimerRef.current);
      }
    };
  }, [isPlaying, currentStep, totalSteps, playSpeed]);

  // Update line highlight when step changes
  useEffect(() => {
    const stepData = getCurrentStepData();
    if (stepData && onLineHighlight) {
      onLineHighlight(stepData.line_number);
    }
  }, [currentStep, debugResult, onLineHighlight]);

  // Get current step for display
  const currentStepData = debugResult?.steps[currentStep] ??
    (quickResult?.steps[currentStep] ? quickStepToExecutionStep(quickResult.steps[currentStep]) : null);

  const hasResult = debugResult !== null || quickResult !== null;

  return (
    <div className="h-full flex flex-col bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bug className="w-5 h-5 text-purple-600" />
          <span className="font-medium text-gray-800">단계별 디버거</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleQuickDebug}
            disabled={isDebugging || !code.trim()}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <Play className="w-4 h-4" />
            빠른 실행
          </button>
          <button
            onClick={handleStartDebug}
            disabled={isDebugging || !code.trim()}
            className="px-3 py-1.5 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <Bug className="w-4 h-4" />
            {isDebugging ? '디버깅 중...' : '디버그 시작'}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-100 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-red-700 whitespace-pre-wrap">{error}</div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-auto p-4">
        {!hasResult ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <Bug className="w-12 h-12 mb-3 text-gray-300" />
            <p className="text-lg font-medium mb-1">디버거 준비 완료</p>
            <p className="text-sm">코드를 작성하고 "디버그 시작" 버튼을 클릭하세요</p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Step Controls */}
            <StepControls
              currentStep={currentStep}
              totalSteps={totalSteps}
              isPlaying={isPlaying}
              onStepForward={handleStepForward}
              onStepBack={handleStepBack}
              onPlay={handlePlay}
              onPause={handlePause}
              onReset={handleReset}
              onGoToStart={handleGoToStart}
              onGoToEnd={handleGoToEnd}
              playSpeed={playSpeed}
              onSpeedChange={setPlaySpeed}
            />

            {/* Current Step Info */}
            {currentStepData && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center gap-3 mb-3">
                  <Code2 className="w-5 h-5 text-gray-500" />
                  <span className="font-medium text-gray-800">현재 실행 위치</span>
                  <span className={`px-2 py-0.5 rounded text-xs ${STEP_TYPE_INFO[currentStepData.step_type].color}`}>
                    {STEP_TYPE_INFO[currentStepData.step_type].label}
                  </span>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-sm text-gray-500 mb-1">
                    줄 {currentStepData.line_number}
                    {currentStepData.function_name !== '<module>' && (
                      <span className="ml-2">
                        in <span className="font-mono">{currentStepData.function_name}()</span>
                      </span>
                    )}
                  </div>
                  <pre className="font-mono text-sm bg-gray-900 text-green-400 p-2 rounded overflow-x-auto">
                    {currentStepData.line_content}
                  </pre>
                </div>
              </div>
            )}

            {/* Variables */}
            {currentStepData && currentStepData.variables.length > 0 && (
              <VariableInspector
                variables={currentStepData.variables}
                title="현재 변수"
              />
            )}

            {/* Call Stack */}
            {currentStepData && currentStepData.call_stack.length > 0 && (
              <CallStack
                frames={currentStepData.call_stack}
                currentLine={currentStepData.line_number}
              />
            )}

            {/* Output */}
            {(debugResult?.output || quickResult?.output) && (
              <div className="bg-white rounded-lg border border-gray-200">
                <div className="px-3 py-2 bg-gray-50 rounded-t-lg border-b flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-gray-500" />
                  <span className="font-medium text-gray-700 text-sm">출력</span>
                </div>
                <pre className="p-3 font-mono text-sm bg-gray-900 text-gray-100 rounded-b-lg whitespace-pre-wrap max-h-40 overflow-auto">
                  {debugResult?.output || quickResult?.output || '(출력 없음)'}
                </pre>
              </div>
            )}

            {/* Execution Info */}
            <div className="text-xs text-gray-500 flex items-center gap-4">
              <span>
                총 {totalSteps} 스텝
              </span>
              <span>
                실행 시간: {(debugResult?.execution_time_ms ?? quickResult?.execution_time_ms ?? 0).toFixed(1)}ms
              </span>
              {debugResult?.session_id && (
                <span className="font-mono">
                  세션: {debugResult.session_id.slice(0, 8)}...
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
