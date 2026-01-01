/**
 * Main Performance Panel Component
 */

import { useState } from 'react';
import { Activity, Play, Loader2, Gauge, RefreshCw } from 'lucide-react';
import { analyzePerformance, quickAnalyze } from '../../api/performance';
import type { PerformanceResult, QuickAnalyzeResult } from '../../api/performance';
import ComplexityDisplay, { CompactComplexity } from './ComplexityDisplay';
import { RuntimeMetricsDisplay, MemoryMetricsDisplay, HotspotDisplay } from './MetricsDisplay';
import IssuesDisplay from './IssuesDisplay';
import OptimizationScore from './OptimizationScore';

interface PerformancePanelProps {
  code: string;
  input?: string;
}

export default function PerformancePanel({ code, input }: PerformancePanelProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PerformanceResult | null>(null);
  const [quickResult, setQuickResult] = useState<QuickAnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisType, setAnalysisType] = useState<'quick' | 'full'>('quick');

  const handleQuickAnalyze = async () => {
    if (!code.trim()) {
      setError('코드를 입력해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await quickAnalyze({ code });
      setQuickResult(response);
      setAnalysisType('quick');
    } catch (err) {
      setError(err instanceof Error ? err.message : '분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFullAnalyze = async () => {
    if (!code.trim()) {
      setError('코드를 입력해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setQuickResult(null);

    try {
      const response = await analyzePerformance({
        code,
        input_data: input,
        include_runtime: true,
        include_memory: true,
      });
      setResult(response);
      setAnalysisType('full');
    } catch (err) {
      setError(err instanceof Error ? err.message : '분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setQuickResult(null);
    setError(null);
  };

  const hasResult = result !== null || quickResult !== null;

  return (
    <div className="h-full flex flex-col bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-600" />
          <span className="font-medium text-gray-800">성능 분석</span>
        </div>
        <div className="flex items-center gap-2">
          {hasResult && (
            <button
              onClick={handleReset}
              className="p-1.5 text-gray-500 hover:text-gray-700 rounded"
              title="초기화"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={handleQuickAnalyze}
            disabled={isAnalyzing || !code.trim()}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <Gauge className="w-4 h-4" />
            빠른 분석
          </button>
          <button
            onClick={handleFullAnalyze}
            disabled={isAnalyzing || !code.trim()}
            className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {isAnalyzing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {isAnalyzing ? '분석 중...' : '전체 분석'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-100 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {!hasResult ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <Activity className="w-12 h-12 mb-3 text-gray-300" />
            <p className="text-lg font-medium mb-1">성능 분석 준비 완료</p>
            <p className="text-sm text-center">
              코드를 작성하고 "빠른 분석" 또는 "전체 분석"을 클릭하세요
            </p>
            <div className="mt-4 text-xs text-gray-400 space-y-1">
              <p>빠른 분석: 시간/공간 복잡도만 분석</p>
              <p>전체 분석: 런타임 성능 + 메모리 프로파일링 포함</p>
            </div>
          </div>
        ) : analysisType === 'quick' && quickResult ? (
          <QuickAnalysisResult result={quickResult} />
        ) : result ? (
          <FullAnalysisResult result={result} />
        ) : null}
      </div>
    </div>
  );
}

interface QuickAnalysisResultProps {
  result: QuickAnalyzeResult;
}

function QuickAnalysisResult({ result }: QuickAnalysisResultProps) {
  if (result.status === 'error') {
    return (
      <div className="text-red-600 p-4 bg-red-50 rounded-lg">
        분석 오류: {result.error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-700 mb-3">복잡도 요약</h3>
        <CompactComplexity
          timeComplexity={result.time_complexity}
          spaceComplexity={result.space_complexity}
        />
        <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">시간:</span>{' '}
            <span className="text-gray-700">{result.time_explanation}</span>
          </div>
          <div>
            <span className="text-gray-500">공간:</span>{' '}
            <span className="text-gray-700">{result.space_explanation}</span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded-lg p-3">
          <span className="text-sm text-gray-500">중첩 깊이</span>
          <p className="text-2xl font-bold text-gray-800">
            {result.max_nesting_depth}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <span className="text-sm text-gray-500">이슈</span>
          <p className="text-2xl font-bold text-gray-800">
            {result.issues_count}개
          </p>
        </div>
      </div>

      <p className="text-sm text-gray-500 text-center">
        더 자세한 분석은 "전체 분석"을 사용하세요
      </p>
    </div>
  );
}

interface FullAnalysisResultProps {
  result: PerformanceResult;
}

function FullAnalysisResult({ result }: FullAnalysisResultProps) {
  if (result.status === 'error') {
    return (
      <div className="text-red-600 p-4 bg-red-50 rounded-lg">
        분석 오류: {result.error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Score */}
      <OptimizationScore score={result.optimization_score} />

      {/* Complexity */}
      {result.complexity && <ComplexityDisplay complexity={result.complexity} />}

      {/* Runtime */}
      {result.runtime && <RuntimeMetricsDisplay metrics={result.runtime} />}

      {/* Memory */}
      {result.memory && <MemoryMetricsDisplay metrics={result.memory} />}

      {/* Hotspots */}
      {result.hotspots && <HotspotDisplay hotspots={result.hotspots} />}

      {/* Issues */}
      <IssuesDisplay issues={result.issues} />
    </div>
  );
}
