/**
 * Main Performance Panel Component - Enhanced with modern design
 */

import { useState } from 'react';
import {
  Activity,
  Play,
  Loader2,
  Gauge,
  RefreshCw,
  Zap,
  Sparkles,
} from 'lucide-react';
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

export function PerformancePanel({ code, input }: PerformancePanelProps) {
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
    <div className="h-full flex flex-col bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-rose-600 dark:text-rose-400" />
          <span className="font-medium text-slate-800 dark:text-slate-200">성능 분석</span>
        </div>
        <div className="flex items-center gap-2">
          {hasResult && (
            <button
              onClick={handleReset}
              className="p-2 text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
              title="초기화"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={handleQuickAnalyze}
            disabled={isAnalyzing || !code.trim()}
            className="px-3 py-1.5 text-sm bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 transition-colors"
          >
            <Gauge className="w-4 h-4" />
            빠른 분석
          </button>
          <button
            onClick={handleFullAnalyze}
            disabled={isAnalyzing || !code.trim()}
            className="px-3 py-1.5 text-sm bg-gradient-to-r from-rose-500 to-pink-500 hover:from-rose-600 hover:to-pink-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 shadow-lg shadow-rose-500/25 transition-all"
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
        <div className="px-5 py-3 bg-red-50 dark:bg-red-900/20 border-b border-red-100 dark:border-red-800 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto p-5">
        {!hasResult ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 dark:text-slate-400">
            <div className="w-20 h-20 mb-4 rounded-2xl bg-gradient-to-br from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 flex items-center justify-center">
              <Activity className="w-10 h-10 text-rose-400 dark:text-rose-500" />
            </div>
            <p className="text-lg font-medium text-slate-700 dark:text-slate-300 mb-1">
              성능 분석 준비 완료
            </p>
            <p className="text-sm text-center max-w-xs">
              코드를 작성하고 "빠른 분석" 또는 "전체 분석"을 클릭하세요
            </p>
            <div className="mt-6 flex flex-col gap-2 text-xs text-slate-400 dark:text-slate-500">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-slate-400" />
                <span>빠른 분석: 시간/공간 복잡도만 분석</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-rose-400" />
                <span>전체 분석: 런타임 성능 + 메모리 프로파일링</span>
              </div>
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
      <div className="text-red-600 dark:text-red-400 p-4 bg-red-50 dark:bg-red-900/20 rounded-xl">
        분석 오류: {result.error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-xl p-5 border border-rose-200 dark:border-rose-800">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-5 h-5 text-rose-500 dark:text-rose-400" />
          <h3 className="font-medium text-slate-700 dark:text-slate-300">복잡도 요약</h3>
        </div>
        <CompactComplexity
          timeComplexity={result.time_complexity}
          spaceComplexity={result.space_complexity}
        />
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3">
            <span className="text-slate-500 dark:text-slate-400">시간:</span>
            <p className="text-slate-700 dark:text-slate-300 mt-1">{result.time_explanation}</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3">
            <span className="text-slate-500 dark:text-slate-400">공간:</span>
            <p className="text-slate-700 dark:text-slate-300 mt-1">{result.space_explanation}</p>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-4 border border-amber-200 dark:border-amber-800">
          <span className="text-sm text-amber-600 dark:text-amber-400">중첩 깊이</span>
          <p className="text-3xl font-bold text-amber-700 dark:text-amber-300 mt-1">
            {result.max_nesting_depth}
          </p>
        </div>
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
          <span className="text-sm text-blue-600 dark:text-blue-400">이슈</span>
          <p className="text-3xl font-bold text-blue-700 dark:text-blue-300 mt-1">
            {result.issues_count}개
          </p>
        </div>
      </div>

      <p className="text-sm text-slate-500 dark:text-slate-400 text-center py-2">
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
      <div className="text-red-600 dark:text-red-400 p-4 bg-red-50 dark:bg-red-900/20 rounded-xl">
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

export default PerformancePanel;
