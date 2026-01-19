import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip
} from 'recharts';
import type { QualityStats } from '@/types';
import { ProgressRing } from './ProgressRing';

interface CodeQualityCardProps {
  stats: QualityStats;
}

// Grade configuration with colors
const gradeConfig: Record<string, { color: string; darkColor: string; bgClass: string }> = {
  A: { color: '#22c55e', darkColor: '#4ade80', bgClass: 'bg-green-500' },
  B: { color: '#3b82f6', darkColor: '#60a5fa', bgClass: 'bg-blue-500' },
  C: { color: '#eab308', darkColor: '#facc15', bgClass: 'bg-yellow-500' },
  D: { color: '#f97316', darkColor: '#fb923c', bgClass: 'bg-orange-500' },
  F: { color: '#ef4444', darkColor: '#f87171', bgClass: 'bg-red-500' },
};

// Dimension configuration
const dimensionConfig = [
  { key: 'correctness', label: '정확성', color: '#22c55e', icon: '✓' },
  { key: 'efficiency', label: '효율성', color: '#3b82f6', icon: '⚡' },
  { key: 'readability', label: '가독성', color: '#a855f7', icon: '📖' },
  { key: 'best_practices', label: '모범사례', color: '#f97316', icon: '⭐' },
];

// Custom tooltip for grade chart (defined outside to avoid recreation on each render)
interface GradeTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: { grade: string; count: number } }>;
}

function GradeTooltip({ active, payload }: GradeTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;
  return (
    <div className="bg-white dark:bg-slate-700 rounded-lg shadow-xl border border-gray-200 dark:border-slate-600 px-3 py-2">
      <p className="text-sm font-bold" style={{ color: gradeConfig[data.grade].color }}>
        {data.grade}등급
      </p>
      <p className="text-sm text-gray-600 dark:text-gray-300">{data.count}회</p>
    </div>
  );
}

export function CodeQualityCard({ stats }: CodeQualityCardProps) {
  const [animatedScores, setAnimatedScores] = useState({
    correctness: 0,
    efficiency: 0,
    readability: 0,
    best_practices: 0,
  });

  const {
    total_analyses,
    avg_overall,
    avg_correctness,
    avg_efficiency,
    avg_readability,
    avg_best_practices,
    avg_cyclomatic,
    total_smells,
    grade_distribution,
  } = stats;

  // Animate scores on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScores({
        correctness: avg_correctness,
        efficiency: avg_efficiency,
        readability: avg_readability,
        best_practices: avg_best_practices,
      });
    }, 100);
    return () => clearTimeout(timer);
  }, [avg_correctness, avg_efficiency, avg_readability, avg_best_practices]);

  // Calculate dominant grade
  const dominantGrade = Object.entries(grade_distribution).reduce(
    (max, [grade, count]) => (count > max.count ? { grade, count } : max),
    { grade: 'C', count: 0 }
  ).grade;

  // Prepare grade distribution data for chart
  const gradeChartData = ['A', 'B', 'C', 'D', 'F'].map(grade => ({
    grade,
    count: grade_distribution[grade] || 0,
    color: gradeConfig[grade].color,
  }));

  if (total_analyses === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">코드 품질 분석</h2>
          <span className="text-xs text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-slate-700 px-2 py-1 rounded-full">CodeBERT</span>
        </div>
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 flex items-center justify-center">
            <span className="text-4xl">🔍</span>
          </div>
          <p className="font-medium">코드 품질 분석 결과가 없습니다</p>
          <p className="text-sm mt-1 text-gray-400 dark:text-gray-500">코드를 제출하면 AI가 품질을 분석해드려요!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">코드 품질 분석</h2>
        <span className="text-xs text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-slate-700 px-2 py-1 rounded-full">CodeBERT</span>
      </div>

      {/* Overall Score with ProgressRing */}
      <div className="flex flex-col md:flex-row items-center gap-6 mb-8">
        <div className="relative">
          <ProgressRing
            progress={avg_overall}
            size="xl"
            color="#6366f1"
            showPercentage={false}
            animate
          />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-bold text-indigo-600 dark:text-indigo-400">
              {avg_overall.toFixed(0)}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">종합점수</span>
          </div>
        </div>

        <div className="flex-1 grid grid-cols-2 gap-3">
          {/* Dominant Grade */}
          <div className="bg-gradient-to-br from-gray-50 to-slate-50 dark:from-slate-700 dark:to-slate-700/50 rounded-xl p-4 text-center">
            <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2">주요 등급</span>
            <span
              className="inline-flex items-center justify-center w-14 h-14 rounded-full text-white text-2xl font-bold shadow-lg"
              style={{ backgroundColor: gradeConfig[dominantGrade]?.color || gradeConfig.C.color }}
            >
              {dominantGrade}
            </span>
          </div>

          {/* Total Analyses */}
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-4 text-center">
            <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2">분석 횟수</span>
            <span className="text-3xl font-bold text-blue-600 dark:text-blue-400">{total_analyses}</span>
            <span className="text-sm text-gray-500 dark:text-gray-400 block">회</span>
          </div>
        </div>
      </div>

      {/* Dimension Scores with animated bars */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 flex items-center gap-2">
          <span className="w-1 h-4 bg-indigo-500 rounded-full"></span>
          품질 차원별 점수
        </h3>
        <div className="space-y-4">
          {dimensionConfig.map((dim) => {
            const score = animatedScores[dim.key as keyof typeof animatedScores];
            return (
              <div key={dim.key} className="group">
                <div className="flex justify-between items-center text-sm mb-1.5">
                  <span className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <span className="text-base">{dim.icon}</span>
                    {dim.label}
                  </span>
                  <span className="font-bold" style={{ color: dim.color }}>
                    {score.toFixed(0)}점
                  </span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-slate-700 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden"
                    style={{
                      width: `${Math.min(score, 100)}%`,
                      backgroundColor: dim.color,
                    }}
                  >
                    {/* Shimmer effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Complexity and Smells */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Complexity */}
        <div className="bg-gray-50 dark:bg-slate-700/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">평균 복잡도</span>
            <span className="text-lg">🔄</span>
          </div>
          <p className={`text-2xl font-bold ${
            avg_cyclomatic > 10
              ? 'text-red-600 dark:text-red-400'
              : avg_cyclomatic > 5
                ? 'text-yellow-600 dark:text-yellow-400'
                : 'text-green-600 dark:text-green-400'
          }`}>
            {avg_cyclomatic.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {avg_cyclomatic > 10 ? '높음 - 리팩토링 권장' : avg_cyclomatic > 5 ? '보통' : '낮음 - 좋음'}
          </p>
        </div>

        {/* Code Smells */}
        <div className="bg-gray-50 dark:bg-slate-700/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">코드 스멜</span>
            <span className="text-lg">🔍</span>
          </div>
          <p className={`text-2xl font-bold ${
            total_smells > 20
              ? 'text-red-600 dark:text-red-400'
              : total_smells > 10
                ? 'text-yellow-600 dark:text-yellow-400'
                : 'text-green-600 dark:text-green-400'
          }`}>
            {total_smells}개
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {total_smells > 20 ? '개선 필요' : total_smells > 10 ? '일부 개선 권장' : '양호'}
          </p>
        </div>
      </div>

      {/* Grade Distribution Chart */}
      {Object.keys(grade_distribution).length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-indigo-500 rounded-full"></span>
            등급 분포
          </h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={gradeChartData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                <XAxis
                  dataKey="grade"
                  tick={{ fontSize: 12, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis hide />
                <Tooltip content={<GradeTooltip />} cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }} />
                <Bar
                  dataKey="count"
                  radius={[6, 6, 0, 0]}
                  animationDuration={1000}
                >
                  {gradeChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
