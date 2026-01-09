import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip, ReferenceLine
} from 'recharts';
import type { SkillPrediction } from '@/types';

interface SkillPredictionsProps {
  predictions: SkillPrediction[];
}

const categoryLabels: Record<string, string> = {
  array: '배열',
  string: '문자열',
  linked_list: '연결 리스트',
  stack: '스택',
  queue: '큐',
  hash_table: '해시 테이블',
  tree: '트리',
  graph: '그래프',
  dp: 'DP',
  greedy: '그리디',
  binary_search: '이진 탐색',
  sorting: '정렬',
  design: '설계',
  dfs: 'DFS',
  bfs: 'BFS',
  math: '수학',
  bit_manipulation: '비트 연산',
  recursion: '재귀',
};

// Category icons
const categoryIcons: Record<string, string> = {
  array: '📊',
  string: '📝',
  linked_list: '🔗',
  stack: '📚',
  queue: '📋',
  hash_table: '🗂️',
  tree: '🌳',
  graph: '🕸️',
  dp: '🧮',
  greedy: '💰',
  binary_search: '🔍',
  sorting: '📈',
  design: '🏗️',
  dfs: '🔎',
  bfs: '🔭',
  math: '➗',
  bit_manipulation: '🔢',
  recursion: '🔄',
};

export function SkillPredictions({ predictions }: SkillPredictionsProps) {
  const [animatedData, setAnimatedData] = useState<SkillPrediction[]>([]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedData(predictions);
    }, 100);
    return () => clearTimeout(timer);
  }, [predictions]);

  if (predictions.length === 0) {
    return null;
  }

  // Sort by recommended_focus first, then by current_level
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (a.recommended_focus !== b.recommended_focus) {
      return a.recommended_focus ? -1 : 1;
    }
    return a.current_level - b.current_level;
  });

  // Prepare chart data
  const chartData = sortedPredictions.slice(0, 8).map(p => ({
    category: categoryLabels[p.category] || p.category,
    current: p.current_level,
    predicted: p.predicted_level,
    recommended: p.recommended_focus,
    confidence: p.confidence,
    icon: categoryIcons[p.category] || '📌',
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: {
    active?: boolean;
    payload?: Array<{
      payload: {
        category: string;
        current: number;
        predicted: number;
        confidence: number;
        icon: string;
      }
    }>
  }) => {
    if (!active || !payload || payload.length === 0) return null;
    const data = payload[0].payload;
    const improvement = data.predicted - data.current;
    return (
      <div className="bg-white dark:bg-slate-700 rounded-lg shadow-xl border border-gray-200 dark:border-slate-600 px-4 py-3 min-w-[160px]">
        <p className="text-sm font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2 mb-2">
          <span>{data.icon}</span>
          {data.category}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">현재</span>
            <span className="font-medium text-blue-600 dark:text-blue-400">{data.current.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">예측</span>
            <span className="font-medium text-green-600 dark:text-green-400">{data.predicted.toFixed(0)}%</span>
          </div>
          {improvement > 0 && (
            <div className="flex justify-between pt-1 border-t border-gray-100 dark:border-slate-600">
              <span className="text-gray-500 dark:text-gray-400">성장</span>
              <span className="font-medium text-emerald-500">+{improvement.toFixed(0)}%</span>
            </div>
          )}
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 dark:border-slate-600">
          <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
            <span>신뢰도:</span>
            <div className="flex gap-0.5">
              {[0.25, 0.5, 0.75, 1].map((threshold, i) => (
                <div
                  key={i}
                  className={`w-1.5 h-1.5 rounded-full ${
                    data.confidence >= threshold ? 'bg-indigo-500' : 'bg-gray-300 dark:bg-slate-500'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">스킬 분석</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 기반 실력 예측</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-blue-500"></span>
            <span className="text-gray-600 dark:text-gray-400">현재</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-green-500"></span>
            <span className="text-gray-600 dark:text-gray-400">예측</span>
          </div>
        </div>
      </div>

      {/* Horizontal Bar Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="category"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              axisLine={false}
              tickLine={false}
              width={60}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }} />
            <ReferenceLine x={50} stroke="#e5e7eb" strokeDasharray="3 3" />
            <Bar
              dataKey="current"
              name="현재 레벨"
              fill="#3b82f6"
              radius={[0, 4, 4, 0]}
              barSize={12}
              animationDuration={1000}
            />
            <Bar
              dataKey="predicted"
              name="예측 레벨"
              fill="#22c55e"
              radius={[0, 4, 4, 0]}
              barSize={12}
              animationDuration={1000}
              animationBegin={500}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.recommended ? '#f59e0b' : '#22c55e'}
                  opacity={0.7}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recommended Focus Items */}
      {predictions.some(p => p.recommended_focus) && (
        <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-xl border border-amber-200 dark:border-amber-800/50">
          <div className="flex items-start gap-3">
            <span className="text-2xl">💡</span>
            <div>
              <p className="font-medium text-amber-800 dark:text-amber-200 mb-1">집중 추천 영역</p>
              <div className="flex flex-wrap gap-2 mt-2">
                {predictions.filter(p => p.recommended_focus).map(p => (
                  <span
                    key={p.category}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-200 rounded-full text-sm font-medium"
                  >
                    <span>{categoryIcons[p.category] || '📌'}</span>
                    {categoryLabels[p.category] || p.category}
                  </span>
                ))}
              </div>
              <p className="text-sm text-amber-700 dark:text-amber-300/80 mt-2">
                이 영역에 집중하면 빠른 실력 향상을 기대할 수 있어요!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Quick Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4 pt-4 border-t border-gray-100 dark:border-slate-700">
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">평균 현재 레벨</p>
          <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
            {(predictions.reduce((sum, p) => sum + p.current_level, 0) / predictions.length).toFixed(0)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">평균 예측 레벨</p>
          <p className="text-xl font-bold text-green-600 dark:text-green-400">
            {(predictions.reduce((sum, p) => sum + p.predicted_level, 0) / predictions.length).toFixed(0)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">평균 성장률</p>
          <p className="text-xl font-bold text-emerald-500">
            +{(predictions.reduce((sum, p) => sum + (p.predicted_level - p.current_level), 0) / predictions.length).toFixed(0)}%
          </p>
        </div>
      </div>
    </div>
  );
}
