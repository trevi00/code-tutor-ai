import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip, CartesianGrid
} from 'recharts';
import type { CategoryProgress } from '@/types';

interface CategoryProgressChartProps {
  data: CategoryProgress[];
  maxItems?: number;
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

// Get color based on progress percentage
function getProgressColor(percentage: number): string {
  if (percentage >= 80) return '#22c55e'; // green
  if (percentage >= 60) return '#3b82f6'; // blue
  if (percentage >= 40) return '#eab308'; // yellow
  if (percentage >= 20) return '#f97316'; // orange
  return '#ef4444'; // red
}

export function CategoryProgressChart({ data, maxItems = 8 }: CategoryProgressChartProps) {
  // Prepare chart data
  const chartData = useMemo(() => {
    return data.slice(0, maxItems).map(cp => {
      const percentage = (cp.solved_problems / cp.total_problems) * 100;
      return {
        category: categoryLabels[cp.category] || cp.category,
        categoryKey: cp.category,
        solved: cp.solved_problems,
        total: cp.total_problems,
        percentage: Math.round(percentage),
        remaining: cp.total_problems - cp.solved_problems,
        icon: categoryIcons[cp.category] || '📌',
        color: getProgressColor(percentage),
      };
    });
  }, [data, maxItems]);

  // Calculate total stats
  const totalStats = useMemo(() => {
    const totalSolved = data.reduce((sum, cp) => sum + cp.solved_problems, 0);
    const totalProblems = data.reduce((sum, cp) => sum + cp.total_problems, 0);
    return {
      solved: totalSolved,
      total: totalProblems,
      percentage: totalProblems > 0 ? Math.round((totalSolved / totalProblems) * 100) : 0,
    };
  }, [data]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: {
    active?: boolean;
    payload?: Array<{
      payload: {
        category: string;
        solved: number;
        total: number;
        percentage: number;
        icon: string;
        color: string;
      }
    }>
  }) => {
    if (!active || !payload || payload.length === 0) return null;
    const item = payload[0].payload;
    return (
      <div className="bg-white dark:bg-slate-700 rounded-lg shadow-xl border border-gray-200 dark:border-slate-600 px-4 py-3 min-w-[140px]">
        <p className="text-sm font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2 mb-2">
          <span>{item.icon}</span>
          {item.category}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">해결</span>
            <span className="font-medium" style={{ color: item.color }}>
              {item.solved} / {item.total}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">진행률</span>
            <span className="font-bold" style={{ color: item.color }}>
              {item.percentage}%
            </span>
          </div>
        </div>
      </div>
    );
  };

  if (data.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">카테고리별 진행률</h2>
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
            <span className="text-3xl">📊</span>
          </div>
          <p className="font-medium">아직 푼 문제가 없습니다</p>
          <p className="text-sm mt-1 text-gray-400 dark:text-gray-500">문제를 풀어보세요!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">카테고리별 진행률</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            총 {totalStats.solved}/{totalStats.total} 문제 해결 ({totalStats.percentage}%)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-24 h-2 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-500"
              style={{ width: `${totalStats.percentage}%` }}
            />
          </div>
          <span className="text-sm font-bold text-gray-700 dark:text-gray-300">{totalStats.percentage}%</span>
        </div>
      </div>

      {/* Horizontal Bar Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 70, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" className="dark:stroke-slate-700" />
            <XAxis
              type="number"
              domain={[0, 'dataMax']}
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
              width={70}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }} />
            <Bar
              dataKey="solved"
              name="해결"
              radius={[0, 6, 6, 0]}
              barSize={20}
              animationDuration={1000}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Progress Legend */}
      <div className="mt-4 pt-4 border-t border-gray-100 dark:border-slate-700">
        <div className="flex items-center justify-center gap-6 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-red-500"></span>
            <span className="text-gray-600 dark:text-gray-400">0-20%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-orange-500"></span>
            <span className="text-gray-600 dark:text-gray-400">20-40%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-yellow-500"></span>
            <span className="text-gray-600 dark:text-gray-400">40-60%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-blue-500"></span>
            <span className="text-gray-600 dark:text-gray-400">60-80%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-green-500"></span>
            <span className="text-gray-600 dark:text-gray-400">80-100%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
