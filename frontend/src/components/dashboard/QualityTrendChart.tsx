import { useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from 'recharts';
import type { QualityTrendPoint } from '@/types';
import clsx from 'clsx';

interface QualityTrendChartProps {
  trends: QualityTrendPoint[];
  days: number;
}

const dimensionConfig = {
  overall: { name: '종합', color: '#6366f1', key: 'avg_overall' },
  correctness: { name: '정확성', color: '#22c55e', key: 'avg_correctness' },
  efficiency: { name: '효율성', color: '#3b82f6', key: 'avg_efficiency' },
  readability: { name: '가독성', color: '#a855f7', key: 'avg_readability' },
  best_practices: { name: '모범사례', color: '#f97316', key: 'avg_best_practices' },
};

type DimensionKey = keyof typeof dimensionConfig;

interface TooltipPayloadItem {
  value: number;
  dataKey: string;
  color: string;
  payload: QualityTrendPoint & { fullDate: string };
}

export function QualityTrendChart({ trends, days }: QualityTrendChartProps) {
  const [selectedDimensions, setSelectedDimensions] = useState<DimensionKey[]>(['overall']);

  if (!trends || trends.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">품질 추이</h2>
          <span className="text-xs text-gray-400 dark:text-gray-500 px-2 py-1 bg-gray-100 dark:bg-slate-700 rounded-full">{days}일</span>
        </div>
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
            <span className="text-3xl">📈</span>
          </div>
          <p className="font-medium">품질 추이 데이터가 없습니다.</p>
          <p className="text-sm mt-1 text-gray-400 dark:text-gray-500">매일 코드를 제출하면 추이를 확인할 수 있어요!</p>
        </div>
      </div>
    );
  }

  // Calculate improvement
  const firstAvg = trends[0]?.avg_overall || 0;
  const lastAvg = trends[trends.length - 1]?.avg_overall || 0;
  const improvement = lastAvg - firstAvg;

  // Format data for Recharts
  const chartData = trends.map((t) => ({
    ...t,
    date: new Date(t.date).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' }),
    fullDate: t.date,
  }));

  // Toggle dimension visibility
  const toggleDimension = (dim: DimensionKey) => {
    setSelectedDimensions(prev => {
      if (prev.includes(dim)) {
        // Don't allow removing the last one
        if (prev.length === 1) return prev;
        return prev.filter(d => d !== dim);
      }
      return [...prev, dim];
    });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[] }) => {
    if (!active || !payload || payload.length === 0) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-white dark:bg-slate-700 rounded-lg shadow-xl border border-gray-200 dark:border-slate-600 p-4 min-w-[180px]">
        <p className="text-sm font-medium text-gray-800 dark:text-gray-100 mb-2 pb-2 border-b border-gray-100 dark:border-slate-600">
          {new Date(data.fullDate).toLocaleDateString('ko-KR', {
            year: 'numeric', month: 'long', day: 'numeric'
          })}
        </p>
        <div className="space-y-1.5">
          {payload.map((entry, index) => {
            const config = Object.values(dimensionConfig).find(c => c.key === entry.dataKey);
            return (
              <div key={index} className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: entry.color }}
                  />
                  {config?.name}
                </span>
                <span className="font-medium text-gray-800 dark:text-gray-100">
                  {entry.value.toFixed(1)}점
                </span>
              </div>
            );
          })}
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 dark:border-slate-600 text-xs text-gray-500 dark:text-gray-400">
          제출 {data.submissions_analyzed}건
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">품질 추이</h2>
        <div className="flex items-center gap-3">
          <span className={clsx(
            'text-sm font-medium flex items-center gap-1 px-2 py-1 rounded-full',
            improvement >= 0
              ? 'text-green-700 bg-green-100 dark:text-green-400 dark:bg-green-900/30'
              : 'text-red-700 bg-red-100 dark:text-red-400 dark:bg-red-900/30'
          )}>
            {improvement >= 0 ? '↑' : '↓'} {Math.abs(improvement).toFixed(1)}점
          </span>
          <span className="text-xs text-gray-400 dark:text-gray-500 px-2 py-1 bg-gray-100 dark:bg-slate-700 rounded-full">{days}일</span>
        </div>
      </div>

      {/* Dimension Toggle Buttons */}
      <div className="flex flex-wrap gap-2 mb-6">
        {Object.entries(dimensionConfig).map(([key, config]) => {
          const isSelected = selectedDimensions.includes(key as DimensionKey);
          return (
            <button
              key={key}
              onClick={() => toggleDimension(key as DimensionKey)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 border',
                isSelected ? 'ring-2 ring-offset-1 dark:ring-offset-slate-800' : 'opacity-50 hover:opacity-75'
              )}
              style={{
                backgroundColor: isSelected ? `${config.color}20` : 'transparent',
                color: config.color,
                borderColor: config.color,
                // @ts-expect-error CSS custom property for ring color
                '--tw-ring-color': isSelected ? config.color : 'transparent',
              }}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: config.color }}
              />
              {config.name}
            </button>
          );
        })}
      </div>

      {/* Recharts Area Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              {Object.entries(dimensionConfig).map(([key, config]) => (
                <linearGradient key={key} id={`gradient-${key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={config.color} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={config.color} stopOpacity={0}/>
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-slate-700" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            {selectedDimensions.map((dim) => {
              const config = dimensionConfig[dim];
              return (
                <Area
                  key={dim}
                  type="monotone"
                  dataKey={config.key}
                  name={config.name}
                  stroke={config.color}
                  strokeWidth={dim === 'overall' ? 2.5 : 1.5}
                  fill={`url(#gradient-${dim})`}
                  animationDuration={500}
                />
              );
            })}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4 pt-4 border-t border-gray-100 dark:border-slate-700">
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">분석 횟수</p>
          <p className="text-xl font-bold text-gray-800 dark:text-gray-100">
            {trends.reduce((sum, t) => sum + t.submissions_analyzed, 0)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">개선 횟수</p>
          <p className="text-xl font-bold text-green-600 dark:text-green-400">
            {trends.reduce((sum, t) => sum + t.improved_count, 0)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">평균 점수</p>
          <p className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
            {(trends.reduce((sum, t) => sum + t.avg_overall, 0) / trends.length).toFixed(1)}
          </p>
        </div>
      </div>
    </div>
  );
}
