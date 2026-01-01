import type { QualityTrendPoint } from '@/types';

interface QualityTrendChartProps {
  trends: QualityTrendPoint[];
  days: number;
}

export function QualityTrendChart({ trends, days }: QualityTrendChartProps) {
  if (!trends || trends.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800">품질 추이</h2>
          <span className="text-xs text-gray-400">{days}일</span>
        </div>
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-2 block">📈</span>
          <p>품질 추이 데이터가 없습니다.</p>
          <p className="text-sm mt-1">매일 코드를 제출하면 추이를 확인할 수 있어요!</p>
        </div>
      </div>
    );
  }

  // Find min/max values for scaling
  const scores = trends.flatMap(t => [
    t.avg_overall,
    t.avg_correctness,
    t.avg_efficiency,
    t.avg_readability,
    t.avg_best_practices,
  ]);
  const minScore = Math.min(...scores, 0);
  const maxScore = Math.max(...scores, 100);
  const range = maxScore - minScore || 1;

  // Calculate improvement
  const firstAvg = trends[0]?.avg_overall || 0;
  const lastAvg = trends[trends.length - 1]?.avg_overall || 0;
  const improvement = lastAvg - firstAvg;

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  // Color for each dimension
  const dimensionColors = {
    overall: '#6366f1', // indigo
    correctness: '#22c55e', // green
    efficiency: '#3b82f6', // blue
    readability: '#a855f7', // purple
    best_practices: '#f97316', // orange
  };

  // Generate SVG path for a dimension
  const generatePath = (getValue: (t: QualityTrendPoint) => number) => {
    const points = trends.map((t, i) => {
      const x = (i / (trends.length - 1 || 1)) * 100;
      const y = 100 - ((getValue(t) - minScore) / range) * 100;
      return `${x},${y}`;
    });
    return `M ${points.join(' L ')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">품질 추이</h2>
        <div className="flex items-center gap-3">
          <span className={`text-sm font-medium ${improvement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {improvement >= 0 ? '↑' : '↓'} {Math.abs(improvement).toFixed(1)}점
          </span>
          <span className="text-xs text-gray-400">{days}일</span>
        </div>
      </div>

      {/* Chart Legend */}
      <div className="flex flex-wrap gap-4 mb-4 text-xs">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: dimensionColors.overall }}></span>
          종합
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: dimensionColors.correctness }}></span>
          정확성
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: dimensionColors.efficiency }}></span>
          효율성
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: dimensionColors.readability }}></span>
          가독성
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: dimensionColors.best_practices }}></span>
          모범사례
        </span>
      </div>

      {/* Simple Line Chart using SVG */}
      <div className="relative h-48 mb-4">
        <svg
          viewBox="0 0 100 100"
          className="w-full h-full"
          preserveAspectRatio="none"
        >
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="#e5e7eb"
              strokeWidth="0.5"
              vectorEffect="non-scaling-stroke"
            />
          ))}

          {/* Dimension lines */}
          <path
            d={generatePath(t => t.avg_overall)}
            fill="none"
            stroke={dimensionColors.overall}
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
          <path
            d={generatePath(t => t.avg_correctness)}
            fill="none"
            stroke={dimensionColors.correctness}
            strokeWidth="1.5"
            strokeDasharray="4 2"
            vectorEffect="non-scaling-stroke"
            opacity="0.7"
          />
          <path
            d={generatePath(t => t.avg_efficiency)}
            fill="none"
            stroke={dimensionColors.efficiency}
            strokeWidth="1.5"
            strokeDasharray="4 2"
            vectorEffect="non-scaling-stroke"
            opacity="0.7"
          />
          <path
            d={generatePath(t => t.avg_readability)}
            fill="none"
            stroke={dimensionColors.readability}
            strokeWidth="1.5"
            strokeDasharray="4 2"
            vectorEffect="non-scaling-stroke"
            opacity="0.7"
          />
          <path
            d={generatePath(t => t.avg_best_practices)}
            fill="none"
            stroke={dimensionColors.best_practices}
            strokeWidth="1.5"
            strokeDasharray="4 2"
            vectorEffect="non-scaling-stroke"
            opacity="0.7"
          />
        </svg>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-400 -ml-6">
          <span>{maxScore.toFixed(0)}</span>
          <span>{((maxScore + minScore) / 2).toFixed(0)}</span>
          <span>{minScore.toFixed(0)}</span>
        </div>
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between text-xs text-gray-400 px-1">
        {trends.length > 0 && <span>{formatDate(trends[0].date)}</span>}
        {trends.length > 2 && (
          <span>{formatDate(trends[Math.floor(trends.length / 2)].date)}</span>
        )}
        {trends.length > 1 && (
          <span>{formatDate(trends[trends.length - 1].date)}</span>
        )}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center">
          <p className="text-xs text-gray-500">분석 횟수</p>
          <p className="text-lg font-bold text-gray-800">
            {trends.reduce((sum, t) => sum + t.submissions_analyzed, 0)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">개선 횟수</p>
          <p className="text-lg font-bold text-green-600">
            {trends.reduce((sum, t) => sum + t.improved_count, 0)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">평균 점수</p>
          <p className="text-lg font-bold text-indigo-600">
            {(trends.reduce((sum, t) => sum + t.avg_overall, 0) / trends.length).toFixed(1)}
          </p>
        </div>
      </div>
    </div>
  );
}
