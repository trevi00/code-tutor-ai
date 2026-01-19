import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, Tooltip
} from 'recharts';
import clsx from 'clsx';

interface SkillData {
  skill: string;
  value: number;
  fullMark?: number;
}

// Custom tooltip component (defined outside to avoid recreation on each render)
interface RadarTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: SkillData }>;
  color?: string;
}

function RadarTooltip({ active, payload, color = '#6366f1' }: RadarTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const item = payload[0].payload;
  return (
    <div className="bg-white dark:bg-slate-700 rounded-lg shadow-xl border border-gray-200 dark:border-slate-600 px-3 py-2">
      <p className="text-sm font-medium text-gray-800 dark:text-gray-100">{item.skill}</p>
      <p className="text-sm text-gray-600 dark:text-gray-300">
        <span className="font-bold" style={{ color }}>{item.value}</span>
        <span className="text-gray-400 dark:text-gray-500"> / {item.fullMark || 100}</span>
      </p>
    </div>
  );
}

interface SkillRadarProps {
  data: SkillData[];
  title?: string;
  description?: string;
  color?: string;
  fillColor?: string;
  className?: string;
}

export function SkillRadar({
  data,
  title = '스킬 분포',
  description,
  color = '#6366f1',
  fillColor,
  className,
}: SkillRadarProps) {
  if (!data || data.length === 0) {
    return (
      <div className={clsx('bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6', className)}>
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-4">{title}</h3>
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
            <span className="text-3xl">🎯</span>
          </div>
          <p className="font-medium">스킬 데이터가 없습니다.</p>
        </div>
      </div>
    );
  }

  // Normalize data with fullMark
  const chartData = data.map(d => ({
    ...d,
    fullMark: d.fullMark || 100,
  }));

  // Calculate average
  const average = Math.round(chartData.reduce((sum, d) => sum + d.value, 0) / chartData.length);

  return (
    <div className={clsx('bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6', className)}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">{title}</h3>
          {description && (
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{description}</p>
          )}
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold" style={{ color }}>{average}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">평균</div>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
            <PolarGrid
              stroke="#e5e7eb"
              className="dark:stroke-slate-700"
            />
            <PolarAngleAxis
              dataKey="skill"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              className="dark:fill-gray-400"
            />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickCount={5}
            />
            <Radar
              name="스킬"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              fill={fillColor || color}
              fillOpacity={0.3}
              animationDuration={1000}
            />
            <Tooltip content={<RadarTooltip color={color} />} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Skills list */}
      <div className="mt-4 pt-4 border-t border-gray-100 dark:border-slate-700">
        <div className="grid grid-cols-2 gap-2">
          {chartData.map((item, index) => (
            <div key={index} className="flex items-center justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">{item.skill}</span>
              <div className="flex items-center gap-2">
                <div className="w-16 h-1.5 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${(item.value / (item.fullMark || 100)) * 100}%`,
                      backgroundColor: color,
                    }}
                  />
                </div>
                <span className="text-gray-800 dark:text-gray-200 font-medium w-8 text-right">
                  {item.value}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Preset for code quality metrics
export function CodeQualityRadar({ metrics }: {
  metrics: {
    correctness: number;
    efficiency: number;
    readability: number;
    best_practices: number;
    maintainability?: number;
  };
}) {
  const data: SkillData[] = [
    { skill: '정확성', value: metrics.correctness },
    { skill: '효율성', value: metrics.efficiency },
    { skill: '가독성', value: metrics.readability },
    { skill: '모범사례', value: metrics.best_practices },
  ];

  if (metrics.maintainability !== undefined) {
    data.push({ skill: '유지보수', value: metrics.maintainability });
  }

  return (
    <SkillRadar
      data={data}
      title="코드 품질 분석"
      description="각 영역별 점수"
      color="#6366f1"
    />
  );
}
