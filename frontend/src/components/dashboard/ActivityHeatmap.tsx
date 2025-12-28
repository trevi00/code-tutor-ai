import { useMemo } from 'react';
import type { HeatmapData } from '@/types';

interface ActivityHeatmapProps {
  data: HeatmapData[];
  months?: number;
}

const levelColors = [
  'bg-gray-100',      // level 0 - no activity
  'bg-green-200',     // level 1 - low
  'bg-green-400',     // level 2 - medium
  'bg-green-600',     // level 3 - high
  'bg-green-800',     // level 4 - very high
];

export function ActivityHeatmap({ data, months = 6 }: ActivityHeatmapProps) {
  const { weeks, monthLabels } = useMemo(() => {
    // Create a map for quick lookup
    const dataMap = new Map(data.map(d => [d.date, d]));

    // Calculate date range
    const today = new Date();
    const startDate = new Date(today);
    startDate.setMonth(startDate.getMonth() - months);
    startDate.setDate(startDate.getDate() - startDate.getDay()); // Start from Sunday

    const weeks: (HeatmapData | null)[][] = [];
    const monthLabels: { month: string; weekIndex: number }[] = [];

    let currentDate = new Date(startDate);
    let currentWeek: (HeatmapData | null)[] = [];
    let lastMonth = -1;
    let weekIndex = 0;

    while (currentDate <= today) {
      const dateStr = currentDate.toISOString().split('T')[0];
      const dayData = dataMap.get(dateStr) || { date: dateStr, count: 0, level: 0 };

      // Track month changes for labels
      if (currentDate.getMonth() !== lastMonth) {
        monthLabels.push({
          month: currentDate.toLocaleDateString('ko-KR', { month: 'short' }),
          weekIndex,
        });
        lastMonth = currentDate.getMonth();
      }

      currentWeek.push(dayData);

      if (currentWeek.length === 7) {
        weeks.push(currentWeek);
        currentWeek = [];
        weekIndex++;
      }

      currentDate.setDate(currentDate.getDate() + 1);
    }

    // Add remaining days
    if (currentWeek.length > 0) {
      while (currentWeek.length < 7) {
        currentWeek.push(null);
      }
      weeks.push(currentWeek);
    }

    return { weeks, monthLabels };
  }, [data, months]);

  const totalContributions = useMemo(() => {
    return data.reduce((sum, d) => sum + d.count, 0);
  }, [data]);

  const dayLabels = ['일', '', '화', '', '목', '', '토'];

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">활동 기록</h2>
        <span className="text-sm text-gray-500">
          최근 {months}개월 동안 {totalContributions}회 제출
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Month labels */}
          <div className="flex mb-1">
            <div className="w-8"></div>
            <div className="flex">
              {monthLabels.map((label, i) => (
                <div
                  key={i}
                  className="text-xs text-gray-500"
                  style={{
                    position: 'relative',
                    left: `${(label.weekIndex * 14) - (i > 0 ? monthLabels[i - 1].weekIndex * 14 : 0)}px`,
                    minWidth: i < monthLabels.length - 1
                      ? `${(monthLabels[i + 1].weekIndex - label.weekIndex) * 14}px`
                      : 'auto',
                  }}
                >
                  {label.month}
                </div>
              ))}
            </div>
          </div>

          {/* Heatmap grid */}
          <div className="flex">
            {/* Day labels */}
            <div className="flex flex-col justify-around w-8 text-xs text-gray-500">
              {dayLabels.map((label, i) => (
                <span key={i} className="h-3 leading-3">{label}</span>
              ))}
            </div>

            {/* Weeks */}
            <div className="flex gap-0.5">
              {weeks.map((week, weekIndex) => (
                <div key={weekIndex} className="flex flex-col gap-0.5">
                  {week.map((day, dayIndex) => (
                    <div
                      key={dayIndex}
                      className={`w-3 h-3 rounded-sm ${
                        day ? levelColors[day.level] : 'bg-transparent'
                      }`}
                      title={day ? `${day.date}: ${day.count}회 제출` : ''}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-end mt-4 gap-1 text-xs text-gray-500">
            <span>적음</span>
            {levelColors.map((color, i) => (
              <div key={i} className={`w-3 h-3 rounded-sm ${color}`} />
            ))}
            <span>많음</span>
          </div>
        </div>
      </div>
    </div>
  );
}
