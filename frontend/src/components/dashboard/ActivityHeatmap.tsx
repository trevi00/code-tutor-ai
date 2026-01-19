import { useMemo, useState } from 'react';
import type { HeatmapData } from '@/types';
import clsx from 'clsx';

interface ActivityHeatmapProps {
  data: HeatmapData[];
  months?: number;
}

const levelColors = {
  light: [
    'bg-gray-100',      // level 0 - no activity
    'bg-green-200',     // level 1 - low
    'bg-green-400',     // level 2 - medium
    'bg-green-500',     // level 3 - high
    'bg-green-700',     // level 4 - very high
  ],
  dark: [
    'dark:bg-slate-700',
    'dark:bg-green-900',
    'dark:bg-green-700',
    'dark:bg-green-600',
    'dark:bg-green-500',
  ],
};

export function ActivityHeatmap({ data, months = 6 }: ActivityHeatmapProps) {
  const [hoveredDay, setHoveredDay] = useState<HeatmapData | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

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

    const currentDate = new Date(startDate);
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

  const { totalContributions, maxStreak, currentStreak } = useMemo(() => {
    const total = data.reduce((sum, d) => sum + d.count, 0);

    // Calculate streaks
    const sortedDates = [...data].filter(d => d.count > 0).map(d => d.date).sort();
    let max = 0;
    let current = 0;
    let streak = 0;

    const now = new Date();
    const today = now.toISOString().split('T')[0];
    const yesterday = new Date(now.getTime() - 86400000).toISOString().split('T')[0];

    for (let i = 0; i < sortedDates.length; i++) {
      if (i === 0) {
        streak = 1;
      } else {
        const prevDate = new Date(sortedDates[i - 1]);
        const currDate = new Date(sortedDates[i]);
        const diff = (currDate.getTime() - prevDate.getTime()) / 86400000;

        if (diff === 1) {
          streak++;
        } else {
          streak = 1;
        }
      }
      max = Math.max(max, streak);

      // Check if this is part of current streak
      if (sortedDates[i] === today || sortedDates[i] === yesterday) {
        current = streak;
      }
    }

    return { totalContributions: total, maxStreak: max, currentStreak: current };
  }, [data]);

  const handleMouseEnter = (day: HeatmapData | null, e: React.MouseEvent) => {
    if (day) {
      setHoveredDay(day);
      const rect = e.currentTarget.getBoundingClientRect();
      setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top - 8 });
    }
  };

  const dayLabels = ['일', '', '화', '', '목', '', '토'];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">활동 기록</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            최근 {months}개월 동안 <span className="font-medium text-green-600 dark:text-green-400">{totalContributions}회</span> 제출
          </p>
        </div>
        <div className="flex gap-4">
          <div className="text-center px-3 py-2 bg-gray-50 dark:bg-slate-700 rounded-lg">
            <div className="text-lg font-bold text-orange-500">{currentStreak}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">현재 스트릭</div>
          </div>
          <div className="text-center px-3 py-2 bg-gray-50 dark:bg-slate-700 rounded-lg">
            <div className="text-lg font-bold text-indigo-500">{maxStreak}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">최장 스트릭</div>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="inline-block min-w-full">
          {/* Month labels */}
          <div className="flex mb-2 ml-8">
            {monthLabels.map((label, i) => (
              <div
                key={i}
                className="text-xs text-gray-500 dark:text-gray-400"
                style={{
                  position: 'relative',
                  left: `${label.weekIndex * 14}px`,
                  width: i < monthLabels.length - 1
                    ? `${(monthLabels[i + 1].weekIndex - label.weekIndex) * 14}px`
                    : 'auto',
                }}
              >
                {label.month}
              </div>
            ))}
          </div>

          {/* Heatmap grid */}
          <div className="flex">
            {/* Day labels */}
            <div className="flex flex-col justify-around w-8 text-xs text-gray-400 dark:text-gray-500 mr-1">
              {dayLabels.map((label, i) => (
                <span key={i} className="h-3 leading-3">{label}</span>
              ))}
            </div>

            {/* Weeks */}
            <div className="flex gap-[3px]">
              {weeks.map((week, weekIndex) => (
                <div key={weekIndex} className="flex flex-col gap-[3px]">
                  {week.map((day, dayIndex) => (
                    <div
                      key={dayIndex}
                      className={clsx(
                        'w-[13px] h-[13px] rounded-sm cursor-pointer transition-all duration-150',
                        day ? [levelColors.light[day.level], levelColors.dark[day.level]] : 'bg-transparent',
                        hoveredDay?.date === day?.date && 'ring-2 ring-gray-400 dark:ring-gray-300 ring-offset-1 dark:ring-offset-slate-800'
                      )}
                      onMouseEnter={(e) => handleMouseEnter(day, e)}
                      onMouseLeave={() => setHoveredDay(null)}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-end mt-4 gap-1.5 text-xs text-gray-500 dark:text-gray-400">
            <span>적음</span>
            {[0, 1, 2, 3, 4].map((level) => (
              <div
                key={level}
                className={clsx(
                  'w-[13px] h-[13px] rounded-sm',
                  levelColors.light[level],
                  levelColors.dark[level]
                )}
              />
            ))}
            <span>많음</span>
          </div>
        </div>
      </div>

      {/* Floating Tooltip */}
      {hoveredDay && (
        <div
          className="fixed z-50 pointer-events-none transform -translate-x-1/2 -translate-y-full"
          style={{ left: tooltipPos.x, top: tooltipPos.y }}
        >
          <div className="bg-gray-900 dark:bg-slate-600 text-white text-xs rounded-lg px-3 py-2 shadow-xl">
            <div className="font-medium">
              {new Date(hoveredDay.date).toLocaleDateString('ko-KR', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                weekday: 'short',
              })}
            </div>
            <div className="text-gray-300 dark:text-gray-200">
              {hoveredDay.count > 0 ? `${hoveredDay.count}회 제출` : '활동 없음'}
            </div>
            {/* Arrow */}
            <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900 dark:border-t-slate-600" />
          </div>
        </div>
      )}
    </div>
  );
}
