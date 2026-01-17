/**
 * StreakCalendar - Visual streak tracking calendar component
 * Shows daily activity with streak highlighting
 */

import { useMemo } from 'react';
import { Flame, Calendar, TrendingUp } from 'lucide-react';
import clsx from 'clsx';

interface DayActivity {
  date: string;
  count: number;
  has_activity: boolean;
}

interface StreakCalendarProps {
  data: DayActivity[];
  currentStreak: number;
  longestStreak: number;
  weeks?: number;
}

export function StreakCalendar({
  data,
  currentStreak,
  longestStreak,
  weeks = 12,
}: StreakCalendarProps) {
  // Build calendar grid
  const calendarData = useMemo(() => {
    const today = new Date();
    const dayMap = new Map(data.map((d) => [d.date, d]));
    const days: (DayActivity | null)[] = [];

    // Calculate start date (weeks ago, aligned to Sunday)
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - weeks * 7 + 1);
    // Align to Sunday
    const dayOfWeek = startDate.getDay();
    startDate.setDate(startDate.getDate() - dayOfWeek);

    // Fill calendar
    const currentDate = new Date(startDate);
    while (currentDate <= today) {
      const dateStr = currentDate.toISOString().split('T')[0];
      const activity = dayMap.get(dateStr);
      days.push(
        activity || {
          date: dateStr,
          count: 0,
          has_activity: false,
        }
      );
      currentDate.setDate(currentDate.getDate() + 1);
    }

    // Pad remaining cells to complete the week
    const remaining = 7 - (days.length % 7);
    if (remaining < 7) {
      for (let i = 0; i < remaining; i++) {
        days.push(null);
      }
    }

    // Group by weeks
    const weeksData: (DayActivity | null)[][] = [];
    for (let i = 0; i < days.length; i += 7) {
      weeksData.push(days.slice(i, i + 7));
    }

    return weeksData;
  }, [data, weeks]);

  // Get month labels
  const monthLabels = useMemo(() => {
    const labels: { month: string; index: number }[] = [];
    let lastMonth = -1;

    calendarData.forEach((week, weekIndex) => {
      const firstDay = week.find((d) => d !== null);
      if (firstDay) {
        const date = new Date(firstDay.date);
        const month = date.getMonth();
        if (month !== lastMonth) {
          labels.push({
            month: date.toLocaleDateString('ko-KR', { month: 'short' }),
            index: weekIndex,
          });
          lastMonth = month;
        }
      }
    });

    return labels;
  }, [calendarData]);

  // Get intensity class based on count
  const getIntensityClass = (count: number, hasActivity: boolean): string => {
    if (!hasActivity || count === 0) {
      return 'bg-slate-100 dark:bg-slate-700/50';
    }
    if (count === 1) {
      return 'bg-emerald-200 dark:bg-emerald-900/50';
    }
    if (count <= 3) {
      return 'bg-emerald-400 dark:bg-emerald-700';
    }
    if (count <= 5) {
      return 'bg-emerald-500 dark:bg-emerald-600';
    }
    return 'bg-emerald-600 dark:bg-emerald-500';
  };

  // Format date for tooltip
  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'short',
    });
  };

  const dayLabels = ['일', '월', '화', '수', '목', '금', '토'];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
            <Flame className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-white">스트릭 캘린더</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">일일 학습 기록</p>
          </div>
        </div>

        {/* Streak Stats */}
        <div className="flex items-center gap-4">
          <div className="text-center">
            <div className="flex items-center gap-1 text-orange-500">
              <Flame className="w-4 h-4" />
              <span className="text-xl font-bold">{currentStreak}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">현재 스트릭</p>
          </div>
          <div className="text-center">
            <div className="flex items-center gap-1 text-emerald-500">
              <TrendingUp className="w-4 h-4" />
              <span className="text-xl font-bold">{longestStreak}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">최장 스트릭</p>
          </div>
        </div>
      </div>

      {/* Calendar Grid */}
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Month Labels */}
          <div className="flex mb-2 ml-8">
            {monthLabels.map((label, idx) => (
              <div
                key={idx}
                className="text-xs text-slate-500 dark:text-slate-400"
                style={{
                  marginLeft: idx === 0 ? 0 : `${(label.index - (monthLabels[idx - 1]?.index || 0)) * 14 - 28}px`,
                }}
              >
                {label.month}
              </div>
            ))}
          </div>

          <div className="flex gap-0.5">
            {/* Day Labels */}
            <div className="flex flex-col gap-0.5 mr-1">
              {dayLabels.map((day, idx) => (
                <div
                  key={idx}
                  className="w-6 h-3 flex items-center justify-end text-[10px] text-slate-400 dark:text-slate-500 pr-1"
                >
                  {idx % 2 === 1 ? day : ''}
                </div>
              ))}
            </div>

            {/* Weeks */}
            {calendarData.map((week, weekIdx) => (
              <div key={weekIdx} className="flex flex-col gap-0.5">
                {week.map((day, dayIdx) => (
                  <div
                    key={dayIdx}
                    className={clsx(
                      'w-3 h-3 rounded-sm transition-all duration-200',
                      day
                        ? getIntensityClass(day.count, day.has_activity)
                        : 'bg-transparent',
                      day && 'hover:ring-2 hover:ring-slate-400 dark:hover:ring-slate-500 hover:ring-offset-1 cursor-pointer'
                    )}
                    title={
                      day
                        ? `${formatDate(day.date)}\n${day.count > 0 ? `${day.count}개 활동` : '활동 없음'}`
                        : undefined
                    }
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-100 dark:border-slate-700">
        <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <Calendar className="w-3.5 h-3.5" />
          <span>최근 {weeks}주</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <span>적음</span>
          <div className="flex gap-0.5">
            <div className="w-3 h-3 rounded-sm bg-slate-100 dark:bg-slate-700/50" />
            <div className="w-3 h-3 rounded-sm bg-emerald-200 dark:bg-emerald-900/50" />
            <div className="w-3 h-3 rounded-sm bg-emerald-400 dark:bg-emerald-700" />
            <div className="w-3 h-3 rounded-sm bg-emerald-500 dark:bg-emerald-600" />
            <div className="w-3 h-3 rounded-sm bg-emerald-600 dark:bg-emerald-500" />
          </div>
          <span>많음</span>
        </div>
      </div>
    </div>
  );
}

export default StreakCalendar;
