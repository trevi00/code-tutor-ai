/**
 * XP Progress Bar Component
 */

import type { UserStats } from '../../api/gamification';

interface XPBarProps {
  stats: UserStats;
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
}

export default function XPBar({ stats, size = 'md', showDetails = true }: XPBarProps) {
  const heightClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4',
  };

  const textClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  return (
    <div className="w-full">
      {showDetails && (
        <div className="flex justify-between items-center mb-1">
          <div className="flex items-center gap-2">
            <span className={`font-bold text-indigo-600 dark:text-indigo-400 ${textClasses[size]}`}>
              Lv.{stats.level}
            </span>
            <span className={`text-neutral-600 dark:text-slate-300 ${textClasses[size]}`}>
              {stats.level_title}
            </span>
          </div>
          <span className={`text-neutral-500 dark:text-slate-400 ${textClasses[size]}`}>
            {stats.xp_progress} / {stats.xp_for_next_level} XP
          </span>
        </div>
      )}

      {/* Progress Bar */}
      <div className={`w-full bg-neutral-200 dark:bg-slate-600 rounded-full ${heightClasses[size]} overflow-hidden`}>
        <div
          className={`${heightClasses[size]} bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-700 ease-out relative overflow-hidden`}
          style={{ width: `${Math.min(100, stats.xp_percentage)}%` }}
        >
          {/* Shimmer effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
        </div>
      </div>

      {showDetails && (
        <div className="flex justify-between items-center mt-1">
          <span className="text-xs text-neutral-400 dark:text-slate-500">
            총 {stats.total_xp.toLocaleString()} XP
          </span>
          <span className="text-xs text-indigo-500 dark:text-indigo-400 font-medium">
            {stats.xp_percentage.toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
}

// Compact level badge
interface LevelBadgeProps {
  level: number;
  title: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LevelBadge({ level, title, size = 'md' }: LevelBadgeProps) {
  const sizeClasses = {
    sm: 'text-sm px-2 py-1',
    md: 'text-base px-3 py-1.5',
    lg: 'text-lg px-4 py-2',
  };

  return (
    <div className={`inline-flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-full font-medium ${sizeClasses[size]}`}>
      <span>Lv.{level}</span>
      <span className="opacity-80">{title}</span>
    </div>
  );
}
