/**
 * Optimization Score Display Component - Enhanced with modern design
 */

import { Award, TrendingUp, TrendingDown, Minus, Sparkles } from 'lucide-react';

interface OptimizationScoreProps {
  score: number;
  previousScore?: number;
}

export default function OptimizationScore({ score, previousScore }: OptimizationScoreProps) {
  const getScoreColor = (s: number) => {
    if (s >= 80) return 'text-emerald-600 dark:text-emerald-400';
    if (s >= 60) return 'text-amber-600 dark:text-amber-400';
    if (s >= 40) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreLabel = (s: number) => {
    if (s >= 90) return '최적화 우수';
    if (s >= 80) return '양호';
    if (s >= 60) return '개선 필요';
    if (s >= 40) return '최적화 권장';
    return '성능 문제';
  };

  const getScoreGradient = (s: number) => {
    if (s >= 80) return 'from-emerald-500 to-teal-500';
    if (s >= 60) return 'from-amber-500 to-yellow-500';
    if (s >= 40) return 'from-orange-500 to-amber-500';
    return 'from-red-500 to-rose-500';
  };

  const getBgGradient = (s: number) => {
    if (s >= 80) return 'from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 border-emerald-200 dark:border-emerald-800';
    if (s >= 60) return 'from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 border-amber-200 dark:border-amber-800';
    if (s >= 40) return 'from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 border-orange-200 dark:border-orange-800';
    return 'from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 border-red-200 dark:border-red-800';
  };

  const getIconBg = (s: number) => {
    if (s >= 80) return 'bg-emerald-100 dark:bg-emerald-900/50';
    if (s >= 60) return 'bg-amber-100 dark:bg-amber-900/50';
    if (s >= 40) return 'bg-orange-100 dark:bg-orange-900/50';
    return 'bg-red-100 dark:bg-red-900/50';
  };

  const diff = previousScore !== undefined ? score - previousScore : null;

  return (
    <div className={`rounded-2xl p-5 bg-gradient-to-r ${getBgGradient(score)} border`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={`w-14 h-14 rounded-xl ${getIconBg(score)} flex items-center justify-center`}>
            <Award className={`w-8 h-8 ${getScoreColor(score)}`} />
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">최적화 점수</p>
            <div className="flex items-baseline gap-2">
              <p className={`text-4xl font-bold ${getScoreColor(score)}`}>{score}</p>
              <span className="text-lg text-slate-400 dark:text-slate-500">/100</span>
            </div>
          </div>
        </div>

        <div className="text-right">
          <div className="flex items-center gap-2 justify-end">
            <Sparkles className={`w-5 h-5 ${getScoreColor(score)}`} />
            <p className={`text-lg font-bold ${getScoreColor(score)}`}>
              {getScoreLabel(score)}
            </p>
          </div>
          {diff !== null && (
            <div className="flex items-center justify-end gap-1.5 text-sm mt-2">
              {diff > 0 ? (
                <>
                  <TrendingUp className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">+{diff}</span>
                </>
              ) : diff < 0 ? (
                <>
                  <TrendingDown className="w-4 h-4 text-red-600 dark:text-red-400" />
                  <span className="text-red-600 dark:text-red-400 font-medium">{diff}</span>
                </>
              ) : (
                <>
                  <Minus className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                  <span className="text-slate-500 dark:text-slate-400">변동 없음</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-5">
        <div className="h-3 bg-white/60 dark:bg-slate-800/60 rounded-full overflow-hidden shadow-inner">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${getScoreGradient(score)} transition-all duration-700 ease-out`}
            style={{ width: `${score}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-2 px-1">
          <span>0</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
      </div>
    </div>
  );
}

// Compact score badge
interface ScoreBadgeProps {
  score: number;
}

export function ScoreBadge({ score }: ScoreBadgeProps) {
  const getColor = (s: number) => {
    if (s >= 80) return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800';
    if (s >= 60) return 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800';
    if (s >= 40) return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-800';
    return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800';
  };

  return (
    <span className={`px-3 py-1 rounded-lg text-sm font-bold border ${getColor(score)}`}>
      {score}점
    </span>
  );
}
