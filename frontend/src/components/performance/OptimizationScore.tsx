/**
 * Optimization Score Display Component
 */

import { Award, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface OptimizationScoreProps {
  score: number;
  previousScore?: number;
}

export default function OptimizationScore({ score, previousScore }: OptimizationScoreProps) {
  const getScoreColor = (s: number) => {
    if (s >= 80) return 'text-green-600';
    if (s >= 60) return 'text-yellow-600';
    if (s >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (s: number) => {
    if (s >= 80) return 'bg-green-100';
    if (s >= 60) return 'bg-yellow-100';
    if (s >= 40) return 'bg-orange-100';
    return 'bg-red-100';
  };

  const getScoreLabel = (s: number) => {
    if (s >= 90) return '최적화 우수';
    if (s >= 80) return '양호';
    if (s >= 60) return '개선 필요';
    if (s >= 40) return '최적화 권장';
    return '성능 문제';
  };

  const getScoreGradient = (s: number) => {
    if (s >= 80) return 'from-green-500 to-emerald-500';
    if (s >= 60) return 'from-yellow-500 to-amber-500';
    if (s >= 40) return 'from-orange-500 to-amber-600';
    return 'from-red-500 to-rose-600';
  };

  const diff = previousScore !== undefined ? score - previousScore : null;

  return (
    <div className={`rounded-lg p-4 ${getScoreBgColor(score)}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-full bg-white ${getScoreColor(score)}`}>
            <Award className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm text-gray-600">최적화 점수</p>
            <p className={`text-3xl font-bold ${getScoreColor(score)}`}>{score}</p>
          </div>
        </div>

        <div className="text-right">
          <p className={`text-lg font-medium ${getScoreColor(score)}`}>
            {getScoreLabel(score)}
          </p>
          {diff !== null && (
            <div className="flex items-center justify-end gap-1 text-sm">
              {diff > 0 ? (
                <>
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  <span className="text-green-600">+{diff}</span>
                </>
              ) : diff < 0 ? (
                <>
                  <TrendingDown className="w-4 h-4 text-red-600" />
                  <span className="text-red-600">{diff}</span>
                </>
              ) : (
                <>
                  <Minus className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-500">변동 없음</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-4">
        <div className="h-3 bg-white rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${getScoreGradient(score)} transition-all duration-500`}
            style={{ width: `${score}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0</span>
          <span>50</span>
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
    if (s >= 80) return 'bg-green-100 text-green-700';
    if (s >= 60) return 'bg-yellow-100 text-yellow-700';
    if (s >= 40) return 'bg-orange-100 text-orange-700';
    return 'bg-red-100 text-red-700';
  };

  return (
    <span className={`px-2 py-1 rounded-full text-sm font-medium ${getColor(score)}`}>
      {score}점
    </span>
  );
}
