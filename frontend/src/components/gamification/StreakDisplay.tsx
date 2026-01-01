/**
 * Streak Display Component
 */

interface StreakDisplayProps {
  currentStreak: number;
  longestStreak: number;
  size?: 'sm' | 'md' | 'lg';
}

export default function StreakDisplay({
  currentStreak,
  longestStreak,
  size = 'md',
}: StreakDisplayProps) {
  const sizeClasses = {
    sm: { icon: 'text-2xl', text: 'text-sm', number: 'text-xl' },
    md: { icon: 'text-3xl', text: 'text-base', number: 'text-2xl' },
    lg: { icon: 'text-4xl', text: 'text-lg', number: 'text-3xl' },
  };

  const classes = sizeClasses[size];

  // Streak fire intensity based on streak length
  const getFireIntensity = () => {
    if (currentStreak >= 30) return 'animate-pulse text-orange-500';
    if (currentStreak >= 7) return 'text-orange-500';
    if (currentStreak >= 3) return 'text-yellow-500';
    return 'text-gray-400';
  };

  return (
    <div className="flex items-center gap-6">
      {/* Current Streak */}
      <div className="flex items-center gap-2">
        <span className={`${classes.icon} ${getFireIntensity()}`}>🔥</span>
        <div>
          <div className={`font-bold ${classes.number} text-gray-900`}>
            {currentStreak}
          </div>
          <div className={`${classes.text} text-gray-500`}>연속 일</div>
        </div>
      </div>

      {/* Divider */}
      <div className="h-12 w-px bg-gray-200" />

      {/* Longest Streak */}
      <div className="flex items-center gap-2">
        <span className={`${classes.icon} text-yellow-500`}>🏆</span>
        <div>
          <div className={`font-bold ${classes.number} text-gray-900`}>
            {longestStreak}
          </div>
          <div className={`${classes.text} text-gray-500`}>최장 기록</div>
        </div>
      </div>
    </div>
  );
}

// Compact streak badge
interface StreakBadgeProps {
  streak: number;
}

export function StreakBadge({ streak }: StreakBadgeProps) {
  const bgColor = streak >= 7 ? 'bg-orange-100' : streak >= 3 ? 'bg-yellow-100' : 'bg-gray-100';
  const textColor = streak >= 7 ? 'text-orange-700' : streak >= 3 ? 'text-yellow-700' : 'text-gray-600';

  return (
    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full ${bgColor} ${textColor} text-sm font-medium`}>
      <span>🔥</span>
      <span>{streak}일</span>
    </div>
  );
}
