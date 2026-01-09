/**
 * ArrayBar component for visualizing array elements - Enhanced with gradients
 */

interface ArrayBarProps {
  value: number;
  maxValue: number;
  state: string;
  index: number;
  showValue?: boolean;
}

const STATE_STYLES: Record<string, { gradient: string; shadow: string }> = {
  default: {
    gradient: 'from-blue-500 to-blue-400',
    shadow: 'shadow-blue-500/30',
  },
  comparing: {
    gradient: 'from-amber-500 to-yellow-400',
    shadow: 'shadow-amber-500/30',
  },
  swapping: {
    gradient: 'from-red-500 to-rose-400',
    shadow: 'shadow-red-500/30',
  },
  sorted: {
    gradient: 'from-emerald-500 to-green-400',
    shadow: 'shadow-emerald-500/30',
  },
  pivot: {
    gradient: 'from-purple-500 to-violet-400',
    shadow: 'shadow-purple-500/30',
  },
  current: {
    gradient: 'from-orange-500 to-amber-400',
    shadow: 'shadow-orange-500/30',
  },
  found: {
    gradient: 'from-emerald-500 to-green-400',
    shadow: 'shadow-emerald-500/30',
  },
  visited: {
    gradient: 'from-slate-500 to-slate-400',
    shadow: 'shadow-slate-500/30',
  },
  active: {
    gradient: 'from-blue-600 to-blue-500',
    shadow: 'shadow-blue-600/30',
  },
};

export default function ArrayBar({
  value,
  maxValue,
  state,
  index,
  showValue = true,
}: ArrayBarProps) {
  const heightPercent = (value / maxValue) * 100;
  const styles = STATE_STYLES[state] || STATE_STYLES.default;

  return (
    <div className="flex flex-col items-center group">
      {/* Value tooltip on hover */}
      {showValue && (
        <div className="relative mb-1 h-6">
          <span
            className={`absolute bottom-0 left-1/2 -translate-x-1/2 text-xs font-bold transition-all duration-200 ${
              state === 'comparing' || state === 'swapping' || state === 'current'
                ? 'text-slate-900 dark:text-white scale-110'
                : 'text-slate-600 dark:text-slate-400'
            }`}
          >
            {value}
          </span>
        </div>
      )}

      {/* Bar */}
      <div
        className={`relative transition-all duration-300 ease-in-out bg-gradient-to-t ${styles.gradient} rounded-t-lg min-w-[24px] shadow-lg ${styles.shadow} hover:scale-105`}
        style={{
          height: `${heightPercent}%`,
          minHeight: '24px',
        }}
      >
        {/* Shine effect */}
        <div className="absolute inset-0 rounded-t-lg overflow-hidden">
          <div className="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-white/30 to-transparent" />
        </div>

        {/* Pulse effect for active states */}
        {(state === 'comparing' || state === 'swapping' || state === 'current') && (
          <div
            className={`absolute inset-0 rounded-t-lg bg-gradient-to-t ${styles.gradient} animate-pulse opacity-50`}
          />
        )}
      </div>

      {/* Index label */}
      <span className="text-[10px] text-slate-400 dark:text-slate-500 mt-1.5 font-mono">
        {index}
      </span>
    </div>
  );
}
