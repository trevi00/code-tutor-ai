/**
 * ArrayBar component for visualizing array elements
 */

interface ArrayBarProps {
  value: number;
  maxValue: number;
  state: string;
  index: number;
  showValue?: boolean;
}

const STATE_COLORS: Record<string, string> = {
  default: 'bg-blue-400',
  comparing: 'bg-yellow-400',
  swapping: 'bg-red-400',
  sorted: 'bg-green-500',
  pivot: 'bg-purple-500',
  current: 'bg-orange-400',
  found: 'bg-green-400',
  visited: 'bg-gray-400',
  active: 'bg-blue-500',
};

export default function ArrayBar({
  value,
  maxValue,
  state,
  index,
  showValue = true,
}: ArrayBarProps) {
  const heightPercent = (value / maxValue) * 100;
  const colorClass = STATE_COLORS[state] || STATE_COLORS.default;

  return (
    <div className="flex flex-col items-center">
      <div
        className={`relative transition-all duration-300 ease-in-out ${colorClass} rounded-t-sm min-w-[20px]`}
        style={{
          height: `${heightPercent}%`,
          minHeight: '20px',
        }}
      >
        {showValue && (
          <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-medium text-gray-700">
            {value}
          </span>
        )}
      </div>
      <span className="text-xs text-gray-500 mt-1">{index}</span>
    </div>
  );
}
