import { useEffect, useState } from 'react';
import clsx from 'clsx';

interface ProgressRingProps {
  progress: number; // 0-100
  size?: 'sm' | 'md' | 'lg' | 'xl';
  strokeWidth?: number;
  color?: string;
  trackColor?: string;
  label?: string;
  sublabel?: string;
  showPercentage?: boolean;
  animate?: boolean;
  className?: string;
}

const sizeConfig = {
  sm: { width: 64, fontSize: 'text-sm', labelSize: 'text-xs' },
  md: { width: 96, fontSize: 'text-lg', labelSize: 'text-sm' },
  lg: { width: 128, fontSize: 'text-2xl', labelSize: 'text-base' },
  xl: { width: 160, fontSize: 'text-3xl', labelSize: 'text-lg' },
};

export function ProgressRing({
  progress,
  size = 'md',
  strokeWidth = 8,
  color = '#6366f1',
  trackColor = '#e5e7eb',
  label,
  sublabel,
  showPercentage = true,
  animate = true,
  className,
}: ProgressRingProps) {
  const [animatedProgress, setAnimatedProgress] = useState(animate ? 0 : progress);
  const config = sizeConfig[size];
  const radius = (config.width - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (animatedProgress / 100) * circumference;

  useEffect(() => {
    if (animate) {
      // Animate progress from 0 to target
      const timer = setTimeout(() => {
        setAnimatedProgress(progress);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [progress, animate]);

  return (
    <div className={clsx('relative inline-flex items-center justify-center', className)}>
      <svg
        width={config.width}
        height={config.width}
        className="transform -rotate-90"
      >
        {/* Background track */}
        <circle
          cx={config.width / 2}
          cy={config.width / 2}
          r={radius}
          stroke={trackColor}
          strokeWidth={strokeWidth}
          fill="none"
          className="dark:stroke-slate-700"
        />
        {/* Progress arc */}
        <circle
          cx={config.width / 2}
          cy={config.width / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transition: animate ? 'stroke-dashoffset 1s ease-out' : 'none',
          }}
        />
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {showPercentage && (
          <span className={clsx('font-bold text-gray-800 dark:text-gray-100', config.fontSize)}>
            {Math.round(animatedProgress)}%
          </span>
        )}
        {label && (
          <span className={clsx('text-gray-600 dark:text-gray-400 font-medium', config.labelSize)}>
            {label}
          </span>
        )}
        {sublabel && (
          <span className="text-xs text-gray-400 dark:text-gray-500">
            {sublabel}
          </span>
        )}
      </div>
    </div>
  );
}

// Multi-ring variant for comparing multiple metrics
interface MultiProgressRingProps {
  rings: Array<{
    progress: number;
    color: string;
    label: string;
  }>;
  size?: 'md' | 'lg' | 'xl';
  className?: string;
}

export function MultiProgressRing({ rings, size = 'lg', className }: MultiProgressRingProps) {
  const [animatedProgress, setAnimatedProgress] = useState(rings.map(() => 0));
  const baseSize = sizeConfig[size].width;
  const strokeWidth = 6;
  const gap = strokeWidth + 4;

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedProgress(rings.map(r => r.progress));
    }, 100);
    return () => clearTimeout(timer);
  }, [rings]);

  return (
    <div className={clsx('relative inline-flex flex-col items-center', className)}>
      <div className="relative" style={{ width: baseSize, height: baseSize }}>
        <svg width={baseSize} height={baseSize} className="transform -rotate-90">
          {rings.map((ring, index) => {
            const radius = (baseSize - strokeWidth) / 2 - index * gap;
            const circumference = radius * 2 * Math.PI;
            const offset = circumference - (animatedProgress[index] / 100) * circumference;

            return (
              <g key={index}>
                {/* Track */}
                <circle
                  cx={baseSize / 2}
                  cy={baseSize / 2}
                  r={radius}
                  stroke="#e5e7eb"
                  strokeWidth={strokeWidth}
                  fill="none"
                  className="dark:stroke-slate-700"
                  opacity={0.5}
                />
                {/* Progress */}
                <circle
                  cx={baseSize / 2}
                  cy={baseSize / 2}
                  r={radius}
                  stroke={ring.color}
                  strokeWidth={strokeWidth}
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={offset}
                  style={{
                    transition: 'stroke-dashoffset 1s ease-out',
                    transitionDelay: `${index * 0.1}s`,
                  }}
                />
              </g>
            );
          })}
        </svg>
      </div>
      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-3 mt-4">
        {rings.map((ring, index) => (
          <div key={index} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: ring.color }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400">
              {ring.label} ({ring.progress}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
