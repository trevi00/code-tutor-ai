/**
 * Performance Metrics Display Component - Enhanced with modern design
 */

import { Timer, Cpu, HardDrive, Layers, Zap, Activity } from 'lucide-react';
import type { RuntimeMetrics, MemoryMetrics, HotspotResult } from '../../api/performance';

interface RuntimeMetricsDisplayProps {
  metrics: RuntimeMetrics;
}

export function RuntimeMetricsDisplay({ metrics }: RuntimeMetricsDisplayProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-emerald-100 to-teal-100 dark:from-emerald-900/30 dark:to-teal-900/30 border-b border-emerald-200 dark:border-emerald-800">
        <h3 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
          <Timer className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          런타임 성능
        </h3>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            icon={<Timer className="w-5 h-5" />}
            label="실행 시간"
            value={`${metrics.execution_time_ms.toFixed(2)}ms`}
            color="emerald"
          />
          <MetricCard
            icon={<Cpu className="w-5 h-5" />}
            label="CPU 시간"
            value={`${metrics.cpu_time_ms.toFixed(2)}ms`}
            color="blue"
          />
          <MetricCard
            icon={<Zap className="w-5 h-5" />}
            label="함수 호출"
            value={metrics.function_calls.toLocaleString()}
            color="purple"
          />
          <MetricCard
            icon={<Layers className="w-5 h-5" />}
            label="최대 호출 깊이"
            value={metrics.peak_call_depth.toString()}
            color="orange"
          />
        </div>
      </div>
    </div>
  );
}

interface MemoryMetricsDisplayProps {
  metrics: MemoryMetrics;
}

export function MemoryMetricsDisplay({ metrics }: MemoryMetricsDisplayProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-purple-100 to-violet-100 dark:from-purple-900/30 dark:to-violet-900/30 border-b border-purple-200 dark:border-purple-800">
        <h3 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          메모리 사용량
        </h3>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <MetricCard
            icon={<HardDrive className="w-5 h-5" />}
            label="최대 메모리"
            value={`${metrics.peak_memory_mb.toFixed(2)} MB`}
            color="purple"
          />
          <MetricCard
            icon={<Activity className="w-5 h-5" />}
            label="평균 메모리"
            value={`${metrics.average_memory_mb.toFixed(2)} MB`}
            color="blue"
          />
          <MetricCard
            icon={<Layers className="w-5 h-5" />}
            label="할당 횟수"
            value={metrics.allocations_count.toLocaleString()}
            color="emerald"
          />
        </div>

        {metrics.largest_object_mb > 0.01 && (
          <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
            <span className="text-sm text-purple-600 dark:text-purple-400">최대 객체: </span>
            <span className="font-mono font-bold text-purple-700 dark:text-purple-300">
              {metrics.largest_object_mb.toFixed(3)} MB
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

interface HotspotDisplayProps {
  hotspots: HotspotResult;
}

export function HotspotDisplay({ hotspots }: HotspotDisplayProps) {
  if (hotspots.hotspot_functions.length === 0) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-orange-100 to-amber-100 dark:from-orange-900/30 dark:to-amber-900/30 border-b border-orange-200 dark:border-orange-800">
        <div className="flex items-center justify-between">
          <h3 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
            <Zap className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            성능 핫스팟
          </h3>
          {hotspots.bottleneck_function && (
            <span className="px-3 py-1 bg-orange-200 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300 text-sm rounded-lg font-medium">
              병목: <span className="font-mono">{hotspots.bottleneck_function}()</span>
            </span>
          )}
        </div>
      </div>

      <div className="p-5">
        <div className="space-y-3">
          {hotspots.hotspot_functions.slice(0, 5).map((func, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between text-sm bg-slate-50 dark:bg-slate-700/50 px-4 py-3 rounded-xl"
            >
              <div className="flex items-center gap-3">
                <span className={`w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold ${
                  idx === 0
                    ? 'bg-orange-500 text-white'
                    : 'bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-300'
                }`}>
                  {idx + 1}
                </span>
                <span className="font-mono text-slate-800 dark:text-slate-200 font-medium">
                  {func.name}()
                </span>
                <span className="text-slate-400 dark:text-slate-500 text-xs">
                  x{func.calls}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-slate-600 dark:text-slate-400 font-mono">
                  {func.total_time_ms.toFixed(2)}ms
                </span>
                <div className="w-24">
                  <div className="h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        idx === 0
                          ? 'bg-gradient-to-r from-orange-500 to-amber-500'
                          : 'bg-gradient-to-r from-slate-400 to-slate-500 dark:from-slate-500 dark:to-slate-400'
                      }`}
                      style={{ width: `${Math.min(100, func.percentage)}%` }}
                    />
                  </div>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400 w-14 text-right font-mono">
                  {func.percentage.toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: 'emerald' | 'blue' | 'purple' | 'orange' | 'red';
}

function MetricCard({ icon, label, value, color }: MetricCardProps) {
  const colorClasses = {
    emerald: {
      bg: 'bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20',
      border: 'border border-emerald-200 dark:border-emerald-800',
      icon: 'text-emerald-600 dark:text-emerald-400',
      text: 'text-emerald-700 dark:text-emerald-300',
    },
    blue: {
      bg: 'bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20',
      border: 'border border-blue-200 dark:border-blue-800',
      icon: 'text-blue-600 dark:text-blue-400',
      text: 'text-blue-700 dark:text-blue-300',
    },
    purple: {
      bg: 'bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20',
      border: 'border border-purple-200 dark:border-purple-800',
      icon: 'text-purple-600 dark:text-purple-400',
      text: 'text-purple-700 dark:text-purple-300',
    },
    orange: {
      bg: 'bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20',
      border: 'border border-orange-200 dark:border-orange-800',
      icon: 'text-orange-600 dark:text-orange-400',
      text: 'text-orange-700 dark:text-orange-300',
    },
    red: {
      bg: 'bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20',
      border: 'border border-red-200 dark:border-red-800',
      icon: 'text-red-600 dark:text-red-400',
      text: 'text-red-700 dark:text-red-300',
    },
  };

  const styles = colorClasses[color];

  return (
    <div className={`p-4 rounded-xl ${styles.bg} ${styles.border}`}>
      <div className={`flex items-center gap-2 text-xs ${styles.icon} mb-2`}>
        {icon}
        <span className="text-slate-600 dark:text-slate-400">{label}</span>
      </div>
      <div className={`text-xl font-bold ${styles.text}`}>{value}</div>
    </div>
  );
}
