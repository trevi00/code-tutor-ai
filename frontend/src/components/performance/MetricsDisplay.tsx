/**
 * Performance Metrics Display Component
 */

import { Timer, Cpu, HardDrive, Layers, Zap } from 'lucide-react';
import type { RuntimeMetrics, MemoryMetrics, HotspotResult } from '../../api/performance';

interface RuntimeMetricsDisplayProps {
  metrics: RuntimeMetrics;
}

export function RuntimeMetricsDisplay({ metrics }: RuntimeMetricsDisplayProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b">
        <h3 className="font-medium text-gray-800 flex items-center gap-2">
          <Timer className="w-5 h-5 text-green-600" />
          런타임 성능
        </h3>
      </div>

      <div className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            icon={<Timer className="w-4 h-4" />}
            label="실행 시간"
            value={`${metrics.execution_time_ms.toFixed(2)}ms`}
            color="green"
          />
          <MetricCard
            icon={<Cpu className="w-4 h-4" />}
            label="CPU 시간"
            value={`${metrics.cpu_time_ms.toFixed(2)}ms`}
            color="blue"
          />
          <MetricCard
            icon={<Zap className="w-4 h-4" />}
            label="함수 호출"
            value={metrics.function_calls.toLocaleString()}
            color="purple"
          />
          <MetricCard
            icon={<Layers className="w-4 h-4" />}
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
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b">
        <h3 className="font-medium text-gray-800 flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-purple-600" />
          메모리 사용량
        </h3>
      </div>

      <div className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <MetricCard
            icon={<HardDrive className="w-4 h-4" />}
            label="최대 메모리"
            value={`${metrics.peak_memory_mb.toFixed(2)} MB`}
            color="purple"
          />
          <MetricCard
            icon={<HardDrive className="w-4 h-4" />}
            label="평균 메모리"
            value={`${metrics.average_memory_mb.toFixed(2)} MB`}
            color="blue"
          />
          <MetricCard
            icon={<Layers className="w-4 h-4" />}
            label="할당 횟수"
            value={metrics.allocations_count.toLocaleString()}
            color="green"
          />
        </div>

        {metrics.largest_object_mb > 0.01 && (
          <div className="mt-4 text-sm text-gray-600">
            <span>최대 객체: </span>
            <span className="font-mono text-purple-600">
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
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b">
        <h3 className="font-medium text-gray-800 flex items-center gap-2">
          <Zap className="w-5 h-5 text-orange-600" />
          성능 핫스팟
        </h3>
        {hotspots.bottleneck_function && (
          <p className="text-sm text-gray-500 mt-1">
            병목 함수:{' '}
            <span className="font-mono text-orange-600">
              {hotspots.bottleneck_function}()
            </span>
          </p>
        )}
      </div>

      <div className="p-4">
        <div className="space-y-2">
          {hotspots.hotspot_functions.slice(0, 5).map((func, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between text-sm bg-gray-50 px-3 py-2 rounded"
            >
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-4">{idx + 1}.</span>
                <span className="font-mono text-gray-800">{func.name}()</span>
                <span className="text-gray-400">x{func.calls}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600">{func.total_time_ms.toFixed(2)}ms</span>
                <div className="w-20">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-orange-500 rounded-full"
                      style={{ width: `${Math.min(100, func.percentage)}%` }}
                    />
                  </div>
                </div>
                <span className="text-xs text-gray-500 w-12 text-right">
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
  color: 'green' | 'blue' | 'purple' | 'orange' | 'red';
}

function MetricCard({ icon, label, value, color }: MetricCardProps) {
  const colorClasses = {
    green: 'text-green-600 bg-green-50',
    blue: 'text-blue-600 bg-blue-50',
    purple: 'text-purple-600 bg-purple-50',
    orange: 'text-orange-600 bg-orange-50',
    red: 'text-red-600 bg-red-50',
  };

  return (
    <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
      <div className="flex items-center gap-1 text-xs opacity-75 mb-1">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-lg font-bold">{value}</div>
    </div>
  );
}
