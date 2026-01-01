/**
 * Complexity Display Component
 */

import { Clock, Database, TrendingUp, AlertTriangle } from 'lucide-react';
import type { ComplexityResult, ComplexityClass } from '../../api/performance';
import { COMPLEXITY_COLORS } from '../../api/performance';

interface ComplexityDisplayProps {
  complexity: ComplexityResult;
}

export default function ComplexityDisplay({ complexity }: ComplexityDisplayProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b">
        <h3 className="font-medium text-gray-800 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          복잡도 분석
        </h3>
      </div>

      <div className="p-4 space-y-4">
        {/* Time & Space Complexity */}
        <div className="grid grid-cols-2 gap-4">
          <ComplexityCard
            icon={<Clock className="w-5 h-5" />}
            label="시간 복잡도"
            complexity={complexity.time_complexity}
            explanation={complexity.time_explanation}
          />
          <ComplexityCard
            icon={<Database className="w-5 h-5" />}
            label="공간 복잡도"
            complexity={complexity.space_complexity}
            explanation={complexity.space_explanation}
          />
        </div>

        {/* Nesting Depth */}
        {complexity.max_nesting_depth > 0 && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-600">최대 중첩 깊이:</span>
            <span
              className={`px-2 py-0.5 rounded font-medium ${
                complexity.max_nesting_depth >= 3
                  ? 'bg-red-100 text-red-700'
                  : complexity.max_nesting_depth >= 2
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-green-100 text-green-700'
              }`}
            >
              {complexity.max_nesting_depth}단계
            </span>
          </div>
        )}

        {/* Recursive Functions */}
        {complexity.recursive_functions.length > 0 && (
          <div className="flex items-start gap-2 text-sm">
            <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <span className="text-gray-600">재귀 함수: </span>
              <span className="font-mono text-purple-600">
                {complexity.recursive_functions.join(', ')}
              </span>
            </div>
          </div>
        )}

        {/* Loops Summary */}
        {complexity.loops.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">루프 분석</h4>
            <div className="space-y-2">
              {complexity.loops.map((loop, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 text-sm bg-gray-50 px-3 py-2 rounded"
                >
                  <span className="text-gray-500">줄 {loop.line_number}</span>
                  <span className="font-mono text-blue-600">{loop.loop_type}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs ${
                      loop.nesting_level >= 2
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    중첩 {loop.nesting_level}
                  </span>
                  {loop.iteration_variable && (
                    <span className="text-gray-500">
                      ({loop.iteration_variable})
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Functions Summary */}
        {complexity.functions.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">함수 분석</h4>
            <div className="space-y-2">
              {complexity.functions.map((func, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-sm bg-gray-50 px-3 py-2 rounded"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-purple-600">{func.name}()</span>
                    <span className="text-gray-500">줄 {func.line_number}</span>
                    {func.is_recursive && (
                      <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">
                        재귀
                      </span>
                    )}
                  </div>
                  {func.calls_count > 0 && (
                    <span className="text-gray-500 text-xs">
                      {func.calls_count}회 호출
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface ComplexityCardProps {
  icon: React.ReactNode;
  label: string;
  complexity: ComplexityClass;
  explanation: string;
}

function ComplexityCard({ icon, label, complexity, explanation }: ComplexityCardProps) {
  const colorClass = COMPLEXITY_COLORS[complexity] || COMPLEXITY_COLORS['Unknown'];

  return (
    <div className="p-3 bg-gray-50 rounded-lg">
      <div className="flex items-center gap-2 text-gray-600 mb-2">
        {icon}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className={`inline-block px-3 py-1 rounded-full text-lg font-bold ${colorClass}`}>
        {complexity}
      </div>
      <p className="text-xs text-gray-500 mt-2">{explanation}</p>
    </div>
  );
}

// Compact display for inline use
interface CompactComplexityProps {
  timeComplexity: ComplexityClass;
  spaceComplexity: ComplexityClass;
}

export function CompactComplexity({ timeComplexity, spaceComplexity }: CompactComplexityProps) {
  return (
    <div className="flex items-center gap-3 text-sm">
      <div className="flex items-center gap-1">
        <Clock className="w-4 h-4 text-gray-400" />
        <span className={`px-2 py-0.5 rounded ${COMPLEXITY_COLORS[timeComplexity]}`}>
          {timeComplexity}
        </span>
      </div>
      <div className="flex items-center gap-1">
        <Database className="w-4 h-4 text-gray-400" />
        <span className={`px-2 py-0.5 rounded ${COMPLEXITY_COLORS[spaceComplexity]}`}>
          {spaceComplexity}
        </span>
      </div>
    </div>
  );
}
