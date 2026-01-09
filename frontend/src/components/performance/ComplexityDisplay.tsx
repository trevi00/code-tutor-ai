/**
 * Complexity Display Component - Enhanced with modern design
 */

import { Clock, Database, TrendingUp, AlertTriangle, Repeat, Code2 } from 'lucide-react';
import type { ComplexityResult, ComplexityClass } from '../../api/performance';
import { COMPLEXITY_COLORS } from '../../api/performance';

interface ComplexityDisplayProps {
  complexity: ComplexityResult;
}

export default function ComplexityDisplay({ complexity }: ComplexityDisplayProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-violet-100 to-purple-100 dark:from-violet-900/30 dark:to-purple-900/30 border-b border-violet-200 dark:border-violet-800">
        <h3 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-violet-600 dark:text-violet-400" />
          복잡도 분석
        </h3>
      </div>

      <div className="p-5 space-y-5">
        {/* Time & Space Complexity */}
        <div className="grid grid-cols-2 gap-4">
          <ComplexityCard
            icon={<Clock className="w-5 h-5" />}
            label="시간 복잡도"
            complexity={complexity.time_complexity}
            explanation={complexity.time_explanation}
            color="blue"
          />
          <ComplexityCard
            icon={<Database className="w-5 h-5" />}
            label="공간 복잡도"
            complexity={complexity.space_complexity}
            explanation={complexity.space_explanation}
            color="purple"
          />
        </div>

        {/* Nesting Depth */}
        {complexity.max_nesting_depth > 0 && (
          <div className="flex items-center gap-3 p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
            <span className="text-sm text-slate-600 dark:text-slate-400">최대 중첩 깊이:</span>
            <span
              className={`px-3 py-1 rounded-lg font-bold text-sm ${
                complexity.max_nesting_depth >= 3
                  ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                  : complexity.max_nesting_depth >= 2
                  ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                  : 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
              }`}
            >
              {complexity.max_nesting_depth}단계
            </span>
          </div>
        )}

        {/* Recursive Functions */}
        {complexity.recursive_functions.length > 0 && (
          <div className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
            <AlertTriangle className="w-5 h-5 text-amber-500 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="text-sm font-medium text-amber-700 dark:text-amber-300">재귀 함수 감지</span>
              <div className="mt-1 flex flex-wrap gap-2">
                {complexity.recursive_functions.map((func, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded font-mono text-sm"
                  >
                    {func}()
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Loops Summary */}
        {complexity.loops.length > 0 && (
          <div>
            <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <Repeat className="w-4 h-4 text-blue-500" />
              루프 분석
            </h4>
            <div className="space-y-2">
              {complexity.loops.map((loop, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 text-sm bg-slate-50 dark:bg-slate-700/50 px-4 py-2.5 rounded-xl"
                >
                  <span className="text-slate-400 dark:text-slate-500 font-mono text-xs">
                    L{loop.line_number}
                  </span>
                  <span className="font-mono text-blue-600 dark:text-blue-400 font-medium">
                    {loop.loop_type}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-lg text-xs font-medium ${
                      loop.nesting_level >= 2
                        ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                        : 'bg-slate-100 dark:bg-slate-600 text-slate-600 dark:text-slate-300'
                    }`}
                  >
                    중첩 {loop.nesting_level}
                  </span>
                  {loop.iteration_variable && (
                    <span className="text-slate-500 dark:text-slate-400 text-xs">
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
          <div>
            <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <Code2 className="w-4 h-4 text-purple-500" />
              함수 분석
            </h4>
            <div className="space-y-2">
              {complexity.functions.map((func, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-sm bg-slate-50 dark:bg-slate-700/50 px-4 py-2.5 rounded-xl"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-purple-600 dark:text-purple-400 font-medium">
                      {func.name}()
                    </span>
                    <span className="text-slate-400 dark:text-slate-500 text-xs">
                      줄 {func.line_number}
                    </span>
                    {func.is_recursive && (
                      <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-xs rounded-lg font-medium">
                        재귀
                      </span>
                    )}
                  </div>
                  {func.calls_count > 0 && (
                    <span className="text-slate-500 dark:text-slate-400 text-xs">
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
  color: 'blue' | 'purple';
}

function ComplexityCard({ icon, label, complexity, explanation, color }: ComplexityCardProps) {
  const colorClass = COMPLEXITY_COLORS[complexity] || COMPLEXITY_COLORS['Unknown'];
  const bgColors = {
    blue: 'from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border-blue-200 dark:border-blue-800',
    purple: 'from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 border-purple-200 dark:border-purple-800',
  };
  const iconColors = {
    blue: 'text-blue-600 dark:text-blue-400',
    purple: 'text-purple-600 dark:text-purple-400',
  };

  return (
    <div className={`p-4 bg-gradient-to-br ${bgColors[color]} rounded-xl border`}>
      <div className={`flex items-center gap-2 ${iconColors[color]} mb-3`}>
        {icon}
        <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{label}</span>
      </div>
      <div className={`inline-block px-4 py-1.5 rounded-xl text-lg font-bold ${colorClass}`}>
        {complexity}
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mt-3 leading-relaxed">{explanation}</p>
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
    <div className="flex items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <Clock className="w-4 h-4 text-blue-500 dark:text-blue-400" />
        <span className={`px-3 py-1 rounded-lg font-bold ${COMPLEXITY_COLORS[timeComplexity]}`}>
          {timeComplexity}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <Database className="w-4 h-4 text-purple-500 dark:text-purple-400" />
        <span className={`px-3 py-1 rounded-lg font-bold ${COMPLEXITY_COLORS[spaceComplexity]}`}>
          {spaceComplexity}
        </span>
      </div>
    </div>
  );
}
