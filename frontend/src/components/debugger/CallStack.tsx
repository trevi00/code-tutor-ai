/**
 * Call Stack Component - Enhanced with modern design
 */

import { ChevronDown, ChevronRight, Layers, GitBranch } from 'lucide-react';
import { useState } from 'react';
import type { StackFrame } from '../../api/debugger';
import VariableInspector from './VariableInspector';

interface CallStackProps {
  frames: StackFrame[];
  currentLine?: number;
}

export default function CallStack({ frames, currentLine: _currentLine }: CallStackProps) {
  const [expandedFrame, setExpandedFrame] = useState<number | null>(
    frames.length > 0 ? frames.length - 1 : null
  );

  if (frames.length === 0) {
    return (
      <div className="text-sm text-slate-500 dark:text-slate-400 italic p-4 bg-slate-50 dark:bg-slate-800 rounded-xl flex items-center gap-2">
        <GitBranch className="w-4 h-4" />
        호출 스택 비어있음
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-sm">
      <div className="px-5 py-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 border-b border-orange-200 dark:border-orange-800 flex items-center gap-2">
        <Layers className="w-4 h-4 text-orange-600 dark:text-orange-400" />
        <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">호출 스택</span>
        <span className="px-2 py-0.5 bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-400 text-xs rounded-lg font-medium">{frames.length}개</span>
      </div>

      <div className="divide-y divide-slate-100 dark:divide-slate-700">
        {frames
          .slice()
          .reverse()
          .map((frame, idx) => {
            const originalIdx = frames.length - 1 - idx;
            const isExpanded = expandedFrame === originalIdx;
            const isCurrent = originalIdx === frames.length - 1;

            return (
              <div
                key={`${frame.function_name}-${originalIdx}`}
                className={`${isCurrent ? 'bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-900/10 dark:to-violet-900/10' : ''}`}
              >
                <button
                  onClick={() => setExpandedFrame(isExpanded ? null : originalIdx)}
                  className={`w-full px-5 py-3 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors ${
                    isCurrent ? 'hover:bg-purple-100/50 dark:hover:bg-purple-900/20' : ''
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                    )}
                    <span className="font-mono text-sm">
                      <span className={`font-semibold ${isCurrent ? 'text-purple-700 dark:text-purple-400' : 'text-slate-900 dark:text-slate-100'}`}>
                        {frame.function_name}
                      </span>
                      <span className="text-slate-400 dark:text-slate-500">()</span>
                    </span>
                    {isCurrent && (
                      <span className="px-2 py-0.5 bg-gradient-to-r from-purple-500 to-violet-500 text-white text-xs rounded-lg font-medium shadow-sm">
                        현재
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 dark:text-slate-400 font-mono px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded">
                    줄 {frame.line_number}
                  </span>
                </button>

                {isExpanded && frame.local_variables.length > 0 && (
                  <div className="px-5 pb-4 pl-12">
                    <VariableInspector
                      variables={frame.local_variables}
                      title="지역 변수"
                    />
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </div>
  );
}

// Compact call stack for inline display
interface CompactCallStackProps {
  frames: StackFrame[];
}

export function CompactCallStack({ frames }: CompactCallStackProps) {
  if (frames.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-400">
      <Layers className="w-4 h-4 text-orange-500" />
      {frames.map((frame, idx) => (
        <span key={idx} className="flex items-center">
          {idx > 0 && <span className="mx-1.5 text-slate-400 dark:text-slate-500">→</span>}
          <span className="font-mono px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded">{frame.function_name}</span>
        </span>
      ))}
    </div>
  );
}
