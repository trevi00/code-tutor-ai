/**
 * Call Stack Component
 */

import { ChevronDown, ChevronRight, Layers } from 'lucide-react';
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
      <div className="text-sm text-gray-500 italic p-2">호출 스택 비어있음</div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="px-3 py-2 bg-gray-50 rounded-t-lg border-b flex items-center gap-2">
        <Layers className="w-4 h-4 text-gray-500" />
        <span className="font-medium text-gray-700 text-sm">호출 스택</span>
        <span className="text-xs text-gray-500">({frames.length}개)</span>
      </div>

      <div className="divide-y divide-gray-100">
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
                className={`${isCurrent ? 'bg-blue-50' : ''}`}
              >
                <button
                  onClick={() => setExpandedFrame(isExpanded ? null : originalIdx)}
                  className={`w-full px-3 py-2 flex items-center justify-between hover:bg-gray-50 ${
                    isCurrent ? 'hover:bg-blue-100' : ''
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-gray-400" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                    )}
                    <span className="font-mono text-sm">
                      <span className={`font-medium ${isCurrent ? 'text-blue-700' : 'text-gray-900'}`}>
                        {frame.function_name}
                      </span>
                      <span className="text-gray-400">()</span>
                    </span>
                    {isCurrent && (
                      <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                        현재
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500">
                    줄 {frame.line_number}
                  </span>
                </button>

                {isExpanded && frame.local_variables.length > 0 && (
                  <div className="px-3 pb-3 pl-8">
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
    <div className="flex items-center gap-1 text-sm text-gray-600">
      {frames.map((frame, idx) => (
        <span key={idx} className="flex items-center">
          {idx > 0 && <span className="mx-1 text-gray-400">→</span>}
          <span className="font-mono">{frame.function_name}</span>
        </span>
      ))}
    </div>
  );
}
