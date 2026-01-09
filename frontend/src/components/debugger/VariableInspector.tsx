/**
 * Variable Inspector Component - Enhanced with modern design
 */

import { ChevronDown, ChevronRight, Variable } from 'lucide-react';
import { useState } from 'react';
import type { Variable as VariableData, VariableType } from '../../api/debugger';
import { VARIABLE_TYPE_COLORS } from '../../api/debugger';

interface VariableInspectorProps {
  variables: VariableData[];
  title?: string;
  compact?: boolean;
}

export default function VariableInspector({
  variables,
  title = '변수',
  compact = false,
}: VariableInspectorProps) {
  const [expanded, setExpanded] = useState(true);

  if (variables.length === 0) {
    return (
      <div className="text-sm text-slate-500 dark:text-slate-400 italic p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">변수 없음</div>
    );
  }

  if (compact) {
    return (
      <div className="flex flex-wrap gap-2">
        {variables.map((v) => (
          <VariableBadge key={v.name} variable={v} />
        ))}
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3 flex items-center justify-between bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 hover:from-blue-100 hover:to-cyan-100 dark:hover:from-blue-900/30 dark:hover:to-cyan-900/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Variable className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400 text-xs rounded-lg font-medium">{variables.length}개</span>
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-slate-500 dark:text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-500 dark:text-slate-400" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="divide-y divide-slate-100 dark:divide-slate-700">
          {variables.map((v) => (
            <VariableRow key={v.name} variable={v} />
          ))}
        </div>
      )}
    </div>
  );
}

interface VariableRowProps {
  variable: VariableData;
}

function VariableRow({ variable }: VariableRowProps) {
  const [showFull, setShowFull] = useState(false);
  const isLongValue = variable.value.length > 50;

  return (
    <div className="px-5 py-3 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-3 min-w-0">
          <span className="font-mono text-sm font-semibold text-slate-900 dark:text-slate-100">
            {variable.name}
          </span>
          <TypeBadge type={variable.type} />
        </div>
      </div>
      <div className="mt-2">
        <code
          className={`text-sm font-mono px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700/50 ${VARIABLE_TYPE_COLORS[variable.type]} break-all inline-block`}
          onClick={() => isLongValue && setShowFull(!showFull)}
          style={{ cursor: isLongValue ? 'pointer' : 'default' }}
        >
          {showFull || !isLongValue
            ? variable.value
            : variable.value.slice(0, 50) + '...'}
        </code>
        {isLongValue && (
          <button
            onClick={() => setShowFull(!showFull)}
            className="ml-2 text-xs text-purple-600 dark:text-purple-400 hover:underline font-medium"
          >
            {showFull ? '접기' : '더보기'}
          </button>
        )}
      </div>
    </div>
  );
}

interface VariableBadgeProps {
  variable: VariableData;
}

function VariableBadge({ variable }: VariableBadgeProps) {
  const shortValue =
    variable.value.length > 20
      ? variable.value.slice(0, 20) + '...'
      : variable.value;

  return (
    <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 rounded-xl text-sm border border-slate-200 dark:border-slate-600">
      <span className="font-mono font-semibold text-slate-900 dark:text-slate-100">{variable.name}</span>
      <span className="text-slate-400 dark:text-slate-500">=</span>
      <span className={`font-mono ${VARIABLE_TYPE_COLORS[variable.type]}`}>
        {shortValue}
      </span>
    </div>
  );
}

interface TypeBadgeProps {
  type: VariableType;
}

function TypeBadge({ type }: TypeBadgeProps) {
  const typeLabels: Record<VariableType, string> = {
    int: 'int',
    float: 'float',
    string: 'str',
    boolean: 'bool',
    list: 'list',
    dict: 'dict',
    tuple: 'tuple',
    set: 'set',
    none: 'None',
    object: 'obj',
    function: 'func',
    class: 'class',
  };

  const typeColors: Record<VariableType, string> = {
    int: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400',
    float: 'bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400',
    string: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400',
    boolean: 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400',
    list: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
    dict: 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400',
    tuple: 'bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-400',
    set: 'bg-pink-100 dark:bg-pink-900/30 text-pink-700 dark:text-pink-400',
    none: 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400',
    object: 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400',
    function: 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400',
    class: 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-400',
  };

  return (
    <span className={`px-2 py-0.5 rounded-lg text-xs font-medium ${typeColors[type]}`}>
      {typeLabels[type]}
    </span>
  );
}

export { VariableRow, VariableBadge, TypeBadge };
