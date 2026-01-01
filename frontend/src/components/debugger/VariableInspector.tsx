/**
 * Variable Inspector Component
 */

import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import type { Variable, VariableType } from '../../api/debugger';
import { VARIABLE_TYPE_COLORS } from '../../api/debugger';

interface VariableInspectorProps {
  variables: Variable[];
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
      <div className="text-sm text-gray-500 italic p-2">변수 없음</div>
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
    <div className="bg-white rounded-lg border border-gray-200">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between bg-gray-50 rounded-t-lg hover:bg-gray-100"
      >
        <span className="font-medium text-gray-700 text-sm">{title}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">{variables.length}개</span>
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="divide-y divide-gray-100">
          {variables.map((v) => (
            <VariableRow key={v.name} variable={v} />
          ))}
        </div>
      )}
    </div>
  );
}

interface VariableRowProps {
  variable: Variable;
}

function VariableRow({ variable }: VariableRowProps) {
  const [showFull, setShowFull] = useState(false);
  const isLongValue = variable.value.length > 50;

  return (
    <div className="px-3 py-2 hover:bg-gray-50">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-sm font-medium text-gray-900">
            {variable.name}
          </span>
          <TypeBadge type={variable.type} />
        </div>
      </div>
      <div className="mt-1">
        <code
          className={`text-sm font-mono ${VARIABLE_TYPE_COLORS[variable.type]} break-all`}
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
            className="ml-2 text-xs text-blue-600 hover:underline"
          >
            {showFull ? '접기' : '더보기'}
          </button>
        )}
      </div>
    </div>
  );
}

interface VariableBadgeProps {
  variable: Variable;
}

function VariableBadge({ variable }: VariableBadgeProps) {
  const shortValue =
    variable.value.length > 20
      ? variable.value.slice(0, 20) + '...'
      : variable.value;

  return (
    <div className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 rounded text-sm">
      <span className="font-mono font-medium text-gray-900">{variable.name}</span>
      <span className="text-gray-400">=</span>
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

  return (
    <span className="px-1.5 py-0.5 bg-gray-200 rounded text-xs text-gray-600">
      {typeLabels[type]}
    </span>
  );
}

export { VariableRow, VariableBadge, TypeBadge };
