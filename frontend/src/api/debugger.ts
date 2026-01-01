/**
 * Debugger API client
 */

import api from './client';

// Types
export type StepType = 'call' | 'line' | 'return' | 'exception';
export type VariableType =
  | 'int'
  | 'float'
  | 'string'
  | 'boolean'
  | 'list'
  | 'dict'
  | 'tuple'
  | 'set'
  | 'none'
  | 'object'
  | 'function'
  | 'class';
export type DebugStatus = 'pending' | 'running' | 'paused' | 'completed' | 'error' | 'timeout';

export interface Variable {
  name: string;
  value: string;
  type: VariableType;
}

export interface StackFrame {
  function_name: string;
  filename: string;
  line_number: number;
  local_variables: Variable[];
}

export interface ExecutionStep {
  step_number: number;
  step_type: StepType;
  line_number: number;
  line_content: string;
  function_name: string;
  variables: Variable[];
  call_stack: StackFrame[];
  output: string;
  return_value: string | null;
  exception: string | null;
}

export interface DebugRequest {
  code: string;
  input_data?: string;
  breakpoints?: number[];
}

export interface DebugResponse {
  session_id: string;
  status: DebugStatus;
  total_steps: number;
  output: string;
  error: string | null;
  execution_time_ms: number;
  steps: ExecutionStep[];
}

export interface QuickDebugStep {
  step: number;
  line: number;
  code: string;
  type: StepType;
  vars: Record<string, string>;
}

export interface QuickDebugResponse {
  status: DebugStatus;
  total_steps: number;
  output: string;
  error: string | null;
  execution_time_ms: number;
  steps: QuickDebugStep[];
}

export interface StepInfoResponse {
  step: ExecutionStep;
  has_previous: boolean;
  has_next: boolean;
  is_breakpoint: boolean;
}

export interface DebugSummaryResponse {
  session_id: string;
  status: DebugStatus;
  total_steps: number;
  total_lines: number;
  functions_called: string[];
  variables_used: string[];
  has_error: boolean;
  error_line: number | null;
  execution_time_ms: number;
}

// API functions
export async function debugCode(request: DebugRequest): Promise<DebugResponse> {
  const response = await api.post<{ data: DebugResponse }>('/debugger', request);
  return response.data.data;
}

export async function quickDebug(request: DebugRequest): Promise<QuickDebugResponse> {
  const response = await api.post<{ data: QuickDebugResponse }>('/debugger/quick', request);
  return response.data.data;
}

export async function getDebugSession(sessionId: string): Promise<DebugResponse> {
  const response = await api.get<{ data: DebugResponse }>(`/debugger/${sessionId}`);
  return response.data.data;
}

export async function getStep(
  sessionId: string,
  stepNumber: number,
  breakpoints?: number[]
): Promise<StepInfoResponse> {
  const params = breakpoints ? { breakpoints: breakpoints.join(',') } : {};
  const response = await api.get<{ data: StepInfoResponse }>(
    `/debugger/${sessionId}/step/${stepNumber}`,
    { params }
  );
  return response.data.data;
}

export async function getDebugSummary(sessionId: string): Promise<DebugSummaryResponse> {
  const response = await api.get<{ data: DebugSummaryResponse }>(
    `/debugger/${sessionId}/summary`
  );
  return response.data.data;
}

// Variable type colors
export const VARIABLE_TYPE_COLORS: Record<VariableType, string> = {
  int: 'text-blue-600',
  float: 'text-blue-500',
  string: 'text-green-600',
  boolean: 'text-purple-600',
  list: 'text-orange-600',
  dict: 'text-yellow-600',
  tuple: 'text-pink-600',
  set: 'text-red-600',
  none: 'text-gray-500',
  object: 'text-indigo-600',
  function: 'text-cyan-600',
  class: 'text-teal-600',
};

// Step type icons/labels
export const STEP_TYPE_INFO: Record<StepType, { label: string; color: string }> = {
  call: { label: '함수 호출', color: 'bg-blue-100 text-blue-700' },
  line: { label: '실행', color: 'bg-gray-100 text-gray-700' },
  return: { label: '반환', color: 'bg-green-100 text-green-700' },
  exception: { label: '예외', color: 'bg-red-100 text-red-700' },
};
