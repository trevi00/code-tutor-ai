/**
 * Performance Analysis API client
 */

import api from './client';

// Types
export type ComplexityClass =
  | 'O(1)'
  | 'O(log n)'
  | 'O(n)'
  | 'O(n log n)'
  | 'O(n²)'
  | 'O(n³)'
  | 'O(2^n)'
  | 'O(n!)'
  | 'Unknown';

export type AnalysisStatus = 'pending' | 'running' | 'completed' | 'error' | 'timeout';
export type IssueSeverity = 'info' | 'warning' | 'error' | 'critical';
export type IssueType =
  | 'nested_loop'
  | 'inefficient_algorithm'
  | 'memory_leak'
  | 'excessive_recursion'
  | 'unnecessary_computation'
  | 'large_data_structure'
  | 'string_concatenation'
  | 'global_variable';

export interface LoopInfo {
  line_number: number;
  loop_type: string;
  nesting_level: number;
  iteration_variable?: string;
  iterable?: string;
  estimated_iterations?: string;
}

export interface FunctionInfo {
  name: string;
  line_number: number;
  parameters: string[];
  is_recursive: boolean;
  calls_count: number;
  complexity: ComplexityClass;
}

export interface PerformanceIssue {
  issue_type: IssueType;
  severity: IssueSeverity;
  line_number: number;
  message: string;
  suggestion: string;
  code_snippet?: string;
}

export interface ComplexityResult {
  time_complexity: ComplexityClass;
  space_complexity: ComplexityClass;
  time_explanation: string;
  space_explanation: string;
  max_nesting_depth: number;
  loops: LoopInfo[];
  functions: FunctionInfo[];
  recursive_functions: string[];
}

export interface RuntimeMetrics {
  execution_time_ms: number;
  cpu_time_ms: number;
  function_calls: number;
  line_executions: number;
  peak_call_depth: number;
}

export interface MemoryMetrics {
  peak_memory_mb: number;
  average_memory_mb: number;
  allocations_count: number;
  deallocations_count: number;
  largest_object_mb: number;
  largest_object_type?: string;
}

export interface FunctionProfile {
  name: string;
  calls: number;
  total_time_ms: number;
  own_time_ms: number;
  avg_time_ms: number;
  percentage: number;
}

export interface HotspotResult {
  hotspot_functions: FunctionProfile[];
  total_execution_time_ms: number;
  bottleneck_function?: string;
  bottleneck_line?: number;
}

export interface PerformanceResult {
  status: AnalysisStatus;
  complexity?: ComplexityResult;
  runtime?: RuntimeMetrics;
  memory?: MemoryMetrics;
  hotspots?: HotspotResult;
  issues: PerformanceIssue[];
  optimization_score: number;
  error?: string;
}

export interface QuickAnalyzeResult {
  status: AnalysisStatus;
  time_complexity: ComplexityClass;
  space_complexity: ComplexityClass;
  time_explanation: string;
  space_explanation: string;
  max_nesting_depth: number;
  issues_count: number;
  error?: string;
}

export interface AnalyzeRequest {
  code: string;
  input_data?: string;
  include_runtime?: boolean;
  include_memory?: boolean;
}

export interface QuickAnalyzeRequest {
  code: string;
}

// API functions
export async function analyzePerformance(request: AnalyzeRequest): Promise<PerformanceResult> {
  const response = await api.post<{ data: PerformanceResult }>('/performance', request);
  return response.data.data;
}

export async function quickAnalyze(request: QuickAnalyzeRequest): Promise<QuickAnalyzeResult> {
  const response = await api.post<{ data: QuickAnalyzeResult }>('/performance/quick', request);
  return response.data.data;
}

// Complexity colors and labels
export const COMPLEXITY_COLORS: Record<ComplexityClass, string> = {
  'O(1)': 'text-green-600 bg-green-100',
  'O(log n)': 'text-green-500 bg-green-50',
  'O(n)': 'text-blue-600 bg-blue-100',
  'O(n log n)': 'text-blue-500 bg-blue-50',
  'O(n²)': 'text-yellow-600 bg-yellow-100',
  'O(n³)': 'text-orange-600 bg-orange-100',
  'O(2^n)': 'text-red-600 bg-red-100',
  'O(n!)': 'text-red-700 bg-red-200',
  'Unknown': 'text-gray-600 bg-gray-100',
};

export const SEVERITY_COLORS: Record<IssueSeverity, string> = {
  info: 'text-blue-600 bg-blue-100',
  warning: 'text-yellow-600 bg-yellow-100',
  error: 'text-red-600 bg-red-100',
  critical: 'text-red-700 bg-red-200',
};

export const ISSUE_TYPE_LABELS: Record<IssueType, string> = {
  nested_loop: '중첩 루프',
  inefficient_algorithm: '비효율적 알고리즘',
  memory_leak: '메모리 누수',
  excessive_recursion: '과도한 재귀',
  unnecessary_computation: '불필요한 연산',
  large_data_structure: '대용량 데이터 구조',
  string_concatenation: '문자열 연결',
  global_variable: '전역 변수',
};
