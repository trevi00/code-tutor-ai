/**
 * Visualization API client
 */

import { apiClient } from './client';

// === Types ===

export interface AlgorithmInfo {
  id: string;
  name: string;
  name_en: string;
  category: string;
  time_complexity: string;
  space_complexity: string;
  description: string;
}

export interface AlgorithmListResponse {
  algorithms: AlgorithmInfo[];
  total: number;
}

export interface VisualizationStep {
  step_number: number;
  action: string;
  indices: number[];
  values: number[];
  array_state: number[];
  element_states: string[];
  description: string;
  code_line: number | null;
  auxiliary_data: Record<string, unknown>;
}

export interface SortingVisualization {
  algorithm_type: string;
  category: string;
  initial_data: number[];
  steps: VisualizationStep[];
  final_data: number[];
  code: string;
  total_steps: number;
  total_comparisons: number;
  total_swaps: number;
}

export interface GraphNode {
  id: string;
  value: string;
  x: number;
  y: number;
  state: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  state: string;
}

export interface GraphStep {
  step_number: number;
  action: string;
  current_node: string | null;
  queue?: string[];
  stack?: string[];
  visited: string[];
  node_states: Record<string, string>;
  edge_states: Record<string, string>;
  description: string;
  exploring?: string;
  back_to?: string;
}

export interface GraphVisualization {
  algorithm_type: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  steps: GraphStep[];
  code: string;
  total_steps: number;
}

// === API Functions ===

export async function getAlgorithms(
  category?: string
): Promise<AlgorithmListResponse> {
  const response = await apiClient.get('/visualization/algorithms', {
    params: category ? { category } : undefined,
  });
  return response.data.data;
}

export async function getAlgorithmInfo(algorithmId: string): Promise<AlgorithmInfo> {
  const response = await apiClient.get(`/visualization/algorithms/${algorithmId}`);
  return response.data.data;
}

export async function getRandomArray(
  size: number = 10,
  minVal: number = 1,
  maxVal: number = 100
): Promise<{ array: number[]; size: number }> {
  const response = await apiClient.get('/visualization/random-array', {
    params: { size, min_val: minVal, max_val: maxVal },
  });
  return response.data.data;
}

export async function generateSortingVisualization(
  algorithm: string,
  data?: number[],
  size: number = 10
): Promise<SortingVisualization> {
  const response = await apiClient.post('/visualization/sorting', {
    algorithm,
    data,
    size,
  });
  return response.data.data;
}

export async function generateSearchVisualization(
  algorithm: string,
  data?: number[],
  target?: number,
  size: number = 10
): Promise<SortingVisualization> {
  const response = await apiClient.post('/visualization/searching', {
    algorithm,
    data,
    target,
    size,
  });
  return response.data.data;
}

export async function generateGraphVisualization(
  algorithm: string,
  startNode: string = 'A'
): Promise<GraphVisualization> {
  const response = await apiClient.post('/visualization/graph', {
    algorithm,
    start_node: startNode,
  });
  return response.data.data;
}

// Quick endpoints with GET
export async function getSortingVisualization(
  algorithmId: string,
  size: number = 10
): Promise<SortingVisualization> {
  const response = await apiClient.get(`/visualization/sorting/${algorithmId}`, {
    params: { size },
  });
  return response.data.data;
}

export async function getSearchVisualization(
  algorithmId: string,
  size: number = 10,
  target?: number
): Promise<SortingVisualization> {
  const response = await apiClient.get(`/visualization/searching/${algorithmId}`, {
    params: { size, target },
  });
  return response.data.data;
}

export async function getGraphVisualization(
  algorithmId: string,
  startNode: string = 'A'
): Promise<GraphVisualization> {
  const response = await apiClient.get(`/visualization/graph/${algorithmId}`, {
    params: { start_node: startNode },
  });
  return response.data.data;
}
