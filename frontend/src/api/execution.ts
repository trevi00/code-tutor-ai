import apiClient from './client';
import type { ExecuteCodeRequest, ExecuteCodeResponse, Submission } from '@/types';

export interface SubmitCodeRequest {
  problem_id: string;
  code: string;
  language?: string;
}

// Helper to unwrap success_response format: { success, data, meta? }
function unwrapResponse<T>(response: T | { success: boolean; data: T; meta?: unknown }): T {
  if (response && typeof response === 'object' && 'success' in response && 'data' in response) {
    return (response as { success: boolean; data: T }).data;
  }
  return response as T;
}

export const executionApi = {
  execute: async (data: ExecuteCodeRequest): Promise<ExecuteCodeResponse> => {
    const response = await apiClient.post<ExecuteCodeResponse | { success: boolean; data: ExecuteCodeResponse }>('/execute/run', data);
    return unwrapResponse(response.data);
  },

  submit: async (data: SubmitCodeRequest): Promise<Submission> => {
    // Use /submit endpoint for immediate evaluation
    const response = await apiClient.post<Submission>('/submit', data);
    return response.data;
  },

  getSubmission: async (id: string): Promise<Submission> => {
    const response = await apiClient.get<Submission>(`/submissions/${id}`);
    return response.data;
  },

  listSubmissions: async (problemId?: string): Promise<Submission[]> => {
    const params = problemId ? { problem_id: problemId } : {};
    const response = await apiClient.get<Submission[]>('/submissions', { params });
    return response.data;
  },
};
