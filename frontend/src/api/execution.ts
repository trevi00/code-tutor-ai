import apiClient from './client';
import type { ExecuteCodeRequest, ExecuteCodeResponse, Submission } from '@/types';

export interface SubmitCodeRequest {
  problem_id: string;
  code: string;
  language?: string;
}

export const executionApi = {
  execute: async (data: ExecuteCodeRequest): Promise<ExecuteCodeResponse> => {
    const response = await apiClient.post<ExecuteCodeResponse>('/execute/run', data);
    return response.data;
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
