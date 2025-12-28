import apiClient from './client';
import type { Problem, ProblemSummary, PaginatedResponse } from '@/types';

export interface ListProblemsParams {
  page?: number;
  size?: number;
  difficulty?: string;
  category?: string;
  search?: string;
}

export const problemsApi = {
  list: async (params: ListProblemsParams = {}): Promise<PaginatedResponse<ProblemSummary>> => {
    const response = await apiClient.get<PaginatedResponse<ProblemSummary>>('/problems', {
      params,
    });
    return response.data;
  },

  get: async (id: string): Promise<Problem> => {
    const response = await apiClient.get<Problem>(`/problems/${id}`);
    return response.data;
  },
};
