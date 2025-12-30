import apiClient from './client';

export interface Pattern {
  id: string;
  name: string;
  name_ko: string;
  description: string;
  description_ko: string;
  time_complexity: string;
  space_complexity: string;
  use_cases: string[];
  keywords: string[];
  example_code?: string;
}

export interface PatternsResponse {
  patterns: Pattern[];
  total: number;
}

export interface PatternSearchResult {
  query: string;
  patterns: Pattern[];
  total: number;
}

export const patternsApi = {
  list: async (): Promise<PatternsResponse> => {
    const response = await apiClient.get<{ data: PatternsResponse }>('/patterns');
    return response.data.data;
  },

  get: async (id: string): Promise<Pattern> => {
    const response = await apiClient.get<{ data: Pattern }>(`/patterns/${id}`);
    return response.data.data;
  },

  search: async (query: string, topK: number = 3): Promise<PatternSearchResult> => {
    const response = await apiClient.post<{ data: PatternSearchResult }>('/patterns/search', null, {
      params: { query, top_k: topK },
    });
    return response.data.data;
  },
};
