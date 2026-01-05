/**
 * Playground API client
 */

import { apiClient } from './client';

// Helper to unwrap success_response format: { success, data, meta? }
function unwrapResponse<T>(response: T | { success: boolean; data: T; meta?: unknown }): T {
  if (response && typeof response === 'object' && 'success' in response && 'data' in response) {
    return (response as { success: boolean; data: T }).data;
  }
  return response as T;
}

// === Types ===

export interface PlaygroundResponse {
  id: string;
  owner_id: string;
  title: string;
  description: string;
  language: string;
  visibility: 'private' | 'unlisted' | 'public';
  share_code: string;
  is_forked: boolean;
  forked_from_id: string | null;
  run_count: number;
  fork_count: number;
  created_at: string;
  updated_at: string;
}

export interface PlaygroundDetailResponse extends PlaygroundResponse {
  code: string;
  stdin: string;
}

export interface PlaygroundListResponse {
  playgrounds: PlaygroundResponse[];
  total: number;
}

export interface CreatePlaygroundRequest {
  title: string;
  description?: string;
  code?: string;
  language?: string;
  visibility?: 'private' | 'unlisted' | 'public';
  stdin?: string;
}

export interface UpdatePlaygroundRequest {
  title?: string;
  description?: string;
  code?: string;
  language?: string;
  visibility?: 'private' | 'unlisted' | 'public';
  stdin?: string;
}

export interface ExecutePlaygroundRequest {
  code?: string;
  stdin?: string;
  timeout_seconds?: number;
}

export interface ExecutionResponse {
  execution_id: string;
  status: string;
  stdout: string;
  stderr: string;
  exit_code: number;
  execution_time_ms: number;
  is_success: boolean;
}

export interface LanguageInfo {
  id: string;
  display_name: string;
  extension: string;
}

export interface LanguagesResponse {
  languages: LanguageInfo[];
}

export interface TemplateResponse {
  id: string;
  title: string;
  description: string;
  code: string;
  language: string;
  category: string;
  tags: string[];
  usage_count: number;
}

export interface TemplateListResponse {
  templates: TemplateResponse[];
  total: number;
}

// === API Functions ===

export async function createPlayground(
  request: CreatePlaygroundRequest
): Promise<PlaygroundDetailResponse> {
  const response = await apiClient.post('/playground', request);
  return response.data;
}

export async function getPlayground(
  playgroundId: string
): Promise<PlaygroundDetailResponse> {
  const response = await apiClient.get(`/playground/${playgroundId}`);
  return response.data;
}

export async function getPlaygroundByShareCode(
  shareCode: string
): Promise<PlaygroundDetailResponse> {
  const response = await apiClient.get(`/playground/share/${shareCode}`);
  return response.data;
}

export async function updatePlayground(
  playgroundId: string,
  request: UpdatePlaygroundRequest
): Promise<PlaygroundDetailResponse> {
  const response = await apiClient.put(`/playground/${playgroundId}`, request);
  return response.data;
}

export async function deletePlayground(playgroundId: string): Promise<void> {
  await apiClient.delete(`/playground/${playgroundId}`);
}

export async function executePlayground(
  playgroundId: string,
  request: ExecutePlaygroundRequest
): Promise<ExecutionResponse> {
  const response = await apiClient.post<ExecutionResponse | { success: boolean; data: ExecutionResponse }>(
    `/playground/${playgroundId}/execute`,
    request
  );
  return unwrapResponse(response.data);
}

export async function forkPlayground(
  playgroundId: string,
  title?: string
): Promise<PlaygroundDetailResponse> {
  const response = await apiClient.post(`/playground/${playgroundId}/fork`, {
    title,
  });
  return response.data;
}

export async function regenerateShareCode(
  playgroundId: string
): Promise<{ share_code: string }> {
  const response = await apiClient.post(
    `/playground/${playgroundId}/regenerate-share-code`
  );
  return response.data;
}

export async function listMyPlaygrounds(
  limit: number = 20,
  offset: number = 0
): Promise<PlaygroundListResponse> {
  const response = await apiClient.get('/playground/mine', {
    params: { limit, offset },
  });
  return response.data;
}

export async function listPublicPlaygrounds(
  language?: string,
  limit: number = 20,
  offset: number = 0
): Promise<PlaygroundListResponse> {
  const response = await apiClient.get('/playground/public', {
    params: { language, limit, offset },
  });
  return response.data;
}

export async function listPopularPlaygrounds(
  limit: number = 10
): Promise<PlaygroundListResponse> {
  const response = await apiClient.get('/playground/popular', {
    params: { limit },
  });
  return response.data;
}

export async function searchPlaygrounds(
  query: string,
  language?: string,
  limit: number = 20
): Promise<PlaygroundListResponse> {
  const response = await apiClient.get('/playground/search', {
    params: { q: query, language, limit },
  });
  return response.data;
}

export async function getLanguages(): Promise<LanguagesResponse> {
  const response = await apiClient.get('/playground/languages');
  return response.data;
}

export async function getDefaultCode(
  language: string
): Promise<{ language: string; code: string }> {
  const response = await apiClient.get('/playground/default-code', {
    params: { language },
  });
  return response.data;
}

export async function listTemplates(
  category?: string,
  language?: string
): Promise<TemplateListResponse> {
  const response = await apiClient.get('/playground/templates/list', {
    params: { category, language },
  });
  return response.data;
}

export async function getTemplate(templateId: string): Promise<TemplateResponse> {
  const response = await apiClient.get(`/playground/templates/${templateId}`);
  return response.data;
}

export async function listPopularTemplates(
  limit: number = 10
): Promise<TemplateListResponse> {
  const response = await apiClient.get('/playground/templates/popular', {
    params: { limit },
  });
  return response.data;
}
