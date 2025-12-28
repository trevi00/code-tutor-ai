import apiClient from './client';
import type { ChatRequest, ChatResponse, Conversation } from '@/types';

export const tutorApi = {
  chat: async (data: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/tutor/chat', data);
    return response.data;
  },

  getConversation: async (id: string): Promise<Conversation> => {
    const response = await apiClient.get<Conversation>(`/tutor/conversations/${id}`);
    return response.data;
  },

  listConversations: async (limit = 20, offset = 0): Promise<Conversation[]> => {
    const response = await apiClient.get<Conversation[]>('/tutor/conversations', {
      params: { limit, offset },
    });
    return response.data;
  },

  closeConversation: async (id: string): Promise<Conversation> => {
    const response = await apiClient.post<Conversation>(`/tutor/conversations/${id}/close`);
    return response.data;
  },
};
