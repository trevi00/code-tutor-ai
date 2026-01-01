/**
 * Collaboration API client
 */

import { apiClient } from './client';

// === Types ===

export interface Participant {
  id: string;
  user_id: string;
  username: string;
  cursor_line: number | null;
  cursor_column: number | null;
  selection_start_line: number | null;
  selection_start_column: number | null;
  selection_end_line: number | null;
  selection_end_column: number | null;
  is_active: boolean;
  color: string;
  joined_at: string;
}

export interface SessionResponse {
  id: string;
  problem_id: string | null;
  host_id: string;
  title: string;
  status: 'waiting' | 'active' | 'closed';
  language: string;
  version: number;
  participant_count: number;
  max_participants: number;
  created_at: string;
  updated_at: string;
}

export interface SessionDetailResponse extends SessionResponse {
  code_content: string;
  participants: Participant[];
}

export interface SessionListResponse {
  sessions: SessionResponse[];
  total: number;
}

export interface CreateSessionRequest {
  title: string;
  problem_id?: string;
  language?: string;
  max_participants?: number;
}

// === API Functions ===

export async function createSession(
  request: CreateSessionRequest
): Promise<SessionDetailResponse> {
  const response = await apiClient.post('/collaboration/sessions', request);
  return response.data;
}

export async function listSessions(
  activeOnly: boolean = true
): Promise<SessionListResponse> {
  const response = await apiClient.get('/collaboration/sessions', {
    params: { active_only: activeOnly },
  });
  return response.data;
}

export async function listActiveSessions(
  limit: number = 10
): Promise<SessionListResponse> {
  const response = await apiClient.get('/collaboration/sessions/active', {
    params: { limit },
  });
  return response.data;
}

export async function getSession(
  sessionId: string
): Promise<SessionDetailResponse> {
  const response = await apiClient.get(`/collaboration/sessions/${sessionId}`);
  return response.data;
}

export async function joinSession(
  sessionId: string
): Promise<SessionDetailResponse> {
  const response = await apiClient.post(`/collaboration/sessions/${sessionId}/join`);
  return response.data;
}

export async function leaveSession(sessionId: string): Promise<void> {
  await apiClient.post(`/collaboration/sessions/${sessionId}/leave`);
}

export async function closeSession(sessionId: string): Promise<void> {
  await apiClient.post(`/collaboration/sessions/${sessionId}/close`);
}

// === WebSocket Message Types ===

export const MessageType = {
  // Client -> Server
  JOIN: 'join',
  LEAVE: 'leave',
  CODE_CHANGE: 'code_change',
  CURSOR_MOVE: 'cursor_move',
  SELECTION_CHANGE: 'selection_change',
  CHAT: 'chat',

  // Server -> Client
  SESSION_STATE: 'session_state',
  USER_JOINED: 'user_joined',
  USER_LEFT: 'user_left',
  CODE_UPDATE: 'code_update',
  CURSOR_UPDATE: 'cursor_update',
  SELECTION_UPDATE: 'selection_update',
  CHAT_MESSAGE: 'chat_message',
  ERROR: 'error',
  SYNC: 'sync',
} as const;

export interface WebSocketMessage {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface SessionStateData {
  session_id: string;
  code_content: string;
  language: string;
  version: number;
  participants: Array<{
    user_id: string;
    username: string;
    color: string;
    cursor: { line: number; column: number } | null;
    selection: {
      start: { line: number; column: number };
      end: { line: number; column: number };
    } | null;
    is_active: boolean;
  }>;
}

export interface CodeUpdateData {
  user_id: string;
  username: string;
  operation_type: 'insert' | 'delete' | 'replace';
  position: number;
  content: string;
  length: number;
  version: number;
}

export interface CursorUpdateData {
  user_id: string;
  username: string;
  line: number;
  column: number;
  color: string;
}

export interface SelectionUpdateData {
  user_id: string;
  username: string;
  start_line: number;
  start_column: number;
  end_line: number;
  end_column: number;
  color: string;
}

export interface ChatMessageData {
  user_id: string;
  username: string;
  message: string;
  timestamp: string;
}

export interface UserJoinedData {
  user_id: string;
  username: string;
  color: string;
}

export interface UserLeftData {
  user_id: string;
  username: string;
}
