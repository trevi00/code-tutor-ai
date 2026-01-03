/**
 * Typing Practice API client
 */
import { apiClient } from './client';

// Types
export interface TypingExercise {
  id: string;
  title: string;
  source_code: string;
  language: string;
  category: 'template' | 'method' | 'algorithm' | 'pattern';
  difficulty: 'easy' | 'medium' | 'hard';
  description: string;
  required_completions: number;
  char_count: number;
  line_count: number;
  created_at: string;
}

export interface TypingExerciseListResponse {
  exercises: TypingExercise[];
  total: number;
  page: number;
  page_size: number;
}

export interface TypingAttempt {
  id: string;
  user_id: string;
  exercise_id: string;
  attempt_number: number;
  accuracy: number;
  wpm: number;
  time_seconds: number;
  status: 'in_progress' | 'completed' | 'abandoned';
  started_at: string;
  completed_at: string | null;
}

export interface UserProgress {
  user_id: string;
  exercise_id: string;
  completed_attempts: number;
  required_completions: number;
  best_accuracy: number;
  best_wpm: number;
  total_time_seconds: number;
  is_mastered: boolean;
  attempts: TypingAttempt[];
}

export interface UserTypingStats {
  total_exercises_attempted: number;
  total_exercises_mastered: number;
  total_attempts: number;
  average_accuracy: number;
  average_wpm: number;
  total_time_seconds: number;
  best_wpm: number;
}

export interface StartAttemptRequest {
  exercise_id: string;
}

export interface CompleteAttemptRequest {
  user_code: string;
  accuracy: number;
  wpm: number;
  time_seconds: number;
}

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  username: string;
  best_wpm: number;
  average_accuracy: number;
  exercises_mastered: number;
}

// API functions
export const typingPracticeApi = {
  // Get list of exercises
  listExercises: async (params?: {
    category?: string;
    page?: number;
    page_size?: number;
  }): Promise<TypingExerciseListResponse> => {
    const response = await apiClient.get('/typing-practice/exercises', { params });
    return response.data;
  },

  // Get single exercise
  getExercise: async (exerciseId: string): Promise<TypingExercise> => {
    const response = await apiClient.get(`/typing-practice/exercises/${exerciseId}`);
    return response.data;
  },

  // Get user progress on an exercise
  getProgress: async (exerciseId: string): Promise<UserProgress> => {
    const response = await apiClient.get(`/typing-practice/exercises/${exerciseId}/progress`);
    return response.data;
  },

  // Start a new attempt
  startAttempt: async (request: StartAttemptRequest): Promise<TypingAttempt> => {
    const response = await apiClient.post('/typing-practice/attempts', request);
    return response.data;
  },

  // Complete an attempt
  completeAttempt: async (
    attemptId: string,
    request: CompleteAttemptRequest
  ): Promise<TypingAttempt> => {
    const response = await apiClient.post(
      `/typing-practice/attempts/${attemptId}/complete`,
      request
    );
    return response.data;
  },

  // Get user stats
  getStats: async (): Promise<UserTypingStats> => {
    const response = await apiClient.get('/typing-practice/stats');
    return response.data;
  },

  // Get leaderboard
  getLeaderboard: async (limit = 10): Promise<{ entries: LeaderboardEntry[] }> => {
    const response = await apiClient.get('/typing-practice/leaderboard', {
      params: { limit },
    });
    return response.data;
  },
};
