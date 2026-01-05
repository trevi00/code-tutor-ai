/**
 * Learning Roadmap API client
 */
import { apiClient } from './client';

// Types
export type PathLevel = 'beginner' | 'elementary' | 'intermediate' | 'advanced';
export type LessonType = 'concept' | 'problem' | 'typing' | 'pattern' | 'quiz';
export type ProgressStatus = 'not_started' | 'in_progress' | 'completed';

export interface Lesson {
  id: string;
  module_id: string;
  title: string;
  description: string;
  lesson_type: LessonType;
  content: string;
  content_id: string | null;
  order: number;
  xp_reward: number;
  estimated_minutes: number;
  status: ProgressStatus | null;
  completed_at: string | null;
  score: number | null;
}

export interface Module {
  id: string;
  path_id: string;
  title: string;
  description: string;
  order: number;
  lesson_count: number;
  total_xp: number;
  estimated_minutes: number;
  lessons: Lesson[];
  completed_lessons: number;
  completion_rate: number;
}

export interface LearningPath {
  id: string;
  level: PathLevel;
  level_display: string;
  title: string;
  description: string;
  icon: string;
  order: number;
  estimated_hours: number;
  module_count: number;
  lesson_count: number;
  total_xp: number;
  prerequisites: string[];
  modules: Module[];
  status: ProgressStatus;
  completed_lessons: number;
  completion_rate: number;
  started_at: string | null;
  completed_at: string | null;
}

export interface LearningPathListResponse {
  items: LearningPath[];
  total: number;
}

export interface UserProgress {
  total_paths: number;
  completed_paths: number;
  in_progress_paths: number;
  total_lessons: number;
  completed_lessons: number;
  total_xp_earned: number;
  current_path: LearningPath | null;
  next_lesson: Lesson | null;
  paths: LearningPath[];
}

export interface PathProgress {
  path_id: string;
  status: ProgressStatus;
  started_at: string | null;
  completed_at: string | null;
  completed_lessons: number;
  total_lessons: number;
  completion_rate: number;
}

export interface LessonProgress {
  lesson_id: string;
  status: ProgressStatus;
  started_at: string | null;
  completed_at: string | null;
  score: number | null;
  attempts: number;
}

export interface CompleteLessonRequest {
  score?: number;
}

// API functions
export const roadmapApi = {
  // List all learning paths
  listPaths: async (): Promise<LearningPathListResponse> => {
    const response = await apiClient.get('/roadmap/paths');
    return response.data;
  },

  // Get a specific path with modules and lessons
  getPath: async (pathId: string): Promise<LearningPath> => {
    const response = await apiClient.get(`/roadmap/paths/${pathId}`);
    return response.data;
  },

  // Get path by level
  getPathByLevel: async (level: PathLevel): Promise<LearningPath> => {
    const response = await apiClient.get(`/roadmap/paths/level/${level}`);
    return response.data;
  },

  // Get modules for a path
  getPathModules: async (pathId: string): Promise<Module[]> => {
    const response = await apiClient.get(`/roadmap/paths/${pathId}/modules`);
    return response.data;
  },

  // Get a specific module
  getModule: async (moduleId: string): Promise<Module> => {
    const response = await apiClient.get(`/roadmap/modules/${moduleId}`);
    return response.data;
  },

  // Get lessons for a module
  getModuleLessons: async (moduleId: string): Promise<Lesson[]> => {
    const response = await apiClient.get(`/roadmap/modules/${moduleId}/lessons`);
    return response.data;
  },

  // Get a specific lesson
  getLesson: async (lessonId: string): Promise<Lesson> => {
    const response = await apiClient.get(`/roadmap/lessons/${lessonId}`);
    return response.data;
  },

  // Get user progress
  getProgress: async (): Promise<UserProgress> => {
    const response = await apiClient.get('/roadmap/progress');
    return response.data;
  },

  // Get progress on a specific path
  getPathProgress: async (pathId: string): Promise<PathProgress> => {
    const response = await apiClient.get(`/roadmap/progress/paths/${pathId}`);
    return response.data;
  },

  // Start a learning path
  startPath: async (pathId: string): Promise<PathProgress> => {
    const response = await apiClient.post(`/roadmap/paths/${pathId}/start`);
    return response.data;
  },

  // Complete a lesson
  completeLesson: async (
    lessonId: string,
    request: CompleteLessonRequest = {}
  ): Promise<LessonProgress> => {
    const response = await apiClient.post(
      `/roadmap/lessons/${lessonId}/complete`,
      request
    );
    return response.data;
  },

  // Get next lesson recommendation
  getNextLesson: async (pathId?: string): Promise<Lesson | null> => {
    const params = pathId ? { path_id: pathId } : {};
    const response = await apiClient.get('/roadmap/next-lesson', { params });
    return response.data;
  },
};
