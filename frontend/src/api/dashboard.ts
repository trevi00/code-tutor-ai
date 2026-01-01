import type { ApiResponse, DashboardData, InsightsData, PredictionData, SubmissionSummary } from '@/types';
import apiClient from './client';

/**
 * Get user dashboard data
 */
export const getDashboard = async (): Promise<DashboardData> => {
  const response = await apiClient.get<ApiResponse<DashboardData>>('/dashboard');
  return response.data.data;
};

/**
 * Get learning predictions
 */
export const getPrediction = async (): Promise<PredictionData> => {
  const response = await apiClient.get<ApiResponse<PredictionData>>('/dashboard/prediction');
  return response.data.data;
};

/**
 * Get AI-powered learning insights (LSTM predictions)
 */
export const getInsights = async (): Promise<InsightsData> => {
  const response = await apiClient.get<ApiResponse<InsightsData>>('/dashboard/insights');
  return response.data.data;
};

/**
 * Get user submissions
 */
export const getSubmissions = async (
  limit: number = 20,
  offset: number = 0
): Promise<SubmissionSummary[]> => {
  const response = await apiClient.get<SubmissionSummary[]>('/submissions', {
    params: { limit, offset },
  });
  return response.data;
};
