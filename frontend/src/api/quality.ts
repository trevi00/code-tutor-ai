import type {
  ApiResponse,
  QualityAnalysis,
  QualityStats,
  QualityTrendPoint,
  RecentQualityAnalysis,
} from '@/types';
import apiClient from './client';

/**
 * Get quality analysis for a submission
 */
export const getSubmissionQuality = async (
  submissionId: string
): Promise<QualityAnalysis> => {
  const response = await apiClient.get<ApiResponse<QualityAnalysis>>(
    `/submissions/${submissionId}/quality`
  );
  return response.data.data;
};

/**
 * Get user quality statistics
 */
export const getQualityStats = async (): Promise<QualityStats> => {
  const response = await apiClient.get<ApiResponse<QualityStats>>(
    '/dashboard/quality'
  );
  return response.data.data;
};

/**
 * Get quality trends over time
 */
export const getQualityTrends = async (
  days: number = 30
): Promise<{ trends: QualityTrendPoint[]; days: number }> => {
  const response = await apiClient.get<
    ApiResponse<{ trends: QualityTrendPoint[]; days: number }>
  >('/dashboard/quality/trends', {
    params: { days },
  });
  return response.data.data;
};

/**
 * Get recent quality analyses
 */
export const getRecentQuality = async (
  limit: number = 10
): Promise<{ analyses: RecentQualityAnalysis[]; total: number }> => {
  const response = await apiClient.get<
    ApiResponse<{ analyses: RecentQualityAnalysis[]; total: number }>
  >('/dashboard/quality/recent', {
    params: { limit },
  });
  return response.data.data;
};
