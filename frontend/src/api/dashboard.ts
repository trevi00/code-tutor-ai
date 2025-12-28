import type { ApiResponse, DashboardData } from '@/types';
import apiClient from './client';

/**
 * Get user dashboard data
 */
export const getDashboard = async (): Promise<DashboardData> => {
  const response = await apiClient.get<ApiResponse<DashboardData>>('/dashboard');
  return response.data.data;
};
