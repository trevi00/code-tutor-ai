/**
 * Gamification API client
 */

import api from './client';

// Types
export type BadgeRarity = 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
export type BadgeCategory = 'problem_solving' | 'streak' | 'mastery' | 'social' | 'special';
export type ChallengeType = 'daily' | 'weekly' | 'monthly' | 'special';
export type ChallengeStatus = 'active' | 'completed' | 'expired';

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: BadgeRarity;
  category: BadgeCategory;
  requirement: string;
  requirement_value: number;
  xp_reward: number;
}

export interface UserBadge {
  id: string;
  badge: Badge;
  earned_at: string;
}

export interface UserBadgesResponse {
  earned: UserBadge[];
  available: Badge[];
  total_earned: number;
  total_available: number;
}

export interface UserStats {
  total_xp: number;
  level: number;
  level_title: string;
  xp_progress: number;
  xp_for_next_level: number;
  xp_percentage: number;
  current_streak: number;
  longest_streak: number;
  problems_solved: number;
  problems_solved_first_try: number;
  patterns_mastered: number;
  collaborations_count: number;
  playgrounds_created: number;
  playgrounds_shared: number;
}

export interface XPAddedResponse {
  xp_added: number;
  total_xp: number;
  level: number;
  level_title: string;
  leveled_up: boolean;
  new_badges: Badge[];
}

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  username: string;
  total_xp: number;
  level: number;
  level_title: string;
  problems_solved: number;
  current_streak: number;
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[];
  period: string;
  total_users: number;
  user_rank: number | null;
}

export interface Challenge {
  id: string;
  name: string;
  description: string;
  challenge_type: ChallengeType;
  target_action: string;
  target_value: number;
  xp_reward: number;
  start_date: string;
  end_date: string;
  time_remaining: string | null;
}

export interface UserChallenge {
  id: string;
  challenge: Challenge;
  current_progress: number;
  status: ChallengeStatus;
  progress_percentage: number;
  started_at: string;
  completed_at: string | null;
}

export interface ChallengesResponse {
  active: UserChallenge[];
  completed: UserChallenge[];
  available: Challenge[];
}

export interface GamificationOverview {
  stats: UserStats;
  recent_badges: UserBadge[];
  active_challenges: UserChallenge[];
  leaderboard_rank: number;
  next_badge_progress: {
    badge: string;
    current: number;
    required: number;
    percentage: number;
  } | null;
}

// API functions
export async function getGamificationOverview(): Promise<GamificationOverview> {
  const response = await api.get<{ data: GamificationOverview }>('/gamification/overview');
  return response.data.data;
}

export async function getAllBadges(): Promise<Badge[]> {
  const response = await api.get<{ data: Badge[] }>('/gamification/badges');
  return response.data.data;
}

export async function getMyBadges(): Promise<UserBadgesResponse> {
  const response = await api.get<{ data: UserBadgesResponse }>('/gamification/badges/me');
  return response.data.data;
}

export async function checkBadges(): Promise<Badge[]> {
  const response = await api.post<{ data: Badge[] }>('/gamification/badges/check');
  return response.data.data;
}

export async function getMyStats(): Promise<UserStats> {
  const response = await api.get<{ data: UserStats }>('/gamification/stats');
  return response.data.data;
}

export async function recordActivity(action: string): Promise<XPAddedResponse> {
  const response = await api.post<{ data: XPAddedResponse }>(`/gamification/activity/${action}`);
  return response.data.data;
}

export async function getLeaderboard(
  period: 'all' | 'weekly' | 'monthly' = 'all',
  limit: number = 100,
  offset: number = 0
): Promise<LeaderboardResponse> {
  const response = await api.get<{ data: LeaderboardResponse }>('/gamification/leaderboard', {
    params: { period, limit, offset },
  });
  return response.data.data;
}

export async function getMyChallenges(): Promise<ChallengesResponse> {
  const response = await api.get<{ data: ChallengesResponse }>('/gamification/challenges');
  return response.data.data;
}

export async function joinChallenge(challengeId: string): Promise<UserChallenge> {
  const response = await api.post<{ data: UserChallenge }>(`/gamification/challenges/${challengeId}/join`);
  return response.data.data;
}

// Rarity colors
export const RARITY_COLORS: Record<BadgeRarity, { bg: string; text: string; border: string }> = {
  common: { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-300' },
  uncommon: { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-300' },
  rare: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-300' },
  epic: { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-300' },
  legendary: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-300' },
};

export const RARITY_LABELS: Record<BadgeRarity, string> = {
  common: '일반',
  uncommon: '희귀',
  rare: '레어',
  epic: '에픽',
  legendary: '전설',
};
