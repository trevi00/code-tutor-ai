// User & Auth types
export interface User {
  id: string;
  email: string;
  username: string;
  role: 'student' | 'admin';
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login_at?: string;
  bio?: string;
}

export interface UpdateProfileRequest {
  username?: string;
  bio?: string;
}

export interface ChangePasswordRequest {
  old_password: string;
  new_password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginResponse {
  user: User;
  tokens: AuthTokens;
}

// Problem types
export type Difficulty = 'easy' | 'medium' | 'hard';
export type Category =
  | 'array' | 'string' | 'hash_table' | 'linked_list'
  | 'stack' | 'queue' | 'tree' | 'graph'
  | 'dp' | 'greedy' | 'binary_search' | 'sorting'
  | 'math' | 'bit_manipulation' | 'recursion';

export interface TestCase {
  id: string;
  input_data: string;
  expected_output: string;
  is_sample: boolean;
}

export interface Problem {
  id: string;
  title: string;
  description: string;
  difficulty: Difficulty;
  category: Category;
  constraints: string;
  hints: string[];
  solution_template: string;
  reference_solution?: string;
  time_limit_ms: number;
  memory_limit_mb: number;
  is_published: boolean;
  test_cases: TestCase[];
  // Pattern-related fields
  pattern_ids: string[];
  pattern_explanation: string;
  approach_hint: string;
  time_complexity_hint: string;
  space_complexity_hint: string;
  created_at: string;
}

export interface ProblemSummary {
  id: string;
  title: string;
  difficulty: Difficulty;
  category: Category;
  is_published: boolean;
  pattern_ids: string[];
}

// Submission types
export type SubmissionStatus =
  | 'pending' | 'running' | 'accepted'
  | 'wrong_answer' | 'runtime_error'
  | 'time_limit_exceeded' | 'memory_limit_exceeded';

export interface TestResult {
  test_case_id: string;
  is_passed: boolean;
  actual_output: string;
  execution_time_ms: number;
  memory_usage_mb: number;
  error?: string;
}

export interface Submission {
  id: string;
  user_id: string;
  problem_id: string;
  code: string;
  language: string;
  status: SubmissionStatus;
  test_results: TestResult[];
  total_tests: number;
  passed_tests: number;
  execution_time_ms: number;
  memory_usage_mb: number;
  error_message?: string;
  submitted_at: string;
  evaluated_at?: string;
}

// Chat types
export type ConversationType = 'general' | 'problem_help' | 'code_review' | 'concept';
export type MessageRole = 'user' | 'assistant';

export interface CodeContext {
  code: string;
  language: string;
  problem_id?: string;
  submission_id?: string;
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  code_context?: CodeContext;
  tokens_used: number;
  created_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  problem_id?: string;
  conversation_type: ConversationType;
  title: string;
  messages: Message[];
  total_tokens: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  conversation_type?: ConversationType;
  problem_id?: string;
  code_context?: CodeContext;
}

export interface ChatResponse {
  conversation_id: string;
  message: Message;
  is_new_conversation: boolean;
}

// Code Execution types
export interface ExecuteCodeRequest {
  code: string;
  language: string;
  stdin?: string;
  problem_id?: string;
}

export type ExecutionStatus =
  | 'success' | 'runtime_error' | 'timeout'
  | 'memory_exceeded' | 'compilation_error';

export interface ExecuteCodeResponse {
  execution_id: string;
  status: ExecutionStatus;
  stdout: string;
  stderr: string;
  exit_code: number;
  execution_time_ms: number;
  memory_usage_mb?: number;
  error_message?: string;
}

// Dashboard types
export interface CategoryProgress {
  category: string;
  total_problems: number;
  solved_problems: number;
  success_rate: number;
}

export interface StreakInfo {
  current_streak: number;
  longest_streak: number;
  last_activity_date: string | null;
}

export interface UserStats {
  total_problems_attempted: number;
  total_problems_solved: number;
  total_submissions: number;
  overall_success_rate: number;
  easy_solved: number;
  medium_solved: number;
  hard_solved: number;
  streak: StreakInfo;
}

export interface RecentSubmission {
  id: string;
  problem_id: string;
  problem_title: string;
  status: string;
  submitted_at: string;
}

export interface HeatmapData {
  date: string;
  count: number;
  level: number;
}

export interface SkillPrediction {
  category: string;
  current_level: number;
  predicted_level: number;
  confidence: number;
  recommended_focus: boolean;
}

export interface DashboardData {
  stats: UserStats;
  category_progress: CategoryProgress[];
  recent_submissions: RecentSubmission[];
  heatmap?: HeatmapData[];
  skill_predictions?: SkillPrediction[];
}

// Prediction types
export interface PredictionInsight {
  type: string;
  message: string;
}

export interface PredictionRecommendation {
  type: string;
  message: string;
  problem_id: string | null;
  reason: string;
}

export interface PredictionData {
  current_success_rate: number;
  predicted_success_rate: number;
  prediction_period: string;
  confidence: number;
  insights: PredictionInsight[];
  recommendations: PredictionRecommendation[];
  model_version: string;
}

// Submission summary for list
export interface SubmissionSummary {
  id: string;
  problem_id: string;
  problem_title: string;
  status: SubmissionStatus;
  execution_time_ms: number;
  memory_usage_mb: number;
  submitted_at: string;
}

// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  meta: {
    request_id: string;
    timestamp: string;
  };
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}

// Pagination
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}
