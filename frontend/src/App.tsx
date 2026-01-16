import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Suspense, lazy, useEffect } from 'react';
import { MainLayout } from '@/components/layout/MainLayout';
import { useAuthStore } from '@/store/authStore';

// Loading fallback component
function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-neutral-50 dark:bg-slate-900">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto" />
        <p className="mt-4 text-neutral-500 dark:text-slate-400">로딩 중...</p>
      </div>
    </div>
  );
}

// Lazy load pages - Core pages (loaded early)
const HomePage = lazy(() => import('@/pages/HomePage').then(m => ({ default: m.HomePage })));
const LoginPage = lazy(() => import('@/pages/auth/LoginPage').then(m => ({ default: m.LoginPage })));
const RegisterPage = lazy(() => import('@/pages/auth/RegisterPage').then(m => ({ default: m.RegisterPage })));

// Lazy load pages - Problems & Patterns
const ProblemsPage = lazy(() => import('@/pages/problems/ProblemsPage').then(m => ({ default: m.ProblemsPage })));
const ProblemSolvePage = lazy(() => import('@/pages/problems/ProblemSolvePage').then(m => ({ default: m.ProblemSolvePage })));
const PatternsPage = lazy(() => import('@/pages/patterns/PatternsPage').then(m => ({ default: m.PatternsPage })));
const PatternDetailPage = lazy(() => import('@/pages/patterns/PatternDetailPage').then(m => ({ default: m.PatternDetailPage })));

// Lazy load pages - Chat (uses markdown/syntax highlighter)
const ChatPage = lazy(() => import('@/pages/chat/ChatPage').then(m => ({ default: m.ChatPage })));

// Lazy load pages - Dashboard (uses Recharts)
const DashboardPage = lazy(() => import('@/pages/dashboard/DashboardPage'));
const SubmissionsPage = lazy(() => import('@/pages/dashboard/SubmissionsPage'));

// Lazy load pages - Profile & Settings
const ProfilePage = lazy(() => import('@/pages/profile/ProfilePage'));
const SettingsPage = lazy(() => import('@/pages/settings/SettingsPage'));

// Lazy load pages - Collaboration (uses Monaco)
const SessionsPage = lazy(() => import('@/pages/collaboration/SessionsPage'));
const CollaborationPage = lazy(() => import('@/pages/collaboration/CollaborationPage'));

// Lazy load pages - Playground (uses Monaco - largest)
const PlaygroundListPage = lazy(() => import('@/pages/playground').then(m => ({ default: m.PlaygroundListPage })));
const PlaygroundEditorPage = lazy(() => import('@/pages/playground').then(m => ({ default: m.PlaygroundEditorPage })));
const SharedPlaygroundPage = lazy(() => import('@/pages/playground').then(m => ({ default: m.SharedPlaygroundPage })));

// Lazy load pages - Visualization
const VisualizationPage = lazy(() => import('@/pages/visualization').then(m => ({ default: m.VisualizationPage })));

// Lazy load pages - Debugger (uses Monaco)
const DebuggerPage = lazy(() => import('@/pages/debugger/DebuggerPage'));

// Lazy load pages - Performance
const PerformancePage = lazy(() => import('@/pages/performance/PerformancePage'));

// Lazy load pages - Gamification
const LeaderboardPage = lazy(() => import('@/pages/gamification').then(m => ({ default: m.LeaderboardPage })));
const BadgesPage = lazy(() => import('@/pages/gamification').then(m => ({ default: m.BadgesPage })));

// Lazy load pages - Typing Practice
const TypingPracticeListPage = lazy(() => import('@/pages/typing-practice/TypingPracticeListPage'));
const TypingExercisePage = lazy(() => import('@/pages/typing-practice/TypingExercisePage'));

// Lazy load pages - Roadmap
const RoadmapPage = lazy(() => import('@/pages/roadmap').then(m => ({ default: m.RoadmapPage })));
const PathDetailPage = lazy(() => import('@/pages/roadmap').then(m => ({ default: m.PathDetailPage })));
const LessonPage = lazy(() => import('@/pages/roadmap').then(m => ({ default: m.LessonPage })));

// Protected Route component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore();

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  if (isLoading) {
    return <PageLoader />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Routes with layout */}
          <Route element={<MainLayout />}>
            <Route path="/" element={<HomePage />} />

            {/* Protected routes */}
            <Route
              path="/problems"
              element={
                <ProtectedRoute>
                  <ProblemsPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/problems/:id/solve"
              element={
                <ProtectedRoute>
                  <ProblemSolvePage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/patterns"
              element={
                <ProtectedRoute>
                  <PatternsPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/patterns/:id"
              element={
                <ProtectedRoute>
                  <PatternDetailPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/chat"
              element={
                <ProtectedRoute>
                  <ChatPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <ProfilePage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/submissions"
              element={
                <ProtectedRoute>
                  <SubmissionsPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/settings"
              element={
                <ProtectedRoute>
                  <SettingsPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/collaboration"
              element={
                <ProtectedRoute>
                  <SessionsPage />
                </ProtectedRoute>
              }
            />

            {/* Playground routes */}
            <Route path="/playground" element={<PlaygroundListPage />} />
            <Route
              path="/playground/share/:shareCode"
              element={<SharedPlaygroundPage />}
            />
            <Route
              path="/playground/:playgroundId"
              element={<PlaygroundEditorPage />}
            />

            {/* Visualization route */}
            <Route path="/visualization" element={<VisualizationPage />} />

            {/* Debugger route */}
            <Route path="/debugger" element={<DebuggerPage />} />

            {/* Performance route */}
            <Route path="/performance" element={<PerformancePage />} />
            {/* Typing Practice */}
            <Route path="/typing-practice" element={<TypingPracticeListPage />} />
            <Route path="/typing-practice/:exerciseId" element={<TypingExercisePage />} />

            {/* Roadmap routes */}
            <Route path="/roadmap" element={<RoadmapPage />} />
            <Route path="/roadmap/:pathId" element={<PathDetailPage />} />
            <Route path="/roadmap/lesson/:lessonId" element={<LessonPage />} />

            {/* Gamification routes */}
            <Route
              path="/leaderboard"
              element={
                <ProtectedRoute>
                  <LeaderboardPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/badges"
              element={
                <ProtectedRoute>
                  <BadgesPage />
                </ProtectedRoute>
              }
            />
          </Route>

          {/* Full-screen collaboration page (no layout) */}
          <Route
            path="/collaboration/:sessionId"
            element={
              <ProtectedRoute>
                <CollaborationPage />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
