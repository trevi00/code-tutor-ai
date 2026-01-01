import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { MainLayout } from '@/components/layout/MainLayout';
import { HomePage } from '@/pages/HomePage';
import { LoginPage } from '@/pages/auth/LoginPage';
import { RegisterPage } from '@/pages/auth/RegisterPage';
import { ProblemsPage } from '@/pages/problems/ProblemsPage';
import { ProblemSolvePage } from '@/pages/problems/ProblemSolvePage';
import { PatternsPage } from '@/pages/patterns/PatternsPage';
import { PatternDetailPage } from '@/pages/patterns/PatternDetailPage';
import { ChatPage } from '@/pages/chat/ChatPage';
import DashboardPage from '@/pages/dashboard/DashboardPage';
import SubmissionsPage from '@/pages/dashboard/SubmissionsPage';
import ProfilePage from '@/pages/profile/ProfilePage';
import SettingsPage from '@/pages/settings/SettingsPage';
import SessionsPage from '@/pages/collaboration/SessionsPage';
import CollaborationPage from '@/pages/collaboration/CollaborationPage';
import { useAuthStore } from '@/store/authStore';

// Protected Route component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore();

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}
