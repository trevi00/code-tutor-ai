import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { MainLayout } from '@/components/layout/MainLayout';
import { HomePage } from '@/pages/HomePage';
import { LoginPage } from '@/pages/auth/LoginPage';
import { RegisterPage } from '@/pages/auth/RegisterPage';
import { ProblemsPage } from '@/pages/problems/ProblemsPage';
import { ProblemSolvePage } from '@/pages/problems/ProblemSolvePage';
import { ChatPage } from '@/pages/chat/ChatPage';
import DashboardPage from '@/pages/dashboard/DashboardPage';
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
                <div className="container mx-auto px-4 py-8">
                  <h1 className="text-2xl font-bold">Profile (Coming Soon)</h1>
                </div>
              </ProtectedRoute>
            }
          />
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
