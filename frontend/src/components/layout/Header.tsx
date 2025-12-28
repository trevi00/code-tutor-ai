import { Link, useNavigate } from 'react-router-dom';
import { BookOpen, MessageSquare, LayoutDashboard, User, LogOut } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function Header() {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuthStore();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-white/95 backdrop-blur">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 font-bold text-xl text-blue-600">
          <BookOpen className="h-6 w-6" />
          <span>Code Tutor AI</span>
        </Link>

        {/* Navigation */}
        {isAuthenticated ? (
          <nav className="flex items-center gap-6">
            <Link
              to="/problems"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <BookOpen className="h-5 w-5" />
              <span>Problems</span>
            </Link>
            <Link
              to="/chat"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <MessageSquare className="h-5 w-5" />
              <span>AI Tutor</span>
            </Link>
            <Link
              to="/dashboard"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <LayoutDashboard className="h-5 w-5" />
              <span>Dashboard</span>
            </Link>

            {/* User Menu */}
            <div className="flex items-center gap-4 ml-4 pl-4 border-l border-neutral-200">
              <Link
                to="/profile"
                className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
              >
                <User className="h-5 w-5" />
                <span>{user?.username}</span>
              </Link>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 text-neutral-600 hover:text-red-600 transition-colors"
              >
                <LogOut className="h-5 w-5" />
              </button>
            </div>
          </nav>
        ) : (
          <nav className="flex items-center gap-4">
            <Link
              to="/login"
              className="px-4 py-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              Login
            </Link>
            <Link
              to="/register"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Sign Up
            </Link>
          </nav>
        )}
      </div>
    </header>
  );
}
