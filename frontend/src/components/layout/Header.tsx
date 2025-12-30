import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BookOpen, MessageSquare, LayoutDashboard, User, LogOut, Settings, FileText, ChevronDown, Sparkles } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function Header() {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuthStore();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
              <span>문제</span>
            </Link>
            <Link
              to="/patterns"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <Sparkles className="h-5 w-5" />
              <span>패턴</span>
            </Link>
            <Link
              to="/chat"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <MessageSquare className="h-5 w-5" />
              <span>AI 튜터</span>
            </Link>
            <Link
              to="/dashboard"
              className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              <LayoutDashboard className="h-5 w-5" />
              <span>대시보드</span>
            </Link>

            {/* User Menu Dropdown */}
            <div className="relative ml-4 pl-4 border-l border-neutral-200" ref={menuRef}>
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 text-neutral-600 hover:text-blue-600 transition-colors"
              >
                <User className="h-5 w-5" />
                <span>{user?.username}</span>
                <ChevronDown className={`h-4 w-4 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {showUserMenu && (
                <div className="absolute right-0 mt-2 w-48 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 py-1">
                  <Link
                    to="/profile"
                    className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <User className="h-4 w-4" />
                    <span>내 프로필</span>
                  </Link>
                  <Link
                    to="/submissions"
                    className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <FileText className="h-4 w-4" />
                    <span>제출 기록</span>
                  </Link>
                  <Link
                    to="/settings"
                    className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <Settings className="h-4 w-4" />
                    <span>설정</span>
                  </Link>
                  <hr className="my-1 border-gray-200" />
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      handleLogout();
                    }}
                    className="flex w-full items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>로그아웃</span>
                  </button>
                </div>
              )}
            </div>
          </nav>
        ) : (
          <nav className="flex items-center gap-4">
            <Link
              to="/login"
              className="px-4 py-2 text-neutral-600 hover:text-blue-600 transition-colors"
            >
              로그인
            </Link>
            <Link
              to="/register"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              회원가입
            </Link>
          </nav>
        )}
      </div>
    </header>
  );
}
