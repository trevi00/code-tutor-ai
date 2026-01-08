import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import {
  BookOpen, MessageSquare, LayoutDashboard, User, LogOut, Settings,
  FileText, ChevronDown, Sparkles, Code2, PlayCircle, Trophy, Award,
  Bug, Activity, Keyboard, Map, Sun, Moon, Menu, GraduationCap,
  Dumbbell, BarChart3
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useTheme } from '@/hooks/useTheme';
import { NavDropdown } from '@/components/navigation/NavDropdown';
import { MobileDrawer } from '@/components/navigation/MobileDrawer';
import clsx from 'clsx';

// Navigation group configuration
const navGroups = {
  learning: {
    label: '학습',
    icon: GraduationCap,
    items: [
      { to: '/roadmap', icon: Map, label: '로드맵' },
      { to: '/problems', icon: BookOpen, label: '문제' },
      { to: '/patterns', icon: Sparkles, label: '패턴' },
    ],
  },
  practice: {
    label: '연습',
    icon: Dumbbell,
    items: [
      { to: '/playground', icon: Code2, label: '플레이그라운드' },
      { to: '/typing-practice', icon: Keyboard, label: '받아쓰기' },
      { to: '/visualization', icon: PlayCircle, label: '시각화' },
    ],
  },
  analysis: {
    label: '분석',
    icon: BarChart3,
    items: [
      { to: '/dashboard', icon: LayoutDashboard, label: '대시보드' },
      { to: '/performance', icon: Activity, label: '성능' },
      { to: '/debugger', icon: Bug, label: '디버거' },
    ],
  },
};

const communityItems = [
  { to: '/chat', icon: MessageSquare, label: 'AI 튜터' },
  { to: '/leaderboard', icon: Trophy, label: '리더보드' },
];

export function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, isAuthenticated, logout } = useAuthStore();
  const { isDark, toggleTheme } = useTheme();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
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

  // Close user menu on route change
  useEffect(() => {
    setShowUserMenu(false);
  }, [location.pathname]);

  const isActive = (path: string) => location.pathname === path;

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b border-neutral-200/50 dark:border-slate-700/50 glass">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 font-bold text-xl gradient-text">
            <BookOpen className="h-6 w-6" />
            <span className="hidden sm:inline">Code Tutor AI</span>
          </Link>

          {/* Desktop Navigation */}
          {isAuthenticated ? (
            <nav className="hidden lg:flex items-center gap-1">
              {/* Grouped Navigation */}
              <NavDropdown
                label={navGroups.learning.label}
                icon={navGroups.learning.icon}
                items={navGroups.learning.items}
                colorClass="hover:text-indigo-600"
              />
              <NavDropdown
                label={navGroups.practice.label}
                icon={navGroups.practice.icon}
                items={navGroups.practice.items}
                colorClass="hover:text-emerald-600"
              />
              <NavDropdown
                label={navGroups.analysis.label}
                icon={navGroups.analysis.icon}
                items={navGroups.analysis.items}
                colorClass="hover:text-orange-600"
              />

              {/* Community Links */}
              <div className="flex items-center gap-1 ml-1 pl-2 border-l border-neutral-200 dark:border-slate-700">
                {communityItems.map((item) => {
                  const ItemIcon = item.icon;
                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={clsx(
                        'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                        isActive(item.to)
                          ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30'
                          : 'text-neutral-600 dark:text-neutral-300 hover:text-blue-600 hover:bg-neutral-100 dark:hover:bg-slate-800'
                      )}
                    >
                      <ItemIcon className="h-4 w-4" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>

              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="ml-2 p-2 rounded-lg text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:text-neutral-200 dark:hover:bg-slate-800 transition-all duration-200"
                aria-label={isDark ? '라이트 모드로 전환' : '다크 모드로 전환'}
              >
                {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>

              {/* User Menu Dropdown */}
              <div className="relative ml-2 pl-3 border-l border-neutral-200 dark:border-slate-700" ref={menuRef}>
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                    showUserMenu
                      ? 'bg-neutral-100 dark:bg-slate-800 text-neutral-800 dark:text-neutral-100'
                      : 'text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-slate-800'
                  )}
                >
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white text-xs font-medium">
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <span className="max-w-[100px] truncate">{user?.username}</span>
                  <ChevronDown className={clsx(
                    'h-4 w-4 transition-transform duration-200',
                    showUserMenu && 'rotate-180'
                  )} />
                </button>

                {/* Dropdown Menu */}
                <div
                  className={clsx(
                    'absolute right-0 mt-2 w-52 rounded-xl bg-white dark:bg-slate-800 shadow-xl ring-1 ring-black/5 dark:ring-white/10 py-1.5 transition-all duration-200 origin-top-right z-50',
                    showUserMenu
                      ? 'opacity-100 scale-100 pointer-events-auto'
                      : 'opacity-0 scale-95 pointer-events-none'
                  )}
                >
                  {/* User Info */}
                  <div className="px-4 py-3 border-b border-neutral-100 dark:border-slate-700">
                    <p className="text-sm font-medium text-neutral-800 dark:text-neutral-100">{user?.username}</p>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate">{user?.email}</p>
                  </div>

                  <Link
                    to="/profile"
                    className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <User className="h-4 w-4" />
                    <span>내 프로필</span>
                  </Link>
                  <Link
                    to="/submissions"
                    className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <FileText className="h-4 w-4" />
                    <span>제출 기록</span>
                  </Link>
                  <Link
                    to="/badges"
                    className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <Award className="h-4 w-4" />
                    <span>내 배지</span>
                  </Link>
                  <Link
                    to="/settings"
                    className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <Settings className="h-4 w-4" />
                    <span>설정</span>
                  </Link>
                  <hr className="my-1.5 border-neutral-200 dark:border-slate-700" />
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      handleLogout();
                    }}
                    className="flex w-full items-center gap-2.5 px-4 py-2.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>로그아웃</span>
                  </button>
                </div>
              </div>
            </nav>
          ) : (
            <nav className="hidden sm:flex items-center gap-4">
              <Link
                to="/login"
                className="px-4 py-2 text-neutral-600 dark:text-neutral-300 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-200"
              >
                로그인
              </Link>
              <Link
                to="/register"
                className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300"
              >
                회원가입
              </Link>
            </nav>
          )}

          {/* Mobile Menu Button */}
          <div className="flex items-center gap-2 lg:hidden">
            {/* Theme Toggle (Mobile) */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:text-neutral-200 dark:hover:bg-slate-800 transition-all duration-200"
              aria-label={isDark ? '라이트 모드로 전환' : '다크 모드로 전환'}
            >
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>

            {isAuthenticated && (
              <button
                onClick={() => setShowMobileMenu(true)}
                className="p-2 rounded-lg text-neutral-600 hover:text-neutral-800 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:text-neutral-100 dark:hover:bg-slate-800 transition-all duration-200"
                aria-label="메뉴 열기"
              >
                <Menu className="h-6 w-6" />
              </button>
            )}

            {!isAuthenticated && (
              <div className="flex items-center gap-2 sm:hidden">
                <Link
                  to="/login"
                  className="px-3 py-1.5 text-sm text-neutral-600 dark:text-neutral-300"
                >
                  로그인
                </Link>
                <Link
                  to="/register"
                  className="px-3 py-1.5 text-sm bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg"
                >
                  회원가입
                </Link>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Mobile Drawer */}
      {isAuthenticated && (
        <MobileDrawer
          isOpen={showMobileMenu}
          onClose={() => setShowMobileMenu(false)}
        />
      )}
    </>
  );
}
