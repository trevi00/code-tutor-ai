import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  X, ChevronDown, BookOpen, MessageSquare, LayoutDashboard,
  User, LogOut, Settings, FileText, Sparkles, Code2, PlayCircle,
  Trophy, Award, Bug, Activity, Keyboard, Map, Sun, Moon,
  GraduationCap, Dumbbell, BarChart3
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useTheme } from '@/hooks/useTheme';
import clsx from 'clsx';

interface MobileDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

interface NavGroup {
  label: string;
  icon: React.ElementType;
  items: {
    to: string;
    icon: React.ElementType;
    label: string;
  }[];
}

const navGroups: NavGroup[] = [
  {
    label: '학습',
    icon: GraduationCap,
    items: [
      { to: '/roadmap', icon: Map, label: '로드맵' },
      { to: '/problems', icon: BookOpen, label: '문제' },
      { to: '/patterns', icon: Sparkles, label: '패턴' },
    ],
  },
  {
    label: '연습',
    icon: Dumbbell,
    items: [
      { to: '/playground', icon: Code2, label: '플레이그라운드' },
      { to: '/typing-practice', icon: Keyboard, label: '받아쓰기' },
      { to: '/visualization', icon: PlayCircle, label: '시각화' },
    ],
  },
  {
    label: '분석',
    icon: BarChart3,
    items: [
      { to: '/dashboard', icon: LayoutDashboard, label: '대시보드' },
      { to: '/performance', icon: Activity, label: '성능' },
      { to: '/debugger', icon: Bug, label: '디버거' },
    ],
  },
];

const communityItems = [
  { to: '/chat', icon: MessageSquare, label: 'AI 튜터' },
  { to: '/leaderboard', icon: Trophy, label: '리더보드' },
];

const userMenuItems = [
  { to: '/profile', icon: User, label: '내 프로필' },
  { to: '/submissions', icon: FileText, label: '제출 기록' },
  { to: '/badges', icon: Award, label: '내 배지' },
  { to: '/settings', icon: Settings, label: '설정' },
];

export function MobileDrawer({ isOpen, onClose }: MobileDrawerProps) {
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { isDark, toggleTheme } = useTheme();
  const [openGroups, setOpenGroups] = useState<string[]>(['학습']);

  // Close drawer when route changes
  useEffect(() => {
    onClose();
  }, [location.pathname, onClose]);

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const toggleGroup = (label: string) => {
    setOpenGroups(prev =>
      prev.includes(label)
        ? prev.filter(g => g !== label)
        : [...prev, label]
    );
  };

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = async () => {
    await logout();
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className={clsx(
          'fixed inset-0 z-50 bg-black/50 backdrop-blur-sm transition-opacity duration-300',
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        )}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className={clsx(
          'fixed inset-y-0 left-0 z-50 w-80 max-w-[85vw] bg-white dark:bg-slate-900 shadow-2xl transition-transform duration-300 ease-out',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-200 dark:border-slate-700">
          <Link to="/" className="flex items-center gap-2 font-bold text-xl gradient-text" onClick={onClose}>
            <BookOpen className="h-6 w-6" />
            <span>Code Tutor AI</span>
          </Link>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:text-neutral-200 dark:hover:bg-slate-800 transition-colors"
            aria-label="메뉴 닫기"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-4 space-y-2">
          {/* Grouped Navigation */}
          {navGroups.map((group) => (
            <div key={group.label} className="space-y-1">
              <button
                onClick={() => toggleGroup(group.label)}
                className="flex items-center justify-between w-full px-3 py-3 rounded-lg text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-slate-800 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <group.icon className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
                  <span className="font-medium">{group.label}</span>
                </div>
                <ChevronDown
                  className={clsx(
                    'h-4 w-4 text-neutral-400 transition-transform duration-200',
                    openGroups.includes(group.label) && 'rotate-180'
                  )}
                />
              </button>

              {/* Accordion Content */}
              <div
                className={clsx(
                  'overflow-hidden transition-all duration-200',
                  openGroups.includes(group.label) ? 'max-h-48' : 'max-h-0'
                )}
              >
                <div className="pl-4 space-y-1">
                  {group.items.map((item) => (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={clsx(
                        'flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[44px]',
                        isActive(item.to)
                          ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                          : 'text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-slate-800'
                      )}
                    >
                      <item.icon className="h-5 w-5" />
                      <span>{item.label}</span>
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          ))}

          {/* Community Section */}
          <div className="pt-2 border-t border-neutral-200 dark:border-slate-700">
            <p className="px-3 py-2 text-xs font-semibold text-neutral-400 dark:text-neutral-500 uppercase tracking-wider">
              커뮤니티
            </p>
            {communityItems.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className={clsx(
                  'flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[44px]',
                  isActive(item.to)
                    ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                    : 'text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-slate-800'
                )}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.label}</span>
              </Link>
            ))}
          </div>

          {/* User Section */}
          <div className="pt-2 border-t border-neutral-200 dark:border-slate-700">
            <p className="px-3 py-2 text-xs font-semibold text-neutral-400 dark:text-neutral-500 uppercase tracking-wider">
              내 계정
            </p>
            <div className="px-3 py-2 mb-2">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-medium">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </div>
                <div>
                  <p className="font-medium text-neutral-800 dark:text-neutral-100">{user?.username}</p>
                  <p className="text-sm text-neutral-500 dark:text-neutral-400">{user?.email}</p>
                </div>
              </div>
            </div>
            {userMenuItems.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className={clsx(
                  'flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[44px]',
                  isActive(item.to)
                    ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                    : 'text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-slate-800'
                )}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.label}</span>
              </Link>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-neutral-200 dark:border-slate-700 space-y-2">
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="flex items-center justify-between w-full px-3 py-3 rounded-lg text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-slate-800 transition-colors min-h-[44px]"
          >
            <div className="flex items-center gap-3">
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              <span>{isDark ? '라이트 모드' : '다크 모드'}</span>
            </div>
            <div className={clsx(
              'w-10 h-6 rounded-full relative transition-colors',
              isDark ? 'bg-blue-600' : 'bg-neutral-300'
            )}>
              <div className={clsx(
                'absolute top-1 w-4 h-4 rounded-full bg-white transition-transform',
                isDark ? 'translate-x-5' : 'translate-x-1'
              )} />
            </div>
          </button>

          {/* Logout */}
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-3 py-3 rounded-lg text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors min-h-[44px]"
          >
            <LogOut className="h-5 w-5" />
            <span>로그아웃</span>
          </button>
        </div>
      </div>
    </>
  );
}
