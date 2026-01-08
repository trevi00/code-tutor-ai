import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';
import clsx from 'clsx';

interface BreadcrumbItem {
  label: string;
  to?: string;
}

// Route to breadcrumb mapping
const routeLabels: Record<string, string> = {
  '': '홈',
  'roadmap': '로드맵',
  'problems': '문제',
  'patterns': '패턴',
  'playground': '플레이그라운드',
  'typing-practice': '받아쓰기',
  'visualization': '시각화',
  'dashboard': '대시보드',
  'performance': '성능',
  'debugger': '디버거',
  'chat': 'AI 튜터',
  'leaderboard': '리더보드',
  'profile': '내 프로필',
  'submissions': '제출 기록',
  'badges': '내 배지',
  'settings': '설정',
  'login': '로그인',
  'register': '회원가입',
};

interface BreadcrumbProps {
  items?: BreadcrumbItem[];
  className?: string;
}

export function Breadcrumb({ items, className }: BreadcrumbProps) {
  const location = useLocation();

  // Auto-generate breadcrumbs from pathname if items not provided
  const breadcrumbItems: BreadcrumbItem[] = items || (() => {
    const pathSegments = location.pathname.split('/').filter(Boolean);
    const generated: BreadcrumbItem[] = [];

    let currentPath = '';
    for (let i = 0; i < pathSegments.length; i++) {
      const segment = pathSegments[i];
      currentPath += `/${segment}`;

      const label = routeLabels[segment] || segment;
      const isLast = i === pathSegments.length - 1;

      generated.push({
        label,
        to: isLast ? undefined : currentPath,
      });
    }

    return generated;
  })();

  if (breadcrumbItems.length === 0) {
    return null;
  }

  return (
    <nav
      aria-label="Breadcrumb"
      className={clsx('flex items-center text-sm', className)}
    >
      <ol className="flex items-center gap-1">
        {/* Home */}
        <li>
          <Link
            to="/"
            className="flex items-center gap-1 text-neutral-500 hover:text-blue-600 dark:text-neutral-400 dark:hover:text-blue-400 transition-colors"
          >
            <Home className="h-4 w-4" />
            <span className="sr-only">홈</span>
          </Link>
        </li>

        {breadcrumbItems.map((item, index) => (
          <li key={index} className="flex items-center gap-1">
            <ChevronRight className="h-4 w-4 text-neutral-400 dark:text-neutral-500" />
            {item.to ? (
              <Link
                to={item.to}
                className="text-neutral-500 hover:text-blue-600 dark:text-neutral-400 dark:hover:text-blue-400 transition-colors"
              >
                {item.label}
              </Link>
            ) : (
              <span className="text-neutral-800 dark:text-neutral-100 font-medium">
                {item.label}
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}

// Export for convenience
export function PageHeader({
  title,
  description,
  breadcrumbItems,
  actions,
}: {
  title: string;
  description?: string;
  breadcrumbItems?: BreadcrumbItem[];
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 space-y-2">
      <Breadcrumb items={breadcrumbItems} />
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-50">
            {title}
          </h1>
          {description && (
            <p className="mt-1 text-neutral-600 dark:text-neutral-400">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="flex-shrink-0">{actions}</div>}
      </div>
    </div>
  );
}
