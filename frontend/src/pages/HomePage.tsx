/**
 * Home Page - Enhanced with modern design
 */

import { Link } from 'react-router-dom';
import {
  BookOpen,
  Code,
  MessageSquare,
  BarChart3,
  ArrowRight,
  Sparkles,
  Zap,
  Trophy,
  Users,
  Target,
  Brain,
  Rocket,
  Code2,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function HomePage() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen overflow-hidden bg-white dark:bg-slate-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-neutral-50 via-emerald-100/20 to-neutral-50 dark:from-slate-900 dark:via-emerald-900/20 dark:to-slate-900 text-neutral-900 dark:text-white">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-emerald-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-teal-500/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 right-1/3 w-[300px] h-[300px] bg-cyan-500/5 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Code2 className="absolute top-20 left-[10%] w-12 h-12 text-emerald-500/20 animate-float" />
          <Brain className="absolute top-32 right-[15%] w-10 h-10 text-teal-500/20 animate-float-delayed" />
          <Zap className="absolute bottom-40 left-[20%] w-8 h-8 text-cyan-500/20 animate-float" />
          <Target className="absolute bottom-32 right-[25%] w-10 h-10 text-emerald-500/15 animate-float-delayed" />
        </div>

        <div className="container mx-auto px-4 py-24 md:py-32 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-emerald-300 text-sm mb-8">
              <Sparkles className="w-4 h-4" />
              AI 기반 알고리즘 학습 플랫폼
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent">
                AI 튜터
              </span>
              와 함께하는{' '}
              <br className="hidden md:block" />
              알고리즘 마스터
            </h1>
            <p className="text-xl md:text-2xl text-neutral-600 dark:text-slate-300 mb-10 max-w-2xl mx-auto">
              맞춤형 AI 가이드, 실시간 코드 피드백, 인터랙티브 문제 풀이로
              Python 알고리즘을 단계별로 학습하세요.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAuthenticated ? (
                <Link
                  to="/problems"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:-translate-y-1"
                >
                  학습 시작하기
                  <ArrowRight className="h-5 w-5" />
                </Link>
              ) : (
                <>
                  <Link
                    to="/register"
                    className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:-translate-y-1"
                  >
                    무료로 시작하기
                    <ArrowRight className="h-5 w-5" />
                  </Link>
                  <Link
                    to="/login"
                    className="inline-flex items-center justify-center px-8 py-4 border-2 border-neutral-300 dark:border-slate-600 text-neutral-900 dark:text-white rounded-xl font-semibold hover:bg-neutral-100 dark:hover:bg-slate-800 hover:border-neutral-400 dark:hover:border-slate-500 transition-all duration-300"
                  >
                    로그인
                  </Link>
                </>
              )}
            </div>

            {/* Stats */}
            <div className="flex flex-wrap justify-center gap-8 mt-16 pt-8 border-t border-neutral-200 dark:border-slate-700/50">
              <div className="text-center">
                <div className="text-3xl font-bold text-emerald-400">25+</div>
                <div className="text-sm text-neutral-500 dark:text-slate-400">알고리즘 패턴</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-teal-400">100+</div>
                <div className="text-sm text-neutral-500 dark:text-slate-400">연습 문제</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">24/7</div>
                <div className="text-sm text-neutral-500 dark:text-slate-400">AI 튜터 지원</div>
              </div>
            </div>
          </div>
        </div>

        {/* Wave decoration */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M0 120L60 105C120 90 240 60 360 45C480 30 600 30 720 37.5C840 45 960 60 1080 67.5C1200 75 1320 75 1380 75L1440 75V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z"
              className="fill-neutral-100 dark:fill-slate-800"
            />
          </svg>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-neutral-100 dark:bg-slate-800">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-400 text-sm mb-4">
              <Zap className="w-4 h-4" />
              핵심 기능
            </div>
            <h2 className="text-3xl md:text-5xl font-bold text-neutral-900 dark:text-white mb-4">
              왜{' '}
              <span className="bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                Code Tutor AI
              </span>
              인가요?
            </h2>
            <p className="text-xl text-neutral-500 dark:text-slate-400 max-w-2xl mx-auto">
              나만의 속도로 알고리즘을 마스터할 수 있도록 설계된 지능형 학습 플랫폼입니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <FeatureCard
              icon={<BookOpen className="h-7 w-7" />}
              title="엄선된 문제"
              description="25개 이상의 알고리즘 패턴을 단계별 설명과 함께 학습하세요."
              color="emerald"
            />
            <FeatureCard
              icon={<MessageSquare className="h-7 w-7" />}
              title="AI 튜터 채팅"
              description="AI 튜터에게 맞춤형 힌트와 설명을 받으세요."
              color="teal"
            />
            <FeatureCard
              icon={<Code className="h-7 w-7" />}
              title="코드 샌드박스"
              description="안전한 환경에서 코드를 작성하고 실행하고 테스트하세요."
              color="cyan"
            />
            <FeatureCard
              icon={<BarChart3 className="h-7 w-7" />}
              title="진도 추적"
              description="상세한 분석으로 학습 여정을 모니터링하세요."
              color="blue"
            />
          </div>
        </div>
      </section>

      {/* Patterns Section */}
      <section className="py-24 bg-white dark:bg-slate-900">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-full text-amber-400 text-sm mb-4">
              <Sparkles className="w-4 h-4" />
              알고리즘 패턴
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 dark:text-white mb-4">
              리트코드 패턴 학습
            </h2>
            <p className="text-xl text-neutral-500 dark:text-slate-400 max-w-2xl mx-auto">
              코딩 테스트에 자주 나오는 25개 핵심 알고리즘 패턴을 체계적으로 학습하세요.
            </p>
          </div>

          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-4 max-w-5xl mx-auto">
            {[
              { name: 'Two Pointers', icon: '👆' },
              { name: 'Sliding Window', icon: '🪟' },
              { name: 'BFS/DFS', icon: '🌳' },
              { name: 'Binary Search', icon: '🔍' },
              { name: 'Dynamic Programming', icon: '📊' },
              { name: 'Backtracking', icon: '🔙' },
              { name: 'Greedy', icon: '💰' },
              { name: 'Stack/Queue', icon: '📚' },
              { name: 'Graph', icon: '🕸️' },
              { name: 'Tree', icon: '🌲' },
            ].map((pattern) => (
              <div
                key={pattern.name}
                className="group bg-neutral-50 dark:bg-slate-800/50 hover:bg-neutral-100 dark:hover:bg-slate-800 rounded-xl p-5 text-center border border-neutral-200 dark:border-slate-700/50 hover:border-emerald-500/30 transition-all duration-300 cursor-pointer hover:-translate-y-1"
              >
                <div className="text-2xl mb-2">{pattern.icon}</div>
                <span className="text-sm font-medium text-neutral-600 dark:text-slate-300 group-hover:text-emerald-500 dark:group-hover:text-emerald-400 transition-colors">
                  {pattern.name}
                </span>
              </div>
            ))}
          </div>

          <div className="text-center mt-10">
            <Link
              to="/patterns"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl transition-all duration-300 shadow-lg shadow-emerald-500/25 hover:-translate-y-1 font-medium"
            >
              모든 패턴 보기
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Social Proof Section */}
      <section className="py-20 bg-neutral-100 dark:bg-slate-800">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white dark:bg-slate-900/50 rounded-2xl p-8 border border-neutral-200 dark:border-slate-700/50 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-emerald-500/20 flex items-center justify-center">
                <Users className="w-8 h-8 text-emerald-400" />
              </div>
              <div className="text-3xl font-bold text-neutral-900 dark:text-white mb-2">1,000+</div>
              <div className="text-neutral-500 dark:text-slate-400">활성 학습자</div>
            </div>
            <div className="bg-white dark:bg-slate-900/50 rounded-2xl p-8 border border-neutral-200 dark:border-slate-700/50 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-teal-500/20 flex items-center justify-center">
                <Trophy className="w-8 h-8 text-teal-400" />
              </div>
              <div className="text-3xl font-bold text-neutral-900 dark:text-white mb-2">50,000+</div>
              <div className="text-neutral-500 dark:text-slate-400">해결된 문제</div>
            </div>
            <div className="bg-white dark:bg-slate-900/50 rounded-2xl p-8 border border-neutral-200 dark:border-slate-700/50 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-cyan-500/20 flex items-center justify-center">
                <Rocket className="w-8 h-8 text-cyan-400" />
              </div>
              <div className="text-3xl font-bold text-neutral-900 dark:text-white mb-2">95%</div>
              <div className="text-neutral-500 dark:text-slate-400">만족도</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-b from-white to-neutral-100 dark:from-slate-900 dark:to-slate-800 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-500/5 rounded-full blur-3xl" />
        </div>

        <div className="container mx-auto px-4 text-center relative z-10">
          <div className="max-w-2xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-400 text-sm mb-6">
              <Rocket className="w-4 h-4" />
              지금 시작하세요
            </div>
            <h2 className="text-3xl md:text-5xl font-bold text-neutral-900 dark:text-white mb-4">
              알고리즘 마스터가 되어보세요
            </h2>
            <p className="text-xl text-neutral-500 dark:text-slate-400 mb-10">
              AI와 함께 알고리즘을 마스터하는 수천 명의 학습자와 함께하세요.
            </p>
            {!isAuthenticated && (
              <Link
                to="/register"
                className="inline-flex items-center gap-2 px-10 py-5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-bold text-lg transition-all duration-300 shadow-xl shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:-translate-y-1"
              >
                무료 계정 만들기
                <ArrowRight className="h-5 w-5" />
              </Link>
            )}
          </div>
        </div>
      </section>

      {/* Animations */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(-5deg); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 7s ease-in-out infinite;
          animation-delay: 1.5s;
        }
      `}</style>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'emerald' | 'teal' | 'cyan' | 'blue';
}) {
  const colorClasses = {
    emerald: 'bg-emerald-500/20 text-emerald-400 group-hover:bg-emerald-500/30',
    teal: 'bg-teal-500/20 text-teal-400 group-hover:bg-teal-500/30',
    cyan: 'bg-cyan-500/20 text-cyan-400 group-hover:bg-cyan-500/30',
    blue: 'bg-blue-500/20 text-blue-400 group-hover:bg-blue-500/30',
  };

  return (
    <div className="group bg-white dark:bg-slate-900/50 hover:bg-neutral-50 dark:hover:bg-slate-900 rounded-2xl p-7 border border-neutral-200 dark:border-slate-700/50 hover:border-emerald-500/30 transition-all duration-300 hover:-translate-y-1">
      <div
        className={`w-14 h-14 rounded-xl flex items-center justify-center mb-5 transition-colors ${colorClasses[color]}`}
      >
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-3 text-neutral-900 dark:text-white">{title}</h3>
      <p className="text-neutral-500 dark:text-slate-400">{description}</p>
    </div>
  );
}
