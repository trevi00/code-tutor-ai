import { Link } from 'react-router-dom';
import { BookOpen, Code, MessageSquare, BarChart3, ArrowRight, Sparkles } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function HomePage() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen overflow-hidden">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-700 animate-gradient text-white">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,...')] opacity-10" />
        <div className="container mx-auto px-4 py-28 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="gradient-text-gold">AI 튜터</span>와 함께하는{' '}
              알고리즘 마스터
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 mb-8">
              맞춤형 AI 가이드, 실시간 코드 피드백, 인터랙티브 문제 풀이로
              Python 알고리즘을 단계별로 학습하세요.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAuthenticated ? (
                <Link
                  to="/problems"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:bg-blue-50 transition-all duration-300 shadow-lg hover:shadow-xl hover:-translate-y-1"
                >
                  학습 시작하기
                  <ArrowRight className="h-5 w-5" />
                </Link>
              ) : (
                <>
                  <Link
                    to="/register"
                    className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:bg-blue-50 transition-all duration-300 shadow-lg hover:shadow-xl hover:-translate-y-1"
                  >
                    무료로 시작하기
                    <ArrowRight className="h-5 w-5" />
                  </Link>
                  <Link
                    to="/login"
                    className="inline-flex items-center justify-center px-8 py-4 border-2 border-white/50 text-white backdrop-blur-sm rounded-xl font-semibold hover:bg-white/10 hover:border-white transition-all duration-300"
                  >
                    로그인
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
        {/* Wave decoration */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M0 120L60 105C120 90 240 60 360 45C480 30 600 30 720 37.5C840 45 960 60 1080 67.5C1200 75 1320 75 1380 75L1440 75V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z"
              fill="#f8fafc"
            />
          </svg>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-neutral-50">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl md:text-5xl font-bold text-center mb-4 gradient-text">
            왜 Code Tutor AI인가요?
          </h2>
          <p className="text-xl text-neutral-600 text-center mb-12 max-w-2xl mx-auto">
            나만의 속도로 알고리즘을 마스터할 수 있도록 설계된 지능형 학습 플랫폼입니다.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <FeatureCard
              icon={<BookOpen className="h-8 w-8" />}
              title="엄선된 문제"
              description="25개 이상의 알고리즘 패턴을 단계별 설명과 함께 학습하세요."
            />
            <FeatureCard
              icon={<MessageSquare className="h-8 w-8" />}
              title="AI 튜터 채팅"
              description="AI 튜터에게 맞춤형 힌트와 설명을 받으세요."
            />
            <FeatureCard
              icon={<Code className="h-8 w-8" />}
              title="코드 샌드박스"
              description="안전한 환경에서 코드를 작성하고 실행하고 테스트하세요."
            />
            <FeatureCard
              icon={<BarChart3 className="h-8 w-8" />}
              title="진도 추적"
              description="상세한 분석으로 학습 여정을 모니터링하세요."
            />
          </div>
        </div>
      </section>

      {/* Patterns Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="h-6 w-6 text-yellow-500" />
            <h2 className="text-3xl md:text-4xl font-bold text-center">
              리트코드 패턴 학습
            </h2>
          </div>
          <p className="text-xl text-neutral-600 text-center mb-12 max-w-2xl mx-auto">
            코딩 테스트에 자주 나오는 25개 핵심 알고리즘 패턴을 체계적으로 학습하세요.
          </p>
          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-4 max-w-4xl mx-auto">
            {[
              'Two Pointers',
              'Sliding Window',
              'BFS/DFS',
              'Binary Search',
              'Dynamic Programming',
              'Backtracking',
              'Greedy',
              'Stack/Queue',
              'Graph',
              'Tree',
            ].map((pattern) => (
              <div
                key={pattern}
                className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-5 text-center border border-blue-100/50 group cursor-pointer card-hover"
              >
                <span className="text-sm font-medium text-blue-700">{pattern}</span>
              </div>
            ))}
          </div>
          <div className="text-center mt-8">
            <Link
              to="/patterns"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300 hover:-translate-y-1 font-medium"
            >
              모든 패턴 보기
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-neutral-50">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            지금 바로 학습을 시작하세요
          </h2>
          <p className="text-xl text-neutral-600 mb-8">
            AI와 함께 알고리즘을 마스터하는 수천 명의 학습자와 함께하세요.
          </p>
          {!isAuthenticated && (
            <Link
              to="/register"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300 hover:-translate-y-1"
            >
              무료 계정 만들기
              <ArrowRight className="h-5 w-5" />
            </Link>
          )}
        </div>
      </section>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="card-hover bg-white rounded-2xl p-7 shadow-soft border border-neutral-100">
      <div className="w-14 h-14 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center mb-5 shadow-lg">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-3 text-neutral-800">{title}</h3>
      <p className="text-neutral-600">{description}</p>
    </div>
  );
}
