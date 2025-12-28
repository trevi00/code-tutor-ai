import { Link } from 'react-router-dom';
import { BookOpen, Code, MessageSquare, BarChart3, ArrowRight } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function HomePage() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 text-white">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,...')] opacity-10" />
        <div className="container mx-auto px-4 py-24 relative">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Master Algorithms with{' '}
              <span className="text-yellow-300">AI-Powered</span> Tutoring
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 mb-8">
              Learn Python algorithms step-by-step with personalized AI guidance,
              instant code feedback, and interactive problem-solving.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAuthenticated ? (
                <Link
                  to="/problems"
                  className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:bg-blue-50 transition-colors"
                >
                  Start Learning
                  <ArrowRight className="h-5 w-5" />
                </Link>
              ) : (
                <>
                  <Link
                    to="/register"
                    className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:bg-blue-50 transition-colors"
                  >
                    Get Started Free
                    <ArrowRight className="h-5 w-5" />
                  </Link>
                  <Link
                    to="/login"
                    className="inline-flex items-center justify-center px-8 py-4 border-2 border-white text-white rounded-xl font-semibold hover:bg-white/10 transition-colors"
                  >
                    Sign In
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
      <section className="py-20 bg-neutral-50">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            Why Code Tutor AI?
          </h2>
          <p className="text-xl text-neutral-600 text-center mb-12 max-w-2xl mx-auto">
            An intelligent learning platform designed to help you master algorithms
            at your own pace.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <FeatureCard
              icon={<BookOpen className="h-8 w-8" />}
              title="Curated Problems"
              description="25+ algorithm patterns with step-by-step explanations in Korean."
            />
            <FeatureCard
              icon={<MessageSquare className="h-8 w-8" />}
              title="AI Tutor Chat"
              description="Get personalized hints and explanations from our AI tutor."
            />
            <FeatureCard
              icon={<Code className="h-8 w-8" />}
              title="Code Sandbox"
              description="Write, run, and test your code in a secure environment."
            />
            <FeatureCard
              icon={<BarChart3 className="h-8 w-8" />}
              title="Track Progress"
              description="Monitor your learning journey with detailed analytics."
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to Start Learning?
          </h2>
          <p className="text-xl text-neutral-600 mb-8">
            Join thousands of learners mastering algorithms with AI assistance.
          </p>
          {!isAuthenticated && (
            <Link
              to="/register"
              className="inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors"
            >
              Create Free Account
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
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-neutral-200 hover:shadow-md transition-shadow">
      <div className="w-14 h-14 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-neutral-600">{description}</p>
    </div>
  );
}
