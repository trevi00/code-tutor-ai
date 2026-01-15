/**
 * Login Page - Enhanced with modern design
 */

import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Code2,
  Mail,
  Lock,
  AlertCircle,
  Loader2,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isLoading, error, clearError } = useAuthStore();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await login({ email, password });
      navigate('/problems');
    } catch {
      // Error is handled by store
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-neutral-100 via-white to-neutral-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 py-12 px-4 relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl" />
        <Code2 className="absolute top-20 left-[15%] w-12 h-12 text-emerald-500/20" />
        <Sparkles className="absolute bottom-32 right-[20%] w-10 h-10 text-teal-500/20" />
      </div>

      <div className="max-w-md w-full relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-3 group">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:shadow-emerald-500/40 transition-shadow">
              <Code2 className="h-7 w-7 text-white" />
            </div>
            <span className="text-2xl font-bold text-neutral-900 dark:text-white">Code Tutor AI</span>
          </Link>
          <p className="mt-4 text-neutral-500 dark:text-slate-400">학습을 계속하려면 로그인하세요</p>
        </div>

        {/* Form Card */}
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-neutral-200 dark:border-slate-700/50">
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="flex items-center gap-3 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300">
                <AlertCircle className="h-5 w-5 flex-shrink-0" />
                <p className="text-sm flex-1">{error}</p>
                <button
                  type="button"
                  onClick={clearError}
                  className="text-red-400 hover:text-red-300 text-xl leading-none"
                >
                  &times;
                </button>
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                이메일
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400 dark:text-slate-500" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="example@email.com"
                  required
                  className="w-full pl-12 pr-4 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                비밀번호
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400 dark:text-slate-500" />
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="비밀번호를 입력하세요"
                  required
                  className="w-full pl-12 pr-4 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  로그인 중...
                </>
              ) : (
                <>
                  로그인
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </form>

          <div className="mt-8 pt-6 border-t border-neutral-200 dark:border-slate-700/50">
            <p className="text-center text-sm text-neutral-500 dark:text-slate-400">
              계정이 없으신가요?{' '}
              <Link
                to="/register"
                className="text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
              >
                회원가입
              </Link>
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="mt-8 text-center text-xs text-neutral-400 dark:text-slate-500">
          로그인하면 서비스 이용약관 및 개인정보처리방침에 동의하게 됩니다.
        </p>
      </div>
    </div>
  );
}
