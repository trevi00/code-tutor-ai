/**
 * Register Page - Enhanced with modern design
 */

import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Code2,
  Mail,
  Lock,
  User,
  AlertCircle,
  CheckCircle,
  Loader2,
  ArrowRight,
  Sparkles,
  Shield,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

export function RegisterPage() {
  const navigate = useNavigate();
  const { register, isLoading, error, clearError } = useAuthStore();
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      return;
    }

    try {
      await register({ email, username, password });
      setSuccess(true);
      setTimeout(() => navigate('/login'), 2000);
    } catch {
      // Error is handled by store
    }
  };

  const passwordsMatch = password === confirmPassword && password.length > 0;
  const passwordValid = password.length >= 8;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-neutral-100 via-white to-neutral-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 py-12 px-4 relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <Code2 className="absolute top-24 right-[15%] w-12 h-12 text-violet-500/20" />
        <Sparkles className="absolute bottom-24 left-[20%] w-10 h-10 text-purple-500/20" />
        <Shield className="absolute top-1/2 left-[10%] w-8 h-8 text-violet-500/15" />
      </div>

      <div className="max-w-md w-full relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-3 group">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-violet-500 to-purple-500 flex items-center justify-center shadow-lg shadow-violet-500/25 group-hover:shadow-violet-500/40 transition-shadow">
              <Code2 className="h-7 w-7 text-white" />
            </div>
            <span className="text-2xl font-bold text-neutral-900 dark:text-white">Code Tutor AI</span>
          </Link>
          <p className="mt-4 text-neutral-500 dark:text-slate-400">계정을 만들고 학습을 시작하세요</p>
        </div>

        {/* Form Card */}
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-neutral-200 dark:border-slate-700/50">
          {success ? (
            <div className="text-center py-8">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle className="h-10 w-10 text-emerald-400" />
              </div>
              <h2 className="text-2xl font-bold text-neutral-900 dark:text-white mb-2">회원가입 완료!</h2>
              <p className="text-neutral-500 dark:text-slate-400">로그인 페이지로 이동합니다...</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-5">
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
                    className="w-full pl-12 pr-4 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="username" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                  사용자명
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400 dark:text-slate-500" />
                  <input
                    id="username"
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="사용자명을 입력하세요"
                    required
                    minLength={3}
                    maxLength={20}
                    className="w-full pl-12 pr-4 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
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
                    placeholder="8자 이상 입력하세요"
                    required
                    minLength={8}
                    className={`w-full pl-12 pr-4 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all ${
                      password && !passwordValid
                        ? 'border-red-500/50'
                        : passwordValid
                        ? 'border-emerald-500/50'
                        : 'border-neutral-200 dark:border-slate-600'
                    }`}
                  />
                </div>
                {password && !passwordValid && (
                  <p className="text-xs text-red-400 mt-1.5">비밀번호는 8자 이상이어야 합니다</p>
                )}
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                  비밀번호 확인
                </label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400 dark:text-slate-500" />
                  <input
                    id="confirmPassword"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="비밀번호를 다시 입력하세요"
                    required
                    className={`w-full pl-12 pr-12 py-3.5 bg-neutral-100 dark:bg-slate-700/50 border rounded-xl text-neutral-900 dark:text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all ${
                      confirmPassword && !passwordsMatch
                        ? 'border-red-500/50'
                        : passwordsMatch
                        ? 'border-emerald-500/50'
                        : 'border-neutral-200 dark:border-slate-600'
                    }`}
                  />
                  {passwordsMatch && (
                    <CheckCircle className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-emerald-400" />
                  )}
                </div>
                {confirmPassword && !passwordsMatch && (
                  <p className="text-xs text-red-400 mt-1.5">비밀번호가 일치하지 않습니다</p>
                )}
              </div>

              <button
                type="submit"
                disabled={isLoading || !passwordsMatch || !passwordValid}
                className="w-full py-3.5 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 flex items-center justify-center gap-2 mt-6"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    계정 생성 중...
                  </>
                ) : (
                  <>
                    계정 만들기
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </form>
          )}

          {!success && (
            <div className="mt-8 pt-6 border-t border-neutral-200 dark:border-slate-700/50">
              <p className="text-center text-sm text-neutral-500 dark:text-slate-400">
                이미 계정이 있으신가요?{' '}
                <Link
                  to="/login"
                  className="text-violet-400 hover:text-violet-300 font-medium transition-colors"
                >
                  로그인
                </Link>
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="mt-8 text-center text-xs text-neutral-400 dark:text-slate-500">
          가입하면 서비스 이용약관 및 개인정보처리방침에 동의하게 됩니다.
        </p>
      </div>
    </div>
  );
}
