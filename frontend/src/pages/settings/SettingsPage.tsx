/**
 * Settings Page - Enhanced with modern design
 */

import { useState } from 'react';
import {
  Settings,
  User,
  Bell,
  Palette,
  Lock,
  AlertTriangle,
  Loader2,
  CheckCircle,
  XCircle,
  Sun,
  Moon,
  Monitor,
  LogOut,
  Sparkles,
  Mail,
  Shield,
  Code2,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { authApi } from '@/api';
import type { ChangePasswordRequest } from '@/types';

type SettingsTab = 'account' | 'notifications' | 'appearance';

export default function SettingsPage() {
  const { user, logout } = useAuthStore();
  const [activeTab, setActiveTab] = useState<SettingsTab>('account');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Password change form
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Notification settings (local state for demo)
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(false);
  const [weeklyReport, setWeeklyReport] = useState(true);

  // Appearance settings
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [language, setLanguage] = useState<'ko' | 'en'>('ko');

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    if (newPassword !== confirmPassword) {
      setError('새 비밀번호가 일치하지 않습니다.');
      setLoading(false);
      return;
    }

    if (newPassword.length < 8) {
      setError('비밀번호는 최소 8자 이상이어야 합니다.');
      setLoading(false);
      return;
    }

    try {
      const data: ChangePasswordRequest = {
        old_password: oldPassword,
        new_password: newPassword,
      };
      await authApi.changePassword(data);
      setSuccess('비밀번호가 변경되었습니다.');
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setError(err.response?.data?.detail || '비밀번호 변경에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm('정말로 계정을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
      return;
    }
    alert('계정 삭제 기능은 아직 구현 중입니다.');
  };

  const tabs: { id: SettingsTab; label: string; icon: React.ReactNode }[] = [
    { id: 'account', label: '계정', icon: <User className="w-4 h-4" /> },
    { id: 'notifications', label: '알림', icon: <Bell className="w-4 h-4" /> },
    { id: 'appearance', label: '화면', icon: <Palette className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-slate-200 via-slate-300 to-slate-400 dark:from-slate-700 dark:via-slate-800 dark:to-slate-900 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/5 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-slate-500/10 rounded-full blur-3xl" />
          <Settings className="absolute top-10 right-[10%] w-12 h-12 text-slate-500/20 dark:text-white/10 animate-float" />
          <Shield className="absolute bottom-10 left-[15%] w-10 h-10 text-slate-500/20 dark:text-white/10 animate-float-delayed" />
        </div>

        <div className="max-w-4xl mx-auto px-6 py-10 relative">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-slate-600/20 dark:bg-white/10 backdrop-blur-sm flex items-center justify-center">
              <Settings className="w-7 h-7 text-slate-700 dark:text-white" />
            </div>
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-600/20 dark:bg-white/10 rounded-full text-slate-700/80 dark:text-white/80 text-sm mb-2">
                <Sparkles className="w-3.5 h-3.5" />
                설정 관리
              </div>
              <h1 className="text-2xl md:text-3xl font-bold text-slate-800 dark:text-white">설정</h1>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8 -mt-6">
        {/* Alert Messages */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center flex-shrink-0">
              <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            </div>
            <span className="text-red-700 dark:text-red-300">{error}</span>
          </div>
        )}
        {success && (
          <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <span className="text-green-700 dark:text-green-300">{success}</span>
          </div>
        )}

        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl overflow-hidden border border-slate-200 dark:border-slate-700">
          {/* Tabs */}
          <div className="border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
            <nav className="flex">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-all ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400 bg-white dark:bg-slate-800'
                      : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:border-slate-300 dark:hover:border-slate-600'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {/* Account Tab */}
            {activeTab === 'account' && (
              <div className="space-y-8">
                {/* Account Info */}
                <div>
                  <h2 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                    <User className="w-5 h-5 text-blue-500" />
                    계정 정보
                  </h2>
                  <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-5 space-y-4">
                    <div className="flex items-center justify-between py-2">
                      <div className="flex items-center gap-3">
                        <Mail className="w-5 h-5 text-slate-400" />
                        <span className="text-slate-600 dark:text-slate-400">이메일</span>
                      </div>
                      <span className="font-medium text-slate-800 dark:text-white">{user?.email}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-t border-slate-200 dark:border-slate-700">
                      <div className="flex items-center gap-3">
                        <User className="w-5 h-5 text-slate-400" />
                        <span className="text-slate-600 dark:text-slate-400">사용자명</span>
                      </div>
                      <span className="font-medium text-slate-800 dark:text-white">{user?.username}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-t border-slate-200 dark:border-slate-700">
                      <div className="flex items-center gap-3">
                        <Shield className="w-5 h-5 text-slate-400" />
                        <span className="text-slate-600 dark:text-slate-400">역할</span>
                      </div>
                      <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                        user?.role === 'admin'
                          ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                          : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                      }`}>
                        {user?.role === 'admin' ? '관리자' : '학생'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-t border-slate-200 dark:border-slate-700">
                      <div className="flex items-center gap-3">
                        <Sparkles className="w-5 h-5 text-slate-400" />
                        <span className="text-slate-600 dark:text-slate-400">가입일</span>
                      </div>
                      <span className="font-medium text-slate-800 dark:text-white">
                        {user?.created_at && new Date(user.created_at).toLocaleDateString('ko-KR')}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Password Change */}
                <div>
                  <h2 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                    <Lock className="w-5 h-5 text-purple-500" />
                    비밀번호 변경
                  </h2>
                  <form onSubmit={handleChangePassword} className="space-y-4 max-w-md">
                    <div>
                      <label htmlFor="oldPassword" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        현재 비밀번호
                      </label>
                      <input
                        type="password"
                        id="oldPassword"
                        value={oldPassword}
                        onChange={(e) => setOldPassword(e.target.value)}
                        className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-800 dark:text-white transition-all"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="newPassword" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        새 비밀번호
                      </label>
                      <input
                        type="password"
                        id="newPassword"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-800 dark:text-white transition-all"
                        minLength={8}
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        새 비밀번호 확인
                      </label>
                      <input
                        type="password"
                        id="confirmPassword"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-800 dark:text-white transition-all"
                        minLength={8}
                        required
                      />
                    </div>
                    <button
                      type="submit"
                      disabled={loading}
                      className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white font-medium rounded-xl shadow-lg shadow-blue-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          변경 중...
                        </>
                      ) : (
                        <>
                          <Lock className="w-4 h-4" />
                          비밀번호 변경
                        </>
                      )}
                    </button>
                  </form>
                </div>

                {/* Danger Zone */}
                <div className="border-t border-slate-200 dark:border-slate-700 pt-8">
                  <h2 className="text-lg font-bold text-red-600 dark:text-red-400 mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    위험 구역
                  </h2>
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-5">
                    <h3 className="font-semibold text-red-800 dark:text-red-300 mb-2">계정 삭제</h3>
                    <p className="text-sm text-red-600 dark:text-red-400 mb-4">
                      계정을 삭제하면 모든 데이터가 영구적으로 삭제됩니다. 이 작업은 되돌릴 수 없습니다.
                    </p>
                    <button
                      onClick={handleDeleteAccount}
                      className="px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white font-medium rounded-xl transition-colors"
                    >
                      계정 삭제
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Notifications Tab */}
            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <h2 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                  <Bell className="w-5 h-5 text-amber-500" />
                  알림 설정
                </h2>

                <div className="space-y-4">
                  <ToggleItem
                    title="이메일 알림"
                    description="중요한 업데이트와 알림을 이메일로 받습니다"
                    icon={<Mail className="w-5 h-5 text-blue-500" />}
                    checked={emailNotifications}
                    onChange={setEmailNotifications}
                  />
                  <ToggleItem
                    title="푸시 알림"
                    description="브라우저 푸시 알림을 받습니다"
                    icon={<Bell className="w-5 h-5 text-purple-500" />}
                    checked={pushNotifications}
                    onChange={setPushNotifications}
                  />
                  <ToggleItem
                    title="주간 학습 리포트"
                    description="매주 학습 통계 요약을 이메일로 받습니다"
                    icon={<Sparkles className="w-5 h-5 text-amber-500" />}
                    checked={weeklyReport}
                    onChange={setWeeklyReport}
                  />
                </div>

                <p className="text-sm text-slate-500 dark:text-slate-400 mt-6 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-xl">
                  💡 알림 설정은 현재 데모 모드입니다. 실제 알림은 향후 지원될 예정입니다.
                </p>
              </div>
            )}

            {/* Appearance Tab */}
            {activeTab === 'appearance' && (
              <div className="space-y-8">
                {/* Theme Setting */}
                <div>
                  <h2 className="text-lg font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                    <Palette className="w-5 h-5 text-pink-500" />
                    테마
                  </h2>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: 'light', label: '라이트', icon: <Sun className="w-6 h-6" />, gradient: 'from-amber-400 to-orange-400' },
                      { value: 'dark', label: '다크', icon: <Moon className="w-6 h-6" />, gradient: 'from-indigo-500 to-purple-500' },
                      { value: 'system', label: '시스템', icon: <Monitor className="w-6 h-6" />, gradient: 'from-slate-500 to-slate-600' },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setTheme(option.value as typeof theme)}
                        className={`p-5 rounded-xl border-2 transition-all group ${
                          theme === option.value
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg'
                            : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 hover:shadow-md'
                        }`}
                      >
                        <div className={`w-12 h-12 mx-auto mb-3 rounded-xl bg-gradient-to-br ${option.gradient} flex items-center justify-center text-white shadow-lg`}>
                          {option.icon}
                        </div>
                        <div className={`text-sm font-medium ${
                          theme === option.value ? 'text-blue-600 dark:text-blue-400' : 'text-slate-700 dark:text-slate-300'
                        }`}>
                          {option.label}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Language Setting */}
                <div>
                  <h3 className="font-bold text-slate-800 dark:text-white mb-3">언어</h3>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as typeof language)}
                    className="w-full max-w-xs px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-800 dark:text-white"
                  >
                    <option value="ko">🇰🇷 한국어</option>
                    <option value="en">🇺🇸 English</option>
                  </select>
                </div>

                {/* Editor Settings */}
                <div>
                  <h3 className="font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                    <Code2 className="w-5 h-5 text-emerald-500" />
                    코드 에디터
                  </h3>
                  <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-700 dark:text-slate-300">폰트 크기</span>
                      <select className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm text-slate-800 dark:text-white">
                        <option>12px</option>
                        <option>14px</option>
                        <option>16px</option>
                        <option>18px</option>
                      </select>
                    </div>
                    <div className="flex items-center justify-between border-t border-slate-200 dark:border-slate-700 pt-4">
                      <span className="text-slate-700 dark:text-slate-300">탭 크기</span>
                      <select className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm text-slate-800 dark:text-white">
                        <option>2 spaces</option>
                        <option>4 spaces</option>
                      </select>
                    </div>
                  </div>
                </div>

                <p className="text-sm text-slate-500 dark:text-slate-400 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-xl">
                  💡 다크 모드와 일부 설정은 향후 지원될 예정입니다.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Logout Button */}
        <div className="mt-8">
          <button
            onClick={logout}
            className="px-6 py-3 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors flex items-center gap-2"
          >
            <LogOut className="w-4 h-4" />
            로그아웃
          </button>
        </div>
      </div>

      {/* Styles */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 5s ease-in-out infinite;
          animation-delay: 1s;
        }
      `}</style>
    </div>
  );
}

// Toggle Item Component
interface ToggleItemProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function ToggleItem({ title, description, icon, checked, onChange }: ToggleItemProps) {
  return (
    <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900/50 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-lg bg-white dark:bg-slate-800 shadow-sm flex items-center justify-center">
          {icon}
        </div>
        <div>
          <h3 className="font-medium text-slate-800 dark:text-white">{title}</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">{description}</p>
        </div>
      </div>
      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="w-11 h-6 bg-slate-200 dark:bg-slate-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
      </label>
    </div>
  );
}
