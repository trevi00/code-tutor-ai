import { useState } from 'react';
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

    // In production, this would call the delete account API
    alert('계정 삭제 기능은 아직 구현 중입니다.');
  };

  const tabs: { id: SettingsTab; label: string; icon: string }[] = [
    { id: 'account', label: '계정', icon: '👤' },
    { id: 'notifications', label: '알림', icon: '🔔' },
    { id: 'appearance', label: '화면', icon: '🎨' },
  ];

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-3xl font-bold text-gray-800 mb-8">설정</h1>

      {/* Alert Messages */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}
      {success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700">
          {success}
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden">
        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
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
                <h2 className="text-lg font-semibold text-gray-800 mb-4">계정 정보</h2>
                <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">이메일</span>
                    <span className="text-gray-800">{user?.email}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">사용자명</span>
                    <span className="text-gray-800">{user?.username}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">역할</span>
                    <span className="text-gray-800">{user?.role === 'admin' ? '관리자' : '학생'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">가입일</span>
                    <span className="text-gray-800">
                      {user?.created_at && new Date(user.created_at).toLocaleDateString('ko-KR')}
                    </span>
                  </div>
                </div>
              </div>

              {/* Password Change */}
              <div>
                <h2 className="text-lg font-semibold text-gray-800 mb-4">비밀번호 변경</h2>
                <form onSubmit={handleChangePassword} className="space-y-4 max-w-md">
                  <div>
                    <label htmlFor="oldPassword" className="block text-sm font-medium text-gray-700 mb-1">
                      현재 비밀번호
                    </label>
                    <input
                      type="password"
                      id="oldPassword"
                      value={oldPassword}
                      onChange={(e) => setOldPassword(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 mb-1">
                      새 비밀번호
                    </label>
                    <input
                      type="password"
                      id="newPassword"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      minLength={8}
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-1">
                      새 비밀번호 확인
                    </label>
                    <input
                      type="password"
                      id="confirmPassword"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      minLength={8}
                      required
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                  >
                    {loading ? '변경 중...' : '비밀번호 변경'}
                  </button>
                </form>
              </div>

              {/* Danger Zone */}
              <div className="border-t border-gray-200 pt-8">
                <h2 className="text-lg font-semibold text-red-600 mb-4">위험 구역</h2>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="font-medium text-red-800 mb-2">계정 삭제</h3>
                  <p className="text-sm text-red-600 mb-4">
                    계정을 삭제하면 모든 데이터가 영구적으로 삭제됩니다. 이 작업은 되돌릴 수 없습니다.
                  </p>
                  <button
                    onClick={handleDeleteAccount}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
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
              <h2 className="text-lg font-semibold text-gray-800 mb-4">알림 설정</h2>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-800">이메일 알림</h3>
                    <p className="text-sm text-gray-500">중요한 업데이트와 알림을 이메일로 받습니다</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={emailNotifications}
                      onChange={(e) => setEmailNotifications(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-800">푸시 알림</h3>
                    <p className="text-sm text-gray-500">브라우저 푸시 알림을 받습니다</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={pushNotifications}
                      onChange={(e) => setPushNotifications(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h3 className="font-medium text-gray-800">주간 학습 리포트</h3>
                    <p className="text-sm text-gray-500">매주 학습 통계 요약을 이메일로 받습니다</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={weeklyReport}
                      onChange={(e) => setWeeklyReport(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              </div>

              <p className="text-sm text-gray-500 mt-4">
                * 알림 설정은 현재 데모 모드입니다. 실제 알림은 향후 지원될 예정입니다.
              </p>
            </div>
          )}

          {/* Appearance Tab */}
          {activeTab === 'appearance' && (
            <div className="space-y-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">화면 설정</h2>

              <div className="space-y-6">
                {/* Theme Setting */}
                <div>
                  <h3 className="font-medium text-gray-800 mb-3">테마</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: 'light', label: '라이트', icon: '☀️' },
                      { value: 'dark', label: '다크', icon: '🌙' },
                      { value: 'system', label: '시스템', icon: '💻' },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setTheme(option.value as typeof theme)}
                        className={`p-4 rounded-lg border-2 transition-colors ${
                          theme === option.value
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="text-2xl mb-2">{option.icon}</div>
                        <div className="text-sm font-medium">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Language Setting */}
                <div>
                  <h3 className="font-medium text-gray-800 mb-3">언어</h3>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as typeof language)}
                    className="w-full max-w-xs px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="ko">한국어</option>
                    <option value="en">English</option>
                  </select>
                </div>

                {/* Editor Settings */}
                <div>
                  <h3 className="font-medium text-gray-800 mb-3">코드 에디터</h3>
                  <div className="bg-gray-50 rounded-lg p-4 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">폰트 크기</span>
                      <select className="px-3 py-1 border border-gray-300 rounded-lg text-sm">
                        <option>12px</option>
                        <option>14px</option>
                        <option selected>16px</option>
                        <option>18px</option>
                      </select>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">탭 크기</span>
                      <select className="px-3 py-1 border border-gray-300 rounded-lg text-sm">
                        <option>2 spaces</option>
                        <option selected>4 spaces</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <p className="text-sm text-gray-500 mt-4">
                * 다크 모드와 일부 설정은 향후 지원될 예정입니다.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Logout Button */}
      <div className="mt-8">
        <button
          onClick={logout}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          로그아웃
        </button>
      </div>
    </div>
  );
}
