/**
 * Profile Page - Enhanced with modern design
 */

import { useState, useEffect } from 'react';
import {
  User,
  Mail,
  Calendar,
  Clock,
  Shield,
  ShieldCheck,
  Edit3,
  Lock,
  Save,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
  Crown,
  FileText,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { authApi } from '@/api';
import type { UpdateProfileRequest, ChangePasswordRequest } from '@/types';

export default function ProfilePage() {
  const { user, setUser } = useAuthStore();
  const [isEditing, setIsEditing] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Profile form state
  const [username, setUsername] = useState(user?.username || '');
  const [bio, setBio] = useState(user?.bio || '');

  // Password form state
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  useEffect(() => {
    if (user) {
      setUsername(user.username);
      setBio(user.bio || '');
    }
  }, [user]);

  const handleUpdateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const data: UpdateProfileRequest = {};
      if (username !== user?.username) data.username = username;
      if (bio !== user?.bio) data.bio = bio;

      if (Object.keys(data).length === 0) {
        setError('변경된 내용이 없습니다.');
        setLoading(false);
        return;
      }

      const updatedUser = await authApi.updateProfile(data);
      setUser(updatedUser);
      setSuccess('프로필이 업데이트되었습니다.');
      setIsEditing(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || '프로필 업데이트에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

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
      setIsChangingPassword(false);
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setError(err.response?.data?.detail || '비밀번호 변경에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const cancelEdit = () => {
    setIsEditing(false);
    setUsername(user?.username || '');
    setBio(user?.bio || '');
    setError(null);
  };

  const cancelPasswordChange = () => {
    setIsChangingPassword(false);
    setOldPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setError(null);
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-neutral-50 dark:from-slate-900 to-neutral-100 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-neutral-900 dark:text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-neutral-500 dark:text-slate-400">프로필 로딩 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-neutral-50 dark:from-slate-900 to-neutral-100 dark:to-slate-800">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute inset-0 overflow-hidden">
          <User className="absolute top-8 right-[10%] w-16 h-16 text-white/10" />
          <Shield className="absolute bottom-6 right-[25%] w-12 h-12 text-white/10" />
        </div>

        <div className="relative max-w-4xl mx-auto px-4 py-12">
          <div className="flex items-center gap-6">
            <div className="w-24 h-24 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center text-4xl font-bold text-white shadow-2xl border border-white/20">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold text-neutral-900 dark:text-white">{user.username}</h1>
                {user.role === 'admin' && (
                  <span className="flex items-center gap-1 px-3 py-1 text-sm bg-amber-500/20 text-amber-300 rounded-full border border-amber-400/30">
                    <Crown className="w-4 h-4" />
                    관리자
                  </span>
                )}
              </div>
              <p className="text-emerald-100 mt-1">{user.email}</p>
              <div className="flex items-center gap-4 mt-3">
                <span className={`flex items-center gap-1.5 px-3 py-1 text-sm rounded-full ${
                  user.is_active
                    ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-400/30'
                    : 'bg-red-500/30 text-red-200 border border-red-400/30'
                }`}>
                  <ShieldCheck className="w-4 h-4" />
                  {user.is_active ? '활성 계정' : '비활성 계정'}
                </span>
                <span className={`flex items-center gap-1.5 px-3 py-1 text-sm rounded-full ${
                  user.is_verified
                    ? 'bg-blue-500/30 text-blue-200 border border-blue-400/30'
                    : 'bg-amber-500/30 text-amber-200 border border-amber-400/30'
                }`}>
                  <Mail className="w-4 h-4" />
                  {user.is_verified ? '이메일 인증됨' : '이메일 미인증'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-8 -mt-6">
        {/* Alert Messages */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-xl flex items-center gap-3 text-red-300">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-300">
              <X className="w-5 h-5" />
            </button>
          </div>
        )}
        {success && (
          <div className="mb-6 p-4 bg-emerald-500/20 border border-emerald-500/30 rounded-xl flex items-center gap-3 text-emerald-300">
            <CheckCircle className="w-5 h-5 flex-shrink-0" />
            <span>{success}</span>
            <button onClick={() => setSuccess(null)} className="ml-auto text-emerald-400 hover:text-emerald-300">
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Profile Info Card */}
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-neutral-200 dark:border-slate-700/50 overflow-hidden shadow-xl">
          <div className="px-6 py-4 border-b border-neutral-200 dark:border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <User className="w-5 h-5 text-emerald-400" />
              <h2 className="text-lg font-semibold text-neutral-900 dark:text-white">프로필 정보</h2>
            </div>
            {!isEditing && (
              <button
                onClick={() => setIsEditing(true)}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm transition-colors"
              >
                <Edit3 className="w-4 h-4" />
                수정
              </button>
            )}
          </div>

          <div className="p-6">
            {!isEditing ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-1">
                  <label className="text-sm text-neutral-500 dark:text-slate-400 flex items-center gap-2">
                    <User className="w-4 h-4" />
                    사용자명
                  </label>
                  <p className="text-neutral-900 dark:text-white text-lg">{user.username}</p>
                </div>
                <div className="space-y-1">
                  <label className="text-sm text-neutral-500 dark:text-slate-400 flex items-center gap-2">
                    <Mail className="w-4 h-4" />
                    이메일
                  </label>
                  <p className="text-neutral-900 dark:text-white text-lg">{user.email}</p>
                </div>
                <div className="md:col-span-2 space-y-1">
                  <label className="text-sm text-neutral-500 dark:text-slate-400 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    자기소개
                  </label>
                  <p className="text-neutral-900 dark:text-white">{user.bio || '아직 자기소개가 없습니다.'}</p>
                </div>
                <div className="space-y-1">
                  <label className="text-sm text-neutral-500 dark:text-slate-400 flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    가입일
                  </label>
                  <p className="text-neutral-900 dark:text-white">
                    {new Date(user.created_at).toLocaleDateString('ko-KR', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    })}
                  </p>
                </div>
                {user.last_login_at && (
                  <div className="space-y-1">
                    <label className="text-sm text-neutral-500 dark:text-slate-400 flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      마지막 로그인
                    </label>
                    <p className="text-neutral-900 dark:text-white">
                      {new Date(user.last_login_at).toLocaleDateString('ko-KR', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <form onSubmit={handleUpdateProfile} className="space-y-6">
                <div>
                  <label htmlFor="username" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                    사용자명
                  </label>
                  <input
                    type="text"
                    id="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full px-4 py-3 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    minLength={3}
                    maxLength={30}
                    required
                  />
                </div>
                <div>
                  <label htmlFor="bio" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                    자기소개
                  </label>
                  <textarea
                    id="bio"
                    value={bio}
                    onChange={(e) => setBio(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-3 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
                    maxLength={200}
                    placeholder="간단한 자기소개를 입력해주세요 (최대 200자)"
                  />
                  <p className="text-sm text-slate-500 mt-1">{bio.length}/200</p>
                </div>
                <div className="flex gap-3">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl transition-all disabled:opacity-50"
                  >
                    {loading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Save className="w-4 h-4" />
                    )}
                    {loading ? '저장 중...' : '저장'}
                  </button>
                  <button
                    type="button"
                    onClick={cancelEdit}
                    className="flex items-center gap-2 px-5 py-2.5 bg-neutral-100 dark:bg-slate-700 hover:bg-neutral-200 dark:hover:bg-slate-600 text-neutral-600 dark:text-slate-300 rounded-xl transition-colors"
                  >
                    <X className="w-4 h-4" />
                    취소
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>

        {/* Security Card */}
        <div className="mt-6 bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-neutral-200 dark:border-slate-700/50 overflow-hidden shadow-xl">
          <div className="px-6 py-4 border-b border-neutral-200 dark:border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Lock className="w-5 h-5 text-amber-400" />
              <h2 className="text-lg font-semibold text-neutral-900 dark:text-white">보안</h2>
            </div>
            {!isChangingPassword && (
              <button
                onClick={() => setIsChangingPassword(true)}
                className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-sm transition-colors"
              >
                <Lock className="w-4 h-4" />
                비밀번호 변경
              </button>
            )}
          </div>

          <div className="p-6">
            {!isChangingPassword ? (
              <div className="flex items-center gap-4 p-4 bg-neutral-100 dark:bg-slate-700/30 rounded-xl">
                <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center">
                  <Shield className="w-6 h-6 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-neutral-900 dark:text-white font-medium">비밀번호</h3>
                  <p className="text-sm text-neutral-500 dark:text-slate-400">정기적으로 비밀번호를 변경하여 계정을 안전하게 보호하세요.</p>
                </div>
              </div>
            ) : (
              <form onSubmit={handleChangePassword} className="space-y-4">
                <div>
                  <label htmlFor="oldPassword" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                    현재 비밀번호
                  </label>
                  <input
                    type="password"
                    id="oldPassword"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                    새 비밀번호
                  </label>
                  <input
                    type="password"
                    id="newPassword"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    minLength={8}
                    required
                  />
                </div>
                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-neutral-600 dark:text-slate-300 mb-2">
                    새 비밀번호 확인
                  </label>
                  <input
                    type="password"
                    id="confirmPassword"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-neutral-100 dark:bg-slate-700/50 border border-neutral-200 dark:border-slate-600 rounded-xl text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    minLength={8}
                    required
                  />
                  {confirmPassword && newPassword !== confirmPassword && (
                    <p className="text-sm text-red-400 mt-1">비밀번호가 일치하지 않습니다</p>
                  )}
                </div>
                <div className="flex gap-3 pt-2">
                  <button
                    type="submit"
                    disabled={loading || newPassword !== confirmPassword}
                    className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white rounded-xl transition-all disabled:opacity-50"
                  >
                    {loading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Lock className="w-4 h-4" />
                    )}
                    {loading ? '변경 중...' : '비밀번호 변경'}
                  </button>
                  <button
                    type="button"
                    onClick={cancelPasswordChange}
                    className="flex items-center gap-2 px-5 py-2.5 bg-neutral-100 dark:bg-slate-700 hover:bg-neutral-200 dark:hover:bg-slate-600 text-neutral-600 dark:text-slate-300 rounded-xl transition-colors"
                  >
                    <X className="w-4 h-4" />
                    취소
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
