import { Outlet } from 'react-router-dom';
import { Header } from './Header';

export function MainLayout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-neutral-200 py-4 text-center text-sm text-neutral-500">
        Code Tutor AI - AI-powered Algorithm Learning Platform
      </footer>
    </div>
  );
}
