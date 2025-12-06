'use client';

import { useServer } from '@/contexts/ServerContext';

export default function AdminPanel() {
  const { status, toggleServer } = useServer();

  return (
    <div className="fixed top-5 right-5 z-50 flex flex-col items-end gap-2">
      {/* Khung điều khiển */}
      <div className="bg-slate-900/90 backdrop-blur-md text-white p-4 rounded-xl shadow-2xl border border-slate-700 w-64">
        <div className="flex justify-between items-center mb-3">
          <span className="text-xs font-bold text-slate-400 tracking-widest uppercase">
            System Control
          </span>
          {/* Đèn trạng thái */}
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === 'offline'
                  ? 'bg-red-500'
                  : status === 'booting'
                    ? 'bg-yellow-400 animate-pulse'
                    : 'bg-green-500 shadow-[0_0_8px_#22c55e]'
              }`}
            />
            <span className="text-xs font-mono font-bold">
              {status === 'offline'
                ? 'OFFLINE'
                : status === 'booting'
                  ? 'BOOTING'
                  : 'LIVE'}
            </span>
          </div>
        </div>
        {/* Nút bấm Công tắc */}
        <button
          onClick={toggleServer}
          disabled={status === 'booting'}
          className={`
            w-full py-2 rounded-lg font-bold text-sm transition-all duration-300 flex items-center justify-center gap-2
            ${
              status === 'online'
                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/50'
                : status === 'booting'
                  ? 'bg-yellow-500/20 text-yellow-400 cursor-wait border border-yellow-500/50'
                  : 'bg-green-600 hover:bg-green-500 text-white shadow-lg transform hover:-translate-y-1'
            }
          `}
        >
          {status === 'online'
            ? '⛔ SHUT DOWN'
            : status === 'booting'
              ? 'CONNECTING...'
              : '🚀 START SERVER'}
        </button>
        <div className="mt-2 text-[10px] text-slate-500 text-center">
          Mode: Manual Override (Demo)
        </div>
      </div>
    </div>
  );
}

