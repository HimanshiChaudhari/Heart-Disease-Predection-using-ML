import React from 'react';
import { Activity } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ 
  message = 'Executing Machine Learning inference pipeline...' 
}) => {
  return (
    <div className="glow-card rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-4">
      <div className="relative w-16 h-16 flex items-center justify-center">
        <div className="absolute inset-0 rounded-full border-4 border-slate-800" />
        <div className="absolute inset-0 rounded-full border-4 border-rose-500 border-t-transparent animate-spin" />
        <Activity className="w-6 h-6 text-rose-500 animate-pulse" />
      </div>
      <div>
        <h3 className="text-base font-bold text-white mb-1">Evaluating Clinical Parameters</h3>
        <p className="text-xs text-slate-400 max-w-sm">{message}</p>
      </div>
    </div>
  );
};
