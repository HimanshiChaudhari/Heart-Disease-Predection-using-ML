import React from 'react';
import { LucideIcon } from 'lucide-react';

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  badge?: string;
  badgeColor?: string;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({
  icon: Icon,
  title,
  description,
  badge,
  badgeColor = 'text-indigo-400 bg-indigo-500/10 border-indigo-500/20',
}) => {
  return (
    <div className="glow-card rounded-2xl p-6 relative overflow-hidden group">
      <div className="flex items-center justify-between mb-4">
        <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-indigo-400 group-hover:text-rose-400 group-hover:border-rose-500/30 transition-all">
          <Icon className="w-5 h-5" />
        </div>
        {badge && (
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider border ${badgeColor}`}>
            {badge}
          </span>
        )}
      </div>
      <h3 className="text-base font-bold text-white mb-2 group-hover:text-slate-100 transition-colors">
        {title}
      </h3>
      <p className="text-xs text-slate-400 leading-relaxed">
        {description}
      </p>
    </div>
  );
};
