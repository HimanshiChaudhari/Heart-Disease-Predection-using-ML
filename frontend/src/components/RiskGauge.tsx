import React from 'react';

interface RiskGaugeProps {
  probability: number | null;
  riskLevel: 'Low' | 'Moderate' | 'High';
  predictionLabel: string;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ probability, riskLevel, predictionLabel }) => {
  const percentage = probability !== null && probability !== undefined 
    ? Math.round(probability * 100) 
    : (riskLevel === 'High' ? 85 : 15);

  // SVG circle calculations
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  let strokeColor = '#10b981'; // Emerald
  let badgeBg = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
  let badgeText = 'LOWER PREDICTED RISK';

  if (riskLevel === 'High' || percentage >= 50) {
    strokeColor = '#f43f5e'; // Rose
    badgeBg = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
    badgeText = 'HIGHER PREDICTED RISK';
  } else if (riskLevel === 'Moderate' || percentage >= 35) {
    strokeColor = '#f59e0b'; // Amber
    badgeBg = 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    badgeText = 'MODERATE PREDICTED RISK';
  }

  return (
    <div className="flex flex-col items-center text-center py-4">
      
      {/* Circular SVG Gauge */}
      <div className="relative w-48 h-48 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 170 170">
          {/* Background track */}
          <circle
            cx="85"
            cy="85"
            r={radius}
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            className="text-slate-800"
          />
          {/* Progress stroke */}
          <circle
            cx="85"
            cy="85"
            r={radius}
            stroke={strokeColor}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-extrabold tracking-tight font-mono text-white">
            {percentage}%
          </span>
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mt-0.5">
            Estimated Risk
          </span>
        </div>
      </div>

      {/* Structured Label Badge */}
      <div className="mt-4">
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase border ${badgeBg}`}>
          {badgeText}
        </span>
      </div>

      {/* Model-estimated probability notice */}
      <div className="mt-3 max-w-xs">
        <p className="text-sm font-semibold text-slate-200">
          {predictionLabel}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          Model-estimated probability based on patient clinical indicators.
        </p>
      </div>

    </div>
  );
};
