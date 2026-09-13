import React from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

interface DisclaimerProps {
  variant?: 'banner' | 'card';
  className?: string;
}

export const Disclaimer: React.FC<DisclaimerProps> = ({ variant = 'banner', className = '' }) => {
  if (variant === 'banner') {
    return (
      <aside className={`bg-amber-950/40 border-b border-amber-800/40 text-amber-200/90 px-4 py-2.5 text-xs sm:text-sm ${className}`} role="alert">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <div className="leading-snug">
            <span className="font-semibold text-amber-300">Important Health Screening Notice: </span>
            This ML-based result is for educational and screening purposes only. It is not a medical diagnosis and should not replace professional medical evaluation. If you have symptoms or concerns about your health, consult a qualified healthcare professional.
          </div>
        </div>
      </aside>
    );
  }

  return (
    <div className={`rounded-xl bg-slate-900/80 border border-amber-500/20 p-4 text-xs leading-relaxed text-slate-300 ${className}`}>
      <div className="flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-semibold text-amber-300 text-sm mb-1">Clinical Screening & Medical Ethics Notice</h4>
          <p className="mb-1.5">
            This tool generates a <strong>model-estimated probability</strong> based on pattern matching with the 920-patient UCI Heart Disease dataset.
            The system does <strong>not</strong> make clinical determinations such as &ldquo;You have heart disease&rdquo; or &ldquo;You do not have heart disease&rdquo;.
          </p>
          <p className="text-amber-200/80 font-medium">
            &bull; This ML-based result is for educational and screening purposes only. It is not a medical diagnosis and should not replace professional medical evaluation.
          </p>
          <p className="text-amber-200/80 font-medium mt-1">
            &bull; If you have symptoms or concerns about your health, consult a qualified healthcare professional.
          </p>
        </div>
      </div>
    </div>
  );
};
