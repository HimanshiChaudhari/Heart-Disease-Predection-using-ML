import React from 'react';
import { PredictionResult as ResultType } from '../types/prediction';
import { PredictionResult } from '../components/PredictionResult';
import { Activity, ArrowLeft } from 'lucide-react';

interface ResultsProps {
  result: ResultType | null;
  onBackToForm: () => void;
}

export const Results: React.FC<ResultsProps> = ({ result, onBackToForm }) => {
  if (!result) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center space-y-4">
        <h2 className="text-xl font-bold text-white">No Assessment on Record</h2>
        <p className="text-xs text-slate-400">
          Please submit a patient clinical profile to generate a risk screening report.
        </p>
        <button
          onClick={onBackToForm}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 text-white font-semibold text-xs"
        >
          <ArrowLeft className="w-4 h-4" /> Go to Screening Form
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-rose-500" />
          <h1 className="text-xl font-bold text-white">Patient Assessment Report</h1>
        </div>
        <button
          onClick={onBackToForm}
          className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-semibold"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Form
        </button>
      </div>

      <PredictionResult result={result} onReset={onBackToForm} />
    </div>
  );
};
