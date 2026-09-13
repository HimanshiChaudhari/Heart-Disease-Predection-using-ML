import React from 'react';
import { Activity, Shield, Database, Github } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800 bg-slate-950 text-slate-400 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/30 flex items-center justify-center">
                <Activity className="w-4 h-4 text-rose-400" />
              </div>
              <span className="text-base font-bold text-white tracking-tight">HeartAI</span>
            </div>
            <p className="text-xs text-slate-400 max-w-md leading-relaxed mb-4">
              A high-precision Machine Learning heart disease risk screening application built on the UCI Heart Disease dataset.
              Providing calibrated risk probability estimations and interpretable clinical factor breakdowns.
            </p>
            <div className="text-xs text-slate-500">
              Original Project: <span className="font-mono text-slate-400">HimanshiChaudhari/Heart-Disease-Predection-using-ML</span>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-indigo-400" />
              ML Architecture
            </h4>
            <ul className="space-y-2 text-xs">
              <li className="hover:text-slate-200 transition-colors">Scikit-Learn ColumnTransformer</li>
              <li className="hover:text-slate-200 transition-colors">Logistic Regression (~80% Acc)</li>
              <li className="hover:text-slate-200 transition-colors">Balanced Decision Tree (Entropy)</li>
              <li className="hover:text-slate-200 transition-colors">Support Vector Classifiers (Linear & RBF)</li>
              <li className="hover:text-slate-200 transition-colors">K-Nearest Neighbors (k=21)</li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              Ethics & Transparency
            </h4>
            <ul className="space-y-2 text-xs">
              <li>Non-Diagnostic Demonstration</li>
              <li>Calibrated Screening Probabilities</li>
              <li>Log-Odds Feature Contributions</li>
              <li>Zero Training-Serving Skew</li>
              <li>No Treatment Recommendations</li>
            </ul>
          </div>

        </div>

        <div className="border-t border-slate-900 pt-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-3">
          <p>&copy; {new Date().getFullYear()} HeartAI &bull; Educational Machine Learning Screening Demonstration</p>
          <p className="text-slate-500 text-center sm:text-right">
            Not for clinical emergencies &bull; Consult a certified medical doctor for clinical diagnosis
          </p>
        </div>

      </div>
    </footer>
  );
};
