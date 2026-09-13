import React from 'react';
import { Hero } from '../components/Hero';
import { HowItWorks } from '../components/HowItWorks';
import { ProjectStats } from '../components/ProjectStats';
import { FeatureCard } from '../components/FeatureCard';
import { Disclaimer } from '../components/Disclaimer';
import { ShieldCheck, Cpu, Database, Stethoscope, HeartPulse, LineChart } from 'lucide-react';

interface HomeProps {
  onStartScreening: () => void;
  onViewModelBenchmark: () => void;
}

export const Home: React.FC<HomeProps> = ({ onStartScreening, onViewModelBenchmark }) => {
  const handleScrollToHowItWorks = () => {
    const el = document.getElementById('how-it-works-section');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="space-y-4">
      
      {/* 14. Hero Section with exact headings and pure SVG/CSS medical AI visual */}
      <Hero
        onStartScreening={onStartScreening}
        onHowItWorks={handleScrollToHowItWorks}
      />

      {/* 16. Project Statistics with verified actual values from the project */}
      <ProjectStats />

      {/* 15. How It Works - 4 Clean Visual Process Steps */}
      <HowItWorks />

      {/* Medical Ethics & Screening Notice */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <Disclaimer variant="card" />
      </div>

      {/* Feature Capabilities Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="text-center max-w-2xl mx-auto mb-10">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Data Science & Cardiovascular Intelligence
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-2">
            Translating multi-dimensional cardiac diagnostic markers into interpretable risk probabilities.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <FeatureCard
            icon={Cpu}
            title="Logistic Regression Flagship"
            description="Trained on 644 patients and validated on 276 test samples to yield ~80.07% accuracy and 83.56% screening sensitivity."
            badge="Flagship Model"
            badgeColor="text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
          />

          <FeatureCard
            icon={LineChart}
            title="Calibrated Risk Probabilities"
            description="Outputs model-estimated percentages (0% – 100%) rather than binary yes/no decisions, allowing clinical risk stratification."
            badge="Continuous Metric"
            badgeColor="text-indigo-400 bg-indigo-500/10 border-indigo-500/20"
          />

          <FeatureCard
            icon={Database}
            title="Zero Training-Serving Skew"
            description="Encapsulated in Scikit-Learn ColumnTransformers, ensuring exact imputation means, modes, and ordinal encodings match training data."
            badge="Production Pipeline"
            badgeColor="text-rose-400 bg-rose-500/10 border-rose-500/20"
          />

          <FeatureCard
            icon={HeartPulse}
            title="13 Diagnostic Biomarkers"
            description="Incorporates exercise ST depression (Oldpeak), resting blood pressure, cholesterol, resting ECG, and fluoroscopy vessel counts."
            badge="Full Schema"
            badgeColor="text-amber-400 bg-amber-500/10 border-amber-500/20"
          />

          <FeatureCard
            icon={Stethoscope}
            title="Screening Sensitivity Priority"
            description="Designed specifically to minimize False Negatives in screening scenarios, ensuring high-risk patient indicators are surfaced."
            badge="Clinical Context"
            badgeColor="text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
          />

          <FeatureCard
            icon={ShieldCheck}
            title="Transparent Model Benchmarks"
            description="Compare Logistic Regression with Decision Trees, Linear SVM, RBF SVM, and KNN (k=21) side-by-side on identical test splits."
            badge="Open Evaluation"
            badgeColor="text-indigo-400 bg-indigo-500/10 border-indigo-500/20"
          />
        </div>
      </section>

    </div>
  );
};
