import React from 'react';
import { ClipboardEdit, Binary, Cpu, CheckCircle2, ArrowRight } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const steps = [
    {
      number: '01',
      title: 'Enter Health Information',
      description:
        'Input available cardiovascular indicators: patient demographics, resting blood pressure, cholesterol, resting ECG, and exercise stress testing parameters.',
      icon: ClipboardEdit,
      badge: 'Step 1 &bull; Clinical Input',
      badgeColor: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    },
    {
      number: '02',
      title: 'Data Processing',
      description:
        'The Scikit-Learn ColumnTransformer imputes missing numerical values using dataset means and missing categorical values using modes, followed by standard scaling and ordinal encoding.',
      icon: Binary,
      badge: 'Step 2 &bull; Preprocessing',
      badgeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    },
    {
      number: '03',
      title: 'Machine Learning Analysis',
      description:
        'The preprocessed feature vector is evaluated by the trained model (Logistic Regression, Decision Tree, SVM, or KNN) to estimate risk likelihood and log-odds contributions.',
      icon: Cpu,
      badge: 'Step 3 &bull; Model Inference',
      badgeColor: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    },
    {
      number: '04',
      title: 'Receive Prediction',
      description:
        'View the model-estimated probability percentage, screening risk level, and transparent clinical risk factors. Results include clear educational disclaimers.',
      icon: CheckCircle2,
      badge: 'Step 4 &bull; Result Delivery',
      badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    },
  ];

  return (
    <section id="how-it-works-section" className="py-16 border-b border-slate-800/80 relative">
      
      {/* Background glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[250px] bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-semibold text-indigo-400 mb-3">
            <span>Visual Process Flow</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            How It Works
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-2">
            A 4-step pipeline ensuring complete reproducibility from raw clinical input to model-based screening.
          </p>
        </div>

        {/* Process Steps Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative">
          
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isLast = idx === steps.length - 1;

            return (
              <div key={idx} className="relative flex flex-col">
                
                {/* Step Card */}
                <div className="glow-card rounded-2xl p-6 flex-1 flex flex-col justify-between border border-slate-800/80 hover:border-slate-700 transition-all">
                  
                  <div>
                    {/* Top Step Header */}
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-3xl font-black font-mono text-slate-700 group-hover:text-indigo-400 transition-colors">
                        {step.number}
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider border ${step.badgeColor}`}
                        dangerouslySetInnerHTML={{ __html: step.badge }}
                      />
                    </div>

                    {/* Step Icon */}
                    <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-indigo-400 mb-4 shadow-sm">
                      <Icon className="w-6 h-6" />
                    </div>

                    {/* Step Title */}
                    <h3 className="text-base font-bold text-white mb-2">
                      {step.title}
                    </h3>

                    {/* Step Description */}
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {step.description}
                    </p>
                  </div>

                  {/* Step Footer Indicator */}
                  <div className="mt-5 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500 font-mono">
                    <span>Phase {step.number} of 04</span>
                    <span className="text-slate-400 font-semibold">{step.number === '04' ? 'Complete' : 'Next &rarr;'}</span>
                  </div>

                </div>

                {/* Connector Arrow (Visible on large screens) */}
                {!isLast && (
                  <div className="hidden lg:flex absolute -right-3.5 top-1/2 -translate-y-1/2 z-20 w-7 h-7 rounded-full bg-slate-900 border border-slate-700 text-slate-400 items-center justify-center shadow-lg pointer-events-none">
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                )}

              </div>
            );
          })}

        </div>

      </div>
    </section>
  );
};
