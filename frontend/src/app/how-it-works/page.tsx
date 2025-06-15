import React from 'react';

const BrainIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.25 12l.813-2.846a4.5 4.5 0 00-3.09-3.09L13.125 5.25l-.813 2.846a4.5 4.5 0 003.09 3.09L18.25 12z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5.25l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 2.25l2.846.813a4.5 4.5 0 003.09 3.09L9 9l.813-2.846a4.5 4.5 0 003.09-3.09L15.75 2.25l-2.846.813a4.5 4.5 0 00-3.09 3.09L9 9z" />
    </svg>
);

const DocumentTextIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
);

const ChartBarIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
);

const AcademicCapIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path d="M12 14l9-5-9-5-9 5 9 5z" />
      <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" />
    </svg>
);


export default function HowItWorksPage() {
    return (
        <div className="bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto">
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">How Our Matching Works</h1>
                    <p className="mt-4 text-lg text-gray-600">
                        We use a hybrid approach to provide you with the most relevant job matches. Here's a breakdown of the key components that create your Match Score.
                    </p>
                </div>

                <div className="mt-12 space-y-10">
                    {/* Semantic Profile Match */}
                    <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
                        <div className="flex-shrink-0 h-20 w-20 flex items-center justify-center bg-indigo-100 rounded-lg">
                           <BrainIcon />
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">1. Semantic Profile Match</h2>
                            <p className="mt-2 text-gray-600">
                                This is the core "AI" component. We convert your professional profile (skills and work experience descriptions) into a rich numerical representation called a semantic embedding. We do the same for each job's description. The system then calculates how similar the *meaning and context* of your profile is to the job description, going far beyond simple keywords. This helps find jobs that are a good conceptual fit.
                            </p>
                        </div>
                    </div>

                    {/* Keyword Skill Match */}
                    <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
                        <div className="flex-shrink-0 h-20 w-20 flex items-center justify-center bg-indigo-100 rounded-lg">
                            <DocumentTextIcon />
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">2. Keyword Skill Match</h2>
                            <p className="mt-2 text-gray-600">
                                Explicit skills are critical. This component directly compares the skills you've listed against the specific skills we've extracted from the job posting using NLP. It calculates a score based on the percentage of the job's required skills that you possess. This ensures jobs with hard requirements that you meet are rated highly.
                            </p>
                        </div>
                    </div>

                    {/* Experience Level Match */}
                    <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
                         <div className="flex-shrink-0 h-20 w-20 flex items-center justify-center bg-indigo-100 rounded-lg">
                            <ChartBarIcon />
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">3. Experience Level Match</h2>
                            <p className="mt-2 text-gray-600">
                                We calculate your total years of experience from your profile and categorize it (e.g., Junior, Mid-Level, Senior). This is compared against the required experience level listed for the job. This score ensures you're matched with roles appropriate for your seniority, and it penalizes mismatches for both under- and over-qualification.
                            </p>
                        </div>
                    </div>
                    
                    {/* Education Match */}
                    <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8">
                         <div className="flex-shrink-0 h-20 w-20 flex items-center justify-center bg-indigo-100 rounded-lg">
                            <AcademicCapIcon />
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">4. Education Match</h2>
                            <p className="mt-2 text-gray-600">
                                Your education contributes in two ways. First, you get a small bonus just for having a degree listed. More importantly, we semantically compare your degree (e.g., "Computer Science") against the job's required technical skills. A degree that is highly relevant to the job's technical domain will receive a higher score, adding a targeted boost to your match.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="mt-12 p-6 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <h2 className="text-2xl font-semibold text-center text-indigo-800">The Final Hybrid Score</h2>
                    <p className="mt-3 text-center text-indigo-700">
                        These individual scores (semantic, keyword, experience, and education) are combined using different weights to produce the final "Match Score %" you see. This hybrid approach ensures that the recommendations are balanced, considering both the nuanced meaning of your profile and the explicit requirements of the job, giving you a more accurate and reliable set of opportunities.
                    </p>
                </div>
            </div>
        </div>
    );
}