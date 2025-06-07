// frontend/src/app/job-matches/page.tsx
'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Modal from '@/components/ui/modal';
import Toast from '@/components/ui/toast';

// --- SVG Icons for Save Button ---
const BookmarkIconOutline = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
    </svg>
);
const BookmarkIconSolid = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" viewBox="0 0 20 20" fill="currentColor">
        <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
    </svg>
);

// --- NEW: Helper Component for Visualizing Scores ---
interface ProgressBarProps {
    score: number; // A value between 0 and 1
    label: string;
}

const ProgressBar = ({ score, label }: ProgressBarProps) => {
    const percent = Math.round(score * 100);
    let bgColor = 'bg-red-500';
    if (percent >= 70) {
        bgColor = 'bg-green-500';
    } else if (percent >= 40) {
        bgColor = 'bg-yellow-500';
    }

    return (
        <div>
            <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">{label}</span>
                <span className="text-sm font-medium text-gray-700">{percent}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                    className={`${bgColor} h-2.5 rounded-full transition-all duration-500`} 
                    style={{ width: `${percent}%` }}
                ></div>
            </div>
        </div>
    );
};


// --- UPDATED TypeScript Interfaces ---
interface ScoreComponents {
    semantic_profile_score: number;
    keyword_skill_score: number;
    experience_level_score: number;
    education_semantic_score: number;
}

interface MatchedJob {
    id: string;
    title: string;
    company: string | null;
    location?: string | null;
    description?: string | null;
    requiredSkills: string[];
    matchScore: number;
    experience_level_required?: string | null;
    score_components: ScoreComponents; // <-- NEW
    matching_keywords: string[];     // <-- NEW
}

export default function JobMatchesPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    const [matches, setMatches] = useState<MatchedJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sortOrder, setSortOrder] = useState('matchScore_desc');
    
    const [savedJobIds, setSavedJobIds] = useState<Set<string>>(new Set());
    const [toastMessage, setToastMessage] = useState<string | null>(null);
    const [toastType, setToastType] = useState<'success' | 'error'>('success');
    
    const [isJobModalOpen, setIsJobModalOpen] = useState(false);
    const [selectedJob, setSelectedJob] = useState<MatchedJob | null>(null);

    const openJobModal = (job: MatchedJob) => {
        setSelectedJob(job);
        setIsJobModalOpen(true);
    };

    const closeJobModal = () => {
        setIsJobModalOpen(false);
        setTimeout(() => setSelectedJob(null), 300);
    };

    const fetchInitialData = useCallback(async () => {
        if (!user) return;
        setIsLoading(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            const [profileRes, matchesRes] = await Promise.all([
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, { headers: { 'Authorization': `Bearer ${token}` } }),
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/jobs/match`, { headers: { 'Authorization': `Bearer ${token}` } })
            ]);

            if (profileRes.ok) {
                const profileData = await profileRes.json();
                setSavedJobIds(new Set(profileData.saved_job_ids || []));
            } else {
                console.warn("Could not fetch user profile to get saved jobs.");
            }

            if (!matchesRes.ok) {
                let errorDetail = `Failed to fetch job matches (${matchesRes.status})`;
                try {
                    const errorData = await matchesRes.json();
                    errorDetail = errorData.detail || errorDetail;
                } catch (e) {}
                throw new Error(errorDetail);
            }
            const matchesData: MatchedJob[] = await matchesRes.json();
            setMatches(matchesData);

        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    useEffect(() => {
        if (!authLoading && !user) router.push('/');
        if (user) fetchInitialData();
    }, [user, authLoading, router, fetchInitialData]);

    const showAndClearMessage = (message: string, type: 'success' | 'error') => {
        setToastType(type);
        setToastMessage(message);
        setTimeout(() => setToastMessage(null), 3000);
    };

    const handleToggleSave = async (jobId: string, isCurrentlySaved: boolean) => {
        if (!user) return;
        
        const originalSavedIds = new Set(savedJobIds);
        const newSavedJobIds = new Set(savedJobIds);
        if (isCurrentlySaved) {
            newSavedJobIds.delete(jobId);
        } else {
            newSavedJobIds.add(jobId);
        }
        setSavedJobIds(newSavedJobIds);

        try {
            const token = await user.getIdToken();
            const method = isCurrentlySaved ? 'DELETE' : 'POST';
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/saved-jobs/${jobId}`, {
                method,
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (!response.ok) throw new Error("Failed to update status.");
            showAndClearMessage(isCurrentlySaved ? 'Job unsaved' : 'Job saved!', 'success');
        } catch (err) {
            setSavedJobIds(originalSavedIds);
            showAndClearMessage((err as Error).message || "An error occurred.", 'error');
        }
    };

    const sortedMatches = useMemo(() => {
        const sortableMatches = [...matches];
        switch (sortOrder) {
            case 'matchScore_asc': return sortableMatches.sort((a, b) => a.matchScore - b.matchScore);
            case 'title_asc': return sortableMatches.sort((a, b) => a.title.localeCompare(b.title));
            case 'title_desc': return sortableMatches.sort((a, b) => b.title.localeCompare(a.title));
            case 'matchScore_desc': default: return sortableMatches.sort((a, b) => b.matchScore - a.matchScore);
        }
    }, [matches, sortOrder]);
    
    if (authLoading) { return <div className="flex justify-center items-center min-h-screen text-gray-500">Checking authentication...</div>; }
    if (!user) { return <div className="flex justify-center items-center min-h-screen text-gray-500">Redirecting to login...</div>; }

    return (
        <>
            <div className="bg-gray-50 min-h-screen">
                <div className="container mx-auto p-4 sm:p-6 lg:p-8">
                    {/* Header & Sorting Controls */}
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-800">Your Job Matches</h1>
                            {!isLoading && matches.length > 0 && (
                                <p className="text-gray-600 mt-1">Found {matches.length} relevant opportunities for you.</p>
                            )}
                        </div>
                        <div>
                            <label htmlFor="sort" className="block text-sm font-medium text-gray-700 mb-1">Sort by</label>
                            <select id="sort" name="sort" className="font-sans block w-full sm:w-auto pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 rounded-md shadow-sm" value={sortOrder} onChange={(e) => setSortOrder(e.target.value)} disabled={isLoading}>
                                <option value="matchScore_desc">Match Score: High to Low</option>
                                <option value="matchScore_asc">Match Score: Low to High</option>
                                <option value="title_asc">Alphabetical: A-Z</option>
                                <option value="title_desc">Alphabetical: Z-A</option>
                            </select>
                        </div>
                    </div>
                    
                    {isLoading && <div className="space-y-6 max-w-4xl mx-auto">{[...Array(5)].map((_, i) => <div key={i} className="p-6 bg-white rounded-lg shadow-md animate-pulse"><div className="h-6 bg-gray-300 rounded w-2/3 mb-3"></div><div className="h-4 bg-gray-300 rounded w-1/3 mb-5"></div><div className="flex flex-wrap gap-2"><div className="h-5 bg-gray-200 rounded-full w-20"></div><div className="h-5 bg-gray-200 rounded-full w-24"></div><div className="h-5 bg-gray-200 rounded-full w-16"></div></div></div>)}</div>}
                    {error && <div className="my-4 text-center text-red-600 p-4 bg-red-100 rounded shadow max-w-4xl mx-auto"><p><strong>Error fetching job matches:</strong></p><p>{error}</p><button onClick={fetchInitialData} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Retry</button></div>}

                    {!isLoading && !error && (
                        <div className="max-w-4xl mx-auto">
                            {sortedMatches.length > 0 ? (
                                <div className="space-y-6">
                                    {sortedMatches.map((job) => {
                                        const isSaved = savedJobIds.has(job.id);
                                        return (
                                            <div key={job.id} className="p-6 bg-white rounded-lg shadow-md border border-gray-200 transition hover:shadow-lg hover:border-indigo-300 relative group">
                                                <button onClick={(e) => { e.stopPropagation(); handleToggleSave(job.id, isSaved); }} className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 text-gray-500 hover:text-indigo-600 transition-colors z-10" aria-label={isSaved ? 'Unsave job' : 'Save job'}>
                                                    {isSaved ? <BookmarkIconSolid /> : <BookmarkIconOutline />}
                                                </button>
                                                <div className="cursor-pointer pr-12" onClick={() => openJobModal(job)}>
                                                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-2 gap-2">
                                                        <div className="flex-grow"><h2 className="text-xl font-semibold text-indigo-700 group-hover:underline">{job.title}</h2><p className="text-md text-gray-800">{job.company}</p>{job.location && job.location !== 'N/A' && <p className="text-sm text-gray-500">{job.location}</p>}</div>
                                                        <div className={`flex-shrink-0 px-3 py-1 rounded-full text-sm font-medium ${job.matchScore >= 0.7 ? 'bg-green-100 text-green-800' : job.matchScore >= 0.5 ? 'bg-yellow-100 text-yellow-800' : 'bg-orange-100 text-orange-800'}`}>{`${(job.matchScore * 100).toFixed(0)}% Match`}</div>
                                                    </div>
                                                    {job.experience_level_required && job.experience_level_required !== 'Unknown' && (<div className="my-3"><p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Experience Level</p><p className="text-sm text-gray-700">{job.experience_level_required}</p></div>)}
                                                    {job.requiredSkills && job.requiredSkills.length > 0 && (<div className="mt-4"><h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Top Skills</h4><div className="flex flex-wrap gap-2 mt-1">{job.requiredSkills.slice(0, 5).map(skill => (<span key={skill} className="text-xs bg-gray-200 text-gray-800 px-2 py-1 rounded-md">{skill}</span>))}</div></div>)}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : ( <div className="text-center py-12 bg-white rounded-lg shadow-md"><h3 className="text-xl font-semibold text-gray-700">No Job Matches Found</h3><p className="mt-2 text-gray-500">Based on your current profile, we couldn't find any relevant opportunities.</p><Link href="/profile" className="mt-6 inline-block px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">Update Your Profile</Link></div>)}
                        </div>
                    )}
                </div>
            </div>
            
            <Toast show={!!toastMessage} message={toastMessage || ''} type={toastType} />
            
            <Modal isOpen={isJobModalOpen} onClose={closeJobModal}>
                {selectedJob && (
                    <div className="space-y-6">
                        {/* Main Job Info */}
                        <div className="flex justify-between items-start">
                            <div>
                                <h2 className="text-2xl font-bold text-indigo-700">{selectedJob.title}</h2>
                                <p className="text-lg text-gray-800">{selectedJob.company}</p>
                                {selectedJob.location && selectedJob.location !== 'N/A' && <p className="text-md text-gray-500">{selectedJob.location}</p>}
                            </div>
                            <span className={`flex-shrink-0 px-4 py-1.5 rounded-full text-md font-semibold ${selectedJob.matchScore >= 0.7 ? 'bg-green-100 text-green-800' : selectedJob.matchScore >= 0.5 ? 'bg-yellow-100 text-yellow-800' : 'bg-orange-100 text-orange-800'}`}>{`${(selectedJob.matchScore * 100).toFixed(0)}% Match`}</span>
                        </div>

                        {/* NEW: Match Insights Section */}
                        <div className="border-t border-gray-200 pt-4 space-y-4">
                            <h3 className="text-lg font-semibold text-gray-800">Match Insights</h3>
                            <ProgressBar score={selectedJob.score_components.semantic_profile_score} label="Profile Context Match" />
                            <div>
                                <ProgressBar score={selectedJob.score_components.keyword_skill_score} label="Keyword Skill Match" />
                                {selectedJob.matching_keywords.length > 0 && (
                                    <div className="mt-2 pl-2">
                                        <p className="text-xs text-gray-500">Your matching skills:</p>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                            {selectedJob.matching_keywords.map(skill => (
                                                <span key={skill} className="text-xs bg-green-100 text-green-800 font-medium px-2 py-0.5 rounded-full">{skill}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                            <ProgressBar score={selectedJob.score_components.experience_level_score} label="Experience Level Match" />
                            <ProgressBar score={selectedJob.score_components.education_semantic_score} label="Education Relevance" />
                        </div>

                        {/* Full Job Details */}
                        <div className="border-t border-gray-200 pt-4 space-y-4">
                            {selectedJob.description && (
                                <div>
                                    <h3 className="font-semibold text-gray-700 mb-1">Job Description</h3>
                                    <p className="text-gray-600 whitespace-pre-wrap">{selectedJob.description}</p>
                                </div>
                            )}
                             {selectedJob.requiredSkills && selectedJob.requiredSkills.length > 0 && (
                                <div>
                                    <h3 className="font-semibold text-gray-700 mb-2">All Required Skills</h3>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedJob.requiredSkills.map(skill => (
                                            <span key={skill} className="text-sm bg-gray-200 text-gray-800 px-3 py-1 rounded-md">{skill}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </Modal>
        </>
    );
}