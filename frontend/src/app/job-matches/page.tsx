// frontend/src/app/job-matches/page.tsx
'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Modal from '@/components/ui/modal'; // Import the reusable Modal component

// --- Interface for Matched Job (Matches backend MatchedJob Pydantic model) ---
interface MatchedJob {
    id: string;
    title: string;
    company: string | null;
    location?: string | null;
    description?: string | null; // Added for modal display
    requiredSkills: string[];
    matchScore: number;
    experience_level_required?: string | null;
}

export default function JobMatchesPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    const [matches, setMatches] = useState<MatchedJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sortOrder, setSortOrder] = useState('matchScore_desc'); 

    // --- State for the Job Details Modal ---
    const [isJobModalOpen, setIsJobModalOpen] = useState(false);
    const [selectedJob, setSelectedJob] = useState<MatchedJob | null>(null);

    const openJobModal = (job: MatchedJob) => {
        setSelectedJob(job);
        setIsJobModalOpen(true);
    };

    const closeJobModal = () => {
        setIsJobModalOpen(false);
        // Deselect job after a delay to allow for closing animation
        setTimeout(() => setSelectedJob(null), 300);
    };

    const fetchMatches = useCallback(async () => {
        if (!user) return;

        setIsLoading(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/jobs/match`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (!response.ok) {
                let errorDetail = `Failed to fetch job matches (${response.status})`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || errorDetail;
                } catch (e) {
                    // Ignore if response is not JSON
                }
                throw new Error(errorDetail);
            }

            const data: MatchedJob[] = await response.json();
            setMatches(data);
        } catch (err: any) {
            console.error("Error fetching job matches:", err);
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/');
        }
        if (user) {
            fetchMatches();
        }
    }, [user, authLoading, router, fetchMatches]);

    const sortedMatches = useMemo(() => {
        const sortableMatches = [...matches];
        switch (sortOrder) {
            case 'matchScore_asc':
                return sortableMatches.sort((a, b) => a.matchScore - b.matchScore);
            case 'title_asc':
                return sortableMatches.sort((a, b) => a.title.localeCompare(b.title));
            case 'title_desc':
                return sortableMatches.sort((a, b) => b.title.localeCompare(a.title));
            case 'matchScore_desc':
            default:
                return sortableMatches.sort((a, b) => b.matchScore - a.matchScore);
        }
    }, [matches, sortOrder]);


    if (authLoading) {
        return <div className="flex justify-center items-center min-h-screen text-gray-500">Checking authentication...</div>;
    }
    if (!user) {
        return <div className="flex justify-center items-center min-h-screen text-gray-500">Redirecting to login...</div>;
    }

    return (
        <>
            <div className="bg-gray-50 min-h-screen">
                <div className="container mx-auto p-4 sm:p-6 lg:p-8">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-800">Your Job Matches</h1>
                            {!isLoading && matches.length > 0 && (
                                <p className="text-gray-600 mt-1">Found {matches.length} relevant opportunities for you.</p>
                            )}
                        </div>
                        <div>
                            <label htmlFor="sort" className="block text-sm font-medium text-gray-700 mb-1">Sort by</label>
                            <select
                                id="sort"
                                name="sort"
                                // Applied font-sans to ensure font consistency
                                className="font-sans block w-full sm:w-auto pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 rounded-md shadow-sm"
                                value={sortOrder}
                                onChange={(e) => setSortOrder(e.target.value)}
                                disabled={isLoading}
                            >
                                <option value="matchScore_desc">Match Score: High to Low</option>
                                <option value="matchScore_asc">Match Score: Low to High</option>
                                <option value="title_asc">Alphabetical: A-Z</option>
                                <option value="title_desc">Alphabetical: Z-A</option>
                            </select>
                        </div>
                    </div>

                    {/* Loading State Skeleton */}
                    {isLoading && (
                        <div className="space-y-6 max-w-4xl mx-auto">
                            {[...Array(5)].map((_, i) => (
                                <div key={i} className="p-6 bg-white rounded-lg shadow-md animate-pulse">
                                    <div className="h-6 bg-gray-300 rounded w-2/3 mb-3"></div>
                                    <div className="h-4 bg-gray-300 rounded w-1/3 mb-5"></div>
                                    <div className="flex flex-wrap gap-2">
                                        <div className="h-5 bg-gray-200 rounded-full w-20"></div>
                                        <div className="h-5 bg-gray-200 rounded-full w-24"></div>
                                        <div className="h-5 bg-gray-200 rounded-full w-16"></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Error State */}
                    {error && (
                        <div className="my-4 text-center text-red-600 p-4 bg-red-100 rounded shadow max-w-4xl mx-auto">
                            <p><strong>Error fetching job matches:</strong></p>
                            <p>{error}</p>
                            <button onClick={fetchMatches} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Retry</button>
                        </div>
                    )}

                    {/* Results Display */}
                    {!isLoading && !error && (
                        <div className="max-w-4xl mx-auto"> {/* Centering and max-width wrapper */}
                            {sortedMatches.length > 0 ? (
                                <div className="space-y-6">
                                    {sortedMatches.map((job) => (
                                        <div 
                                            key={job.id} 
                                            className="p-6 bg-white rounded-lg shadow-md border border-gray-200 transition hover:shadow-lg hover:border-indigo-300 cursor-pointer"
                                            onClick={() => openJobModal(job)} // Open modal on click
                                        >
                                            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-2 gap-2">
                                                <div className="flex-grow">
                                                    <h2 className="text-xl font-semibold text-indigo-700">{job.title}</h2>
                                                    <p className="text-md text-gray-800">{job.company}</p>
                                                    {job.location && job.location !== 'N/A' && <p className="text-sm text-gray-500">{job.location}</p>}
                                                </div>
                                                <div className={`flex-shrink-0 px-3 py-1 rounded-full text-sm font-medium ${
                                                    job.matchScore >= 0.7 ? 'bg-green-100 text-green-800' :
                                                    job.matchScore >= 0.5 ? 'bg-yellow-100 text-yellow-800' :
                                                    'bg-orange-100 text-orange-800'
                                                    }`}>
                                                    {`${(job.matchScore * 100).toFixed(0)}% Match`}
                                                </div>
                                            </div>
                                            
                                            {job.experience_level_required && job.experience_level_required !== 'Unknown' && (
                                                <div className="my-3">
                                                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Experience Level</p>
                                                    <p className="text-sm text-gray-700">{job.experience_level_required}</p>
                                                </div>
                                            )}

                                            {job.requiredSkills && job.requiredSkills.length > 0 && (
                                                <div className="mt-4">
                                                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Top Skills</h4>
                                                    <div className="flex flex-wrap gap-2 mt-1">
                                                        {job.requiredSkills.slice(0, 5).map(skill => (
                                                            <span key={skill} className="text-xs bg-gray-200 text-gray-800 px-2 py-1 rounded-md">
                                                                {skill}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-center py-12 bg-white rounded-lg shadow-md">
                                    <h3 className="text-xl font-semibold text-gray-700">No Job Matches Found</h3>
                                    <p className="mt-2 text-gray-500">Based on your current profile, we couldn't find any relevant opportunities.</p>
                                    <Link href="/profile" className="mt-6 inline-block px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">
                                        Update Your Profile
                                    </Link>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* --- Job Details Modal --- */}
            <Modal isOpen={isJobModalOpen} onClose={closeJobModal}>
                {selectedJob && (
                    <div className="space-y-4">
                        <div className="flex justify-between items-start">
                            <div>
                                <h2 className="text-2xl font-bold text-indigo-700">{selectedJob.title}</h2>
                                <p className="text-lg text-gray-800">{selectedJob.company}</p>
                                {selectedJob.location && selectedJob.location !== 'N/A' && <p className="text-md text-gray-500">{selectedJob.location}</p>}
                            </div>
                            <span className={`flex-shrink-0 px-4 py-1.5 rounded-full text-md font-semibold ${
                                selectedJob.matchScore >= 0.7 ? 'bg-green-100 text-green-800' :
                                selectedJob.matchScore >= 0.5 ? 'bg-yellow-100 text-yellow-800' :
                                'bg-orange-100 text-orange-800'
                                }`}>
                                {`${(selectedJob.matchScore * 100).toFixed(0)}% Match`}
                            </span>
                        </div>

                        <div className="border-t border-gray-200 pt-4 space-y-4">
                            {selectedJob.experience_level_required && selectedJob.experience_level_required !== 'Unknown' && (
                                <div>
                                    <h3 className="font-semibold text-gray-700 mb-1">Experience Level</h3>
                                    <p className="text-gray-600">{selectedJob.experience_level_required}</p>
                                </div>
                            )}

                            {selectedJob.requiredSkills && selectedJob.requiredSkills.length > 0 && (
                                <div>
                                    <h3 className="font-semibold text-gray-700 mb-2">Required Skills</h3>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedJob.requiredSkills.map(skill => (
                                            <span key={skill} className="text-sm bg-gray-200 text-gray-800 px-3 py-1 rounded-md">
                                                {skill}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {selectedJob.description && (
                                <div>
                                    <h3 className="font-semibold text-gray-700 mb-1">Job Description</h3>
                                    <p className="text-gray-600 whitespace-pre-wrap">{selectedJob.description}</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </Modal>
        </>
    );
}

