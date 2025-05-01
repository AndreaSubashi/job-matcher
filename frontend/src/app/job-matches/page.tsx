// frontend/src/app/job-matches/page.tsx
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link'; 

// --- Interface for Matched Job (Matches backend MatchedJob Pydantic model) ---
interface MatchedJob {
    id: string;
    title: string;
    company: string;
    location?: string | null;
    description?: string | null;
    requiredSkills: string[];
    matchScore: number; // Score between 0.0 and 1.0
}

export default function JobMatchesPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    const [matchedJobs, setMatchedJobs] = useState<MatchedJob[]>([]);
    const [loadingMatches, setLoadingMatches] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // --- Route Protection ---
    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/login');
        }
    }, [user, authLoading, router]);

    // --- Fetch Job Matches ---
    const fetchMatches = useCallback(async () => {
        if (!user) return; // Only fetch if user is logged in

        console.log("Fetching job matches for user:", user.uid);
        setLoadingMatches(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            // Adjust threshold via query param if needed, e.g.?min_score_threshold=0.5
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/jobs/match`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                let errorDetail = `Failed to fetch job matches (${response.status})`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || errorDetail;
                } catch (e) { /* Ignore */ }
                throw new Error(errorDetail);
            }

            const data: MatchedJob[] = await response.json();
            setMatchedJobs(data);
            console.log("Job matches received:", data);

        } catch (err: any) {
            console.error("Error fetching job matches:", err);
            setError(err.message);
        } finally {
            setLoadingMatches(false);
        }
    }, [user]); // Depend on user

    // Fetch matches when user is available
    useEffect(() => {
        if (user) {
            fetchMatches();
        }
    }, [user, fetchMatches]);


    // --- Render Logic ---
    if (authLoading) {
        return <div className="flex justify-center items-center min-h-screen">Checking authentication...</div>;
    }
    if (!user) {
        // Should be redirected by the effect, but show message as fallback
        return <div className="flex justify-center items-center min-h-screen">Redirecting to login...</div>;
    }

    return (
        <div className="container mx-auto p-4 md:p-8 max-w-5xl">
            <h1 className="text-3xl font-bold mb-6">Your Job Matches</h1>

            {/* Loading State */}
            {loadingMatches && (
                 <div className="text-center py-10">
                    <p className="text-lg text-gray-600">Finding relevant jobs...</p>
                    {/* Add a spinner here if desired */}
                 </div>
            )}

            {/* Error State */}
            {error && (
                 <div className="my-4 text-center text-red-600 p-4 bg-red-100 rounded shadow">
                     <p><strong>Error fetching job matches:</strong></p>
                     <p>{error}</p>
                     <button onClick={fetchMatches} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Retry</button>
                </div>
            )}

            {/* Results Display */}
            {!loadingMatches && !error && (
                <div>
                    {matchedJobs.length > 0 ? (
                        <div className="space-y-6">
                            {matchedJobs.map((job) => (
                                <div key={job.id} className="p-6 bg-white rounded-lg shadow-md border border-gray-200 transition hover:shadow-lg">
                                    <div className="flex justify-between items-start mb-2">
                                        <h2 className="text-xl font-semibold text-indigo-700">{job.title}</h2>
                                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                            job.matchScore >= 0.7 ? 'bg-green-100 text-green-800' :
                                            job.matchScore >= 0.4 ? 'bg-yellow-100 text-yellow-800' :
                                            'bg-red-100 text-red-800'
                                            }`}>
                                            {/* Format score as percentage */}
                                            {`${(job.matchScore * 100).toFixed(0)}% Match`}
                                        </span>
                                    </div>
                                    <p className="text-md text-gray-800 mb-1">{job.company}</p>
                                    {job.location && <p className="text-sm text-gray-500 mb-3">{job.location}</p>}

                                    {/* Required Skills */}
                                    {job.requiredSkills && job.requiredSkills.length > 0 && (
                                        <div className="mb-4">
                                            <h4 className="text-sm font-medium text-gray-600 mb-1">Required Skills:</h4>
                                            <div className="flex flex-wrap gap-2">
                                                {job.requiredSkills.map(skill => (
                                                    <span key={skill} className="text-xs bg-gray-200 text-gray-800 px-2 py-0.5 rounded">
                                                        {skill}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Optional: Job Description Snippet */}
                                    {job.description && (
                                        <p className="text-sm text-gray-600 line-clamp-3"> {/* Shows first 3 lines */}
                                            {job.description}
                                        </p>
                                        // Add a "Read More" link/button if needed later
                                    )}
                                    {/* Optional: Add link to actual job posting if available */}
                                </div>
                            ))}
                        </div>
                    ) : (
                        // No matches found message
                         <div className="text-center py-10">
                            <p className="text-lg text-gray-600">No job matches found based on your current profile.</p>
                            <p className="mt-2 text-sm text-gray-500">Try adding more skills to your <Link href="/profile" className="text-indigo-600 hover:underline">profile</Link>.</p>
                         </div>
                    )}
                </div>
            )}
        </div>
    );
}