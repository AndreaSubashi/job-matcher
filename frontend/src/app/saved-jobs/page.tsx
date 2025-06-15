'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Toast from '@/components/ui/toast';

//structure for each saved job
interface SavedJob {
    id: string;
    title: string;
    company: string | null;
    location?: string | null;
}

//icon for the unsave (x) button
const XCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
    </svg>
);

export default function SavedJobsPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    const [savedJobs, setSavedJobs] = useState<SavedJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [toastMessage, setToastMessage] = useState<string | null>(null);
    const [toastType, setToastType] = useState<'success' | 'error'>('success');

    //fetch saved jobs from backend
    const fetchSavedJobs = useCallback(async () => {
        if (!user) return;
        setIsLoading(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/saved-jobs`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (!response.ok) throw new Error("failed to fetch saved jobs");
            const data: SavedJob[] = await response.json();
            setSavedJobs(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    //check auth and trigger fetch
    useEffect(() => {
        if (!authLoading && !user) router.push('/');
        if (user) fetchSavedJobs();
    }, [user, authLoading, router, fetchSavedJobs]);

    //toast message trigger
    const showAndClearMessage = (message: string) => {
        setToastMessage(message);
        setTimeout(() => setToastMessage(null), 3000);
    };

    //handle unsave click
    const handleUnsave = async (jobId: string) => {
        if (!user) return;

        const originalJobs = [...savedJobs]; //backup in case it fails
        setSavedJobs(prev => prev.filter(job => job.id !== jobId)); //optimistic update
        setToastType('success');
        showAndClearMessage("job unsaved");

        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/saved-jobs/${jobId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (!response.ok) throw new Error("failed to unsave job");
        } catch (err) {
            console.error(err);
            setSavedJobs(originalJobs); //revert if error
            setToastType('error');
            showAndClearMessage("could not unsave job. please try again");
        }
    };
    
    return (
        <>
            <div className="bg-gray-50 min-h-screen">
                <div className="container mx-auto p-4 sm:p-6 lg:p-8">
                    <div className="mb-8">
                        <h1 className="text-3xl font-bold text-gray-800">your saved jobs</h1>
                        <p className="text-gray-600 mt-1">review and manage the opportunities you're interested in</p>
                    </div>

                    {isLoading && <p>loading saved jobs...</p>}
                    {error && <p className="text-red-500">{error}</p>}
                    
                    {!isLoading && !error && (
                        <div className="max-w-4xl mx-auto">
                            {savedJobs.length > 0 ? (
                                <div className="space-y-4">
                                    {savedJobs.map(job => (
                                        <div key={job.id} className="p-4 bg-white rounded-lg shadow-md border flex items-center justify-between gap-4">
                                            <div className="flex-grow">
                                                <p className="font-semibold text-indigo-700">{job.title}</p>
                                                <p className="text-sm text-gray-700">{job.company}</p>
                                                {job.location && job.location !== 'N/A' && <p className="text-xs text-gray-500">{job.location}</p>}
                                            </div>
                                            <button 
                                                onClick={() => handleUnsave(job.id)}
                                                className="flex-shrink-0 p-2 rounded-full text-gray-400 hover:bg-red-100 hover:text-red-600 transition-colors"
                                                aria-label="unsave job"
                                            >
                                                <XCircleIcon />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-center py-12 bg-white rounded-lg shadow-md">
                                    <h3 className="text-xl font-semibold text-gray-700">no jobs saved yet</h3>
                                    <p className="mt-2 text-gray-500">click the bookmark icon on a job listing to save it here</p>
                                    <Link href="/job-matches" className="mt-6 inline-block px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">
                                        find job matches
                                    </Link>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
            <Toast show={!!toastMessage} message={toastMessage || ''} type={toastType} />
        </>
    );
}
