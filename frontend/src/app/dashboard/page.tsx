// frontend/src/app/dashboard/page.tsx
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

// --- TypeScript Interfaces for the data we expect ---

// Simplified UserProfile to get completeness status
interface UserProfile {
    skills: any[];
    education: any[];
    experience: any[];
    displayName?: string;
}

// Job match data structure
interface MatchedJob {
    id: string;
    title: string;
    company: string | null;
    matchScore: number;
}

// --- Helper Components (defined within the file for simplicity) ---

// Loading Skeleton for a Job Card
const JobCardSkeleton = () => (
    <div className="p-4 bg-gray-100 rounded-lg animate-pulse">
        <div className="h-5 bg-gray-300 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-gray-300 rounded w-1/2 mb-4"></div>
        <div className="h-8 bg-gray-300 rounded-full w-20 ml-auto"></div>
    </div>
);

// Loading Skeleton for the Profile Status card
const ProfileStatusSkeleton = () => (
    <div className="p-6 bg-white rounded-lg shadow-md animate-pulse">
        <div className="h-6 bg-gray-300 rounded w-1/2 mb-4"></div>
        <div className="h-4 bg-gray-300 rounded-full w-full mb-4"></div>
        <div className="space-y-3 mt-4">
            <div className="h-5 bg-gray-300 rounded w-3/5"></div>
            <div className="h-5 bg-gray-300 rounded w-4/5"></div>
            <div className="h-5 bg-gray-300 rounded w-2/5"></div>
        </div>
        <div className="h-10 bg-indigo-200 rounded mt-6"></div>
    </div>
);


export default function DashboardPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    // State for data fetched from our APIs
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [matches, setMatches] = useState<MatchedJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        if (!user) return;
        setIsLoading(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            
            // Fetch both profile and matches in parallel
            const [profileRes, matchesRes] = await Promise.all([
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }),
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/jobs/match`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                })
            ]);

            if (!profileRes.ok) {
                // If profile fetch fails, it might be a new user. We can still show matches.
                console.warn("Could not fetch user profile, it might be new.");
            } else {
                const profileData: UserProfile = await profileRes.json();
                setProfile(profileData);
            }

            if (!matchesRes.ok) {
                const errorData = await matchesRes.json();
                throw new Error(errorData.detail || "Failed to fetch job matches.");
            }
            const matchesData: MatchedJob[] = await matchesRes.json();
            setMatches(matchesData);

        } catch (err: any) {
            console.error("Dashboard fetch error:", err);
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
            fetchData();
        }
    }, [user, authLoading, router, fetchData]);


    // --- Profile Completeness Calculation ---
    const completeness = {
        skills: profile && profile.skills && profile.skills.length > 0,
        education: profile && profile.education && profile.education.length > 0,
        experience: profile && profile.experience && profile.experience.length > 0,
    };
    const completedSections = Object.values(completeness).filter(Boolean).length;
    const completenessPercent = Math.round((completedSections / 3) * 100);

    const greetingName = profile?.displayName || user?.email?.split('@')[0] || 'User';

    return (
        <div className="bg-gray-50 min-h-screen">
            <div className="container mx-auto p-4 sm:p-6 lg:p-8">
                {/* --- Header --- */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Welcome back, {greetingName}!</h1>
                    <p className="text-gray-600 mt-1">Here's your professional dashboard at a glance.</p>
                </div>
                
                {/* --- Main Content Grid --- */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    
                    {/* --- Left Column: Top Job Matches --- */}
                    <div className="lg:col-span-2">
                        <div className="p-6 bg-white rounded-lg shadow-md">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Your Top Job Matches</h2>
                            {isLoading ? (
                                <div className="space-y-4">
                                    <JobCardSkeleton />
                                    <JobCardSkeleton />
                                    <JobCardSkeleton />
                                </div>
                            ) : error ? (
                                <p className="text-red-500">Could not load job matches.</p>
                            ) : matches.length > 0 ? (
                                <div className="space-y-4">
                                    {/* Show top 3 matches */}
                                    {matches.slice(0, 3).map(job => (
                                        <div key={job.id} className="p-4 border border-gray-200 rounded-lg hover:shadow-lg hover:border-indigo-300 transition-shadow duration-200">
                                            <div className="flex justify-between items-center">
                                                <div>
                                                    <h3 className="font-semibold text-indigo-700">{job.title}</h3>
                                                    <p className="text-sm text-gray-600">{job.company}</p>
                                                </div>
                                                <span className="text-lg font-bold text-gray-700">
                                                    {`${(job.matchScore * 100).toFixed(0)}%`}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                    {matches.length > 3 && (
                                        <Link href="/job-matches" className="block text-center mt-6 px-4 py-2 text-indigo-600 font-semibold rounded-lg hover:bg-indigo-50 transition">
                                            View All {matches.length} Matches &rarr;
                                        </Link>
                                    )}
                                </div>
                            ) : (
                                <div className="text-center py-8">
                                    <p className="text-gray-600">No job matches found right now.</p>
                                    <p className="text-sm text-gray-500 mt-1">Try adding more details to your profile for better results.</p>
                                    <Link href="/profile" className="mt-4 inline-block px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">
                                        Update Profile
                                    </Link>
                                </div>
                            )}
                        </div>
                    </div>
                    
                    {/* --- Right Column: Profile Status --- */}
                    <div className="lg:col-span-1">
                        {isLoading ? (
                            <ProfileStatusSkeleton />
                        ) : (
                            <div className="p-6 bg-white rounded-lg shadow-md">
                                <h2 className="text-xl font-semibold text-gray-800 mb-4">Profile Status</h2>
                                
                                {/* Progress Bar */}
                                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
                                    <div 
                                        className="bg-green-500 h-2.5 rounded-full" 
                                        style={{ width: `${completenessPercent}%` }}
                                    ></div>
                                </div>
                                <p className="text-right text-sm font-medium text-gray-700 mb-4">{completenessPercent}% Complete</p>
                                
                                {/* Checklist */}
                                <ul className="space-y-3">
                                    <li className="flex items-center">
                                        {completeness.skills ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Skills Added</span>
                                    </li>
                                    <li className="flex items-center">
                                        {completeness.education ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Education Added</span>
                                    </li>
                                    <li className="flex items-center">
                                        {completeness.experience ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Experience Added</span>
                                    </li>
                                </ul>
                                
                                <Link href="/profile" className="block w-full text-center mt-6 px-4 py-2 text-white font-semibold bg-indigo-600 rounded-lg hover:bg-indigo-700 transition">
                                    {completenessPercent === 100 ? 'Review Profile' : 'Complete Profile'}
                                </Link>
                            </div>
                        )}

                         {/* Quick Stats Card */}
                         {!isLoading && matches.length > 0 && (
                            <div className="p-6 bg-white rounded-lg shadow-md mt-8">
                                <h2 className="text-xl font-semibold text-gray-800 mb-4">Quick Stats</h2>
                                <div className="space-y-2">
                                    <p className="text-gray-700">Total Matches Found: <span className="font-bold text-indigo-600">{matches.length}</span></p>
                                    <p className="text-gray-700">Highest Match Score: <span className="font-bold text-indigo-600">{`${(matches[0].matchScore * 100).toFixed(0)}%`}</span></p>
                                </div>
                            </div>
                         )}

                    </div>
                </div>
            </div>
        </div>
    );
}
