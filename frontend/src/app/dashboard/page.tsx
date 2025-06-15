'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

//basic user profile structure - just what we need for dashboard
interface UserProfile {
    skills: any[];
    education: any[];
    experience: any[];
    displayName?: string;
}

//job match structure from our api
interface MatchedJob {
    id: string;
    title: string;
    company: string | null;
    matchScore: number;
}

//loading placeholder for job cards - shows while we fetch data
const JobCardSkeleton = () => (
    <div className="p-4 bg-gray-100 rounded-lg animate-pulse">
        <div className="h-5 bg-gray-300 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-gray-300 rounded w-1/2 mb-4"></div>
        <div className="h-8 bg-gray-300 rounded-full w-20 ml-auto"></div>
    </div>
);

//loading placeholder for profile status section
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

    //main dashboard data
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [matches, setMatches] = useState<MatchedJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    //fetch user profile and job matches from our backend
    const fetchData = useCallback(async () => {
        if (!user) return;
        
        setIsLoading(true);
        setError(null);
        
        try {
            const token = await user.getIdToken();
            
            //hit both endpoints at once to save time
            const [profileRes, matchesRes] = await Promise.all([
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }),
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/jobs/match`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                })
            ]);

            //profile might not exist for new users - that's ok
            if (!profileRes.ok) {
                console.warn("couldn't fetch user profile, might be a new user");
            } else {
                const profileData: UserProfile = await profileRes.json();
                setProfile(profileData);
            }

            //job matches are more critical - show error if this fails
            if (!matchesRes.ok) {
                const errorData = await matchesRes.json();
                throw new Error(errorData.detail || "failed to fetch job matches");
            }
            const matchesData: MatchedJob[] = await matchesRes.json();
            setMatches(matchesData);

        } catch (err: any) {
            console.error("dashboard fetch error:", err);
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    //redirect to login if not authenticated, otherwise fetch dashboard data
    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/');
        }
        if (user) {
            fetchData();
        }
    }, [user, authLoading, router, fetchData]);

    //calculate how complete the user's profile is
    const completeness = {
        skills: profile && profile.skills && profile.skills.length > 0,
        education: profile && profile.education && profile.education.length > 0,
        experience: profile && profile.experience && profile.experience.length > 0,
    };
    const completedSections = Object.values(completeness).filter(Boolean).length;
    const completenessPercent = Math.round((completedSections / 3) * 100);

    //figure out what to call the user - display name, email prefix, or fallback
    const greetingName = profile?.displayName || user?.email?.split('@')[0] || 'user';

    return (
        <div className="bg-gray-50 min-h-screen">
            <div className="container mx-auto p-4 sm:p-6 lg:p-8">
                {/*welcome header*/}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Welcome back, {greetingName}!</h1>
                    <p className="text-gray-600 mt-1">Here's your professional dashboard at a glance.</p>
                </div>
                
                {/*main dashboard layout - job matches on left, profile status on right*/}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    
                    {/*job matches section - takes up 2/3 of the width on desktop*/}
                    <div className="lg:col-span-2">
                        <div className="p-6 bg-white rounded-lg shadow-md">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Your top job matches</h2>
                            {isLoading ? (
                                //show loading skeletons while fetching
                                <div className="space-y-4">
                                    <JobCardSkeleton />
                                    <JobCardSkeleton />
                                    <JobCardSkeleton />
                                </div>
                            ) : error ? (
                                //simple error message if fetch failed
                                <p className="text-red-500">could not load job matches.</p>
                            ) : matches.length > 0 ? (
                                <div className="space-y-4">
                                    {/*show top 3 matches with hover effects*/}
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
                                    {/*if there are more than 3 matches, show link to see all*/}
                                    {matches.length > 3 && (
                                        <Link href="/job-matches" className="block text-center mt-6 px-4 py-2 text-indigo-600 font-semibold rounded-lg hover:bg-indigo-50 transition">
                                            View all {matches.length} matches &rarr;
                                        </Link>
                                    )}
                                </div>
                            ) : (
                                //empty state - encourage user to complete profile
                                <div className="text-center py-8">
                                    <p className="text-gray-600">no job matches found right now.</p>
                                    <p className="text-sm text-gray-500 mt-1">try adding more details to your profile for better results.</p>
                                    <Link href="/profile" className="mt-4 inline-block px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">
                                        update profile
                                    </Link>
                                </div>
                            )}
                        </div>
                    </div>
                    
                    {/*profile status sidebar - takes up 1/3 of width on desktop*/}
                    <div className="lg:col-span-1">
                        {isLoading ? (
                            <ProfileStatusSkeleton />
                        ) : (
                            <div className="p-6 bg-white rounded-lg shadow-md">
                                <h2 className="text-xl font-semibold text-gray-800 mb-4">Profile status</h2>
                                
                                {/*visual progress bar showing completion percentage*/}
                                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
                                    <div 
                                        className="bg-green-500 h-2.5 rounded-full" 
                                        style={{ width: `${completenessPercent}%` }}
                                    ></div>
                                </div>
                                <p className="text-right text-sm font-medium text-gray-700 mb-4">{completenessPercent}% complete</p>
                                
                                {/*checklist of profile sections*/}
                                <ul className="space-y-3">
                                    <li className="flex items-center">
                                        {completeness.skills ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Skills added</span>
                                    </li>
                                    <li className="flex items-center">
                                        {completeness.education ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Education added</span>
                                    </li>
                                    <li className="flex items-center">
                                        {completeness.experience ? '✅' : '❌'}
                                        <span className="ml-3 text-gray-700">Experience added</span>
                                    </li>
                                </ul>
                                
                                {/*cta button - text changes based on completion status*/}
                                <Link href="/profile" className="block w-full text-center mt-6 px-4 py-2 text-white font-semibold bg-indigo-600 rounded-lg hover:bg-indigo-700 transition">
                                    {completenessPercent === 100 ? 'review profile' : 'complete profile'}
                                </Link>
                            </div>
                        )}

                        {/*quick stats card - only show if we have match data   */}
                        {!isLoading && matches.length > 0 && (
                            <div className="p-6 bg-white rounded-lg shadow-md mt-8">
                                <h2 className="text-xl font-semibold text-gray-800 mb-4">Quick stats</h2>
                                <div className="space-y-2">
                                    <p className="text-gray-700">Total matches found: <span className="font-bold text-indigo-600">{matches.length}</span></p>
                                    <p className="text-gray-700">Highest match score: <span className="font-bold text-indigo-600">{`${(matches[0].matchScore * 100).toFixed(0)}%`}</span></p>
                                </div>
                            </div>
                        )}

                    </div>
                </div>
            </div>
        </div>
    );
}