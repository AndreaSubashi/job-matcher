'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';

// Define interface for profile data consistency (matches backend response model)
interface UserProfile {
    uid: string;
    email?: string;
    displayName?: string;
    photoURL?: string;
    createdAt?: string;
    skills: string[];
    education: any[]; // Use more specific types later
    experience: any[]; // Use more specific types later
}

export default function ProfilePage() {
    const { user, loading: authLoading, logout } = useAuth();
    const router = useRouter();

    // State for profile data
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [skills, setSkills] = useState<string[]>([]); // Local state for editing skills
    const [newSkill, setNewSkill] = useState('');

    // State for API interactions
    const [loadingProfile, setLoadingProfile] = useState(true);
    const [savingSkills, setSavingSkills] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);

    // --- Route Protection ---
    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/login');
        }
    }, [user, authLoading, router]);

    // --- Fetch Profile Data ---
    const fetchProfile = useCallback(async () => {
        if (!user) return; // Don't fetch if user is not logged in

        setLoadingProfile(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, { // Use environment variable for API URL
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to fetch profile (${response.status})`);
            }

            const data: UserProfile = await response.json();
            setProfile(data);
            setSkills(data.skills || []); // Initialize local skills state
            console.log("Profile data fetched:", data);

        } catch (err: any) {
            console.error("Error fetching profile:", err);
            setError(err.message);
            // Maybe redirect or show persistent error if profile can't be fetched
        } finally {
            setLoadingProfile(false);
        }
    }, [user]); // Depend on user object

    useEffect(() => {
        // Fetch profile when user object is available
        if (user) {
            fetchProfile();
        }
    }, [user, fetchProfile]); // Include fetchProfile in dependencies

    // --- Skill Management Handlers ---
    const handleAddSkill = () => {
        const trimmedSkill = newSkill.trim();
        // Prevent adding empty or duplicate skills (case-insensitive check)
        if (trimmedSkill && !skills.some(skill => skill.toLowerCase() === trimmedSkill.toLowerCase())) {
            setSkills([...skills, trimmedSkill]); // Add to local state
            setNewSkill(''); // Clear input
            setSuccessMessage(null); // Clear previous success message
        } else if (trimmedSkill) {
             setError(`Skill "${trimmedSkill}" already added.`);
             setTimeout(() => setError(null), 3000); // Clear error after 3s
        }
        setNewSkill(''); // Clear input regardless
    };

     const handleRemoveSkill = (skillToRemove: string) => {
        setSkills(skills.filter(skill => skill !== skillToRemove)); // Remove from local state
        setSuccessMessage(null); // Clear previous success message
    };


    // --- Save Skills Handler ---
    const handleSaveSkills = async () => {
        if (!user) return;

        setSavingSkills(true);
        setError(null);
        setSuccessMessage(null);
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/skills`, { // Use environment variable
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ skills: skills }), // Send updated skills list
            });

             if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to save skills (${response.status})`);
            }

            const updatedData = await response.json();
            console.log("Skills saved:", updatedData);
            setSuccessMessage("Skills updated successfully!");
             setTimeout(() => setSuccessMessage(null), 3000); // Clear success message after 3s

            // setSkills(updatedData.skills);

        } catch (err: any) {
             console.error("Error saving skills:", err);
             setError(err.message);
        } finally {
            setSavingSkills(false);
        }
    };


    // --- Render Logic ---
    if (authLoading || loadingProfile) {
        return <div className="flex justify-center items-center min-h-screen">Loading Profile...</div>; // 
    }

    if (!user) {
        return <div className="flex justify-center items-center min-h-screen">Redirecting to login...</div>;
    }

    if (!profile) {
         // Handle case where profile fetch failed after loading completed
         return <div className="flex flex-col justify-center items-center min-h-screen">
             <p className="text-red-600">Error loading profile data.</p>
             {error && <p className="text-sm text-red-500 mt-2">{error}</p>}
             <button onClick={fetchProfile} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Retry</button>
        </div>;
    }

    // Main Profile Content
    return (
        <div className="container mx-auto p-4 md:p-8 max-w-4xl">
            <h1 className="text-3xl font-bold mb-6">Your Profile</h1>
            <p className="mb-2"><strong>Email:</strong> {profile.email}</p>
            {profile.displayName && <p className="mb-6"><strong>Name:</strong> {profile.displayName}</p>}

            {/* --- Skills Section --- */}
            <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                <h2 className="text-2xl font-semibold mb-4">Skills</h2>

                 {/* Display existing skills */}
                 <div className="flex flex-wrap gap-2 mb-4">
                    {skills.length > 0 ? (
                        skills.map((skill) => (
                            <span key={skill} className="flex items-center bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full">
                                {skill}
                                <button
                                    onClick={() => handleRemoveSkill(skill)}
                                    className="ml-2 text-indigo-600 hover:text-indigo-800 text-lg leading-none"
                                    aria-label={`Remove ${skill}`}
                                >
                                    &times; {/* Multiplication sign as 'x' */}
                                </button>
                            </span>
                        ))
                    ) : (
                        <p className="text-gray-500 italic">No skills added yet.</p>
                    )}
                </div>

                {/* Add new skill input */}
                <div className="flex items-center gap-2 mb-4">
                   <input
                        type="text"
                        value={newSkill}
                        onChange={(e) => { setNewSkill(e.target.value); setError(null); }}
                        placeholder="Add a new skill (e.g., Python)"
                        className="flex-grow px-3 py-2 text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        onKeyDown={(e) => { if (e.key === 'Enter') handleAddSkill(); }} // Add skill on Enter key
                    />
                    <button
                        onClick={handleAddSkill}
                        className="px-4 py-2 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                        Add
                    </button>
                </div>

                {/* Save button and messages */}
                <div className="flex items-center justify-end gap-4 mt-4">
                     {error && <p className="text-sm text-red-600">{error}</p>}
                     {successMessage && <p className="text-sm text-green-600">{successMessage}</p>}
                     <button
                        onClick={handleSaveSkills}
                        disabled={savingSkills}
                        className="px-5 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
                    >
                        {savingSkills ? 'Saving...' : 'Save Skills'}
                    </button>
                </div>
            </div>

            {/* --- Education Section (Placeholder) --- */}
            <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                 <h2 className="text-2xl font-semibold mb-4">Education</h2>
                 {/* Add UI for Education here later */}
                 <p className="text-gray-500 italic">Education section coming soon.</p>
            </div>


            {/* --- Experience Section (Placeholder) --- */}
             <div className="p-6 bg-white rounded-lg shadow-md">
                 <h2 className="text-2xl font-semibold mb-4">Experience</h2>
                 {/* Add UI for Experience here later */}
                 <p className="text-gray-500 italic">Experience section coming soon.</p>
            </div>

            {/* Maybe add Logout button here too or in a Navbar */}
             <button
                onClick={logout}
                className="px-4 py-2 mt-8 font-bold text-white bg-red-500 rounded hover:bg-red-700"
             >
                Logout
            </button>
        </div>
    );
}