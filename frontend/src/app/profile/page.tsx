// frontend/src/app/profile/page.tsx
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import { v4 as uuidv4 } from 'uuid'; // Import uuid generator

// --- Interfaces (Consider moving to a separate types file later) ---
interface EducationItem {
    id: string; // Using string for UUID representation for simplicity frontend/backend
    degree: string;
    school: string;
    startYear?: number | null; // Allow null or number
    endYear?: number | null;
}

interface EducationUpdateResponse {
    education: EducationItem[];
}

// NEW: Interface for Experience Item
interface ExperienceItem {
    id: string; // Use string for UUID
    title: string;
    company: string;
    // Use string for YYYY-MM format, matches backend model if using string
    startDate?: string | null;
    endDate?: string | null; // null or empty string means 'Present'
    location?: string | null;
    description?: string | null;
}

interface ExperienceUpdateResponse{
    experience: ExperienceItem[];
}

interface UserProfile {
    uid: string;
    email?: string;
    displayName?: string;
    photoURL?: string;
    createdAt?: string; // Keep as string based on backend serialization
    skills: string[];
    education: EducationItem[]; // Use the specific interface
    experience: any[]; // Use more specific types later
}

// Default empty states for forms
const defaultNewEducation: Partial<EducationItem> = { /* ... */ };
const defaultNewExperience: Partial<ExperienceItem> = { // NEW
    title: '',
    company: '',
    startDate: '',
    endDate: '',
    location: '',
    description: ''
};

export default function ProfilePage() {
    const { user, loading: authLoading, logout } = useAuth();
    const router = useRouter();

    // --- State ---
    const [profile, setProfile] = useState<UserProfile | null>(null);
    // Skills state
    const [skills, setSkills] = useState<string[]>([]);
    const [newSkill, setNewSkill] = useState('');
    const [savingSkills, setSavingSkills] = useState(false);
    // Education state
    const [educationItems, setEducationItems] = useState<EducationItem[]>([]);
    const [newEducationEntry, setNewEducationEntry] = useState<Partial<EducationItem>>(defaultNewEducation);
    const [savingEducation, setSavingEducation] = useState(false); // Separate saving state
    // NEW: Experience state
    const [experienceItems, setExperienceItems] = useState<ExperienceItem[]>([]);
    const [newExperienceEntry, setNewExperienceEntry] = useState<Partial<ExperienceItem>>(defaultNewExperience);
    const [savingExperience, setSavingExperience] = useState(false); // Separate saving state
    // General state
    const [loadingProfile, setLoadingProfile] = useState(true);
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
        if (!user) return;
        console.log("Fetching profile for user:", user.uid);
        setLoadingProfile(true);
        setError(null); // Clear previous errors on new fetch
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, {
                method: 'GET',
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (!response.ok) {
                let errorDetail = `Failed to fetch profile (${response.status})`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || errorDetail;
                } catch (e) { /* Ignore parsing error if response isn't JSON */ }
                throw new Error(errorDetail);
            }

            const data: UserProfile = await response.json();
            setProfile(data);
            setSkills(data.skills || []);
             // Ensure IDs are strings, handle potential null/undefined data defensively
            setEducationItems((data.education || []).map(edu => ({
                ...edu,
                id: String(edu.id), // Ensure ID is string
                startYear: edu.startYear ?? null, // Ensure null if missing
                endYear: edu.endYear ?? null, // Ensure null if missing
            })));
            setExperienceItems((data.experience || []).map(exp => ({
                ...exp,
                id: String(exp.id), // Ensure ID is string
                startDate: exp.startDate ?? null, // Handle null/undefined
                endDate: exp.endDate ?? null, // Handle null/undefined
                location: exp.location ?? null,
                description: exp.description ?? null,
            })));

            console.log("Profile data fetched successfully:", data);

        } catch (err: any) {
            console.error("Error fetching profile:", err);
            setError(err.message);
        } finally {
            setLoadingProfile(false);
        }
    }, [user]); // Depend on user object

    useEffect(() => { if (user && !profile) { fetchProfile(); } }, [user, profile, fetchProfile]);

    // --- Clear messages helper ---
    const clearMessages = () => {
        setError(null);
        setSuccessMessage(null);
    }

    // --- Skill Handlers ---
    const handleAddSkill = () => {
        clearMessages();
        const trimmedSkill = newSkill.trim();
        if (trimmedSkill && !skills.some(skill => skill.toLowerCase() === trimmedSkill.toLowerCase())) {
            setSkills([...skills, trimmedSkill]);
            setNewSkill('');
        } else if (trimmedSkill) {
             setError(`Skill "${trimmedSkill}" already added.`);
             setTimeout(clearMessages, 3000);
        }
        setNewSkill(''); // Clear input regardless
    };

     const handleRemoveSkill = (skillToRemove: string) => {
        clearMessages();
        setSkills(skills.filter(skill => skill !== skillToRemove));
    };

    const handleSaveSkills = async () => {
        if (!user) return;
        setSavingSkills(true);
        clearMessages();
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/skills`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', },
                body: JSON.stringify({ skills: skills }),
            });
             if (!response.ok) { /* ... error handling ... */ throw new Error('Failed save skills'); } // Simplified
            const updatedData = await response.json();
            console.log("Skills saved:", updatedData);
            setSuccessMessage("Skills updated successfully!");
             setTimeout(clearMessages, 3000);
        } catch (err: any) { /* ... error handling ... */ setError(err.message); }
        finally { setSavingSkills(false); }
    };

    // --- Education Handlers ---
    const handleEducationInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setNewEducationEntry(prev => ({
            ...prev,
            [name]: name === 'startYear' || name === 'endYear' ? (value === '' ? undefined : parseInt(value, 10)) : value
        }));
    };

    const handleAddNewEducation = () => {
        clearMessages();
        if (!newEducationEntry.degree || !newEducationEntry.school) {
             setError("Degree and School are required."); setTimeout(clearMessages, 3000); return;
        }
        if (newEducationEntry.startYear && newEducationEntry.endYear && newEducationEntry.startYear > newEducationEntry.endYear) {
            setError("Start year cannot be after end year."); setTimeout(clearMessages, 3000); return;
        }
        const newEntry: EducationItem = {
            id: uuidv4(), // Generate unique string ID
            degree: newEducationEntry.degree,
            school: newEducationEntry.school,
            startYear: newEducationEntry.startYear || null,
            endYear: newEducationEntry.endYear || null,
        };
        setEducationItems([...educationItems, newEntry]);
        setNewEducationEntry(defaultNewEducation); // Reset form
    };

     const handleRemoveEducation = (idToRemove: string) => {
        clearMessages();
        setEducationItems(educationItems.filter(item => item.id !== idToRemove));
    };

    const handleSaveEducation = async () => {
        if (!user) return;
        setSavingEducation(true);
        clearMessages(); // Use the helper function
        try {
            const token = await user.getIdToken();
             const payload = {
                education: educationItems.map(item => ({
                    id: item.id, // Already a string from frontend state
                    degree: item.degree,
                    school: item.school,
                    startYear: item.startYear ? Number(item.startYear) : null,
                    endYear: item.endYear ? Number(item.endYear) : null,
                }))
            };
            console.log("Saving education payload:", JSON.stringify(payload, null, 2));

            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/education`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

             if (!response.ok) {
                let errorDetail = `Failed to save education (${response.status})`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) {}
                throw new Error(errorDetail);
            }

            // Explicitly type the response data
            const updatedData: EducationUpdateResponse = await response.json(); // <-- TYPE ADDED HERE
            console.log("Education saved successfully:", updatedData);

            // Explicitly type 'edu' in map callback
            setEducationItems((updatedData.education || []).map((edu: EducationItem) => ({ // <-- TYPE ADDED HERE
                ...edu,
                id: String(edu.id) // Ensure ID remains string, just in case backend sent UUID object
            })));

            setSuccessMessage("Education details updated successfully!");
             setTimeout(clearMessages, 3000); // Use helper

        } catch (err: any) {
             console.error("Error saving education:", err);
             setError(err.message);
        } finally {
            setSavingEducation(false);
        }
    };

    // --- NEW: Experience Handlers ---
    const handleExperienceInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setNewExperienceEntry(prev => ({
            ...prev,
            // Keep dates as strings (YYYY-MM), handle textarea for description
            [name]: value
        }));
    };

     const handleAddNewExperience = () => {
        clearMessages();
        // Basic validation
        if (!newExperienceEntry.title || !newExperienceEntry.company) {
             setError("Job Title and Company are required for experience entries.");
             setTimeout(clearMessages, 3000);
             return;
        }
        // Optional: More robust date validation if needed

        const newEntry: ExperienceItem = {
            id: uuidv4(), // Generate unique string ID
            title: newExperienceEntry.title,
            company: newExperienceEntry.company,
            startDate: newExperienceEntry.startDate || null, // Store as null if empty string
            endDate: newExperienceEntry.endDate || null, // Store as null if empty string means 'Present'
            location: newExperienceEntry.location || null,
            description: newExperienceEntry.description || null,
        };
        setExperienceItems([...experienceItems, newEntry]); // Add to local state
        setNewExperienceEntry(defaultNewExperience); // Reset form
    };

     const handleRemoveExperience = (idToRemove: string) => {
        clearMessages();
        setExperienceItems(experienceItems.filter(item => item.id !== idToRemove));
    };

     const handleSaveExperience = async () => {
        if (!user) return;
        setSavingExperience(true);
        clearMessages();
        try {
            const token = await user.getIdToken();
            // Prepare payload, ensure it matches ExperienceUpdateRequest structure
            const payload = {
                experience: experienceItems.map(item => ({
                    id: item.id, // Send string UUID
                    title: item.title,
                    company: item.company,
                    startDate: item.startDate || null, // Ensure null if empty
                    endDate: item.endDate || null,   // Ensure null if empty
                    location: item.location || null,
                    description: item.description || null,
                }))
            };
             console.log("Saving experience payload:", JSON.stringify(payload, null, 2));

            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/experience`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', },
                body: JSON.stringify(payload),
            });
             if (!response.ok) { /* ... error handling ... */ throw new Error('Failed save experience'); } // Simplified
            const updatedData: ExperienceUpdateResponse = await response.json(); // Should match ExperienceUpdateRequest
            console.log("Experience saved successfully:", updatedData);

             // Re-sync state from response
            setExperienceItems((updatedData.experience || []).map((exp: ExperienceItem) => ({
                ...exp,
                id: String(exp.id),
                startDate: exp.startDate ?? null,
                endDate: exp.endDate ?? null,
                location: exp.location ?? null,
                description: exp.description ?? null,
             })));

            setSuccessMessage("Experience details updated successfully!");
            setTimeout(clearMessages, 3000);
        } catch (err: any) {
             console.error("Error saving experience:", err);
             setError(err.message);
        } finally {
            setSavingExperience(false);
        }
    };


    // --- Render Logic ---
    if (authLoading || loadingProfile) {
        return <div className="flex justify-center items-center min-h-screen">Loading Profile...</div>;
    }
    if (!user) {
        return <div className="flex justify-center items-center min-h-screen">Redirecting to login...</div>;
    }
     if (!profile && !loadingProfile) { // Check !loadingProfile too
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
             {/* Use optional chaining and fallback */}
            <p className="mb-2"><strong>Email:</strong> {profile?.email ?? 'N/A'}</p>
            {profile?.displayName && <p className="mb-6"><strong>Name:</strong> {profile.displayName}</p>}

             {/* Display general error/success messages */}
             {error && <p className="my-4 text-sm text-center text-red-600 p-2 bg-red-100 rounded">{error}</p>}
             {successMessage && <p className="my-4 text-sm text-center text-green-600 p-2 bg-green-100 rounded">{successMessage}</p>}


            {/* --- Skills Section --- */}
            <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                 <h2 className="text-2xl font-semibold mb-4">Skills</h2>
                 <div className="flex flex-wrap gap-2 mb-4">
                    {skills.length > 0 ? skills.map((skill) => (
                        <span key={skill} className="flex items-center bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full">
                            {skill}
                            <button onClick={() => handleRemoveSkill(skill)} className="ml-2 text-indigo-600 hover:text-indigo-800 text-lg leading-none" aria-label={`Remove ${skill}`}>&times;</button>
                        </span> )) : (<p className="text-gray-500 italic">No skills added yet.</p>)}
                </div>
                <div className="flex items-center gap-2 mb-4">
                   <input type="text" value={newSkill} onChange={(e) => setNewSkill(e.target.value)} placeholder="Add a new skill (e.g., Python)" className="flex-grow px-3 py-2 text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" onKeyDown={(e) => { if (e.key === 'Enter') handleAddSkill(); }}/>
                    <button onClick={handleAddSkill} className="px-4 py-2 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">Add</button>
                </div>
                 <div className="flex items-center justify-end gap-4 mt-4">
                     <button onClick={handleSaveSkills} disabled={savingSkills} className="px-5 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50">
                         {savingSkills ? 'Saving...' : 'Save Skills'}
                    </button>
                 </div>
            </div>

            {/* --- Education Section --- */}
            <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                <h2 className="text-2xl font-semibold mb-4">Education</h2>
                <div className="space-y-4 mb-6">
                    {educationItems.length > 0 ? educationItems.map((item) => (
                        <div key={item.id} className="p-4 border border-gray-200 rounded-md flex justify-between items-start">
                            <div>
                                <p className="font-semibold text-lg">{item.degree}</p>
                                <p className="text-gray-700">{item.school}</p>
                                <p className="text-sm text-gray-500">{item.startYear || 'N/A'} - {item.endYear || 'Present'}</p>
                            </div>
                            <button onClick={() => handleRemoveEducation(item.id)} className="ml-4 text-red-500 hover:text-red-700 text-xl font-bold" aria-label={`Remove ${item.degree} at ${item.school}`}>&times;</button>
                        </div> )) : (<p className="text-gray-500 italic">No education added yet.</p>)}
                </div>
                <div className="border-t border-gray-200 pt-4 space-y-3">
                     <h3 className="text-lg font-medium">Add New Education</h3>
                     <div>
                         <label htmlFor="degree" className="block text-sm font-medium text-gray-700 mb-1">Degree</label>
                         <input type="text" name="degree" id="degree" value={newEducationEntry.degree ?? ''} onChange={handleEducationInputChange} required className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>
                      <div>
                         <label htmlFor="school" className="block text-sm font-medium text-gray-700 mb-1">School/University</label>
                         <input type="text" name="school" id="school" value={newEducationEntry.school ?? ''} onChange={handleEducationInputChange} required className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>
                     <div className="grid grid-cols-2 gap-4">
                         <div>
                            <label htmlFor="startYear" className="block text-sm font-medium text-gray-700 mb-1">Start Year</label>
                            <input type="number" name="startYear" id="startYear" value={newEducationEntry.startYear ?? ''} onChange={handleEducationInputChange} placeholder="YYYY" className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                         </div>
                          <div>
                            <label htmlFor="endYear" className="block text-sm font-medium text-gray-700 mb-1">End Year (or leave blank)</label>
                            <input type="number" name="endYear" id="endYear" value={newEducationEntry.endYear ?? ''} onChange={handleEducationInputChange} placeholder="YYYY" className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                         </div>
                     </div>
                     <button onClick={handleAddNewEducation} className="px-4 py-2 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">Add Education Entry</button>
                </div>
                <div className="flex items-center justify-end gap-4 mt-6 border-t border-gray-200 pt-4">
                     <button onClick={handleSaveEducation} disabled={savingEducation} className="px-5 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50">
                         {savingEducation ? 'Saving...' : 'Save Education'}
                     </button>
                </div>
            </div>

            {/* --- NEW: Experience Section --- */}
            <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                <h2 className="text-2xl font-semibold mb-4">Experience</h2>

                 {/* Display existing experience items */}
                 <div className="space-y-4 mb-6">
                    {experienceItems.length > 0 ? (
                        experienceItems.map((item) => (
                            <div key={item.id} className="p-4 border border-gray-200 rounded-md flex justify-between items-start">
                                <div>
                                    <p className="font-semibold text-lg">{item.title}</p>
                                    <p className="text-gray-700">{item.company} {item.location ? `(${item.location})` : ''}</p>
                                    <p className="text-sm text-gray-500">
                                        {item.startDate || 'N/A'} - {item.endDate || 'Present'}
                                    </p>
                                    {item.description && <p className="mt-2 text-sm text-gray-600 whitespace-pre-wrap">{item.description}</p>}
                                </div>
                                <button
                                    onClick={() => handleRemoveExperience(item.id)}
                                    className="ml-4 text-red-500 hover:text-red-700 text-xl font-bold flex-shrink-0"
                                    aria-label={`Remove ${item.title} at ${item.company}`}
                                >
                                    &times;
                                </button>
                            </div>
                        ))
                    ) : (
                        <p className="text-gray-500 italic">No experience added yet.</p>
                    )}
                </div>

                {/* Form to add new experience */}
                <div className="border-t border-gray-200 pt-4 space-y-3">
                     <h3 className="text-lg font-medium">Add New Experience</h3>
                     {/* Job Title */}
                     <div>
                         <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">Job Title</label>
                         <input type="text" name="title" id="title" value={newExperienceEntry.title ?? ''} onChange={handleExperienceInputChange} required className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>
                     {/* Company */}
                      <div>
                         <label htmlFor="company" className="block text-sm font-medium text-gray-700 mb-1">Company</label>
                         <input type="text" name="company" id="company" value={newExperienceEntry.company ?? ''} onChange={handleExperienceInputChange} required className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>
                     {/* Dates */}
                     <div className="grid grid-cols-2 gap-4">
                         <div>
                            <label htmlFor="startDate" className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
                            {/* Using type="month" provides a native month/year picker */}
                            <input type="month" name="startDate" id="startDate" value={newExperienceEntry.startDate ?? ''} onChange={handleExperienceInputChange} className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                         </div>
                          <div>
                            <label htmlFor="endDate" className="block text-sm font-medium text-gray-700 mb-1">End Date (leave blank if current)</label>
                            <input type="month" name="endDate" id="endDate" value={newExperienceEntry.endDate ?? ''} onChange={handleExperienceInputChange} className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                         </div>
                     </div>
                      {/* Location */}
                     <div>
                         <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                         <input type="text" name="location" id="location" value={newExperienceEntry.location ?? ''} onChange={handleExperienceInputChange} placeholder="e.g., City, State or Remote" className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>
                     {/* Description */}
                     <div>
                         <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">Description (Optional)</label>
                         <textarea name="description" id="description" value={newExperienceEntry.description ?? ''} onChange={handleExperienceInputChange} rows={4} className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" />
                     </div>

                     <button onClick={handleAddNewExperience} className="px-4 py-2 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                        Add Experience Entry
                    </button>
                </div>

                {/* Save Experience Button */}
                <div className="flex items-center justify-end gap-4 mt-6 border-t border-gray-200 pt-4">
                     <button onClick={handleSaveExperience} disabled={savingExperience} className="px-5 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50">
                         {savingExperience ? 'Saving...' : 'Save Experience'}
                     </button>
                </div>
            </div>

            {/* --- Logout Button --- */}
             <button onClick={logout} className="px-4 py-2 mt-8 font-bold text-white bg-red-500 rounded hover:bg-red-700">
                Logout
            </button>
        </div>
    );
}