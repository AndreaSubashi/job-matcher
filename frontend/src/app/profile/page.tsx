// frontend/src/app/profile/page.tsx
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import { v4 as uuidv4 } from 'uuid';
import Toast from '@/components/ui/toast';

// --- Interfaces ---
interface EducationItem {
    id: string;
    degree: string;
    school: string;
    startYear?: number | null;
    endYear?: number | null;
}

interface ExperienceItem {
    id: string;
    title: string;
    company: string;
    startDate?: string | null;
    endDate?: string | null;
    location?: string | null;
    description?: string | null;
}

interface UserProfile {
    uid: string;
    email?: string;
    displayName?: string;
    skills: string[];
    education: EducationItem[];
    experience: ExperienceItem[];
}

const defaultNewEducation: Partial<EducationItem> = { degree: '', school: '', startYear: undefined, endYear: undefined };
const defaultNewExperience: Partial<ExperienceItem> = { title: '', company: '', startDate: '', endDate: '', location: '', description: '' };

// --- Reusable Tailwind CSS classes ---
const inputStyle = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-500 border border-gray-300 rounded-md shadow-sm appearance-none focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm";
const btnPrimary = "px-4 py-2 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50";
const btnGreen = "px-5 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50";
const btnSecondary = "px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400";
const btnTextIndigo = "text-sm text-indigo-600 hover:text-indigo-800 font-medium disabled:opacity-50";
const btnTextRed = "text-sm text-red-500 hover:text-red-700 font-medium disabled:opacity-50";
const labelStyle = "block text-sm font-medium text-gray-700 mb-1";

export default function ProfilePage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();

    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    
    const [skills, setSkills] = useState<string[]>([]);
    const [newSkill, setNewSkill] = useState('');
    const [savingSkills, setSavingSkills] = useState(false);

    const [educationItems, setEducationItems] = useState<EducationItem[]>([]);
    const [newEducationEntry, setNewEducationEntry] = useState<Partial<EducationItem>>(defaultNewEducation);
    const [savingEducation, setSavingEducation] = useState(false);

    const [experienceItems, setExperienceItems] = useState<ExperienceItem[]>([]);
    const [newExperienceEntry, setNewExperienceEntry] = useState<Partial<ExperienceItem>>(defaultNewExperience);
    const [savingExperience, setSavingExperience] = useState(false);
    
    const [editingEducationId, setEditingEducationId] = useState<string | null>(null);
    const [editedEducationData, setEditedEducationData] = useState<Partial<EducationItem>>({});
    const [editingExperienceId, setEditingExperienceId] = useState<string | null>(null);
    const [editedExperienceData, setEditedExperienceData] = useState<Partial<ExperienceItem>>({});

    const fetchProfile = useCallback(async () => {
        if (!user) return;
        setIsLoading(true);
        setError(null);
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (!response.ok) {
                throw new Error("Failed to fetch profile. Please try again.");
            }
            const data: UserProfile = await response.json();
            setProfile(data);
            setSkills(data.skills || []);
            setEducationItems(data.education || []);
            setExperienceItems(data.experience || []);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/');
        }
        if (user && !profile) {
            fetchProfile();
        } else if (!authLoading) {
            setIsLoading(false);
        }
    }, [user, authLoading, router, profile, fetchProfile]);

    const showAndClearMessage = (setter: React.Dispatch<React.SetStateAction<string | null>>, message: string) => {
        setter(message);
        setTimeout(() => setter(null), 3000);
    };
    const clearMessages = () => { setError(null); setSuccessMessage(null); };

    const handleAddSkill = () => {
        clearMessages();
        const trimmedSkill = newSkill.trim();
        if (trimmedSkill && !skills.some(skill => skill.toLowerCase() === trimmedSkill.toLowerCase())) {
            setSkills([...skills, trimmedSkill]);
            setNewSkill('');
        }
    };
    const handleRemoveSkill = (skillToRemove: string) => { setSkills(skills.filter(skill => skill !== skillToRemove)); clearMessages(); };
    const handleSaveSkills = async () => {
        if (!user) return;
        setSavingSkills(true); clearMessages();
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/skills`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ skills }),
            });
            if (!response.ok) throw new Error("Failed to save skills.");
            showAndClearMessage(setSuccessMessage, "Skills updated successfully!");
        } catch (err: any) { showAndClearMessage(setError, err.message); } finally { setSavingSkills(false); }
    };

    const handleAddNewEducation = () => {
        clearMessages();
        if (!newEducationEntry.degree || !newEducationEntry.school) { showAndClearMessage(setError, "Degree and School are required."); return; }
        setEducationItems(prev => [...prev, { ...defaultNewEducation, ...newEducationEntry, id: uuidv4() } as EducationItem]);
        setNewEducationEntry(defaultNewEducation);
    };
    const handleRemoveEducation = (idToRemove: string) => { setEducationItems(prev => prev.filter(item => item.id !== idToRemove)); clearMessages(); };
    const handleSaveEducation = async () => {
        if (!user) return;
        setSavingEducation(true); clearMessages();
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/education`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ education: educationItems }),
            });
            if (!response.ok) throw new Error("Failed to save education.");
            showAndClearMessage(setSuccessMessage, "Education updated successfully!");
        } catch (err: any) { showAndClearMessage(setError, err.message); } finally { setSavingEducation(false); }
    };
    const handleStartEditEducation = (eduItem: EducationItem) => {
        setEditingExperienceId(null);
        setEditingEducationId(eduItem.id);
        setEditedEducationData(eduItem);
    };
    const handleCancelEditEducation = () => { setEditingEducationId(null); setEditedEducationData({}); };
    const handleUpdateEducationItem = () => {
        setEducationItems(prev => prev.map(item => item.id === editingEducationId ? { ...item, ...editedEducationData } as EducationItem : item));
        handleCancelEditEducation();
    };
    const handleEditEducationFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setEditedEducationData(prev => ({ ...prev, [name]: name.includes('Year') ? (value === '' ? null : parseInt(value, 10)) : value }));
    };
    const handleNewEducationFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setNewEducationEntry(prev => ({ ...prev, [name]: name.includes('Year') ? (value === '' ? undefined : parseInt(value, 10)) : value }));
    };

    const handleAddNewExperience = () => {
        clearMessages();
        if (!newExperienceEntry.title || !newExperienceEntry.company) { showAndClearMessage(setError, "Job Title and Company are required."); return; }
        setExperienceItems(prev => [...prev, { ...defaultNewExperience, ...newExperienceEntry, id: uuidv4() } as ExperienceItem]);
        setNewExperienceEntry(defaultNewExperience);
    };
    const handleRemoveExperience = (idToRemove: string) => { setExperienceItems(prev => prev.filter(item => item.id !== idToRemove)); clearMessages(); };
    const handleSaveExperience = async () => {
        if (!user) return;
        setSavingExperience(true); clearMessages();
        try {
            const token = await user.getIdToken();
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/profile/experience`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ experience: experienceItems }),
            });
             if (!response.ok) throw new Error("Failed to save experience.");
            showAndClearMessage(setSuccessMessage, "Experience updated successfully!");
        } catch (err: any) { showAndClearMessage(setError, err.message); } finally { setSavingExperience(false); }
    };
    const handleStartEditExperience = (expItem: ExperienceItem) => {
        setEditingEducationId(null); // Close other edit form
        setEditingExperienceId(expItem.id);
        setEditedExperienceData(expItem);
    };
    const handleCancelEditExperience = () => { setEditingExperienceId(null); setEditedExperienceData({}); };
    const handleUpdateExperienceItem = () => {
        setExperienceItems(prev => prev.map(item => item.id === editingExperienceId ? { ...item, ...editedExperienceData } as ExperienceItem : item));
        handleCancelEditExperience();
    };
    const handleEditExperienceFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setEditedExperienceData(prev => ({ ...prev, [name]: value }));
    };
    const handleNewExperienceFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setNewExperienceEntry(prev => ({ ...prev, [name]: value }));
    };

    if (isLoading || authLoading) { return <div className="flex justify-center items-center min-h-screen">Loading Profile...</div>; }
    if (!user || !profile) { return <div className="flex justify-center items-center min-h-screen">Redirecting...</div>; }

    return (
        <>
            <div className="bg-gray-50 min-h-screen">
                <div className="container mx-auto p-4 md:p-8 max-w-4xl">
                    <h1 className="text-3xl font-bold mb-1 text-gray-800">Your Professional Profile</h1>
                    <p className="text-gray-500 mb-6">Keep this information up-to-date to get the best job matches.</p>

                    {/* --- Skills Section --- */}
                    <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                        <h2 className="text-2xl font-semibold mb-4">Skills</h2>
                        <div className="flex flex-wrap gap-2 mb-4 min-h-[40px]">
                            {skills.length > 0 ? (
                                skills.map((skill) => (
                                    <span key={skill} className="flex items-center bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full">
                                        {skill}
                                        <button onClick={() => handleRemoveSkill(skill)} className="ml-2 text-indigo-600 hover:text-indigo-800 text-lg leading-none" aria-label={`Remove ${skill}`}>&times;</button>
                                    </span>
                                ))
                            ) : (
                                <p className="text-gray-500 italic">Add your skills below.</p>
                            )}
                        </div>
                        <div className="flex items-center gap-2 mb-4">
                            <input type="text" value={newSkill} onChange={(e) => setNewSkill(e.target.value)} placeholder="e.g., Python" className={`flex-grow ${inputStyle}`} onKeyDown={(e) => { if (e.key === 'Enter') handleAddSkill(); }} />
                            <button onClick={handleAddSkill} className={btnPrimary}>Add</button>
                        </div>
                        <div className="flex justify-end mt-4">
                            <button onClick={handleSaveSkills} disabled={savingSkills} className={btnGreen}>{savingSkills ? 'Saving...' : 'Save Skills'}</button>
                        </div>
                    </div>

                    {/* --- Education Section --- */}
                    <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                        <h2 className="text-2xl font-semibold mb-4">Education</h2>
                        <div className="space-y-4 mb-6">
                            {educationItems.length > 0 ? (
                                educationItems.map((item) => (
                                    <div key={item.id} className="p-4 border border-gray-200 rounded-md">
                                        {editingEducationId === item.id ? (
                                            <div className="space-y-3">
                                                <h3 className="text-lg font-medium text-indigo-600">Editing Education</h3>
                                                <div><label htmlFor={`degree-edit-${item.id}`} className={labelStyle}>Degree</label><input type="text" id={`degree-edit-${item.id}`} name="degree" value={editedEducationData.degree ?? ''} onChange={handleEditEducationFormChange} className={inputStyle} /></div>
                                                <div><label htmlFor={`school-edit-${item.id}`} className={labelStyle}>School/University</label><input type="text" id={`school-edit-${item.id}`} name="school" value={editedEducationData.school ?? ''} onChange={handleEditEducationFormChange} className={inputStyle} /></div>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div><label htmlFor={`startYear-edit-${item.id}`} className={labelStyle}>Start Year</label><input type="number" id={`startYear-edit-${item.id}`} name="startYear" value={editedEducationData.startYear ?? ''} onChange={handleEditEducationFormChange} className={inputStyle} /></div>
                                                    <div><label htmlFor={`endYear-edit-${item.id}`} className={labelStyle}>End Year</label><input type="number" id={`endYear-edit-${item.id}`} name="endYear" value={editedEducationData.endYear ?? ''} onChange={handleEditEducationFormChange} className={inputStyle} /></div>
                                                </div>
                                                <div className="flex justify-end gap-2 mt-2">
                                                    <button onClick={handleCancelEditEducation} className={btnSecondary}>Cancel</button>
                                                    <button onClick={handleUpdateEducationItem} className={btnGreen}>Update</button>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="flex justify-between items-start"><div className="flex-grow">
                                                <p className="font-semibold text-lg">{item.degree}</p><p className="text-gray-700">{item.school}</p><p className="text-sm text-gray-500">{item.startYear || ''} - {item.endYear || 'Present'}</p>
                                            </div><div className="flex gap-2 flex-shrink-0 ml-2">
                                                <button onClick={() => handleStartEditEducation(item)} className={btnTextIndigo}>Edit</button><button onClick={() => handleRemoveEducation(item.id)} className={btnTextRed}>Remove</button>
                                            </div></div>
                                        )}
                                    </div>
                                ))
                            ) : (
                                <p className="text-gray-500 italic">No education history added yet.</p>
                            )}
                        </div>
                        <div className="border-t pt-4 space-y-3">
                            <h3 className="text-lg font-medium">Add New Education</h3>
                            <div><label htmlFor="degree-new" className={labelStyle}>Degree</label><input type="text" id="degree-new" name="degree" value={newEducationEntry.degree ?? ''} onChange={handleNewEducationFormChange} className={inputStyle} /></div>
                            <div><label htmlFor="school-new" className={labelStyle}>School/University</label><input type="text" id="school-new" name="school" value={newEducationEntry.school ?? ''} onChange={handleNewEducationFormChange} className={inputStyle} /></div>
                            <div className="grid grid-cols-2 gap-4">
                                <div><label htmlFor="startYear-new" className={labelStyle}>Start Year</label><input type="number" id="startYear-new" name="startYear" value={newEducationEntry.startYear ?? ''} onChange={handleNewEducationFormChange} className={inputStyle} /></div>
                                <div><label htmlFor="endYear-new" className={labelStyle}>End Year (or blank)</label><input type="number" id="endYear-new" name="endYear" value={newEducationEntry.endYear ?? ''} onChange={handleNewEducationFormChange} className={inputStyle} /></div>
                            </div>
                            <button onClick={handleAddNewEducation} className={btnPrimary}>Add Education</button>
                        </div>
                        <div className="flex justify-end mt-6 border-t pt-4"><button onClick={handleSaveEducation} disabled={savingEducation} className={btnGreen}>{savingEducation ? 'Saving...' : 'Save All Education'}</button></div>
                    </div>

                    {/* --- Experience Section --- */}
                    <div className="p-6 bg-white rounded-lg shadow-md mb-8">
                        <h2 className="text-2xl font-semibold mb-4">Work Experience</h2>
                        <div className="space-y-4 mb-6">
                            {experienceItems.length > 0 ? (
                                experienceItems.map((item) => (
                                    <div key={item.id} className="p-4 border border-gray-200 rounded-md">
                                        {editingExperienceId === item.id ? (
                                            <div className="space-y-3">
                                                <h3 className="text-lg font-medium text-indigo-600">Editing Experience</h3>
                                                <div><label htmlFor={`title-edit-${item.id}`} className={labelStyle}>Job Title</label><input type="text" id={`title-edit-${item.id}`} name="title" value={editedExperienceData.title ?? ''} onChange={handleEditExperienceFormChange} className={inputStyle} /></div>
                                                <div><label htmlFor={`company-edit-${item.id}`} className={labelStyle}>Company</label><input type="text" id={`company-edit-${item.id}`} name="company" value={editedExperienceData.company ?? ''} onChange={handleEditExperienceFormChange} className={inputStyle} /></div>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div><label htmlFor={`startDate-edit-${item.id}`} className={labelStyle}>Start Date</label><input type="month" id={`startDate-edit-${item.id}`} name="startDate" value={editedExperienceData.startDate ?? ''} onChange={handleEditExperienceFormChange} className={inputStyle} /></div>
                                                    <div><label htmlFor={`endDate-edit-${item.id}`} className={labelStyle}>End Date</label><input type="month" id={`endDate-edit-${item.id}`} name="endDate" value={editedExperienceData.endDate ?? ''} onChange={handleEditExperienceFormChange} className={inputStyle} /></div>
                                                </div>
                                                <div><label htmlFor={`location-edit-${item.id}`} className={labelStyle}>Location</label><input type="text" id={`location-edit-${item.id}`} name="location" value={editedExperienceData.location ?? ''} onChange={handleEditExperienceFormChange} className={inputStyle} /></div>
                                                <div><label htmlFor={`description-edit-${item.id}`} className={labelStyle}>Description</label><textarea id={`description-edit-${item.id}`} name="description" value={editedExperienceData.description ?? ''} onChange={handleEditExperienceFormChange} rows={4} className={inputStyle}></textarea></div>
                                                <div className="flex justify-end gap-2 mt-2"><button onClick={handleCancelEditExperience} className={btnSecondary}>Cancel</button><button onClick={handleUpdateExperienceItem} className={btnGreen}>Update</button></div>
                                            </div>
                                        ) : (
                                            <div className="flex justify-between items-start"><div className="flex-grow">
                                                <p className="font-semibold text-lg">{item.title}</p><p className="text-gray-700">{item.company}</p><p className="text-sm text-gray-500">{item.startDate} - {item.endDate || 'Present'}</p>{item.location && <p className="text-sm text-gray-500">{item.location}</p>}{item.description && <p className="mt-2 text-gray-600 whitespace-pre-wrap">{item.description}</p>}
                                            </div><div className="flex gap-2 flex-shrink-0 ml-2">
                                                <button onClick={() => handleStartEditExperience(item)} className={btnTextIndigo}>Edit</button><button onClick={() => handleRemoveExperience(item.id)} className={btnTextRed}>Remove</button>
                                            </div></div>
                                        )}
                                    </div>
                                ))
                            ) : (
                                <p className="text-gray-500 italic">No work experience added yet.</p>
                            )}
                        </div>
                        <div className="border-t pt-4 space-y-3">
                            <h3 className="text-lg font-medium">Add New Experience</h3>
                            <div><label htmlFor="title-new" className={labelStyle}>Job Title</label><input type="text" id="title-new" name="title" value={newExperienceEntry.title ?? ''} onChange={handleNewExperienceFormChange} className={inputStyle} /></div>
                            <div><label htmlFor="company-new" className={labelStyle}>Company</label><input type="text" id="company-new" name="company" value={newExperienceEntry.company ?? ''} onChange={handleNewExperienceFormChange} className={inputStyle} /></div>
                            <div className="grid grid-cols-2 gap-4">
                                <div><label htmlFor="startDate-new" className={labelStyle}>Start Date</label><input type="month" id="startDate-new" name="startDate" value={newExperienceEntry.startDate ?? ''} onChange={handleNewExperienceFormChange} className={inputStyle} /></div>
                                <div><label htmlFor="endDate-new" className={labelStyle}>End Date (or blank)</label><input type="month" id="endDate-new" name="endDate" value={newExperienceEntry.endDate ?? ''} onChange={handleNewExperienceFormChange} className={inputStyle} /></div>
                            </div>
                            <div><label htmlFor="location-new" className={labelStyle}>Location</label><input type="text" id="location-new" name="location" value={newExperienceEntry.location ?? ''} onChange={handleNewExperienceFormChange} className={inputStyle} /></div>
                            <div><label htmlFor="description-new" className={labelStyle}>Description</label><textarea id="description-new" name="description" value={newExperienceEntry.description ?? ''} onChange={handleNewExperienceFormChange} rows={4} className={inputStyle}></textarea></div>
                            <button onClick={handleAddNewExperience} className={btnPrimary}>Add Experience</button>
                        </div>
                        <div className="flex justify-end mt-6 border-t pt-4"><button onClick={handleSaveExperience} disabled={savingExperience} className={btnGreen}>{savingExperience ? 'Saving...' : 'Save All Experience'}</button></div>
                    </div>
                </div>
            </div>

            {/* --- Toast Notification Component --- */}
            <Toast 
                message={successMessage || error || ''} 
                show={!!successMessage || !!error}
                type={successMessage ? 'success' : 'error'}
            />
        </>
    );
}

