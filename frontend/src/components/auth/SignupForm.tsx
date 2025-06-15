'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { createUserWithEmailAndPassword, GoogleAuthProvider, signInWithPopup, User } from 'firebase/auth';
import { auth, db } from '@/lib/firebase/config';
import { doc, setDoc, getDoc, Timestamp } from "firebase/firestore";

interface SignupFormProps {
    onSuccess?: () => void; //callback when signup works
    onSwitchToLogin?: () => void; //callback to show login form instead
}

//shared styles for consistency, could be moved to a global css file
const inputStyle = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-500 border border-gray-300 rounded-md shadow-sm appearance-none focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm";
const primaryButtonStyle = "relative flex justify-center w-full px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md group hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50";
const googleButtonStyle = "flex items-center justify-center w-full px-4 py-2 mt-4 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50";


export default function SignupForm({ onSuccess, onSwitchToLogin }: SignupFormProps) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loadingEmail, setLoadingEmail] = useState(false);
    const [loadingGoogle, setLoadingGoogle] = useState(false);
    const router = useRouter();

    const clearError = () => setError(null);

    //creates user profile in firestore with default empty arrays
    const createUserProfile = async (user: User) => {
         if (!db) { console.error("Firestore client (db) not available."); return; }
         const userDocRef = doc(db, "users", user.uid);
         //double check if profile already exists, google users might already have one
         const userDocSnap = await getDoc(userDocRef);
         if (!userDocSnap.exists()) {
             try {
                  await setDoc(userDocRef, {
                    uid: user.uid,
                    email: user.email || '',
                    displayName: user.displayName || email.split('@')[0], //use part before @ if no display name
                    photoURL: user.photoURL || null,
                    createdAt: Timestamp.fromDate(new Date()), //firestore timestamp for consistency
                    skills: [],
                    education: [],
                    experience: [],
                 });
                 console.log("Created new user profile in Firestore for:", user.email);
             } catch(err) {
                  console.error("Error creating user profile during signup:", err);
             }
        } else {
            console.log("Profile already existed during signup check for:", user.email);
        }
    }

    //handles email/password signup with validation
    const handleSignup = async (event: React.FormEvent) => {
        event.preventDefault(); clearError();
        if (password !== confirmPassword) { setError("Passwords do not match."); return; }
        if (password.length < 6) { setError("Password should be at least 6 characters."); return; }

        setLoadingEmail(true);
        try {
            const userCredential = await createUserWithEmailAndPassword(auth, email, password);
            await createUserProfile(userCredential.user);
            onSuccess?.(); //call success callback if provided
            router.push('/dashboard');
        } catch (err: any) {
            console.error("Email Signup Error:", err);
            if (err.code === 'auth/email-already-in-use') { setError('This email address is already in use.'); }
            else if (err.code === 'auth/weak-password') { setError('Password is too weak (min 6 characters).'); }
            else { setError(err.message || 'Failed to sign up.'); }
        } finally {
            setLoadingEmail(false);
        }
    };

    //handles google popup signup, creates profile if needed
    const handleGoogleSignIn = async () => {
        clearError(); setLoadingGoogle(true);
        const provider = new GoogleAuthProvider();
        try {
            const result = await signInWithPopup(auth, provider);
            await createUserProfile(result.user); //will create if needed
            onSuccess?.();
            router.push('/dashboard');
        } catch (err: any) {
            console.error("Google Sign-In Error:", err);
             if (err.code === 'auth/popup-closed-by-user') { setError('Sign-in popup closed before completion.'); }
             else { setError(err.message || 'Google Sign-In failed.'); }
        } finally {
            setLoadingGoogle(false);
        }
    };

    return (
        <div>
            <h2 className="text-xl font-semibold text-center text-gray-900 mb-6">Sign Up</h2>
            {/* google signup button with SVG icon */}
             <button
               onClick={handleGoogleSignIn}
               disabled={loadingGoogle || loadingEmail}
               className={googleButtonStyle}
             >
               <svg className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C12.955 4 4 12.955 4 24s8.955 20 20 20s20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.221 0-9.584-3.681-11.283-8.574l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002l6.19 5.238C39.704 35.846 44 30.417 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg>
               {loadingGoogle ? 'Signing in...' : 'Sign up with Google'}
             </button>
            {/* visual separator between google and email options */}
            <div className="relative flex items-center justify-center my-4">
                <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-gray-300"></div></div>
                <div className="relative px-2 text-sm text-gray-500 bg-white">Or sign up with email</div>
            </div>
            {/* email/password signup form with validation */}
            <form onSubmit={handleSignup} className="space-y-4">
                <div>
                    <label htmlFor="signup-email" className="block text-sm font-medium text-gray-700">Email address</label>
                    <input id="signup-email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} className={inputStyle} />
                </div>
                <div>
                    <label htmlFor="signup-password"className="block text-sm font-medium text-gray-700">Password (min 6 characters)</label>
                    <input id="signup-password" name="password" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} className={inputStyle} />
                </div>
                <div>
                    <label htmlFor="confirmPassword"className="block text-sm font-medium text-gray-700">Confirm Password</label>
                    <input id="confirmPassword" name="confirmPassword" type="password" required value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className={inputStyle} />
                </div>
                {error && <p className="text-sm text-red-600 text-center">{error}</p>}
                <div>
                    <button type="submit" disabled={loadingEmail || loadingGoogle} className={primaryButtonStyle}>
                        {loadingEmail ? 'Signing up...' : 'Sign Up'}
                    </button>
                </div>
            </form>
            {/* link to switch to login form if callback provided */}
            {onSwitchToLogin && (
                <p className="text-sm text-center text-gray-600 mt-4">
                    Already have an account?{' '}
                    <button onClick={onSwitchToLogin} type="button" className="font-medium text-indigo-600 hover:text-indigo-500 underline focus:outline-none">
                       Log In
                    </button>
                </p>
            )}
        </div>
    );
}