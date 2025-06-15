'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { signInWithEmailAndPassword, GoogleAuthProvider, signInWithPopup, User } from 'firebase/auth';
import { auth, db } from '@/lib/firebase/config';
import { doc, setDoc, getDoc, Timestamp } from "firebase/firestore";

interface LoginFormProps {
    onSuccess?: () => void; //callback when login works
    onSwitchToSignup?: () => void; //callback to show signup form instead
}

//reusable button and input styles, could move these to a global css file
const inputStyle = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-500 border border-gray-300 rounded-md shadow-sm appearance-none focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm";
const primaryButtonStyle = "relative flex justify-center w-full px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md group hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50";
const googleButtonStyle = "flex items-center justify-center w-full px-4 py-2 mt-4 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50";

export default function LoginForm({ onSuccess, onSwitchToSignup }: LoginFormProps) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loadingEmail, setLoadingEmail] = useState(false);
    const [loadingGoogle, setLoadingGoogle] = useState(false);
    const router = useRouter();

    const clearError = () => setError(null);

    //checks if user has a profile in firestore, creates one if they don't
    const checkOrCreateUserProfile = async (user: User) => {
        if (!db) {
             console.error("Firestore client (db) not available.");
             //could show error message to user here
             return; //exit early if db isn't working
        }
        const userDocRef = doc(db, "users", user.uid);
        try {
            const userDocSnap = await getDoc(userDocRef);
            if (!userDocSnap.exists()) {
                console.log(`Profile not found for ${user.email}, creating...`);
                await setDoc(userDocRef, {
                    uid: user.uid,
                    email: user.email || '', //fallback to empty string
                    displayName: user.displayName || email.split('@')[0], //use part before @ if no display name
                    photoURL: user.photoURL || null,
                    createdAt: Timestamp.fromDate(new Date()), //firestore timestamp for consistency
                    skills: [],
                    education: [],
                    experience: [],
                });
                console.log("Created profile during login for:", user.email);
            } else {
                 console.log("Existing profile found for:", user.email);
            }
        } catch (err) {
            console.error("Error checking/creating user profile:", err);
            //user can still login even if profile creation fails
        }
    }

    //handles email/password login
    const handleLogin = async (event: React.FormEvent) => {
        event.preventDefault(); clearError(); setLoadingEmail(true);
        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            await checkOrCreateUserProfile(userCredential.user);
            onSuccess?.(); //call success callback if provided
            router.push('/dashboard');
        } catch (err: any) {
            console.error("Email Login Error:", err);
            setError(err.message || 'Failed to login. Please check your credentials.');
        } finally {
            setLoadingEmail(false);
        }
    };

    //handles google popup login
    const handleGoogleSignIn = async () => {
        clearError(); setLoadingGoogle(true);
        const provider = new GoogleAuthProvider();
        try {
            const result = await signInWithPopup(auth, provider);
            await checkOrCreateUserProfile(result.user);
            onSuccess?.();
            router.push('/dashboard');
        } catch (err: any) {
            console.error("Google Sign-In Error:", err);
            if (err.code === 'auth/popup-closed-by-user') {
                 setError('Sign-in popup closed before completion.');
            } else {
                 setError(err.message || 'Google Sign-In failed.');
            }
        } finally {
            setLoadingGoogle(false);
        }
    };

    return (
        <div>
            <h2 className="text-xl font-semibold text-center text-gray-900 mb-6">Login</h2>
            {/* Google login button with SVG icon */}
            <button
              onClick={handleGoogleSignIn}
              disabled={loadingGoogle || loadingEmail}
              className={googleButtonStyle}
            >
              <svg className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C12.955 4 4 12.955 4 24s8.955 20 20 20s20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.221 0-9.584-3.681-11.283-8.574l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002l6.19 5.238C39.704 35.846 44 30.417 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg>
              {loadingGoogle ? 'Signing in...' : 'Sign in with Google'}
            </button>
            {/* visual separator between google and email options */}
            <div className="relative flex items-center justify-center my-4">
               <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-gray-300"></div></div>
               <div className="relative px-2 text-sm text-gray-500 bg-white">Or continue with email</div>
           </div>
            {/* traditional email/password form */}
           <form onSubmit={handleLogin} className="space-y-4">
               <div>
                   <label htmlFor="login-email" className="block text-sm font-medium text-gray-700">Email address</label>
                   <input id="login-email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} className={inputStyle} />
               </div>
               <div>
                   <label htmlFor="login-password"className="block text-sm font-medium text-gray-700">Password</label>
                   <input id="login-password" name="password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} className={inputStyle} />
               </div>
               {error && <p className="text-sm text-red-600 text-center">{error}</p>}
               <div>
                   <button type="submit" disabled={loadingEmail || loadingGoogle} className={primaryButtonStyle}>
                       {loadingEmail ? 'Logging in...' : 'Login'}
                   </button>
               </div>
           </form>
           {/* link to switch to signup form if callback provided */}
           {onSwitchToSignup && (
                <p className="text-sm text-center text-gray-600 mt-4">
                   Don't have an account?{' '}
                   <button onClick={onSwitchToSignup} type="button" className="font-medium text-indigo-600 hover:text-indigo-500 underline focus:outline-none">
                      Sign Up
                   </button>
                </p>
            )}
        </div>
   );
}