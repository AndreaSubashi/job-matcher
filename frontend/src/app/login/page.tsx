// frontend/src/app/login/page.tsx
'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
// Import necessary Firebase auth functions
import {
  signInWithEmailAndPassword,
  GoogleAuthProvider, // Import Google Auth Provider
  signInWithPopup     // Import Popup sign-in method
} from 'firebase/auth';
import { auth, db } from '@/lib/firebase/config'; // Adjust path if needed
import { doc, setDoc, getDoc } from "firebase/firestore"; // Import Firestore functions if creating profile on login

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loadingEmail, setLoadingEmail] = useState(false); // Separate loading states
  const [loadingGoogle, setLoadingGoogle] = useState(false);
  const router = useRouter();

  // --- Email/Password Login Handler (from before) ---
  const handleLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoadingEmail(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.push('/dashboard');
    } catch (err: any) {
      console.error("Email Login Error:", err);
      setError(err.message || 'Failed to login. Please check your credentials.');
      if (err.code === 'auth/invalid-credential') {
        setError('Invalid Credentials');
      }
    } finally {
      setLoadingEmail(false);
    }
  };

  // --- Google Sign-In Handler ---
  const handleGoogleSignIn = async () => {
    setError(null);
    setLoadingGoogle(true);
    const provider = new GoogleAuthProvider(); // Create a GoogleAuthProvider instance

    try {
      const result = await signInWithPopup(auth, provider); // Trigger the Google Sign-In popup
      const user = result.user;
      console.log("Google Sign-In successful:", user);

      // Optional: Check if user profile exists in Firestore, create if not
      const userDocRef = doc(db, "users", user.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (!userDocSnap.exists()) {
        // User is new, create their profile document
        await setDoc(userDocRef, {
          uid: user.uid,
          email: user.email,
          displayName: user.displayName, // Comes from Google profile
          photoURL: user.photoURL, // Comes from Google profile
          createdAt: new Date(), // Good practice to timestamp
          // Initialize profile fields needed for your app
          skills: [],
          education: [],
          experience: [],
        });
         console.log("Created new user profile in Firestore for:", user.email);
      } else {
        console.log("User profile already exists for:", user.email);
        // Optional: You could update fields like displayName or photoURL here if needed
      }


      // Redirect after successful Google Sign-In
      router.push('/dashboard');

    } catch (err: any) {
        console.error("Google Sign-In Error:", err);
        console.error("Error Code:", err.code); // Log the specific error code
        // Check for specific error codes
        if (err.code === 'auth/popup-closed-by-user') {
            setError('Sign-in popup closed before completion.');
        } else if (err.code === 'auth/cancelled-popup-request') {
             setError('Multiple sign-in attempts detected. Please try again.');
        }
        // Add other specific codes if needed: https://firebase.google.com/docs/auth/admin/errors
        else {
            setError(err.message || 'Failed to sign in with Google. Please try again.');
        }
    } finally {
      setLoadingGoogle(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-md">
        <h2 className="text-2xl font-bold text-center text-gray-900">Login</h2>

        {/* --- Google Sign-In Button --- */}
        <button
          onClick={handleGoogleSignIn}
          disabled={loadingGoogle || loadingEmail}
          className="flex items-center justify-center w-full px-4 py-2 mt-4 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          {/* You can add a Google icon here */}
          <svg className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C12.955 4 4 12.955 4 24s8.955 20 20 20s20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.221 0-9.584-3.681-11.283-8.574l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002l6.19 5.238C39.704 35.846 44 30.417 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg>
          {loadingGoogle ? 'Signing in...' : 'Sign in with Google'}
        </button>

        {/* --- Divider --- */}
        <div className="relative flex items-center justify-center my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300"></div>
          </div>
          <div className="relative px-2 text-sm text-gray-500 bg-white">
            Or continue with email
          </div>
        </div>

        {/* --- Email/Password Form (from before) --- */}
        <form onSubmit={handleLogin} className="space-y-6">
          {/* Email Input */}
          <div>
            <label htmlFor="email" /* ... */>Email address</label>
            <input id="email" name="email" type="email" /* ... */ value={email} onChange={(e) => setEmail(e.target.value)} className="/* ... */" />
          </div>
          {/* Password Input */}
          <div>
            <label htmlFor="password" /* ... */>Password</label>
            <input id="password" name="password" type="password" /* ... */ value={password} onChange={(e) => setPassword(e.target.value)} className="/* ... */" />
          </div>
          {error && <p className="text-sm text-red-600">{error}</p>}
          {/* Email Login Button */}
          <div>
            <button type="submit" disabled={loadingEmail || loadingGoogle} className="/* ... */">
              {loadingEmail ? 'Logging in...' : 'Login'}
            </button>
          </div>
        </form>

         {/* Optional: Link to Signup page */}
         <p className="text-sm text-center text-gray-600">
            Don't have an account?{' '}
            <a href="/signup" className="font-medium text-indigo-600 hover:text-indigo-500">
              Sign Up
            </a>
         </p>
      </div>
    </div>
  );
}