'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider, // Import Google Auth Provider
  signInWithPopup     // Import Popup sign-in method
} from 'firebase/auth';
import { auth, db } from '@/lib/firebase/config'; 
import { doc, setDoc, getDoc } from "firebase/firestore"; 

export default function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loadingEmail, setLoadingEmail] = useState(false); // Separate loading states
  const [loadingGoogle, setLoadingGoogle] = useState(false);
  const router = useRouter();

  // --- Helper function to create user profile ---
  const createUserProfile = async (user: any) => {
      const userDocRef = doc(db, "users", user.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (!userDocSnap.exists()) {
          await setDoc(userDocRef, {
              uid: user.uid,
              email: user.email,
              displayName: user.displayName || email.split('@')[0], // Use email part if no displayName
              photoURL: user.photoURL || null, // Google provides this, email doesn't
              createdAt: new Date(),
              // Initialize profile fields
              skills: [],
              education: [],
              experience: [],
          });
          console.log("Created new user profile in Firestore for:", user.email);
      } else {
           console.log("User profile already exists during signup check for:", user.email);
      }
  }

  // --- Email/Password Signup Handler ---
  const handleSignup = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 6) {
        setError("Password should be at least 6 characters long.");
        return;
    }

    setLoadingEmail(true);
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      console.log("Email Signup successful:", userCredential.user);

      // Create user profile in Firestore
      await createUserProfile(userCredential.user);

      router.push('/dashboard');
    } catch (err: any) {
      console.error("Email Signup Error:", err);
      if (err.code === 'auth/email-already-in-use') {
          setError('This email address is already in use.');
      } else if (err.code === 'auth/weak-password') {
           setError('Password is too weak. It should be at least 6 characters long.');
      }
       else {
          setError(err.message || 'Failed to sign up. Please try again.');
      }
    } finally {
      setLoadingEmail(false);
    }
  };

  // --- Google Sign-In Handler ---
  const handleGoogleSignIn = async () => {
    setError(null);
    setLoadingGoogle(true);
    const provider = new GoogleAuthProvider();

    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      console.log("Google Sign-In successful on Signup Page:", user);

      // Create user profile in Firestore (handles new or existing Google users)
      await createUserProfile(user);

      router.push('/dashboard');
    } catch (err: any) {
        console.error("Google Sign-In Error:", err);
        console.error("Error Code:", err.code); // Log the specific error code
        // Check for specific error codes
        if (err.code === 'auth/popup-closed-by-user') {
            setError('Sign-up popup closed before completion.');
        } else if (err.code === 'auth/cancelled-popup-request') {
             setError('Multiple sign-up attempts detected. Please try again.');
        }
        // Add other specific codes if needed: https://firebase.google.com/docs/auth/admin/errors
        else {
            setError(err.message || 'Failed to sign up with Google. Please try again.');
        }
    } finally {
      setLoadingGoogle(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-md">
        <h2 className="text-2xl font-bold text-center text-gray-900">Sign Up</h2>

        {/* --- Google Sign-In Button --- */}
        <button
          onClick={handleGoogleSignIn}
          disabled={loadingGoogle || loadingEmail}
          className="flex items-center justify-center w-full px-4 py-2 mt-4 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          <svg className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"> {/* Google Icon SVG */}</svg>
          {loadingGoogle ? 'Signing in...' : 'Sign up with Google'}
        </button>

        {/* --- Divider --- */}
        <div className="relative flex items-center justify-center my-6">
          <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-gray-300"></div></div>
          <div className="relative px-2 text-sm text-gray-500 bg-white">Or sign up with email</div>
        </div>

        {/* --- Email/Password Form --- */}
        <form onSubmit={handleSignup} className="space-y-6">
          {/* Email Input */}
          <div>
            <label htmlFor="email" /* ... */>Email address</label>
            <input id="email" name="email" type="email" required /* ... */ value={email} onChange={(e) => setEmail(e.target.value)} className="/* ... */" />
          </div>
          {/* Password Input */}
          <div>
            <label htmlFor="password" /* ... */>Password</label>
            <input id="password" name="password" type="password" required /* ... */ value={password} onChange={(e) => setPassword(e.target.value)} className="/* ... */" />
          </div>
          {/* Confirm Password Input */}
          <div>
            <label htmlFor="confirmPassword" /* ... */>Confirm Password</label>
            <input id="confirmPassword" name="confirmPassword" type="password" required /* ... */ value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="/* ... */" />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}
          {/* Email Signup Button */}
          <div>
            <button type="submit" disabled={loadingEmail || loadingGoogle} className="/* ... */">
              {loadingEmail ? 'Signing up...' : 'Sign Up'}
            </button>
          </div>
        </form>

        {/* Optional: Link to Login page */}
        <p className="text-sm text-center text-gray-600">
          Already have an account?{' '}
          <a href="/login" className="font-medium text-indigo-600 hover:text-indigo-500">
            Log In
          </a>
        </p>
      </div>
    </div>
  );
}