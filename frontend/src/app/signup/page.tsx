'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup     
} from 'firebase/auth';
import { auth, db } from '@/lib/firebase/config'; 
import { doc, setDoc, getDoc } from "firebase/firestore"; 

export default function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loadingEmail, setLoadingEmail] = useState(false);
  const [loadingGoogle, setLoadingGoogle] = useState(false);
  const router = useRouter();

  //create user profile document in Firestore after successful registration
  const createUserProfile = async (user: any) => {
      const userDocRef = doc(db, "users", user.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (!userDocSnap.exists()) {
          await setDoc(userDocRef, {
              uid: user.uid,
              email: user.email,
              displayName: user.displayName || email.split('@')[0], //use email prefix if no display name
              photoURL: user.photoURL || null,
              createdAt: new Date(),
              //initialize empty profile sections for the app
              skills: [],
              education: [],
              experience: [],
          });
          console.log("Created new user profile in Firestore for:", user.email);
      } else {
           console.log("User profile already exists during signup check for:", user.email);
      }
  }

  //handle email/password registration with validation
  const handleSignup = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    //basic form validation before attempting signup
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

      await createUserProfile(userCredential.user);

      router.push('/dashboard');
    } catch (err: any) {
      console.error("Email Signup Error:", err);
      
      //handle common Firebase signup errors with user-friendly messages
      if (err.code === 'auth/email-already-in-use') {
          setError('This email address is already in use.');
      } else if (err.code === 'auth/weak-password') {
           setError('Password is too weak. It should be at least 6 characters long.');
      } else {
          setError(err.message || 'Failed to sign up. Please try again.');
      }
    } finally {
      setLoadingEmail(false);
    }
  };

  //handle Google OAuth signup with automatic profile creation
  const handleGoogleSignIn = async () => {
    setError(null);
    setLoadingGoogle(true);
    const provider = new GoogleAuthProvider();

    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      console.log("Google Sign-In successful on Signup Page:", user);

      await createUserProfile(user);

      router.push('/dashboard');
    } catch (err: any) {
        console.error("Google Sign-In Error:", err);
        console.error("Error Code:", err.code);
        
        //handle common Google signup errors
        if (err.code === 'auth/popup-closed-by-user') {
            setError('Sign-up popup closed before completion.');
        } else if (err.code === 'auth/cancelled-popup-request') {
             setError('Multiple sign-up attempts detected. Please try again.');
        } else {
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

        {/* google signup button with loading state */}
        <button
          onClick={handleGoogleSignIn}
          disabled={loadingGoogle || loadingEmail}
          className="flex items-center justify-center w-full px-4 py-2 mt-4 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          {/* google logo SVG */}
          <svg className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
            <path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C12.955 4 4 12.955 4 24s8.955 20 20 20s20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
            <path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4C16.318 4 9.656 8.337 6.306 14.691z"/>
            <path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.221 0-9.584-3.681-11.283-8.574l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/>
            <path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002l6.19 5.238C39.704 35.846 44 30.417 44 24c0-1.341-.138-2.65-.389-3.917z"/>
          </svg>
          {loadingGoogle ? 'Signing up...' : 'Sign up with Google'}
        </button>

        {/* visual separator between OAuth and email signup */}
        <div className="relative flex items-center justify-center my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300"></div>
          </div>
          <div className="relative px-2 text-sm text-gray-500 bg-white">Or sign up with email</div>
        </div>

        {/* email signup form with password confirmation */}
        <form onSubmit={handleSignup} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email address</label>
            <input 
              id="email" 
              name="email" 
              type="email" 
              required 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              className="w-full px-3 py-2 mt-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500" 
            />
          </div>
          
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
            <input 
              id="password" 
              name="password" 
              type="password" 
              required 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              className="w-full px-3 py-2 mt-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500" 
            />
          </div>
          
          <div>
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">Confirm Password</label>
            <input 
              id="confirmPassword" 
              name="confirmPassword" 
              type="password" 
              required 
              value={confirmPassword} 
              onChange={(e) => setConfirmPassword(e.target.value)} 
              className="w-full px-3 py-2 mt-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500" 
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}
          
          <div>
            <button 
              type="submit" 
              disabled={loadingEmail || loadingGoogle} 
              className="w-full px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {loadingEmail ? 'Signing up...' : 'Sign Up'}
            </button>
          </div>
        </form>

        {/* link to login page for existing users */}
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