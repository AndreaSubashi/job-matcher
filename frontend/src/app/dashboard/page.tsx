'use client'; // Needs to be a client component to use hooks

import React, { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext'; 
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function DashboardPage() {
  const { user, loading, logout } = useAuth(); // Get user, loading state, and logout function
  const router = useRouter();

  // Effect to redirect if user is not logged in after loading
  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]); // Dependencies

  if (loading) {
    return <div>Loading...</div>; 
  }

  // If loading is finished and still no user, router.push should have initiated redirect
  // But we can return null or a message while redirect happens
  if (!user) {
       return <div>Redirecting to login...</div>; // Or null
  }

  // Render dashboard content only if user is logged in
  return (
    <div className="p-4">
      <h1>Welcome to your Dashboard!</h1>
      <p>You are logged in as: {user.email}</p>
      {user.displayName && <p>Display Name: {user.displayName}</p>}

      {/* Logout Button */}
      <button
        onClick={logout} 
        className="px-4 py-2 mt-4 font-bold text-white bg-red-500 rounded hover:bg-red-700"
      >
        Logout
      </button>

      {/* Add links to profile, find jobs etc. */}
      <div className="mt-6 space-x-4">
            <Link href="/profile" className="text-indigo-600 hover:text-indigo-800">
              Go to Profile
            </Link>
             <Link href="/job-matches" className="text-green-600 hover:text-green-800 font-semibold">
              Find Job Matches
            </Link>
      </div>
    </div>
  );
  

}