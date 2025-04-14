// frontend/src/app/dashboard/page.tsx
'use client'; // Needs to be a client component to use hooks

import React, { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext'; // Import the custom hook
import { useRouter } from 'next/navigation';

export default function DashboardPage() {
  const { user, loading, logout } = useAuth(); // Get user, loading state, and logout function
  const router = useRouter();

  // Effect to redirect if user is not logged in after loading
  useEffect(() => {
    if (!loading && !user) {
      // Redirect to login page if not loading and no user
      router.push('/login');
    }
  }, [user, loading, router]); // Dependencies

  // Show loading state
  if (loading) {
    return <div>Loading...</div>; // Or a spinner component
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
        onClick={logout} // Call the logout function from context
        className="px-4 py-2 mt-4 font-bold text-white bg-red-500 rounded hover:bg-red-700"
      >
        Logout
      </button>

      {/* Add links to profile, find jobs etc. */}
      <div className="mt-6">
          <a href="/profile" className="text-indigo-600 hover:text-indigo-800">Go to Profile</a>
          {/* Add link to Find Jobs later */}
      </div>
    </div>
  );
}