// frontend/src/components/layout/Navbar.tsx
'use client'; // Required because we use hooks (useAuth, useState potentially later)

import Link from 'next/link';
import { useAuth } from '@/context/AuthContext'; // Import useAuth hook
import { useRouter } from 'next/navigation'; // To redirect on logout if needed

export default function Navbar() {
  const { user, loading, logout } = useAuth(); // Get auth state and logout function
  const router = useRouter();

  const handleLogout = async () => {
    try {
      await logout(); // Call the logout function from context
      // Redirect is handled within the logout function in context, but can add here too if needed
      router.push('/');// Responsible for the proper logout
    } catch (error) {
      console.error("Navbar logout error:", error);
      // Handle error display if necessary
    }
  };

  return (
    <nav className="bg-white shadow-md sticky top-0 z-50"> {/* Sticky navbar */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand */}
          <div className="flex-shrink-0">
            <Link href="/" className="text-2xl font-bold text-indigo-600 hover:text-indigo-800">
              JobMatcher AI {/* Or your app name/logo */}
            </Link>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center space-x-4">
            {loading ? (
              // Optional: Show a placeholder or nothing while loading auth state
              <div className="text-sm text-gray-500">Loading...</div>
            ) : user ? (
              // Links shown when user IS logged in
              <>
                <Link href="/dashboard" className="text-gray-600 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">
                  Dashboard
                </Link>
                <Link href="/profile" className="text-gray-600 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">
                  Profile
                </Link>
                <Link href="/job-matches" className="text-gray-600 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">
                  Job Matches
                </Link>
                <button
                  onClick={handleLogout}
                  className="text-gray-600 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium"
                >
                  Logout
                </button>
              </>
            ) : (
                null
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}