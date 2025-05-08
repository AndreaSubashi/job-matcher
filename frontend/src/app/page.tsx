// frontend/src/app/page.tsx
'use client'; // Landing page needs to be client component to manage modal state

import React, { useState } from 'react';
// Remove Link import if not used elsewhere on this page
// import Link from 'next/link';
import Modal from '@/components/ui/modal'; // Import the Modal component
import LoginForm from '@/components/auth/LoginForm'; // Import the LoginForm
import SignupForm from '@/components/auth/SignupForm'; // Import the SignupForm
import { useAuth } from '@/context/AuthContext'; // <-- Import useAuth
import Link from 'next/link'; // <-- Import Link


export default function HomePage() {
  // State to control modal visibility
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const { user, loading } = useAuth(); // <-- Use the auth hook

  // Functions to open/close modals
  const openLoginModal = () => { setIsSignupModalOpen(false); setIsLoginModalOpen(true); };
  const closeLoginModal = () => setIsLoginModalOpen(false);
  const openSignupModal = () => { setIsLoginModalOpen(false); setIsSignupModalOpen(true); };
  const closeSignupModal = () => setIsSignupModalOpen(false);
  const switchToLogin = () => { closeSignupModal(); openLoginModal(); };
  const switchToSignup = () => { closeLoginModal(); openSignupModal(); };


  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12 md:p-24 bg-gradient-to-b from-indigo-50 via-white to-white">
       <div className="text-center max-w-3xl">
        {/* ... Headline and description ... */}
        <h1 className="text-4xl md:text-6xl font-bold text-indigo-800 mb-4 tracking-tight">
          Unlock Your Next Career Move
        </h1>
        <p className="text-lg md:text-xl text-gray-700 mb-8">
          Leverage AI to analyze your profile and discover job opportunities perfectly matched to your skills, education, and experience.
        </p>


        {/* --- Conditionally Render Buttons --- */}
        {/* Added min-height to prevent layout jumps during auth loading */}
        <div className="flex flex-col sm:flex-row justify-center gap-4 min-h-[56px]">
          {loading ? (
            // Show nothing or a placeholder while auth state loads
            <div className="h-14 w-64 bg-gray-200 animate-pulse rounded-md"></div> // Example placeholder
          ) : !user ? (
            // --- RENDER THIS BLOCK ONLY IF NOT LOGGED IN ---
            <>
              <button
                onClick={openSignupModal}
                className="px-8 py-3 text-lg font-semibold text-white bg-indigo-600 rounded-md shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out"
              >
                Get Started - Sign Up
              </button>
              <button
                onClick={openLoginModal}
                className="px-8 py-3 text-lg font-semibold text-indigo-700 bg-white border border-indigo-300 rounded-md shadow-md hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out"
              >
                Login
              </button>
            </>
          ) : (
            // --- RENDER THIS BLOCK ONLY IF LOGGED IN ---
            <Link
                href="/dashboard" // Or '/profile' or '/job-matches'
                className="px-8 py-3 text-lg font-semibold text-white bg-green-600 rounded-md shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition duration-150 ease-in-out"
              >
                Go to Your Dashboard
              </Link>
          )}
        </div>
        {/* --- End Conditional Buttons --- */}

       </div>

       {/* --- Modals (Render conditionally is slightly cleaner) --- */}
       {!loading && !user && (
           <>
              <Modal isOpen={isLoginModalOpen} onClose={closeLoginModal} title="Login">
                   <LoginForm onSuccess={closeLoginModal} onSwitchToSignup={switchToSignup} />
               </Modal>
               <Modal isOpen={isSignupModalOpen} onClose={closeSignupModal} title="Sign Up">
                   <SignupForm onSuccess={closeSignupModal} onSwitchToLogin={switchToLogin} />
               </Modal>
           </>
       )}

    </main>
  );
}