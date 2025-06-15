'use client'; //client component for modal state

import React, { useState } from 'react';
import Modal from '@/components/ui/modal'; 
import LoginForm from '@/components/auth/LoginForm'; 
import SignupForm from '@/components/auth/SignupForm'; 
import { useAuth } from '@/context/AuthContext'; 
import Link from 'next/link';

export default function HomePage() {
  //modal visibility state
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const { user, loading } = useAuth();

  //modal controls
  const openLoginModal = () => { setIsSignupModalOpen(false); setIsLoginModalOpen(true); };
  const closeLoginModal = () => setIsLoginModalOpen(false);
  const openSignupModal = () => { setIsLoginModalOpen(false); setIsSignupModalOpen(true); };
  const closeSignupModal = () => setIsSignupModalOpen(false);
  const switchToLogin = () => { closeSignupModal(); openLoginModal(); };
  const switchToSignup = () => { closeLoginModal(); openSignupModal(); };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12 md:p-24 bg-gradient-to-b from-indigo-50 via-white to-white">
       <div className="text-center max-w-3xl">
        {/* main headline */}
        <h1 className="text-4xl md:text-6xl font-bold text-indigo-800 mb-4 tracking-tight">
          Unlock Your Next Career Move
        </h1>
        <p className="text-lg md:text-xl text-gray-700 mb-8">
          Leverage AI to analyze your profile and discover job opportunities perfectly matched to your skills, education, and experience.
        </p>

        {/* button area with auth logic */}
        <div className="flex flex-col sm:flex-row justify-center gap-4 min-h-[56px]">
          {loading ? (
            //loading placeholder
            <div className="h-14 w-64 bg-gray-200 animate-pulse rounded-md"></div>
          ) : !user ? (
            //not logged in buttons
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
            //logged in dashboard link
            <Link
                href="/dashboard"
                className="px-8 py-3 text-lg font-semibold text-white bg-green-600 rounded-md shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition duration-150 ease-in-out"
              >
                Go to Your Dashboard
              </Link>
          )}
        </div>
       </div>

       {/* auth modals */}
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