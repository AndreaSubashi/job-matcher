// frontend/src/context/AuthContext.tsx
'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { onAuthStateChanged, User, signOut } from 'firebase/auth';
import { auth } from '@/lib/firebase/config'; // Adjust path if needed
import { useRouter } from 'next/navigation';

// Define the shape of the context data
interface AuthContextType {
  user: User | null; // Firebase User object or null
  loading: boolean; // To handle initial auth state loading
  logout: () => Promise<void>; // Function to log out
}

// Create the context with a default value (usually null or undefined)
// Using 'null' initially might require type casting or checks later
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Create the Provider component
interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true); // Start loading until auth state is determined
  const router = useRouter();

  useEffect(() => {
    // Listen for changes in Firebase authentication state
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      console.log("Auth state changed:", currentUser?.email || 'No user');
      setUser(currentUser); // Set user to the Firebase User object or null
      setLoading(false); // Auth state determined, stop loading
    });

    // Cleanup function: Unsubscribe when the component unmounts
    return () => unsubscribe();
  }, []); // Empty dependency array means this runs once on mount

  // Logout function
  const logout = async () => {
    setLoading(true); // Optional: show loading state during logout
    try {
        await signOut(auth);
        // onAuthStateChanged will handle setting user to null
        console.log("User logged out successfully");
        router.push('/login'); // Redirect to login after logout
    } catch (error) {
        console.error("Logout failed:", error);
        // Handle logout error if needed
    } finally {
         setLoading(false); // Ensure loading state is reset
    }

  };

  // Value passed down by the provider
  const value = {
    user,
    loading,
    logout
  };

  // Render children only after initial loading is complete to prevent flashes
  // Or show a loading spinner globally
  // if (loading) {
  //   return <div>Loading Authentication...</div>; // Or a spinner component
  // }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Create a custom hook for easy consumption of the context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}