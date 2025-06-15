'use client';
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { onAuthStateChanged, User, signOut } from 'firebase/auth';
import { auth } from '@/lib/firebase/config';
import { useRouter } from 'next/navigation';

//what auth context looks like, user info, loading state, and logout function
interface AuthContextType {
  user: User | null; //either we have a user or we don't
  loading: boolean; //shows spinner while figuring out if user is logged in
  logout: () => Promise<void>; //logs user out
}

//create context, starts as undefined until provider wraps components
const AuthContext = createContext<AuthContextType | undefined>(undefined);

//props for our provider component
interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true); //assume loading until we know for sure
  const router = useRouter();

  useEffect(() => {
    //firebase tells us whenever auth state changes (login/logout)
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      console.log("auth state changed:", currentUser?.email || 'no user');
      setUser(currentUser); //update user state
      setLoading(false); //we know the auth state now, stop loading
    });

    //clean up listener when component unmounts
    return () => unsubscribe();
  }, []); //only run once when component mounts

  // handles user logout
  const logout = async () => {
    setLoading(true);
    try {
        await signOut(auth);
        //firebase will automatically trigger onAuthStateChanged to set user to null
        console.log("user logged out successfully");
        router.push('/'); //send them back to home/login page
    } catch (error) {
        console.error("logout failed:", error);
        //could show error message to user here
    } finally {
         setLoading(false); //always reset loading state
    }
  };

  //bundle everything up to pass down to child components
  const value = {
    user,
    loading,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

//custom hook to use auth context anywhere in the app
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}