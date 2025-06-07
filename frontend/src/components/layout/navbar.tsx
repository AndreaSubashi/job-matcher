// frontend/src/components/layout/Navbar.tsx
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import { Menu, Transition } from '@headlessui/react';
import { Fragment } from 'react';

// A default user icon for fallback
const UserCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zM12 12a4 4 0 00-4 4h8a4 4 0 00-4-4z" clipRule="evenodd" />
    </svg>
);

// Chevron down icon for the dropdown indicator
const ChevronDownIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400 group-hover:text-gray-500 transition-colors" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
    </svg>
);

export default function Navbar() {
  const { user, loading } = useAuth();
  const pathname = usePathname();

  const activeLinkClass = "text-indigo-600 font-semibold";
  const inactiveLinkClass = "text-gray-500 hover:text-indigo-600";

  return (
    <nav className="bg-white shadow-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex-shrink-0">
            <Link href="/" className="text-2xl font-bold text-indigo-600 hover:text-indigo-800">
              JobMatcher AI
            </Link>
          </div>

          <div className="flex items-center">
            {loading ? (
              <div className="h-8 w-8 bg-gray-200 rounded-full animate-pulse"></div>
            ) : user ? (
              <>
                <div className="hidden sm:flex items-center space-x-8">
                  <Link href="/dashboard" className={`${pathname === '/dashboard' ? activeLinkClass : inactiveLinkClass} text-sm font-medium transition-colors`}>
                    Dashboard
                  </Link>
                  <Link href="/profile" className={`${pathname === '/profile' ? activeLinkClass : inactiveLinkClass} text-sm font-medium transition-colors`}>
                    Profile
                  </Link>
                  <Link href="/job-matches" className={`${pathname === '/job-matches' ? activeLinkClass : inactiveLinkClass} text-sm font-medium transition-colors`}>
                    Job Matches
                  </Link>
                  {/* --- NEW LINK --- */}
                  <Link href="/saved-jobs" className={`${pathname === '/saved-jobs' ? activeLinkClass : inactiveLinkClass} text-sm font-medium transition-colors`}>
                    Saved Jobs
                  </Link>
                  {/* --- NEW LINK --- */}
                  <Link href="/how-it-works" className={`${pathname === '/how-it-works' ? activeLinkClass : inactiveLinkClass} text-sm font-medium transition-colors`}>
                    How It Works
                  </Link>
                </div>
                
                <Menu as="div" className="relative ml-5">
                  <div>
                    <Menu.Button className="group flex items-center rounded-full bg-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                      <span className="sr-only">Open user menu</span>
                      {user.photoURL ? (
                        <img className="h-8 w-8 rounded-full" src={user.photoURL} alt="User profile picture" referrerPolicy="no-referrer" />
                      ) : (
                        <div className="h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center"><UserCircleIcon/></div>
                      )}
                      <div className="ml-1.5"><ChevronDownIcon /></div>
                    </Menu.Button>
                  </div>
                  <Transition
                    as={Fragment}
                    enter="transition ease-out duration-100"
                    enterFrom="transform opacity-0 scale-95"
                    enterTo="transform opacity-100 scale-100"
                    leave="transition ease-in duration-75"
                    leaveFrom="transform opacity-100 scale-100"
                    leaveTo="transform opacity-0 scale-95"
                  >
                    <Menu.Items className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white py-1 shadow-lg border border-gray-200 focus:outline-none divide-y divide-gray-100">
                      <div className="px-4 py-3">
                        <p className="text-sm text-gray-900">Signed in as</p>
                        <p className="truncate text-sm font-medium text-gray-800">{user.displayName || user.email}</p>
                      </div>

                      <div className="py-1">
                        <div className="sm:hidden">
                            <Menu.Item>
                              {({ active }) => (<Link href="/dashboard" className={`${active ? 'bg-gray-100' : ''} block px-4 py-2 text-sm text-gray-700`}>Dashboard</Link>)}
                            </Menu.Item>
                            <Menu.Item>
                              {({ active }) => (<Link href="/profile" className={`${active ? 'bg-gray-100' : ''} block px-4 py-2 text-sm text-gray-700`}>Your Profile</Link>)}
                            </Menu.Item>
                            <Menu.Item>
                              {({ active }) => (<Link href="/job-matches" className={`${active ? 'bg-gray-100' : ''} block px-4 py-2 text-sm text-gray-700`}>Job Matches</Link>)}
                            </Menu.Item>
                             {/* --- NEW LINK (for mobile dropdown) --- */}
                            <Menu.Item>
                              {({ active }) => (<Link href="/saved-jobs" className={`${active ? 'bg-gray-100' : ''} block px-4 py-2 text-sm text-gray-700`}>Saved Jobs</Link>)}
                            </Menu.Item>
                        </div>
                        {/* Always visible 'Your Profile' link in dropdown */}
                        <Menu.Item>
                          {({ active }) => (<Link href="/profile" className={`${active ? 'bg-gray-100' : ''} hidden sm:block px-4 py-2 text-sm text-gray-700`}>Your Profile</Link>)}
                        </Menu.Item>
                      </div>

                      <div className="py-1">
                        <Menu.Item>
                          {({ active }) => {
                            const { logout } = useAuth();
                            const router = useRouter();
                            const handleLogout = async () => { await logout(); router.push('/'); };
                            return (
                              <button
                                onClick={handleLogout}
                                className={`${active ? 'bg-red-50 text-red-800' : 'text-red-700'} block w-full text-left px-4 py-2 text-sm font-medium`}
                              >
                                Sign out
                              </button>
                            );
                          }}
                        </Menu.Item>
                      </div>
                    </Menu.Items>
                  </Transition>
                </Menu>
              </>
            ) : ( null )}
          </div>
        </div>
      </div>
    </nav>
  );
}
