'use client';
import { Transition } from '@headlessui/react';
import { Fragment } from 'react';

interface ToastProps {
  show: boolean;
  message: string;
  type?: 'success' | 'error';
}

//success icon
const CheckCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

//error icon
const XCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2 2m2-2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

export default function Toast({ show, message, type = 'success' }: ToastProps) {
  if (!message) {
    return null; //no empty toasts
  }
 
  const isSuccess = type === 'success';
  //border color based on type
  const borderClass = isSuccess ? 'border-green-500' : 'border-red-500';
  
  return (
    //toast container positioned at top right
    <div
      aria-live="assertive"
      className="pointer-events-none fixed inset-15 flex items-end px-4 py-6 sm:items-start sm:p-6 z-50"
    >
      <div className="flex w-full flex-col items-center space-y-4 sm:items-end">
        {/* slide in animation */}
        <Transition
          show={show}
          as={Fragment}
          enter="transform ease-out duration-300 transition"
          enterFrom="translate-y-2 opacity-0 sm:translate-y-0 sm:translate-x-2"
          enterTo="translate-y-0 opacity-100 sm:translate-x-0"
          leave="transition ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          {/* toast card with colored left border */}
          <div className={`pointer-events-auto w-full max-w-sm overflow-hidden rounded-lg bg-white shadow-lg border-l-4 ${borderClass}`}>
            <div className="p-4">
              <div className="flex items-start">
                {/* icon area */}
                <div className="flex-shrink-0">
                  {isSuccess ? <CheckCircleIcon /> : <XCircleIcon />}
                </div>
                {/* message content */}
                <div className="ml-3 w-0 flex-1 pt-0.5">
                  <p className={`text-sm font-medium text-gray-900`}>
                    {isSuccess ? 'Success' : 'Error'}
                  </p>
                  <p className="mt-1 text-sm text-gray-500">{message}</p>
                </div>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  );
}