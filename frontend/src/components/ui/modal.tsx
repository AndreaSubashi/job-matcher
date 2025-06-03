// frontend/src/components/ui/Modal.tsx
'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, ReactNode } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void; // Function to close the modal
  title?: string;
  children: ReactNode; // Content of the modal
}

export default function Modal({ isOpen, onClose, title, children }: ModalProps) {
  return (
    // Use Transition component for smooth open/close animations
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        {/* Backdrop */}
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          {/* --- MODIFIED THIS DIV --- */}
          {/* Added backdrop-blur-sm and a very light semi-transparent background for better effect */}
          <div className="fixed inset-0 backdrop-blur-lg" />
          {/* You can adjust backdrop-blur-sm to backdrop-blur-md or backdrop-blur-lg for more blur */}
          {/* The bg-gray-500 bg-opacity-25 is optional but can help the blur look more distinct */}
        </Transition.Child>

        {/* Modal Container */}
        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              {/* Modal Panel (ensure this has its own background and shadow) */}
              <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all">
                {/* Optional Title */}
                {title && (
                  <Dialog.Title
                    as="h3"
                    className="text-lg font-medium leading-6 text-gray-900 mb-4"
                  >
                    {title}
                  </Dialog.Title>
                )}

                {/* Close Button (optional but recommended) */}
                 <button
                     onClick={onClose}
                     className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 focus:outline-none"
                     aria-label="Close modal"
                >
                     <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                         <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                     </svg>
                 </button>

                {/* Modal Content */}
                <div className="mt-2">
                  {children}
                </div>

              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}
