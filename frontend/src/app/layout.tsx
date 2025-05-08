import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext"; // Import the AuthProvider
import Navbar from "@/components/layout/navbar"; // <-- Import Navbar

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Resume Analyzer", // Customize title
  description: "Analyze your profile and find matching jobs", // Customize description
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider> {/* Wrap children with the AuthProvider */}
          <Navbar /> {/* <-- Render Navbar here */}
          <main>{children}</main> {/* <-- Wrap page content in main (optional but good practice) */}
        </AuthProvider>
      </body>
    </html>
  );
}