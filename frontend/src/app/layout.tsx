import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext"; 
import Navbar from "@/components/layout/navbar"; 

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Resume Analyzer", 
  description: "Analyze your profile and find matching jobs", 
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider> {/* Wrap children with AuthProvider */}
          <Navbar />
          <main>{children}</main> 
        </AuthProvider>
      </body>
    </html>
  );
}