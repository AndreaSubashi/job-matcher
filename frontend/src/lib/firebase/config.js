import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
//import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyA4v-MoNd_65i6XCztTH-22hr0PhP-NfTY",
  authDomain: "job-matcher-9c692.firebaseapp.com",
  projectId: "job-matcher-9c692",
  storageBucket: "job-matcher-9c692.firebasestorage.app",
  messagingSenderId: "71568991044",
  appId: "1:71568991044:web:1dbd9fc05deb5495606d11",
  measurementId: "G-W4XCW1G14V"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
//const analytics = getAnalytics(app);

export { auth, db, app }; 