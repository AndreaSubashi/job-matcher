import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
//import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCx9H2UHLCCO2GgUAw8n2JJc-f2s5QEKYY",
  authDomain: "job-matcher-b6aec.firebaseapp.com",
  projectId: "job-matcher-b6aec",
  storageBucket: "job-matcher-b6aec.firebasestorage.app",
  messagingSenderId: "828528390854",
  appId: "1:828528390854:web:78e8d961bad39a65f51dc7",
  measurementId: "G-HBNJGDF6E4"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(firebaseApp);
//const analytics = getAnalytics(app);

export { auth, db, firebaseApp }; 