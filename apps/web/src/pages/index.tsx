import { Geist, Geist_Mono } from "next/font/google"
import CanvasViewer from "@/_components/CanvasViewer"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export default function Home() {
  return (
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen`}>
      <main className="relative min-h-screen flex justify-center p-4">
        <CanvasViewer />
      </main>
    </div>
  )
}
