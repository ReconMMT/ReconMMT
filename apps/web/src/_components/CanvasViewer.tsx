'use client'

import { useEffect, useRef, useState } from 'react'

type Detection = {
    cls: string
    conf: number
    bbox: number[]
}

type ObjectMetadata = {
    width_px: number
    height_px: number
    scale_w: number
    scale_h: number
    estimated_age_group?: string
    estimated_height_ratio?: number
    object_type?: string
    structure?: boolean
}

type FoundObject = {
    id: number
    class: string
    first_seen: number
    bbox: number[]
    metadata: ObjectMetadata
}

type StreamMode =
    | 'rgb'
    | 'low_light'
    | 'contrast'
    | 'edge'
    | 'thermal'

type CameraDevice = {
    index: number
    label: string
}

type TabMode = 'live' | 'reconstruction'

const ALL_MODES: StreamMode[] = [
    'rgb',
    'low_light',
    'contrast',
    'edge',
    'thermal'
]

export default function CanvasViewer() {
    const liveCanvasRef = useRef<HTMLCanvasElement | null>(null)
    const reconRefs = useRef<Record<StreamMode, HTMLCanvasElement | null>>({
        rgb: null,
        low_light: null,
        contrast: null,
        edge: null,
        thermal: null
    })
    const wsRef = useRef<WebSocket | null>(null)

    const [tab, setTab] = useState<TabMode>('live')
    const [running, setRunning] = useState(false)
    const [mode, setMode] = useState<StreamMode>('rgb')

    const [realtime, setRealtime] = useState<Record<string, number>>({})
    const [found, setFound] = useState<FoundObject[]>([])

    const [cameras, setCameras] = useState<CameraDevice[]>([])
    const [cameraId, setCameraId] = useState('0')

    const [error, setError] = useState<string | null>(null)
    const [connected, setConnected] = useState(false)

    const [role, setRole] = useState<'transmit' | 'receive'>('receive')
    const [password, setPassword] = useState('')
    const [server, setServer] = useState('127.0.0.1:8000')

    /* ---- camera list ---- */
    useEffect(() => {
        setCameras([
            { index: 0, label: 'Camera 0' },
            { index: 1, label: 'Camera 1' }
        ])
        setCameraId('0')
    }, [])

    /* ---- websocket ---- */
    useEffect(() => {
        if (!running) return
        if (!password) {
            setError("Password required")
            return
        }

        setError(null)
        setConnected(false)

        wsRef.current?.close()
        wsRef.current = null

        let closed = false

        const endpoint = `ws://${server}/ws?camera=${cameraId}`
        const ws = new WebSocket(endpoint)
        wsRef.current = ws

        ws.onopen = () => setConnected(true)

        ws.onerror = () => {
            setError('Unable to connect to backend')
            setConnected(false)
        }

        ws.onclose = () => {
            if (!closed) {
                setError('API Connection closed')
                setConnected(false)
            }
        }

        ws.onmessage = e => {
            let data: any
            try {
                data = JSON.parse(e.data)
            } catch {
                return
            }

            if (!data.streams || !data.metadata) return

            /* ---- object analysis ---- */
            const rt: Record<string, number> = {}
            data.metadata.realtime.forEach((d: Detection) => {
                rt[d.cls] = (rt[d.cls] || 0) + 1
            })

            setRealtime(rt)
            setFound(data.metadata.objects_found || [])

            /* ---- draw helper ---- */
            const draw = (canvas: HTMLCanvasElement | null, src?: string) => {
                if (!canvas || !src) return
                const ctx = canvas.getContext('2d')
                if (!ctx) return

                const img = new Image()
                img.src = 'data:image/jpeg;base64,' + src

                img.onload = () => {
                    if (
                        canvas.width !== img.width ||
                        canvas.height !== img.height
                    ) {
                        canvas.width = img.width
                        canvas.height = img.height
                    }

                    ctx.setTransform(1, 0, 0, 1, 0, 0)
                    ctx.clearRect(0, 0, canvas.width, canvas.height)
                    ctx.drawImage(img, 0, 0)
                }
            }

            /* ---- always draw live ---- */
            draw(liveCanvasRef.current, data.streams[mode])

            /* ---- reconstruction ---- */
            ALL_MODES.forEach(m =>
                draw(reconRefs.current[m], data.streams[m])
            )
        }

        return () => {
            closed = true
            wsRef.current?.close()
            wsRef.current = null
        }
    }, [running, cameraId, mode, role, password, server])

    return (
        <>
            {/* Status */}
            {error && (
                <div className="absolute top-2 w-auto left-1/2 -translate-x-1/2 self-center px-6 py-2 rounded-full bg-red-950 border border-red-900 whitespace-nowrap text-sm text-red-300">
                    {error}
                </div>
            )}

            {!error && running && !connected && (
                <div className="absolute top-2 w-auto left-1/2 -translate-x-1/2 self-center px-6 py-2 rounded-full bg-blue-950 border border-blue-900 whitespace-nowrap text-sm text-blue-300">
                    Connecting to backend…
                </div>
            )}
            <div className="flex flex-col gap-4 mt-10">

                <div className=' max-w-4xl mx-auto w-full gap-4 flex items-center justify-center'>

                    <select
                        value={role}
                        onChange={e => setRole(e.target.value as any)}
                        className="bg-neutral-800 border border-neutral-700 px-2 py-1 rounded text-sm"
                    >
                        <option value="receive">Receiver</option>
                        <option value="transmit">Transmitter</option>
                    </select>

                    <input
                        type="password"
                        placeholder="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        className="bg-neutral-800 border border-neutral-700 px-2 py-1 rounded text-sm"
                    />

                    <input
                        placeholder="server ip"
                        value={server}
                        onChange={e => setServer(e.target.value)}
                        className="bg-neutral-800 border border-neutral-700 px-2 py-1 rounded text-sm"
                    />

                </div>

                <div className=' max-w-4xl mx-auto w-full space-y-4 flex flex-col items-center justify-center'>
                    {/* Tabs (column) */}
                    <div className="flex gap-2">
                        <button
                            onClick={() => setTab('live')}
                            className={`px-4 py-2 rounded-md text-sm text-left cursor-pointer ${tab === 'live'
                                ? 'bg-neutral-800'
                                : 'bg-neutral-900'
                                }`}
                        >
                            Live Mode
                        </button>
                        <button
                            onClick={() => setTab('reconstruction')}
                            className={`px-4 py-2 rounded-md text-sm text-left cursor-pointer ${tab === 'reconstruction'
                                ? 'bg-neutral-800'
                                : 'bg-neutral-900'
                                }`}
                        >
                            Frame Reconstruction
                        </button>
                        <button
                            onClick={() => setRunning(v => !v)}
                            className={`px-3 py-1 rounded-full text-sm cursor-pointer ${running ? 'bg-red-600' : 'bg-emerald-600'
                                }`}
                        >
                            {running ? 'Stop' : 'Start'}
                        </button>
                    </div>

                    {/* Controls */}
                    <div className="flex flex-wrap gap-2">

                        <select
                            value={cameraId}
                            onChange={e => setCameraId(e.target.value)}
                            className="rounded bg-neutral-800 border border-neutral-700 px-2 py-1 text-sm"
                        >
                            {cameras.map(c => (
                                <option key={c.index} value={c.index}>
                                    {c.label}
                                </option>
                            ))}
                        </select>

                        {tab === 'live' && (
                            <select
                                value={mode}
                                onChange={e =>
                                    setMode(e.target.value as StreamMode)
                                }
                                className="rounded bg-neutral-800 border border-neutral-700 px-2 py-1 text-sm"
                            >
                                {ALL_MODES.map(m => (
                                    <option key={m} value={m}>
                                        {m}
                                    </option>
                                ))}
                            </select>
                        )}
                    </div>
                </div>

                {/* Viewport */}
                {tab === 'live' ? (
                    <div className="flex flex-col lg:flex-row gap-4 h-[64vh]">
                        <div className="w-full bg-black lg:min-w-3xl rounded-xl overflow-hidden flex items-center justify-center relative">
                            <canvas
                                ref={liveCanvasRef}
                                className="w-full h-auto"
                            />
                            {/* {!running && (<div className='absolute top-1/2 left-1/2 font-serif text-3xl font-thin! w-full h-full bg-black -translate-x-1/2 -translate-y-1/2 flex items-center justify-center'>Feed Stoped!</div>)} */}
                        </div>
                        <Analysis realtime={realtime} found={found} />
                    </div>
                ) : (
                    <div className="flex flex-col lg:flex-row gap-4">
                        <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {ALL_MODES.map(m => (
                                <div
                                    key={m}
                                    className="bg-black rounded-xl lg:min-w-100 aspect-video flex items-center justify-center"
                                >
                                    <canvas
                                        ref={el => {
                                            reconRefs.current[m] = el
                                        }}
                                        className="w-full h-auto"
                                    />
                                </div>
                            ))}
                        </div>
                        <Analysis realtime={realtime} found={found} />
                    </div>
                )}
            </div>
        </>
    )
}

/* ---- Analysis ---- */

function Analysis({
    realtime,
    found
}: {
    realtime: Record<string, number>
    found: FoundObject[]
}) {
    return (
        <div className="flex flex-col lg:flex-row gap-4 ">
            <div className="flex-1 lg:min-w-60 rounded-xl border border-neutral-800 bg-neutral-900 p-4">
                <h3 className="font-semibold mb-2">Realtime Objects</h3>
                <div className="flex flex-col gap-2 max-h-[64vh] overflow-y-auto">
                    {Object.entries(realtime).map(([k, v]) => (
                        <div key={k} className="flex justify-between text-sm">
                            <span>{k}</span>
                            <span>{v}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="flex-1 lg:min-w-60 rounded-xl border border-neutral-800 bg-neutral-900 p-4">
                <h3 className="font-semibold mb-2">Objects Found Logs</h3>

                <div className="flex flex-col gap-2 max-h-[64vh] overflow-y-auto">
                    {found.map(o => (
                        <div
                            key={o.id}
                            className="text-xs border border-neutral-800 rounded-md p-2"
                        >
                            <div className="flex justify-between">
                                <span className="font-semibold">{o.class}</span>
                                <span>ID {o.id}</span>
                            </div>

                            <div className="text-neutral-400">
                                w: {o.metadata.width_px}px
                            </div>

                            <div className="text-neutral-400">
                                h: {o.metadata.height_px}px
                            </div>

                            <div className="text-neutral-400">
                                scale: {o.metadata.scale_w.toFixed(3)} / {o.metadata.scale_h.toFixed(3)}
                            </div>

                            {o.metadata.estimated_age_group && (
                                <div className="text-neutral-400">
                                    age: {o.metadata.estimated_age_group}
                                </div>
                            )}

                            {o.metadata.object_type && (
                                <div className="text-neutral-400">
                                    type: {o.metadata.object_type}
                                </div>
                            )}

                            {o.metadata.structure && (
                                <div className="text-neutral-400">
                                    structure
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
