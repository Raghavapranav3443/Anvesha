import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { createGlobeMaterial } from './globe'
import { animateOrbits, buildOrbitSystem } from './orbits'
import dayUrl from './assets/earth_day.jpg'
import nightUrl from './assets/earth_night.png'

function Scene({ zoom, reduced }: { zoom: number; reduced: boolean }) {
  const [dayTex, nightTex] = useLoader(THREE.TextureLoader, [dayUrl, nightUrl])
  const globeRef = useRef<THREE.Mesh>(null)
  const orbitRef = useRef<THREE.Group>(null)
  const system = useMemo(() => buildOrbitSystem(), [])

  const material = useMemo(() => {
    dayTex.colorSpace = THREE.SRGBColorSpace
    nightTex.colorSpace = THREE.SRGBColorSpace
    return createGlobeMaterial(dayTex, nightTex)
  }, [dayTex, nightTex])

  useEffect(() => () => material.dispose(), [material])

  useFrame(({ camera }, delta) => {
    if (reduced) return
    const d = Math.min(delta, 0.05)
    if (globeRef.current) globeRef.current.rotation.y += d * 0.045
    if (orbitRef.current) orbitRef.current.rotation.y += d * 0.01
    animateOrbits(system.rings, d)
    // scroll-driven dolly: pull the camera in and drift up toward Europe
    const k = Math.min(1, d * 6)
    camera.position.z += (4.3 - zoom * 2.15 - camera.position.z) * k
    camera.position.y += (zoom * 0.55 - camera.position.y) * k
    camera.lookAt(0, zoom * 0.35, 0)
  })

  return (
    <>
      <group rotation={[0, 0, 0.41]}>
        <mesh ref={globeRef} material={material}>
          <sphereGeometry args={[1, 64, 64]} />
        </mesh>
      </group>
      <group ref={orbitRef}>
        <primitive object={system.root} />
      </group>
    </>
  )
}

/** Re-renders on demand when the frameloop is not 'always'. */
function Invalidator({ dep }: { dep: number }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => { invalidate() }, [dep, invalidate])
  return null
}

export default function HeroGlobe({ zoom, reduced }: { zoom: number; reduced: boolean }) {
  const [visible, setVisible] = useState(true)
  useEffect(() => {
    const onVis = () => setVisible(document.visibilityState === 'visible')
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  const frameloop: 'always' | 'demand' | 'never' =
    reduced ? 'demand' : visible ? 'always' : 'never'

  return (
    <Canvas
      frameloop={frameloop}
      dpr={[1, 2]}
      camera={{ position: [0, 0, 4.3], fov: 40 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
    >
      <Invalidator dep={zoom} />
      <Suspense fallback={null}>
        <Scene zoom={zoom} reduced={reduced} />
      </Suspense>
    </Canvas>
  )
}
