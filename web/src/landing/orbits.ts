import * as THREE from 'three'

export interface RingSpec {
  group: THREE.Group
  satellites: { pivot: THREE.Object3D; speed: number; angle: number }[]
  trail?: { pivot: THREE.Object3D; speed: number; angle: number }
}

const RING_COUNT = 7

/** Deterministic PRNG so the orbit layout is stable across reloads. */
function rng(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

function circlePoints(radius: number, segments = 160): THREE.Vector3[] {
  const pts: THREE.Vector3[] = []
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2
    pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius))
  }
  return pts
}

function makeSatellite(): THREE.Group {
  const g = new THREE.Group()
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.022, 0.022, 0.05),
    new THREE.MeshBasicMaterial({ color: 0xdfe8ff }),
  )
  const panelGeo = new THREE.BoxGeometry(0.085, 0.004, 0.028)
  const panelMat = new THREE.MeshBasicMaterial({ color: 0x6fb2ff })
  const p1 = new THREE.Mesh(panelGeo, panelMat)
  p1.position.x = 0.062
  const p2 = new THREE.Mesh(panelGeo, panelMat)
  p2.position.x = -0.062
  g.add(body, p1, p2)
  return g
}

export function buildOrbitSystem(): { root: THREE.Group; rings: RingSpec[] } {
  const rand = rng(20260826)
  const root = new THREE.Group()
  const rings: RingSpec[] = []

  for (let i = 0; i < RING_COUNT; i++) {
    const radius = 1.42 + rand() * 0.55
    const group = new THREE.Group()
    group.rotation.x = (rand() - 0.5) * Math.PI * 0.9
    group.rotation.z = (rand() - 0.5) * Math.PI * 0.9

    const highlighted = i === 0
    const ringGeo = new THREE.BufferGeometry().setFromPoints(circlePoints(radius))
    const ringMat = new THREE.LineBasicMaterial({
      color: highlighted ? 0x35c5f2 : 0x3d6fe0,
      transparent: true,
      opacity: highlighted ? 0.85 : 0.26 + rand() * 0.16,
    })
    group.add(new THREE.LineLoop(ringGeo, ringMat))

    const satellites: RingSpec['satellites'] = []
    if (rand() < 0.6 || highlighted) {
      const pivot = new THREE.Group()
      const sat = makeSatellite()
      sat.position.x = radius
      pivot.add(sat)
      pivot.rotation.y = rand() * Math.PI * 2
      group.add(pivot)
      satellites.push({ pivot, speed: 0.12 + rand() * 0.1, angle: pivot.rotation.y })
    }

    let trail: RingSpec['trail']
    if (highlighted) {
      // comet: glowing head + additive arc trailing behind it, both share a
      // pivot so the whole comet travels around the ring
      const headPivot = new THREE.Group()
      const arcLen = Math.PI * 0.5
      const curvePts: THREE.Vector3[] = []
      for (let k = 0; k <= 48; k++) {
        const a = -arcLen * (k / 48)
        curvePts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius))
      }
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(
          new THREE.CatmullRomCurve3(curvePts), 64, 0.006, 6, false,
        ),
        new THREE.MeshBasicMaterial({
          color: 0x35c5f2, transparent: true, opacity: 0.5,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }),
      )
      const head = new THREE.Mesh(
        new THREE.SphereGeometry(0.016, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xbff1ff }),
      )
      head.position.x = radius
      headPivot.add(tube, head)
      group.add(headPivot)
      trail = { pivot: headPivot, speed: 0.22, angle: rand() * Math.PI * 2 }
      headPivot.rotation.y = trail.angle
    }

    root.add(group)
    rings.push({ group, satellites, trail })
  }
  return { root, rings }
}

export function animateOrbits(rings: RingSpec[], delta: number) {
  for (const r of rings) {
    for (const s of r.satellites) {
      s.angle += s.speed * delta
      ;(s.pivot as THREE.Group).rotation.y = s.angle
    }
    if (r.trail) {
      r.trail.angle += r.trail.speed * delta
      ;(r.trail.pivot as THREE.Group).rotation.y = r.trail.angle
    }
  }
}
