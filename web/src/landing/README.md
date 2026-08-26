# Landing page (`home` view)

Self-contained, **removable** animated landing page: rotating Earth with
orbiting satellites (three.js), scroll-driven zoom handoff into the app.

## Removing it

Delete this folder and rebuild — no other edits are needed:

```bash
rm -rf web/src/landing
cd web && npm run build
```

The only integration point is `web/src/App.tsx`, which discovers this module
via `import.meta.glob('./landing/index.tsx')`. With the folder gone the glob
resolves to `{}` at build time: the Home nav entry disappears and the app
opens on the Console exactly as before the landing page existed.

## Contents

| File | Role |
|---|---|
| `index.tsx` | lazy entry (loaded as a separate chunk by App.tsx) |
| `LandingPage.tsx` | scroll track, pinned hero, copy block, placeholder next section |
| `HeroGlobe.tsx` | three.js canvas: camera dolly, visibility pause, DPR cap |
| `globe.ts` | day/night shader material (terminator blend + atmosphere rim) |
| `orbits.ts` | deterministic orbit rings, satellites, comet trail |
| `assets/` | NASA Blue Marble day + city-lights textures (public domain) |

## Notes

- `prefers-reduced-motion`: static globe, no zoom, no orbit animation.
- No WebGL: gradient-orb + SVG-rings fallback.
- Textures: NASA Visible Earth / Blue Marble imagery (public domain).
  Visual style inspired by a reference animation; all code and assets here
  are original or public-domain.
