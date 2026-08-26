import * as THREE from 'three'

/**
 * Day/night globe material: blends a NASA Blue Marble day texture with the
 * city-lights night texture across the terminator, plus a fresnel atmosphere
 * rim. Sun direction is fixed so the lit face matches the reference art.
 */
export function createGlobeMaterial(
  dayTex: THREE.Texture,
  nightTex: THREE.Texture,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      dayTex: { value: dayTex },
      nightTex: { value: nightTex },
      sunDir: { value: new THREE.Vector3(0.9, 0.35, 0.6).normalize() },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      void main() {
        vUv = uv;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vViewDir = normalize(cameraPosition - wp.xyz);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D dayTex;
      uniform sampler2D nightTex;
      uniform vec3 sunDir;
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      void main() {
        vec3 n = normalize(vNormalW);
        float sun = dot(n, normalize(sunDir));
        float dayness = smoothstep(-0.12, 0.35, sun);
        vec3 day = texture2D(dayTex, vUv).rgb * (0.35 + 0.75 * clamp(sun, 0.0, 1.0));
        vec3 night = texture2D(nightTex, vUv).rgb * vec3(1.0, 0.82, 0.55) * 1.7;
        vec3 col = mix(night, day, dayness);
        float rim = pow(1.0 - clamp(dot(n, vViewDir), 0.0, 1.0), 2.6);
        col += vec3(0.25, 0.5, 1.0) * rim * 0.55;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  })
}
