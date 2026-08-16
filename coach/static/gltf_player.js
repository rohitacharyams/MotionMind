// gltf_player.js — TIER 2 motion playback (Blender-identical path).
// ===========================================================================
// Plays motion on the avatar's OWN raw skeleton via three.js AnimationMixer —
// the exact glTF animation path Blender / Unity / Unreal use. No three-vrm
// normalization at PLAY time, no runtime band-aids: pure baked keyframes.
//
// IN-BROWSER BAKE (the unlock): no mediapipe / pkls / 5.6 GB of GLBs needed.
// three-vrm already propagates the normalized pose to the RAW bone nodes
// (J_Bip_*). So we drive a MotionPlayer (pure mode) on a hidden avatar from the
// corrected, whitelisted clip JSON, SAMPLE the raw node local quats + hips
// position each frame, and build a THREE.AnimationClip. That clip plays via
// AnimationMixer on the avatar's real skeleton = TRUE Tier 2, for ALL 313
// whitelist clips, instantly.
//
// AVATAR: defaults to 'mymodel1' (Nova) — user's custom VRM, measured close to
// the SMPL mocap body (arm -9% / leg -1% off). Override:
// window.__tier2.setAvatar('<character name>').
//
// API: window.__tier2 = { show(clipId), hide(), toggle(clipId), setAvatar(name),
//   avatar }.  Hotkey Alt+T toggles for the live clip.

(function () {
  if (window.__tier2) return;

  const BONES = ['hips', 'spine', 'chest', 'neck', 'head',
    'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
    'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
    'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
    'rightUpperLeg', 'rightLowerLeg', 'rightFoot'];

  let THREE = null, GLTFLoader = null, VRMLoaderPlugin = null, VRMUtils = null;
  let panel = null, renderer = null, scene = null, camera = null;
  let mixer = null, clock = null, raf = null;
  let avatarName = 'mymodel2';
  let avatarVrm = null, avatarLoadedFor = null;
  let curClip = null, busy = false;

  async function ensureLibs() {
    if (THREE) return;
    THREE = await import('three');
    ({ GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js'));
    ({ VRMLoaderPlugin, VRMUtils } = await import('@pixiv/three-vrm'));
  }

  function buildPanel() {
    if (panel) return;
    panel = document.createElement('div');
    panel.id = '__tier2-panel';
    panel.style.cssText = [
      'position:fixed', 'right:12px', 'top:60px', 'z-index:99997',
      'width:300px', 'height:420px', 'background:#0d0f17',
      'border:2px solid #2ad', 'border-radius:12px', 'overflow:hidden',
      'box-shadow:0 10px 40px rgba(0,0,0,.6)', 'font:11px system-ui',
    ].join(';');
    const bar = document.createElement('div');
    bar.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:#161a28;color:#8df;';
    bar.innerHTML = '<b>TIER 2 \u00b7 AnimationMixer</b>';
    const close = document.createElement('button');
    close.textContent = '\u00d7';
    close.style.cssText = 'background:none;border:none;color:#8df;font-size:16px;cursor:pointer;';
    close.onclick = () => hide();
    bar.appendChild(close);
    panel.appendChild(bar);
    const cv = document.createElement('canvas');
    cv.id = '__tier2-cv'; cv.width = 300; cv.height = 380;
    cv.style.cssText = 'width:300px;height:380px;display:block;background:#0d0f17;';
    panel.appendChild(cv);
    const cap = document.createElement('div');
    cap.id = '__tier2-cap';
    cap.style.cssText = 'position:absolute;bottom:4px;left:8px;color:#9ab;font:10px monospace;';
    panel.appendChild(cap);
    document.body.appendChild(panel);

    renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
    renderer.setPixelRatio(1); renderer.setSize(300, 380, false);
    renderer.setClearColor(0x0d0f17, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.05));
    const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(2, 4, 3);
    scene.add(dl);
    scene.add(new THREE.GridHelper(2, 8, 0x335, 0x223));
    camera = new THREE.PerspectiveCamera(32, 300 / 380, 0.01, 100);
    clock = new THREE.Clock();
  }

  function setCap(t) { const c = document.getElementById('__tier2-cap'); if (c) c.textContent = t; }

  async function ensureAvatar() {
    if (avatarVrm && avatarLoadedFor === avatarName) return avatarVrm;
    setCap('loading avatar ' + avatarName + '\u2026');
    if (avatarVrm) { try { scene.remove(avatarVrm.scene); VRMUtils.deepDispose(avatarVrm.scene); } catch (e) {} avatarVrm = null; }
    const buf = await fetch('/api/vrm/' + encodeURIComponent(avatarName)).then((r) => r.arrayBuffer());
    const loader = new GLTFLoader();
    loader.register((p) => new VRMLoaderPlugin(p));
    const gltf = await new Promise((res, rej) => loader.parse(buf, '', res, rej));
    const vrm = gltf.userData.vrm;
    try { VRMUtils.removeUnnecessaryJoints(gltf.scene); } catch (e) {}
    vrm.scene.traverse((o) => { if (o.isMesh) { o.frustumCulled = false; } });
    scene.add(vrm.scene);
    avatarVrm = vrm; avatarLoadedFor = avatarName;
    return vrm;
  }

  // Drive a MotionPlayer (pure mode) on the avatar, sample raw nodes -> AnimationClip.
  async function bakeClip(clipId, vrm) {
    const d = await fetch('/api/motion/data/' + clipId + '.json').then((r) => r.json());
    const fps = d.fps || 30;
    const N = Math.min(900, d.frames || d.n_frames || 0);
    if (N < 2) throw new Error('empty clip');
    const H = vrm.humanoid;
    const rawNodes = {};
    for (const b of BONES) { const n = H.getRawBoneNode(b); if (n) rawNodes[b] = n; }
    const hipsNode = rawNodes.hips;
    const MP = window.__player && window.__player.constructor;
    if (!MP) throw new Error('MotionPlayer unavailable');
    const livePlayer = window.__player;
    const prevPure = window.__coachPureMode;
    window.__coachPureMode = true;
    const tp = new MP(vrm, {});
    tp.load(d); tp.play({ loop: false });
    tp._easeStartedAt = performance.now() - 1e5;
    const times = new Float32Array(N);
    const quatTracks = {}; for (const b of BONES) quatTracks[b] = new Float32Array(N * 4);
    const hipsPos = new Float32Array(N * 3);
    const dt = 1 / fps;
    for (let i = 0; i < N; i++) {
      tp.update(dt); vrm.update(dt);
      times[i] = i * dt;
      for (const b of BONES) {
        const q = rawNodes[b].quaternion;
        let qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        // Hemisphere continuity: a quaternion and its negation are the SAME
        // rotation, but adjacent keyframes on opposite hemispheres make the
        // AnimationMixer's slerp walk the long way → a limb (foot) snaps
        // 180° for one frame. Force each frame to share a hemisphere with
        // the previous so the baked track interpolates the short arc.
        if (i > 0) {
          const o = i * 4 - 4, t = quatTracks[b];
          const dot = t[o] * qx + t[o + 1] * qy + t[o + 2] * qz + t[o + 3] * qw;
          if (dot < 0) { qx = -qx; qy = -qy; qz = -qz; qw = -qw; }
        }
        quatTracks[b][i * 4] = qx; quatTracks[b][i * 4 + 1] = qy;
        quatTracks[b][i * 4 + 2] = qz; quatTracks[b][i * 4 + 3] = qw;
      }
      if (hipsNode) {
        const p = hipsNode.position;
        hipsPos[i * 3] = p.x; hipsPos[i * 3 + 1] = p.y; hipsPos[i * 3 + 2] = p.z;
      }
    }
    window.__player = livePlayer;
    window.__coachPureMode = prevPure;
    const tracks = [];
    for (const b of BONES) {
      tracks.push(new THREE.QuaternionKeyframeTrack(
        rawNodes[b].name + '.quaternion', times, quatTracks[b]));
    }
    if (hipsNode) {
      tracks.push(new THREE.VectorKeyframeTrack(
        hipsNode.name + '.position', times, hipsPos));
    }
    return new THREE.AnimationClip('t2_' + clipId, (N - 1) * dt, tracks);
  }

  function frameCamera(vrm) {
    vrm.scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(vrm.scene);
    const c = box.getCenter(new THREE.Vector3());
    const s = box.getSize(new THREE.Vector3());
    const dist = s.y * 1.5 + 0.8;
    camera.position.set(c.x, c.y, c.z + dist);
    camera.up.set(0, 1, 0); camera.lookAt(c.x, c.y, c.z);
    camera.updateProjectionMatrix();
  }

  function loop() {
    raf = requestAnimationFrame(loop);
    const dt = clock.getDelta();
    if (mixer) mixer.update(dt);
    // CRITICAL: three-vrm needs its own update() to finalize the humanoid
    // pose after the AnimationMixer sets the raw bones. Without it, some
    // clips (CMU especially) render tilted ~25° because the normalized→raw
    // mapping isn't applied. Measured: mixer-only minUpY 0.70 (leaning) →
    // mixer + vrm.update() minUpY 1.00 (upright).
    if (avatarVrm && avatarVrm.update) avatarVrm.update(dt);
    if (avatarVrm) {
      avatarVrm.scene.position.y = 0;
      avatarVrm.scene.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(avatarVrm.scene);
      if (isFinite(box.min.y)) avatarVrm.scene.position.y = -box.min.y;
    }
    if (renderer) renderer.render(scene, camera);
  }

  async function show(clipId) {
    clipId = clipId || (window.__lastPlay && window.__lastPlay.clipId);
    if (!clipId) { console.warn('[tier2] no clip'); return; }
    if (busy) return;
    busy = true;
    try {
      await ensureLibs();
      buildPanel();
      const vrm = await ensureAvatar();
      setCap('baking ' + clipId + '\u2026');
      if (mixer) { try { mixer.stopAllAction(); } catch (e) {} mixer = null; }
      const clip = await bakeClip(clipId, vrm);
      mixer = new THREE.AnimationMixer(vrm.scene);
      mixer.clipAction(clip).play();
      mixer.update(0.001);
      frameCamera(vrm);
      curClip = clipId;
      setCap(clipId + ' \u00b7 ' + avatarName + ' \u00b7 pure glTF');
      if (!raf) loop();
    } catch (e) {
      setCap('error: ' + (e.message || e));
      console.warn('[tier2]', e);
    } finally {
      busy = false;
    }
  }

  // ---- PRODUCTION / MOBILE PATH: play a pre-exported anim-only file ----
  // Rebuilds a THREE.AnimationClip from the compact /static/anim/<clip>.t2.json
  // (raw bone names + quaternion tracks + hips position). NO runtime bake,
  // NO retarget math — just load + AnimationMixer. This is exactly what the
  // native/Capacitor app does: load the avatar once, stream a tiny clip file.
  function clipFromAnimFile(a) {
    const times = new Float32Array(a.n);
    for (let i = 0; i < a.n; i++) times[i] = i / a.fps;
    const tracks = [];
    for (const boneKey in a.bones) {
      const nodeName = a.names[boneKey];
      if (!nodeName) continue;
      tracks.push(new THREE.QuaternionKeyframeTrack(
        nodeName + '.quaternion', times, Float32Array.from(a.bones[boneKey])));
    }
    if (a.hips && a.names.hips) {
      tracks.push(new THREE.VectorKeyframeTrack(
        a.names.hips + '.position', times, Float32Array.from(a.hips)));
    }
    return new THREE.AnimationClip('t2file_' + a.clip, (a.n - 1) / a.fps, tracks);
  }

  async function showFile(clipId) {
    clipId = clipId || (window.__lastPlay && window.__lastPlay.clipId);
    if (!clipId) { console.warn('[tier2] no clip'); return; }
    if (busy) return;
    busy = true;
    try {
      await ensureLibs();
      buildPanel();
      const vrm = await ensureAvatar();
      setCap('loading anim ' + clipId + '\u2026');
      if (mixer) { try { mixer.stopAllAction(); } catch (e) {} mixer = null; }
      const a = await fetch('/static/anim/' + clipId + '.t2.json').then((r) => {
        if (!r.ok) throw new Error('no anim file (' + r.status + ')'); return r.json();
      });
      const clip = clipFromAnimFile(a);
      mixer = new THREE.AnimationMixer(vrm.scene);
      mixer.clipAction(clip).play();
      mixer.update(0.001);
      frameCamera(vrm);
      curClip = clipId;
      setCap(clipId + ' \u00b7 ' + avatarName + ' \u00b7 anim-only file');
      if (!raf) loop();
    } catch (e) {
      setCap('error: ' + (e.message || e));
      console.warn('[tier2 file]', e);
    } finally {
      busy = false;
    }
  }

  function hide() {
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    if (mixer) { try { mixer.stopAllAction(); } catch (e) {} mixer = null; }
    if (avatarVrm) { try { scene.remove(avatarVrm.scene); VRMUtils.deepDispose(avatarVrm.scene); } catch (e) {} avatarVrm = null; avatarLoadedFor = null; }
    if (panel) { panel.remove(); panel = null; renderer = null; scene = null; camera = null; }
    curClip = null;
  }

  function toggle(clipId) { if (panel) hide(); else show(clipId); }

  window.addEventListener('keydown', (e) => {
    if (e.altKey && (e.key === 't' || e.key === 'T')) {
      toggle((window.__lastPlay && window.__lastPlay.clipId) || null);
    }
  });

  window.__tier2 = {
    show, showFile, hide, toggle,
    setAvatar: (n) => { avatarName = n; avatarLoadedFor = null; },
    get avatar() { return avatarName; },
  };
  console.info('[tier2] ready \u2014 Alt+T to A/B live clip on Nova (mymodel1). __tier2.show(clipId) bakes; __tier2.showFile(clipId) plays the exported anim-only file');
})();
