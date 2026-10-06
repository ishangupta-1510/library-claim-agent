// AR sweep (Android Chrome + ARCore): metric room points and camera frames from one WebXR session.
//
// Metric scale for the room comes from ARCore's motion tracking. A hit test
// from the centre of the screen finds where the crosshair meets a detected
// surface; "Mark corner" records that point (metres, y up) for the floor
// polygon, "Mark ceiling" records a point on a wall at the ceiling line.
//
// The browser owns the camera during an AR session, so frames for the agent
// and the measuring pipeline are read from WebXR raw camera access.

const $ = (id) => document.getElementById(id);

export async function startAr({ onPoint, onEnd }) {
  const overlay = document.body;
  const session = await navigator.xr.requestSession("immersive-ar", {
    requiredFeatures: ["hit-test"],
    optionalFeatures: ["dom-overlay", "camera-access", "local-floor"],
    domOverlay: { root: overlay },
  });
  document.body.classList.add("xr");
  $("reticle").style.display = "block";

  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", { xrCompatible: true, alpha: true });
  await gl.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });

  let refSpace;
  try { refSpace = await session.requestReferenceSpace("local-floor"); } catch { refSpace = await session.requestReferenceSpace("local"); }
  const viewerSpace = await session.requestReferenceSpace("viewer");
  const hitSource = await session.requestHitTestSource({ space: viewerSpace });
  const binding = window.XRWebGLBinding ? new XRWebGLBinding(session, gl) : null;

  let lastHit = null;           // {position: [x,y,z], at: ms}
  let cameraPixels = null;      // latest camera image as RGBA, read on demand
  const pending = [];           // frame requests waiting for the next XR frame
  const framebuffer = gl.createFramebuffer();

  session.addEventListener("end", () => { document.body.classList.remove("xr"); $("reticle").style.display = "none"; onEnd(); });

  function readCamera(frameView) {
    // Raw camera access: the camera image arrives as a WebGL texture.
    if (!binding || !frameView.camera) return null;
    const texture = binding.getCameraImage(frameView.camera);
    const { width, height } = frameView.camera;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const pixels = new Uint8ClampedArray(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
    return { pixels, width, height };
  }

  function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const hits = frame.getHitTestResults(hitSource);
    const reticle = $("reticle");
    if (hits.length) {
      const p = hits[0].getPose(refSpace).transform.position;
      lastHit = { position: [p.x, p.y, p.z], at: time };
      reticle.classList.add("hit");
      $("reticle-label").textContent = `height ${p.y.toFixed(2)} m`;
    } else {
      reticle.classList.remove("hit");
      $("reticle-label").textContent = "aim at a surface";
    }

    const pose = frame.getViewerPose(refSpace);
    if (pose && pending.length) {
      cameraPixels = readCamera(pose.views[0]);
      pending.splice(0).forEach((resolve) => resolve(cameraPixels));
    }
  }
  session.requestAnimationFrame(onFrame);

  function mark(kind) {
    if (!lastHit || performance.now() - lastHit.at > 800) {
      $("status").textContent = "No surface under the crosshair: move slowly until it turns green";
      return;
    }
    onPoint(kind, lastHit.position);
    navigator.vibrate?.(40);
  }
  $("corner").onclick = () => mark("floor_corner");
  $("ceiling").onclick = () => mark("ceiling");

  // Frame grabber used by app.js: returns base64 JPEG of the latest camera image.
  const work = document.createElement("canvas");
  const workCtx = work.getContext("2d");
  const out = document.createElement("canvas");
  let warned = false;

  return function grabArFrame(maxSide, quality) {
    // Ask the next XR frame to read the camera; use the most recent image now.
    pending.push(() => {});
    const image = cameraPixels;
    if (!image) {
      if (!warned && binding === null) { warned = true; $("status").textContent = "This browser has no WebXR camera access"; }
      return null;
    }
    work.width = image.width;
    work.height = image.height;
    // readPixels returns rows bottom-up: flip while drawing.
    workCtx.putImageData(new ImageData(image.pixels, image.width, image.height), 0, 0);
    const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
    out.width = Math.round(image.width * scale);
    out.height = Math.round(image.height * scale);
    const ctx = out.getContext("2d");
    ctx.setTransform(1, 0, 0, -1, 0, out.height);
    ctx.drawImage(work, 0, 0, out.width, out.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return out.toDataURL("image/jpeg", quality).split(",")[1];
  };
}
