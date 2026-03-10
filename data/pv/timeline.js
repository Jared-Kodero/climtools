window.__PV_FRAME_DATA__ = window.__PV_FRAME_DATA__ || {};
window.__PV_TIMELINE_STATE__ = window.__PV_TIMELINE_STATE__ || {
    current: 0,
    initialized: false,
    viewState: null,
    refs: null,
    lastRequestedFrame: 0,
};

const frames = FRAMES_ARRAY;
const state = window.__PV_TIMELINE_STATE__;
const CAMERA_STORAGE_KEY = `__PV_VIEW_STATE__:${location.pathname}`;
const DEFAULT_CAMERA_CONFIG = Object.assign(
    {
        preset: null,
        azimuth: 0,
        elevation: 0,
        zoom: null,
    },
    window.__PV_CAMERA_CONFIG__ || {}
);

const container = document.querySelector('.content');
if (!container) {
    throw new Error('Template does not contain .content');
}

const viewerHost = container.parentElement || container;
if (getComputedStyle(viewerHost).position === 'static') {
    viewerHost.style.position = 'relative';
}
viewerHost.style.isolation = 'isolate';
viewerHost.setAttribute('data-pv-ready', 'false');

const styleEl = document.createElement('style');
styleEl.textContent = `
    .content,
    .content canvas {
        outline: none;
    }

    .content {
        min-height: 240px;
    }

    .content canvas {
        display: block;
        width: 100%;
        height: 100%;
    }
`;
document.head.appendChild(styleEl);

const loadingEl = document.createElement('div');
loadingEl.className = 'pv-loading-chip';
loadingEl.setAttribute('role', 'status');
loadingEl.setAttribute('aria-live', 'polite');
const loadingSpinner = document.createElement('span');
loadingSpinner.className = 'pv-loading-spinner';
const loadingLabel = document.createElement('span');
loadingLabel.className = 'pv-loading-label';
loadingLabel.textContent = 'Loading';
loadingEl.appendChild(loadingSpinner);
loadingEl.appendChild(loadingLabel);
viewerHost.appendChild(loadingEl);

function showLoading(text = 'Loading') {
    loadingLabel.textContent = text;
    loadingEl.classList.add('is-visible');
}

function hideLoading() {
    loadingEl.classList.remove('is-visible');
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function frameToPercent(i) {
    if (frames.length <= 1) return 0;
    return (i / (frames.length - 1)) * 100;
}

function percentToFrame(pct) {
    if (frames.length <= 1) return 0;
    return Math.round((pct / 100) * (frames.length - 1));
}

function safeNumber(value, fallback = null) {
    return Number.isFinite(value) ? value : fallback;
}

function safeVector3(value) {
    if (!Array.isArray(value) || value.length !== 3) return null;
    const out = value.map((v) => Number(v));
    return out.every(Number.isFinite) ? out : null;
}

function safeRange2(value) {
    if (!Array.isArray(value) || value.length !== 2) return null;
    const out = value.map((v) => Number(v));
    return out.every(Number.isFinite) ? out : null;
}

function readPersistedViewState() {
    try {
        const raw = sessionStorage.getItem(CAMERA_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        return normalizeViewState(parsed);
    } catch (_) {
        return null;
    }
}

function persistViewState(viewState) {
    try {
        if (!viewState) {
            sessionStorage.removeItem(CAMERA_STORAGE_KEY);
            return;
        }
        sessionStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify(viewState));
    } catch (_) {
        // Ignore sessionStorage failures.
    }
}

function normalizeViewState(viewState) {
    if (!viewState || typeof viewState !== 'object') return null;

    const normalized = {
        position: safeVector3(viewState.position),
        focalPoint: safeVector3(viewState.focalPoint),
        viewUp: safeVector3(viewState.viewUp),
        clippingRange: safeRange2(viewState.clippingRange),
        parallelScale: safeNumber(Number(viewState.parallelScale)),
        viewAngle: safeNumber(Number(viewState.viewAngle)),
        parallelProjection: typeof viewState.parallelProjection === 'boolean'
            ? viewState.parallelProjection
            : null,
    };

    if (!normalized.position || !normalized.focalPoint || !normalized.viewUp) {
        return null;
    }

    return normalized;
}

function kickRender() {
    const refs = getRefsFast();
    if (!refs) return;

    syncViewportSize(refs);

    if (refs.renderWindow && refs.renderWindow.render) {
        try {
            refs.renderWindow.render();
        } catch (_) {
            // Ignore render errors during transient resize/load phases.
        }
    }
}

function forceRenderBurst() {
    const delays = [0, 16, 40, 80, 140, 220, 320, 500, 800];
    for (const ms of delays) {
        setTimeout(() => {
            kickRender();
        }, ms);
    }
    requestAnimationFrame(() => {
        kickRender();
        requestAnimationFrame(() => kickRender());
    });
}

async function waitForContainerSize(timeoutMs = 8000) {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
        const rect = container.getBoundingClientRect();
        if (rect.width > 8 && rect.height > 8) {
            return true;
        }
        await sleep(32);
    }
    return false;
}

async function stabilizeViewport(token) {
    const delays = [0, 16, 40, 80, 140, 220, 320, 500, 800, 1200];
    for (const ms of delays) {
        if (ms > 0) {
            await sleep(ms);
        }
        if (token !== navToken) return;
        syncViewportSize();
        kickRender();
    }
}

async function reapplyViewStateBurst(viewState, token) {
    if (!viewState) return;

    const delays = [0, 16, 40, 80, 140, 220, 320, 500, 800, 1200];
    for (const ms of delays) {
        if (ms > 0) {
            await sleep(ms);
        }
        if (token !== navToken) return;
        applyViewState(viewState);
        syncViewportSize();
        kickRender();
    }
}

const lane = document.getElementById('timeline-lane');
const ticks = document.getElementById('ticks');
const thumb = document.getElementById('thumb');
const frameLabel = document.getElementById('frameLabel');

const nextBtn = document.getElementById('next');
const prevBtn = document.getElementById('prev');

let dragging = false;
let dragPreviewFrame = null;
let dragStartViewState = null;
let lastLoadedFrame = -1;
let navToken = 0;
let loadingPromise = Promise.resolve();
let saveViewTimer = 0;
let observersAttached = false;
let applyingViewState = false;

const scriptPromises = new Map();

function updateUI(i = state.current) {
    const pct = frameToPercent(i);
    if (thumb) thumb.style.left = pct + '%';
    if (frameLabel) {
        frameLabel.style.left = pct + '%';
        frameLabel.textContent = String(i);
    }
}

function clientXToFrame(clientX) {
    const rect = lane.getBoundingClientRect();
    const x = clamp(clientX - rect.left, 0, rect.width);
    const pct = rect.width === 0 ? 0 : (x / rect.width) * 100;
    return percentToFrame(pct);
}

function ensureFrameData(i) {
    if (window.__PV_FRAME_DATA__[i]) {
        return Promise.resolve(window.__PV_FRAME_DATA__[i]);
    }
    if (scriptPromises.has(i)) {
        return scriptPromises.get(i);
    }

    const p = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = '<<SRC_DIR>>/' + frames[i];
        script.async = true;
        script.onload = () => {
            const payload = window.__PV_FRAME_DATA__[i];
            if (!payload) {
                reject(new Error('Payload script loaded, but frame data missing for ' + i));
                return;
            }
            resolve(payload);
        };
        script.onerror = () => reject(new Error('Failed to load frame payload ' + i));
        document.head.appendChild(script);
    });

    scriptPromises.set(i, p);
    return p;
}

function prefetchNeighbors(i) {
    const neighbors = [i - 2, i - 1, i + 1, i + 2].filter(
        (j) => j >= 0 && j < frames.length
    );
    for (const j of neighbors) {
        void ensureFrameData(j).catch(() => { });
    }
}

function clearContainer() {
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
}

function looksLikeCamera(obj) {
    return !!(
        obj &&
        typeof obj.getPosition === 'function' &&
        typeof obj.setPosition === 'function' &&
        typeof obj.getFocalPoint === 'function' &&
        typeof obj.setFocalPoint === 'function' &&
        typeof obj.getViewUp === 'function' &&
        typeof obj.setViewUp === 'function'
    );
}

function looksLikeRenderer(obj) {
    return !!(obj && typeof obj.getActiveCamera === 'function');
}

function looksLikeRenderWindow(obj) {
    return !!(obj && typeof obj.render === 'function');
}

function looksLikeApiSpecificView(obj) {
    return !!(
        obj &&
        typeof obj.setSize === 'function' &&
        (typeof obj.getCanvas === 'function' || typeof obj.getContainer === 'function')
    );
}

function shallowScan(obj, found) {
    if (!obj || typeof obj !== 'object') return;

    if (!found.renderer && looksLikeRenderer(obj)) found.renderer = obj;
    if (!found.renderWindow && looksLikeRenderWindow(obj)) found.renderWindow = obj;
    if (!found.camera && looksLikeCamera(obj)) found.camera = obj;
    if (!found.apiView && looksLikeApiSpecificView(obj)) found.apiView = obj;

    const keys = Object.keys(obj).slice(0, 120);
    for (const key of keys) {
        let sub;
        try {
            sub = obj[key];
        } catch (_) {
            continue;
        }
        if (!sub || typeof sub !== 'object') continue;

        if (!found.renderer && looksLikeRenderer(sub)) found.renderer = sub;
        if (!found.renderWindow && looksLikeRenderWindow(sub)) found.renderWindow = sub;
        if (!found.camera && looksLikeCamera(sub)) found.camera = sub;
        if (!found.apiView && looksLikeApiSpecificView(sub)) found.apiView = sub;
    }
}

function extractRefsFromKnownObjects() {
    const found = {
        renderer: null,
        renderWindow: null,
        camera: null,
        apiView: null,
    };

    if (state.refs) {
        shallowScan(state.refs, found);
    }

    if (window.__pv_last_load_result__) {
        shallowScan(window.__pv_last_load_result__, found);
    }

    if (found.renderer && !found.camera) {
        try {
            found.camera = found.renderer.getActiveCamera();
        } catch (_) { }
    }

    if (found.renderer || found.renderWindow || found.camera || found.apiView) {
        state.refs = found;
        return found;
    }

    return null;
}

function discoverRefsOnce() {
    const cached = extractRefsFromKnownObjects();
    if (cached && (cached.camera || cached.renderer || cached.renderWindow || cached.apiView)) {
        return cached;
    }

    const found = {
        renderer: null,
        renderWindow: null,
        camera: null,
        apiView: null,
    };

    const winKeys = Object.getOwnPropertyNames(window).slice(0, 300);
    for (const k of winKeys) {
        let obj;
        try {
            obj = window[k];
        } catch (_) {
            continue;
        }
        shallowScan(obj, found);
        if (found.renderer && found.renderWindow && found.camera && found.apiView) break;
    }

    if (!found.camera && found.renderer) {
        try {
            found.camera = found.renderer.getActiveCamera();
        } catch (_) { }
    }

    if (found.renderer || found.renderWindow || found.camera || found.apiView) {
        state.refs = found;
    }

    return state.refs;
}

function getRefsFast() {
    return extractRefsFromKnownObjects() || state.refs || null;
}

function getCameraFromRefs(refs) {
    if (!refs) return null;
    if (refs.camera) return refs.camera;
    if (refs.renderer && typeof refs.renderer.getActiveCamera === 'function') {
        try {
            return refs.renderer.getActiveCamera();
        } catch (_) {
            return null;
        }
    }
    return null;
}

function syncViewportSize(refs = null) {
    refs = refs || getRefsFast() || discoverRefsOnce();
    if (!refs) return false;

    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));

    if (refs.apiView && typeof refs.apiView.setSize === 'function') {
        try {
            refs.apiView.setSize(width, height);
        } catch (_) {
            // Ignore transient sizing errors.
        }
    }

    const canvas = container.querySelector('canvas');
    if (canvas) {
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;
    }

    return true;
}

function captureViewState() {
    const refs = getRefsFast() || discoverRefsOnce();
    const cam = getCameraFromRefs(refs);
    if (!cam) return null;

    const out = {};
    try { out.position = Array.from(cam.getPosition()); } catch (_) { }
    try { out.focalPoint = Array.from(cam.getFocalPoint()); } catch (_) { }
    try { out.viewUp = Array.from(cam.getViewUp()); } catch (_) { }
    try { out.parallelProjection = !!cam.getParallelProjection(); } catch (_) { }
    try { out.parallelScale = cam.getParallelScale(); } catch (_) { }
    try { out.viewAngle = cam.getViewAngle(); } catch (_) { }
    try { out.clippingRange = Array.from(cam.getClippingRange()); } catch (_) { }
    return normalizeViewState(out);
}

function saveCameraForNavigation() {
    if (applyingViewState) return state.viewState || null;
    const v = captureViewState();
    if (v) {
        state.viewState = v;
        persistViewState(v);
    }
    return v;
}

function scheduleViewStateSave(delayMs = 64) {
    if (applyingViewState) return;
    clearTimeout(saveViewTimer);
    saveViewTimer = setTimeout(() => {
        requestAnimationFrame(() => {
            if (!applyingViewState) {
                saveCameraForNavigation();
            }
        });
    }, delayMs);
}

function getPresetCameraVectors(preset) {
    switch ((preset || '').toLowerCase()) {
        case 'xy':
            return {
                position: [0, 0, 1],
                focalPoint: [0, 0, 0],
                viewUp: [0, 1, 0],
            };
        case 'xz':
            return {
                position: [0, -1, 0],
                focalPoint: [0, 0, 0],
                viewUp: [0, 0, 1],
            };
        case 'yz':
            return {
                position: [1, 0, 0],
                focalPoint: [0, 0, 0],
                viewUp: [0, 0, 1],
            };
        case 'iso':
            return {
                position: [1, 1, 1],
                focalPoint: [0, 0, 0],
                viewUp: [0, 0, 1],
            };
        default:
            return null;
    }
}

function applyPyVistaLikeCamera(config = DEFAULT_CAMERA_CONFIG) {
    const refs = getRefsFast() || discoverRefsOnce();
    const renderer = refs && refs.renderer;
    const cam = getCameraFromRefs(refs);
    if (!cam) return null;

    applyingViewState = true;
    try {
        const preset = getPresetCameraVectors(config.preset);
        if (preset) {
            cam.setPosition(...preset.position);
            cam.setFocalPoint(...preset.focalPoint);
            cam.setViewUp(...preset.viewUp);
        }

        if (typeof cam.computeDistance === 'function') {
            cam.computeDistance();
        }
        if (typeof cam.orthogonalizeViewUp === 'function') {
            cam.orthogonalizeViewUp();
        }
        if (typeof config.elevation === 'number' && config.elevation !== 0 && typeof cam.elevation === 'function') {
            cam.elevation(config.elevation);
        }
        if (typeof config.azimuth === 'number' && config.azimuth !== 0 && typeof cam.azimuth === 'function') {
            cam.azimuth(config.azimuth);
        }
        if (typeof config.zoom === 'number' && config.zoom > 0 && typeof cam.zoom === 'function') {
            cam.zoom(config.zoom);
        }
        if (renderer && typeof renderer.resetCameraClippingRange === 'function') {
            renderer.resetCameraClippingRange();
        }
        kickRender();
    } finally {
        requestAnimationFrame(() => {
            applyingViewState = false;
            scheduleViewStateSave(0);
        });
    }

    return captureViewState();
}

function applyViewState(viewState) {
    const normalized = normalizeViewState(viewState);
    if (!normalized) return false;

    const refs = getRefsFast() || discoverRefsOnce();
    if (!refs) return false;

    const renderer = refs.renderer;
    const renderWindow = refs.renderWindow;
    const cam = getCameraFromRefs(refs);
    if (!cam) return false;

    applyingViewState = true;
    try {
        if (normalized.position && cam.setPosition) {
            cam.setPosition(...normalized.position);
        }
        if (normalized.focalPoint && cam.setFocalPoint) {
            cam.setFocalPoint(...normalized.focalPoint);
        }
        if (normalized.viewUp && cam.setViewUp) {
            cam.setViewUp(...normalized.viewUp);
        }
        if (typeof cam.computeDistance === 'function') {
            cam.computeDistance();
        }
        if (typeof cam.orthogonalizeViewUp === 'function') {
            cam.orthogonalizeViewUp();
        }
        if (typeof normalized.parallelProjection === 'boolean' && cam.setParallelProjection) {
            cam.setParallelProjection(normalized.parallelProjection);
        }
        if (typeof normalized.parallelScale === 'number' && cam.setParallelScale) {
            cam.setParallelScale(normalized.parallelScale);
        }
        if (typeof normalized.viewAngle === 'number' && cam.setViewAngle) {
            cam.setViewAngle(normalized.viewAngle);
        }
        if (normalized.clippingRange && cam.setClippingRange) {
            cam.setClippingRange(...normalized.clippingRange);
        }
        if (renderer && renderer.resetCameraClippingRange) {
            renderer.resetCameraClippingRange();
        }
        syncViewportSize(refs);
        if (renderWindow && renderWindow.render) {
            renderWindow.render();
        }
        state.viewState = normalized;
        persistViewState(normalized);
        return true;
    } catch (_) {
        return false;
    } finally {
        requestAnimationFrame(() => {
            applyingViewState = false;
        });
    }
}

async function waitForOfflineLocalView(timeoutMs = 10000) {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
        if (window.OfflineLocalView && typeof window.OfflineLocalView.load === 'function') {
            return;
        }
        await sleep(20);
    }
    throw new Error('OfflineLocalView.load did not become available in time');
}

async function waitForSceneReady(timeoutMs = 2500) {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
        const refs = extractRefsFromKnownObjects() || discoverRefsOnce();
        const cam = getCameraFromRefs(refs);
        const canvas = container.querySelector('canvas');
        const rect = container.getBoundingClientRect();
        if (cam && canvas && rect.width > 8 && rect.height > 8) {
            syncViewportSize(refs);
            return true;
        }
        await sleep(16);
    }
    return false;
}

function prepareInteractiveViewport() {
    const interactiveEl = container.querySelector('canvas, [tabindex]');
    if (!interactiveEl) return;

    if (!interactiveEl.hasAttribute('tabindex')) {
        interactiveEl.setAttribute('tabindex', '0');
    }
    interactiveEl.setAttribute('aria-label', '3D frame viewer');
}

async function loadPayload(base64Str) {
    await waitForContainerSize();
    clearContainer();

    const result = window.OfflineLocalView.load(container, { base64Str });
    if (result && typeof result.then === 'function') {
        window.__pv_last_load_result__ = await result;
    } else {
        window.__pv_last_load_result__ = result || null;
    }

    state.refs = null;
    await nextFrame();
    await nextFrame();
    prepareInteractiveViewport();
    extractRefsFromKnownObjects() || discoverRefsOnce();
    syncViewportSize();
    forceRenderBurst();
}

async function loadFrame(i, preservedViewState = null, token = 0) {
    i = clamp(i, 0, frames.length - 1);

    if (i === lastLoadedFrame && state.initialized) {
        state.current = i;
        updateUI(i);
        if (preservedViewState || state.viewState) {
            applyViewState(preservedViewState || state.viewState);
        }
        forceRenderBurst();
        return;
    }

    showLoading('Loading frame ' + i);

    try {
        await waitForContainerSize();
        if (token !== navToken) return;

        const payload = await ensureFrameData(i);
        if (token !== navToken) return;

        await waitForOfflineLocalView();
        if (token !== navToken) return;

        await loadPayload(payload);
        if (token !== navToken) return;

        await waitForSceneReady();
        if (token !== navToken) return;

        state.current = i;
        state.initialized = true;
        lastLoadedFrame = i;
        updateUI(i);
        viewerHost.setAttribute('data-pv-ready', 'true');

        const viewToApply = normalizeViewState(preservedViewState || state.viewState || null);
        if (viewToApply) {
            state.viewState = viewToApply;
            await reapplyViewStateBurst(viewToApply, token);
        } else {
            const seededView = applyPyVistaLikeCamera(DEFAULT_CAMERA_CONFIG);
            if (seededView) {
                state.viewState = seededView;
                persistViewState(seededView);
            }
        }

        await stabilizeViewport(token);
        await nextFrame();
        await nextFrame();
        prepareInteractiveViewport();
        forceRenderBurst();
        scheduleViewStateSave(0);

        setTimeout(() => {
            if (token === navToken) {
                window.dispatchEvent(new Event('resize'));
                forceRenderBurst();
                scheduleViewStateSave(0);
            }
        }, 0);

        prefetchNeighbors(i);
    } finally {
        if (token === navToken) {
            hideLoading();
        }
    }
}

function requestFrame(i, preservedViewState = null) {
    i = clamp(i, 0, frames.length - 1);

    if (!preservedViewState) {
        preservedViewState = saveCameraForNavigation() || state.viewState || null;
    }

    state.lastRequestedFrame = i;
    updateUI(i);

    const token = ++navToken;

    loadingPromise = loadingPromise
        .catch(() => { })
        .then(() => loadFrame(i, preservedViewState, token))
        .catch((err) => {
            if (token === navToken) {
                console.error(err);
                showLoading('Failed to load');
            }
        });
}

function goNext() {
    requestFrame(state.current + 1, saveCameraForNavigation());
}

function goPrev() {
    requestFrame(state.current - 1, saveCameraForNavigation());
}

function previewDrag(clientX) {
    dragPreviewFrame = clientXToFrame(clientX);
    updateUI(dragPreviewFrame);
}

function commitDrag() {
    if (dragPreviewFrame == null) return;

    const target = dragPreviewFrame;
    const preserved = dragStartViewState || state.viewState || saveCameraForNavigation() || null;

    dragPreviewFrame = null;
    dragStartViewState = null;

    requestFrame(target, preserved);
}

function buildTicks() {
    if (!ticks) return;
    ticks.innerHTML = '';
    const maxLabels = 20;
    const step = Math.max(1, Math.ceil(frames.length / maxLabels));

    frames.forEach((_, i) => {
        const tick = document.createElement('button');
        tick.type = 'button';
        tick.className = 'tick';
        tick.style.left = frameToPercent(i) + '%';

        let label = '';
        if (i % step === 0 || i === frames.length - 1) {
            label = `<span class="tick-label">${i}</span>`;
        }

        tick.innerHTML = `<span class="tick-mark"></span>${label}`;
        tick.addEventListener('click', () => {
            requestFrame(i, saveCameraForNavigation());
        });

        ticks.appendChild(tick);
    });
}

function attachObservers() {
    if (observersAttached) return;
    observersAttached = true;

    const persistViewState = () => scheduleViewStateSave(80);
    const rerenderScene = () => {
        syncViewportSize();
        forceRenderBurst();
        scheduleViewStateSave(120);
    };

    ['pointerup', 'mouseup', 'touchend', 'wheel', 'keyup', 'pointerleave'].forEach((eventName) => {
        viewerHost.addEventListener(eventName, persistViewState, { passive: true });
    });

    viewerHost.addEventListener('focusin', rerenderScene, { passive: true });

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            rerenderScene();
        }
    });

    window.addEventListener('focus', rerenderScene, { passive: true });
    window.addEventListener('pageshow', rerenderScene, { passive: true });
    window.addEventListener('load', rerenderScene, { passive: true });
    window.addEventListener('resize', rerenderScene, { passive: true });

    if (typeof ResizeObserver === 'function') {
        const resizeObserver = new ResizeObserver(() => {
            rerenderScene();
        });
        resizeObserver.observe(container);
        resizeObserver.observe(viewerHost);
    }

    if (typeof MutationObserver === 'function') {
        const mutationObserver = new MutationObserver(() => {
            prepareInteractiveViewport();
            rerenderScene();
        });
        mutationObserver.observe(container, { childList: true, subtree: true });
    }

    if (typeof IntersectionObserver === 'function') {
        const intersectionObserver = new IntersectionObserver((entries) => {
            const entry = entries[0];
            if (entry && entry.isIntersecting) {
                rerenderScene();
            }
        }, { threshold: 0.05 });
        intersectionObserver.observe(viewerHost);
    }
}

if (lane) {
    lane.addEventListener('click', (e) => {
        if (dragging) return;
        if (e.target.closest('.tick')) return;

        requestFrame(clientXToFrame(e.clientX), saveCameraForNavigation());
    });
}

if (thumb) {
    thumb.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        dragging = true;
        thumb.setPointerCapture(e.pointerId);

        dragStartViewState = saveCameraForNavigation();
        dragPreviewFrame = state.current;
    });
}

document.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    previewDrag(e.clientX);
}, { passive: true });

document.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false;
    commitDrag();
});

document.addEventListener('pointercancel', () => {
    if (!dragging) return;
    dragging = false;
    dragPreviewFrame = null;
    dragStartViewState = null;
    updateUI(state.current);
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goPrev();
    }
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        goNext();
    }
});

if (nextBtn) {
    nextBtn.addEventListener('click', (e) => {
        e.preventDefault();
        goNext();
    });
}

if (prevBtn) {
    prevBtn.addEventListener('click', (e) => {
        e.preventDefault();
        goPrev();
    });
}

buildTicks();
updateUI(0);
attachObservers();

let booted = false;

async function init() {
    if (booted) return;
    booted = true;

    const persistedView = readPersistedViewState();
    if (persistedView) {
        state.viewState = persistedView;
    }

    updateUI(state.current || 0);
    await waitForContainerSize();
    requestFrame(state.current || 0, state.viewState || null);

    try {
        await loadingPromise;
    } catch (err) {
        console.error(err);
        showLoading('Failed to load');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        void init();
    }, { once: true });
} else {
    void init();
}
