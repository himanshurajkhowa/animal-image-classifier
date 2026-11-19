export default function (component) {
  const { parentElement, data, setTriggerValue } = component;
  const root = parentElement.querySelector("#animal-app");
  const $ = (selector) => root.querySelector(selector);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const state = (root.animalState ??= {
    stage: "welcome",
    photo: null,
    example: null,
    active: -1,
    galleryOpen: false,
    result: null,
    pending: false,
    requestId: null,
    mode: "softmax",
    stream: null,
  });
  const events = new AbortController();
  const listen = (element, name, callback, options = {}) =>
    element.addEventListener(name, callback, {
      ...options,
      signal: events.signal,
    });
  const gallery = data.gallery ?? [];
  const maxPhotoBytes = 8 * 1024 * 1024;
  const maxPreparedBytes = 700 * 1024;
  const maxPhotoPixels = 24_000_000;
  const heicHint =
    "This browser could not open the HEIC photo. Export it as JPEG or upload a screenshot.";

  function withPhotoDeadline(signal, work) {
    return new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
      if (signal.aborted) reject(signal.reason);
      else Promise.resolve().then(work).then(resolve, reject);
    });
  }

  async function photoFormat(file) {
    const head = new Uint8Array(await file.slice(0, 32).arrayBuffer());
    const tag = (offset, size) =>
      String.fromCharCode(...head.slice(offset, offset + size));
    if (head[0] === 255 && head[1] === 216 && head[2] === 255) return "jpeg";
    if (head[0] === 137 && tag(1, 3) === "PNG") return "png";
    if (tag(0, 4) === "RIFF" && tag(8, 4) === "WEBP") return "webp";
    if (tag(4, 4) === "ftyp" && /^(heic|heix|heif|mif1)$/.test(tag(8, 4)))
      return "heic";
    if (/heic|heif/i.test(file.type) || /\.hei[cf]$/i.test(file.name)) {
      throw new Error(heicHint);
    }
    throw new Error("Choose a valid JPG, PNG, WEBP or HEIC photo.");
  }

  async function preparePhoto(file, image, format, signal) {
    if (file.size <= maxPreparedBytes && format !== "heic") return file;

    // Do not turn an animated upload into an arbitrary still image.
    if (format === "png" || format === "webp") {
      const bytes = new DataView(await file.arrayBuffer());
      if (signal.aborted) throw signal.reason;
      if (
        format === "webp" &&
        bytes.byteLength > 20 &&
        bytes.getUint32(12) === 0x56503858 &&
        bytes.getUint8(20) & 2
      ) {
        throw new Error(
          "Choose a still photograph rather than an animated image.",
        );
      }
      if (format === "png") {
        for (let offset = 8; offset + 12 <= bytes.byteLength; ) {
          const type = bytes.getUint32(offset + 4);
          if (type === 0x6163544c) {
            throw new Error(
              "Choose a still photograph rather than an animated image.",
            );
          }
          if (type === 0x49444154) break;
          offset += 12 + bytes.getUint32(offset);
        }
      }
    }

    // Keep the whole frame. The model's resize and centre crop still run on the server.
    const scale = Math.min(
      1,
      1800 / Math.max(image.naturalWidth, image.naturalHeight),
    );
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    try {
      const ctx = canvas.getContext("2d", { alpha: false });
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      for (const quality of [0.88, 0.72, 0.55]) {
        const blob = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", quality),
        );
        if (signal.aborted) throw signal.reason;
        if (
          blob?.type === "image/jpeg" &&
          blob.size &&
          blob.size <= maxPreparedBytes
        )
          return blob;
      }
      throw new Error(
        "This photo could not be made small enough. Try a smaller copy.",
      );
    } finally {
      canvas.width = canvas.height = 0;
    }
  }

  function readUpload(blob, signal) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      const abort = () => reader.abort();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reader.onabort = () =>
        reject(
          new Error(
            "Could not read the photo. Try again or choose another one.",
          ),
        );
      reader.onloadend = () => signal.removeEventListener("abort", abort);
      signal.addEventListener("abort", abort, { once: true });
      reader.readAsDataURL(blob);
    });
  }
  const showNotice = (message = "") => {
    $("#notice").textContent = message;
    $("#notice").hidden = !message;
  };
  const focus = (element) => element.focus({ preventScroll: true });
  const scrollTo = (element, block = "nearest") =>
    element.scrollIntoView({
      behavior: reduceMotion.matches ? "auto" : "smooth",
      block,
      inline: "nearest",
    });

  if (!root.dataset.built) {
    const main = gallery.find((item) => item.animal === "Dog") ?? gallery[0];
    const small = gallery.find((item) => item.animal === "Cat") ?? gallery[1];
    $("#hero-main").src = main.src;
    $("#hero-small").src = small.src;
    gallery.forEach((item, index) => {
      const button = document.createElement("button");
      button.className = "gallery-card";
      button.dataset.index = index;
      button.setAttribute(
        "aria-label",
        `Select ${item.animal} example ${gallery.slice(0, index + 1).filter((x) => x.animal === item.animal).length}`,
      );
      button.setAttribute("aria-pressed", "false");
      const image = document.createElement("img");
      image.src = item.src;
      image.alt = item.alt;
      image.loading = "lazy";
      image.draggable = false;
      const caption = document.createElement("span");
      caption.className = "animal-caption";
      caption.textContent = item.animal;
      const mark = document.createElement("span");
      mark.className = "selected-mark";
      mark.setAttribute("aria-hidden", "true");
      caption.append(mark);
      button.append(image, caption);
      $("#gallery").append(button);
      const credit = document.createElement("div");
      credit.className = "credit-item";
      const title = document.createElement("strong");
      title.textContent = `${item.animal} · ${item.author}`;
      const source = document.createElement("a");
      source.href = item.source_url;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.textContent = "Original photograph";
      const license = document.createElement("a");
      license.href = item.license_url;
      license.target = "_blank";
      license.rel = "noopener noreferrer";
      license.textContent = item.license_label ?? item.license;
      const modification = document.createElement("div");
      modification.textContent = item.modifications;
      credit.append(title, source, license, modification);
      $("#credits-list").append(credit);
    });
    root.dataset.built = "true";
  }

  function render() {
    $("#welcome").hidden = state.stage !== "welcome";
    $("#classifier").hidden = state.stage !== "classify";
    $("#mode").value = state.mode;
    $("#gallery-section").hidden = !state.galleryOpen;
    $("#examples-open").setAttribute(
      "aria-expanded",
      String(state.galleryOpen),
    );
    const selected = Boolean(state.photo || state.example);
    $(".workbench").classList.toggle("has-photo", selected);
    $("#prediction-panel").hidden = !selected;
    $("#empty").hidden = selected;
    $("#preview").hidden = !selected;
    $("#photo-label").hidden = !selected;
    if (selected) {
      const src = state.example
        ? gallery.find((item) => item.id === state.example)?.src
        : state.photo.src;
      if ($("#preview").getAttribute("src") !== src) $("#preview").src = src;
      $("#photo-label").textContent = state.example
        ? `Gallery · ${gallery.find((item) => item.id === state.example).animal}`
        : state.photo.name;
    } else {
      $("#preview").removeAttribute("src");
    }
    $("#predict").disabled =
      !selected || state.pending || Boolean(root.photoJob);
    $("#predict").setAttribute("aria-busy", String(state.pending));
    $(".loading-ring").hidden = !state.pending;
    $("#predict svg").hidden = state.pending;
    $("#predict-label").textContent = state.pending
      ? "Checking your photo…"
      : "Predict Animal";
    $("#result").hidden = !state.result;
    $("#another").hidden = !selected || state.pending;
    parentElement
      .querySelectorAll(".gallery-card")
      .forEach((card, i) =>
        card.setAttribute("aria-pressed", String(i === state.active)),
      );
    $("#selected-credit").replaceChildren();
    if (state.example) {
      const item = gallery.find((item) => item.id === state.example);
      const link = document.createElement("a");
      link.href = item.source_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `Photo: ${item.author} · ${item.license_label ?? item.license}`;
      $("#selected-credit").append(link);
    }
    if (state.result) {
      const result = state.result;
      $("#animal-name").textContent = result.animal;
      $("#probability").textContent =
        `${(result.probability * 100).toFixed(1)}%`;
      $("#probabilities").replaceChildren();
      Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([animal, probability]) => {
          const row = document.createElement("div");
          row.className =
            "probability-row" + (animal === result.animal ? " winner" : "");
          const label = document.createElement("span");
          label.textContent = animal;
          const track = document.createElement("div");
          track.className = "bar-track";
          track.setAttribute("aria-hidden", "true");
          const bar = document.createElement("div");
          bar.className = "bar-fill";
          bar.style.width = `${Math.max(0, Math.min(1, probability)) * 100}%`;
          track.append(bar);
          const value = document.createElement("span");
          value.textContent = `${(probability * 100).toFixed(1)}%`;
          row.append(label, track, value);
          $("#probabilities").append(row);
        });
    }
  }

  function clearPrediction() {
    clearTimeout(root.pendingTimer);
    state.result = null;
    state.pending = false;
    state.requestId = null;
    showNotice();
  }
  function releasePhoto() {
    if (state.photo?.objectUrl) URL.revokeObjectURL(state.photo.objectUrl);
    state.photo = null;
  }
  function selectExample(index) {
    cancelPhotoJob();
    releasePhoto();
    clearPrediction();
    state.active = index;
    state.example = gallery[index].id;
    $(".source-button.active")?.classList.remove("active");
    $("#examples-open").classList.add("active");
    render();
    if (
      state.mode === "binary" &&
      !["Cat", "Dog"].includes(gallery[index].animal)
    )
      showNotice(
        "Cat and Dog mode only supports those two animals. Choose All five animals for this example.",
      );
  }
  function cancelPhotoJob() {
    root.photoJob?.abort();
    root.photoJob = null;
    showUploadStatus();
  }

  function showUploadStatus(message = "", busy = false) {
    const status = $("#upload-status");
    status.textContent = message;
    status.hidden = !message;
    $("#upload").setAttribute("aria-busy", String(busy));
    $("#upload-label").textContent = busy ? "Preparing…" : "Upload Photo";
    if (message) status.scrollIntoView({ block: "nearest" });
  }

  async function selectFile(file, fromCamera = false) {
    if (!file) return;
    cancelPhotoJob();
    clearPrediction();
    releasePhoto();
    state.example = null;
    state.active = -1;
    state.galleryOpen = false;
    $(".source-button.active")?.classList.remove("active");
    const job = new AbortController();
    const { signal } = job;
    root.photoJob = job;
    showUploadStatus("Preparing your photo…", true);
    render();

    let sourceUrl;
    let image;
    let detachImage;
    const timeout = setTimeout(() => {
      job.abort(
        new Error(
          "This photo took too long to open. Try a smaller copy or a screenshot.",
        ),
      );
    }, 25000);
    try {
      const prepared = await withPhotoDeadline(signal, async () => {
        if (!file.size)
          throw new Error("That photo is empty. Choose another one.");
        if (file.size > maxPhotoBytes)
          throw new Error("Choose a photo no larger than 8 MB.");
        const format = await photoFormat(file);
        if (signal.aborted) throw signal.reason;
        sourceUrl = URL.createObjectURL(file);
        image = new Image();
        await new Promise((resolve, reject) => {
          detachImage = () => {
            image.removeAttribute("src");
            reject(signal.reason);
          };
          signal.addEventListener("abort", detachImage, { once: true });
          image.onload = resolve;
          image.onerror = () =>
            reject(
              new Error(
                format === "heic"
                  ? heicHint
                  : "We could not open that photo. Try another JPG, PNG or WEBP image.",
              ),
            );
          image.src = sourceUrl;
        });
        const { naturalWidth: width, naturalHeight: height } = image;
        if (!width || !height)
          throw new Error("We could not open that photo. Choose another one.");
        if (width * height > maxPhotoPixels) {
          throw new Error(
            "This photo is over 24 megapixels. Try a smaller copy or a screenshot.",
          );
        }
        if (Math.max(width, height) > 32 * Math.min(width, height)) {
          throw new Error(
            "This photo is too narrow. Crop it around the animal and try again.",
          );
        }
        const blob = await preparePhoto(file, image, format, signal);
        if (signal.aborted) throw signal.reason;
        return { blob, encoded: await readUpload(blob, signal) };
      });
      if (root.photoJob !== job) return;
      const previewUrl = URL.createObjectURL(prepared.blob);
      state.photo = {
        src: previewUrl,
        objectUrl: previewUrl,
        encoded: prepared.encoded,
        name: fromCamera ? "Camera photo" : file.name.slice(0, 100),
      };
      $(fromCamera ? "#camera-open" : "#upload").classList.add("active");
      showUploadStatus("Photo ready. Tap Predict Animal.");
    } catch (error) {
      if (root.photoJob === job) {
        showUploadStatus(
          error.name === "Error"
            ? error.message
            : "We could not prepare that photo. Try a smaller copy or choose another one.",
        );
      }
    } finally {
      clearTimeout(timeout);
      if (detachImage) signal.removeEventListener("abort", detachImage);
      if (image) image.removeAttribute("src");
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
      for (const input of [$("#file"), $("#camera-file")]) {
        if (input.files[0] === file) input.value = "";
      }
      if (root.photoJob === job) {
        root.photoJob = null;
        $("#upload").setAttribute("aria-busy", "false");
        $("#upload-label").textContent = "Upload Photo";
        render();
      }
    }
  }
  function closeCamera() {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    $("#video").srcObject = null;
    $("#capture").disabled = true;
    if ($("#camera-dialog").open) $("#camera-dialog").close();
  }
  async function openCamera() {
    $("#camera-dialog").showModal();
    $("#camera-status").textContent =
      "Allow camera access when your browser asks.";
    $("#capture").disabled = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      $("#camera-status").textContent =
        "Live camera is unavailable here. Use your phone camera below or upload a photo.";
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
        audio: false,
      });
      if (!$("#camera-dialog").open) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      state.stream = stream;
      $("#video").srcObject = stream;
      await $("#video").play();
      $("#capture").disabled = false;
      $("#camera-status").textContent = "Your camera is ready.";
    } catch (error) {
      $("#camera-status").textContent =
        error.name === "NotAllowedError"
          ? "Camera access was not allowed. You can upload a photo or use your phone camera below."
          : "We could not open a camera here. Try your phone camera below or upload a photo.";
    }
  }
  function showDialog(id) {
    $(id).showModal();
  }

  listen($("#start"), "click", () => {
    clearTimeout(root.transitionTimer);
    $("#welcome").classList.add("exit");
    const timer = setTimeout(
      () => {
        state.stage = "classify";
        $("#welcome").classList.remove("exit");
        render();
        root.scrollIntoView({
          block: "start",
          behavior: reduceMotion.matches ? "auto" : "smooth",
        });
        focus($("#test-title"));
      },
      reduceMotion.matches ? 0 : 120,
    );
    root.transitionTimer = timer;
  });
  listen($("#home"), "click", () => {
    clearTimeout(root.transitionTimer);
    $("#welcome").classList.remove("exit");
    state.stage = "welcome";
    closeCamera();
    render();
    root.scrollIntoView({
      block: "start",
      behavior: reduceMotion.matches ? "auto" : "smooth",
    });
    focus($("#start"));
  });
  listen($("#about-open"), "click", () => showDialog("#about-dialog"));
  listen($("#credits-open"), "click", () => showDialog("#credits-dialog"));
  parentElement
    .querySelectorAll("[data-close]")
    .forEach((button) =>
      listen(button, "click", () => $("#" + button.dataset.close).close()),
    );
  listen($("#upload"), "click", () => {
    $("#file").value = "";
    $("#file").click();
  });
  listen($("#file"), "change", (event) => {
    selectFile(event.target.files[0]);
  });
  listen($("#camera-file"), "change", (event) => {
    selectFile(event.target.files[0], true);
  });
  listen($("#camera-open"), "click", openCamera);
  listen($("#camera-close"), "click", closeCamera);
  listen($("#camera-dialog"), "cancel", closeCamera);
  listen($("#camera-dialog"), "close", () => {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
  });
  listen($("#camera-fallback"), "click", () => {
    closeCamera();
    $("#camera-file").value = "";
    $("#camera-file").click();
  });
  listen($("#capture"), "click", () => {
    const video = $("#video");
    if (!video.videoWidth) return;
    const canvas = document.createElement("canvas");
    const ratio = Math.min(
      1,
      1800 / Math.max(video.videoWidth, video.videoHeight),
    );
    canvas.width = Math.round(video.videoWidth * ratio);
    canvas.height = Math.round(video.videoHeight * ratio);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          selectFile(
            new File([blob], "camera-photo.jpg", { type: "image/jpeg" }),
            true,
          );
          closeCamera();
        }
      },
      "image/jpeg",
      0.92,
    );
  });
  function scrollGalleryTo(card) {
    const rail = $("#gallery");
    const frame = rail.getBoundingClientRect();
    const item = card.getBoundingClientRect();
    if (item.left >= frame.left && item.right <= frame.right) return;
    rail.scrollBy({
      left: item.left - frame.left - 4,
      behavior: reduceMotion.matches ? "instant" : "smooth",
    });
  }

  listen($("#examples-open"), "click", () => {
    state.galleryOpen = !state.galleryOpen;
    render();
    if (state.galleryOpen) {
      scrollTo($("#gallery-section"));
      focus($("#gallery").children[Math.max(0, state.active)]);
    }
  });
  parentElement.querySelectorAll(".gallery-card").forEach((button) =>
    listen(button, "click", () => {
      selectExample(Number(button.dataset.index));
      scrollGalleryTo(button);
    }),
  );
  listen($("#gallery"), "keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const cards = Array.from($("#gallery").children);
    const current = cards.indexOf(parentElement.activeElement);
    if (current < 0) return;
    event.preventDefault();
    const next = Math.max(
      0,
      Math.min(
        cards.length - 1,
        current + (event.key === "ArrowRight" ? 1 : -1),
      ),
    );
    focus(cards[next]);
    scrollGalleryTo(cards[next]);
  });
  for (const [id, direction] of [
    ["#gallery-prev", -1],
    ["#gallery-next", 1],
  ])
    listen($(id), "click", () =>
      $("#gallery").scrollBy({
        left: direction * $("#gallery").clientWidth * 0.75,
        behavior: reduceMotion.matches ? "instant" : "smooth",
      }),
    );
  listen($("#mode"), "change", (event) => {
    state.mode = event.target.value;
    clearPrediction();
    render();
    if (state.mode === "binary")
      showNotice(
        "This mode distinguishes cats from dogs. Other animals will still receive one of those two labels.",
      );
  });
  listen($("#another"), "click", () => {
    cancelPhotoJob();
    releasePhoto();
    state.example = null;
    state.active = -1;
    clearPrediction();
    $(".source-button.active")?.classList.remove("active");
    render();
    scrollTo($(".source-tabs"));
    focus($("#upload"));
  });
  listen($("#photo-area"), "dragover", (event) => {
    event.preventDefault();
    $("#photo-area").classList.add("dragover");
  });
  listen($("#photo-area"), "dragleave", () =>
    $("#photo-area").classList.remove("dragover"),
  );
  listen($("#photo-area"), "drop", (event) => {
    event.preventDefault();
    $("#photo-area").classList.remove("dragover");
    selectFile(event.dataTransfer.files[0]);
  });
  listen($("#predict"), "click", () => {
    if (state.pending || root.photoJob || (!state.photo && !state.example))
      return;
    if (
      state.example &&
      state.mode === "binary" &&
      !["Cat", "Dog"].includes(
        gallery.find((item) => item.id === state.example).animal,
      )
    ) {
      showNotice(
        "Choose All five animals for this example, or select a cat or dog.",
      );
      return;
    }
    showUploadStatus();
    state.pending = true;
    state.result = null;
    state.requestId = crypto.randomUUID();
    showNotice();
    render();
    const request = { id: state.requestId, mode: state.mode };
    if (state.example) request.example = state.example;
    else request.photo = state.photo.encoded;
    root.pendingTimer = setTimeout(() => {
      if (state.pending) {
        state.pending = false;
        state.requestId = null;
        showNotice("This is taking longer than expected. Please try again.");
        render();
      }
    }, 60000);
    setTriggerValue("predict", request);
  });
  if (data.result && data.result.id === state.requestId) {
    clearTimeout(root.pendingTimer);
    state.pending = false;
    if (data.result.error) {
      state.result = null;
      showNotice(data.result.error);
    } else {
      state.result = data.result;
      showNotice();
    }
  }
  render();
  return () => {
    events.abort();
    clearTimeout(root.transitionTimer);
    queueMicrotask(() => {
      if (!root.isConnected) {
        cancelPhotoJob();
        clearTimeout(root.pendingTimer);
        state.stream?.getTracks().forEach((track) => track.stop());
        releasePhoto();
      }
    });
  };
}
