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
    $("#predict").disabled = !selected || state.pending;
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
  async function selectFile(file, fromCamera = false) {
    if (!file) return;
    if (file.size > 8 * 1024 * 1024) {
      showNotice("Choose a photo smaller than 8 MB.");
      return;
    }
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (
      !allowed.includes(file.type) &&
      !/\.(jpe?g|png|webp)$/i.test(file.name)
    ) {
      showNotice(
        "Choose a JPG, PNG or WEBP photo. If your photo is HEIC, export it as JPEG first.",
      );
      return;
    }
    let objectUrl;
    try {
      const encoded = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      objectUrl = URL.createObjectURL(file);
      await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = resolve;
        image.onerror = reject;
        image.src = objectUrl;
      });
      releasePhoto();
      clearPrediction();
      state.example = null;
      state.active = -1;
      state.galleryOpen = false;
      state.photo = {
        src: objectUrl,
        objectUrl,
        encoded: encoded.split(",")[1],
        name: fromCamera ? "Camera photo" : file.name.slice(0, 100),
      };
      $(".source-button.active")?.classList.remove("active");
      $(fromCamera ? "#camera-open" : "#upload").classList.add("active");
      render();
    } catch {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      showNotice(
        "We could not open that photo. Please choose a valid JPG, PNG or WEBP image.",
      );
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
  listen($("#upload"), "click", () => $("#file").click());
  listen($("#file"), "change", (event) => {
    selectFile(event.target.files[0]);
    event.target.value = "";
  });
  listen($("#camera-file"), "change", (event) => {
    selectFile(event.target.files[0], true);
    event.target.value = "";
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
    if (state.pending || (!state.photo && !state.example)) return;
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
        clearTimeout(root.pendingTimer);
        state.stream?.getTracks().forEach((track) => track.stop());
        releasePhoto();
      }
    });
  };
}
