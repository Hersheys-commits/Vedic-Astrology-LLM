(() => {
  "use strict";

  const STORAGE_KEY = "astro-rag-ui-v1";
  const STATE_VERSION = 1;
  const DEFAULT_SETTINGS = {
    ayanamsa: "lahiri",
    house_system: "whole_sign",
    node_type: "mean",
    strategy: "hybrid",
    top_k: 5,
    response_model: "gpt-5-mini",
    theme: "system",
  };
  const $ = (selector, root = document) => root.querySelector(selector);
  const elements = {
    shell: $("#app-shell"),
    sidebar: $("#sidebar"),
    chatPanel: $(".chat-panel"),
    transcript: $("#transcript"),
    composer: $("#composer"),
    question: $("#question-input"),
    send: $("#send-button"),
    requestStatus: $("#request-status"),
    dialog: $("#profile-dialog"),
    profileForm: $("#profile-form"),
    dialogTitle: $("#dialog-title"),
    dialogClose: $("#dialog-close"),
    profileCancel: $("#profile-cancel"),
    profileSave: $("#profile-save"),
    profileError: $("#profile-error"),
    profilePlace: $("#profile-place"),
    profileMeta: $("#profile-meta"),
    placeSearch: $("#place-search"),
    searchPlace: $("#search-place"),
    placeResults: $("#place-results"),
    selectedPlace: $("#selected-place"),
    latitude: $("#latitude"),
    longitude: $("#longitude"),
    mapDetails: $(".map-picker"),
    mapFallback: $("#map-fallback"),
    mobileMenu: $("#mobile-menu"),
    mobileScrim: $("#mobile-scrim"),
    clearData: $("#clear-data"),
  };

  let state = readState();
  let map = null;
  let mapMarker = null;
  let locationMode = "geocoded";
  let editing = false;
  let opener = null;
  let sending = false;
  let activeRequest = null;

  function readState() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || saved.version !== STATE_VERSION) return emptyState();
      const settings = { ...DEFAULT_SETTINGS, ...(saved.settings || {}) };
      if (!["gpt-5-mini", "astro-qwen-lora", "astro-qwen-npu"].includes(settings.response_model)) {
        settings.response_model = DEFAULT_SETTINGS.response_model;
      }
      return {
        version: STATE_VERSION,
        profile: validProfile(saved.profile) ? saved.profile : null,
        settings,
        messages: Array.isArray(saved.messages) ? saved.messages : [],
      };
    } catch (_) {
      return emptyState();
    }
  }

  function emptyState() {
    return { version: STATE_VERSION, profile: null, settings: { ...DEFAULT_SETTINGS }, messages: [] };
  }

  function validProfile(profile) {
    return Boolean(
      profile && typeof profile.birth_date === "string" &&
      typeof profile.birth_time === "string" && profile.location &&
      Number.isFinite(Number(profile.location.latitude)) &&
      Number.isFinite(Number(profile.location.longitude))
    );
  }

  function persist() {
    try {
      state.version = STATE_VERSION;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      return true;
    } catch (error) {
      showStatus("Your browser could not save this update. Free up browser storage and try again.", "error");
      return false;
    }
  }

  function applyTheme() {
    const mode = state.settings.theme || "system";
    const dark = mode === "dark" || (mode === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const color = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim();
    $("meta[name='theme-color']")?.setAttribute("content", color || (dark ? "#171717" : "#ffffff"));
    const select = $("#theme-mode");
    if (select) select.value = mode;
  }

  function profileLabel(profile) {
    return profile?.location?.display_name ||
      `${Number(profile?.location?.latitude).toFixed(4)}, ${Number(profile?.location?.longitude).toFixed(4)}`;
  }

  function profileSnapshot(profile) {
    return {
      id: profile.id,
      birth_date: profile.birth_date,
      birth_time: profile.birth_time,
      location: { ...profile.location },
    };
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function inlineMarkdown(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, "$1<em>$2</em>");
  }

  function renderMarkdown(value) {
    const lines = String(value || "").replace(/\r/g, "").split("\n");
    const output = [];
    let list = null;
    let paragraph = [];
    const closeList = () => {
      if (list) output.push(`</${list}>`);
      list = null;
    };
    const closeParagraph = () => {
      if (paragraph.length) output.push(`<p>${paragraph.map(inlineMarkdown).join("<br>")}</p>`);
      paragraph = [];
    };
    for (const line of lines) {
      const heading = line.match(/^\s{0,3}(#{1,4})\s+(.+)$/);
      const bullet = line.match(/^\s*[-*]\s+(.+)$/);
      const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (!line.trim()) { closeParagraph(); closeList(); continue; }
      if (heading) {
        closeParagraph(); closeList();
        const level = Math.min(heading[1].length + 1, 6);
        output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      } else if (bullet || numbered) {
        closeParagraph();
        const wanted = bullet ? "ul" : "ol";
        if (list !== wanted) { closeList(); output.push(`<${wanted}>`); list = wanted; }
        output.push(`<li>${inlineMarkdown((bullet || numbered)[1])}</li>`);
      } else {
        closeList(); paragraph.push(line);
      }
    }
    closeParagraph(); closeList();
    return output.join("");
  }

  function formatDate(date) {
    if (!date) return "";
    const parsed = new Date(`${date}T12:00:00`);
    return Number.isNaN(parsed.getTime()) ? date : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(parsed);
  }

  function formatTime(value) {
    if (!value) return "";
    const [hourText, minute] = value.split(":");
    const hour = Number(hourText);
    if (!Number.isFinite(hour)) return value;
    const period = hour >= 12 ? "PM" : "AM";
    return `${hour % 12 || 12}:${minute || "00"} ${period}`;
  }

  function renderProfile() {
    const profile = state.profile;
    if (!profile) return;
    elements.profilePlace.textContent = profileLabel(profile);
    elements.profileMeta.textContent = `${formatDate(profile.birth_date)} · ${formatTime(profile.birth_time)}`;
  }

  function renderSources(response, parent) {
    if (!response || (!response.chart_facts_used?.length && !response.retrieved_passages?.length)) return;
    const details = document.createElement("details");
    details.className = "answer-sources";
    const summary = document.createElement("summary");
    summary.textContent = "Chart facts & source passages";
    details.append(summary);

    if (response.chart_facts_used?.length) {
      const factsHeading = document.createElement("h4");
      factsHeading.textContent = "Chart facts used";
      details.append(factsHeading);
      const factList = document.createElement("ul");
      factList.className = "fact-list";
      response.chart_facts_used.forEach((fact) => {
        const item = document.createElement("li");
        const fields = Object.entries(fact.detail || {}).filter(([key]) => key !== "type")
          .map(([key, val]) => `${key.replaceAll("_", " ")}: ${val}`).join(" · ");
        item.textContent = `${fact.type.replaceAll("_", " ")}${fields ? ` — ${fields}` : ""}`;
        factList.append(item);
      });
      details.append(factList);
    }

    if (response.retrieved_passages?.length) {
      const passagesHeading = document.createElement("h4");
      passagesHeading.textContent = "Retrieved passages";
      details.append(passagesHeading);
      response.retrieved_passages.forEach((passage, index) => {
        const item = document.createElement("details");
        item.className = "source-passage";
        const title = document.createElement("summary");
        const metadata = passage.metadata?.source || {};
        const book = metadata.book || metadata.book_id || "Source text";
        const chapter = metadata.chapter ? ` · Chapter ${metadata.chapter}` : "";
        title.textContent = `Passage ${index + 1} · ${book}${chapter}`;
        const text = document.createElement("p");
        text.textContent = passage.text || "";
        item.append(title, text);
        details.append(item);
      });
    }
    parent.append(details);
  }

  function renderMessage(message) {
    const row = document.createElement("article");
    row.className = `message-row message-${message.role}`;
    if (message.id) row.dataset.messageId = message.id;
    const content = document.createElement("div");
    content.className = "message-content";

    if (message.role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "assistant-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = "✳";
      row.append(avatar);
    }

    const body = document.createElement("div");
    body.className = "message-body";
    if (message.role === "assistant") {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      const profileText = message.profile ? `Based on ${profileLabel(message.profile)}` : "Astro RAG";
      meta.textContent = message.response?.model ? `${profileText} · ${message.response.model}` : profileText;
      body.append(meta);
    }
    const text = document.createElement("div");
    text.className = "message-text";
    if (message.status === "error") text.classList.add("message-error");
    text.innerHTML = renderMarkdown(message.content || "");
    body.append(text);

    if (message.role === "assistant" && message.response) renderSources(message.response, body);
    if (message.status === "error" && message.retry_question) {
      const retry = document.createElement("button");
      retry.className = "retry-button";
      retry.type = "button";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => sendQuestion(
        message.retry_question, message.profile, message.response_model || state.settings.response_model,
      ));
      body.append(retry);
    }
    content.append(body);
    row.append(content);
    return row;
  }

  function renderTranscript() {
    elements.transcript.replaceChildren();
    if (!state.messages.length && !activeRequest) {
      const welcome = document.createElement("section");
      welcome.className = "welcome-state";
      welcome.innerHTML = `<div class="welcome-mark" aria-hidden="true">✳</div>
        <div class="eyebrow">YOUR CHART, IN CONTEXT</div>
        <h1>What would you like to understand?</h1>
        <p>Ask about a placement, a life area, or a traditional astrological concept. Responses are grounded in the chart and source passages.</p>
        <div class="suggestion-list" aria-label="Suggested questions">
          <button type="button" class="suggestion">What does my chart suggest about my career?</button>
          <button type="button" class="suggestion">Which relationship themes appear in my chart?</button>
          <button type="button" class="suggestion">What is my current Mahadasha?</button>
        </div>`;
      welcome.querySelectorAll(".suggestion").forEach((button) => {
        button.addEventListener("click", () => {
          elements.question.value = button.textContent;
          elements.question.focus();
          resizeComposer();
        });
      });
      elements.transcript.append(welcome);
      return;
    }
    state.messages.forEach((message) => elements.transcript.append(renderMessage(message)));
    if (activeRequest) elements.transcript.append(renderProgressMessage());
    requestAnimationFrame(() => { elements.transcript.scrollTop = elements.transcript.scrollHeight; });
  }

  function renderProgressMessage() {
    const row = document.createElement("article");
    row.className = "message-row message-assistant";
    row.setAttribute("role", "status");
    row.setAttribute("aria-live", "polite");

    const avatar = document.createElement("div");
    avatar.className = "assistant-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "✳";

    const body = document.createElement("div");
    body.className = "message-body";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = activeRequest.responseModel === "astro-qwen-npu"
      ? "Astro Qwen · Intel NPU"
      : activeRequest.responseModel === "astro-qwen-lora"
        ? "Astro Qwen · Local"
        : "Preparing your answer";

    const text = document.createElement("div");
    text.className = "message-text progress-message";
    const spinner = document.createElement("span");
    spinner.className = "progress-spinner";
    spinner.setAttribute("aria-hidden", "true");
    const status = document.createElement("span");
    status.textContent = activeRequest.statusText;
    text.append(spinner, status);
    body.append(meta, text);
    row.append(avatar, body);
    return row;
  }

  function updateProgressMessage(statusText) {
    if (!activeRequest) return;
    activeRequest.statusText = statusText;
    const status = elements.transcript.querySelector(".progress-message span:last-child");
    if (status) status.textContent = statusText;
    requestAnimationFrame(() => { elements.transcript.scrollTop = elements.transcript.scrollHeight; });
  }

  function showStatus(message, kind = "") {
    elements.requestStatus.textContent = message;
    elements.requestStatus.className = `request-status${kind ? ` status-${kind}` : ""}`;
  }

  function updateSending(loading) {
    sending = loading;
    elements.send.disabled = loading;
    elements.question.disabled = loading;
    elements.send.setAttribute("aria-label", loading ? "Sending question" : "Send question");
  }

  function newId() {
    return globalThis.crypto?.randomUUID?.() || `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function sendQuestion(question, profile = state.profile, responseModel = state.settings.response_model) {
    const cleanQuestion = String(question || "").trim();
    if (!cleanQuestion || !profile || sending) return;
    const usedProfile = profileSnapshot(profile);
    const userMessage = { id: newId(), role: "user", content: cleanQuestion, profile: usedProfile, created_at: new Date().toISOString() };
    state.messages.push(userMessage);
    persist();
    renderTranscript();
    elements.question.value = "";
    resizeComposer();
    updateSending(true);
    activeRequest = {
      responseModel,
      statusText: responseModel === "astro-qwen-npu"
        ? "Preparing your chart and source passages before NPU generation…"
        : responseModel === "astro-qwen-lora"
          ? "Preparing your chart and source passages before local generation…"
          : "Preparing your chart and source passages…",
      startedAt: Date.now(),
      progressTimer: null,
    };
    showStatus("Your question is being processed. You can follow progress above.", "loading");
    renderTranscript();
    activeRequest.progressTimer = setInterval(() => {
      if (!activeRequest) return;
      const elapsedSeconds = Math.floor((Date.now() - activeRequest.startedAt) / 1000);
      const elapsed = elapsedSeconds < 60
        ? `${elapsedSeconds}s`
        : `${Math.floor(elapsedSeconds / 60)}m ${elapsedSeconds % 60}s`;
      updateProgressMessage(
        `Still working… elapsed ${elapsed}. Local model generation can take a few minutes.`
      );
      showStatus(`Still processing your question · ${elapsed} elapsed.`, "loading");
    }, 10000);

    const payload = {
      birth_date: usedProfile.birth_date,
      birth_time: usedProfile.birth_time,
      location: usedProfile.location,
      settings: {
        ayanamsa: state.settings.ayanamsa,
        house_system: state.settings.house_system,
        node_type: state.settings.node_type,
      },
      question: cleanQuestion,
      strategy: state.settings.strategy,
      top_k: Number(state.settings.top_k),
      response_model: responseModel,
    };
    let streamingMessageId = null;
    try {
      let data = null;
      const streamResponse = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify(payload),
      });
      const contentType = streamResponse.headers.get("content-type") || "";
      if (streamResponse.ok && contentType.includes("text/event-stream") && streamResponse.body) {
        const streamingMsg = {
          id: newId(),
          role: "assistant",
          content: "",
          profile: usedProfile,
          response: null,
          created_at: new Date().toISOString(),
        };
        streamingMessageId = streamingMsg.id;
        let pushedStreamingMsg = false;
        let receivedDone = false;
        const ensureStreamingRow = () => {
          if (!pushedStreamingMsg) {
            pushedStreamingMsg = true;
            state.messages.push(streamingMsg);
            renderTranscript();
          }
          return elements.transcript.querySelector(`[data-message-id="${streamingMsg.id}"]`);
        };
        const handleFrame = async (frame) => {
          let eventType = "message";
          const dataLines = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) eventType = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
          }
          if (!dataLines.length) return;
          const parsed = JSON.parse(dataLines.join("\n"));
          if (eventType === "progress") {
            const message = parsed.message || "Preparing your answer…";
            updateProgressMessage(message);
            showStatus(message, "loading");
          } else if (eventType === "metadata") {
            streamingMsg.response = {
              chart_facts_used: parsed.chart_facts_used || [],
              retrieved_passages: parsed.retrieved_passages || [],
              model: parsed.model,
              generation_mode: parsed.generation_mode,
            };
            ensureStreamingRow();
            updateProgressMessage("Generating answer (streaming live)…");
          } else if (eventType === "delta") {
            streamingMsg.content += parsed.delta || "";
            updateProgressMessage("Answer streaming live…");
            const row = ensureStreamingRow();
            const textEl = row?.querySelector(".message-text");
            if (textEl) textEl.innerHTML = renderMarkdown(streamingMsg.content);
            showStatus("Streaming answer live…", "loading");
            requestAnimationFrame(() => { elements.transcript.scrollTop = elements.transcript.scrollHeight; });
            await new Promise((resolve) => {
              requestAnimationFrame(() => setTimeout(resolve, 0));
            });
          } else if (eventType === "replace") {
            streamingMsg.content = parsed.answer || "";
            const row = ensureStreamingRow();
            const textEl = row?.querySelector(".message-text");
            if (textEl) textEl.innerHTML = renderMarkdown(streamingMsg.content);
          } else if (eventType === "done") {
            data = parsed;
            receivedDone = true;
            streamingMsg.content = parsed.answer || streamingMsg.content || "The service returned an empty answer.";
            streamingMsg.response = {
              chart_facts_used: parsed.chart_facts_used || [],
              retrieved_passages: parsed.retrieved_passages || [],
              model: parsed.model,
              generation_mode: parsed.generation_mode,
            };
            if (!pushedStreamingMsg) {
              pushedStreamingMsg = true;
              state.messages.push(streamingMsg);
            }
          } else if (eventType === "error") {
            if (pushedStreamingMsg) {
              state.messages = state.messages.filter((message) => message.id !== streamingMsg.id);
            }
            throw new Error(parsed?.error?.message || parsed?.detail || "Streaming generation failed.");
          }
        };
        const reader = streamResponse.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split(/\r?\n\r?\n/);
          buffer = frames.pop() || "";
          for (const frame of frames) await handleFrame(frame);
        }
        buffer += decoder.decode();
        if (buffer.trim()) await handleFrame(buffer);
        if (!receivedDone) throw new Error("The answer stream ended before its final response.");
      } else {
        if (!streamResponse.ok && ![404, 405].includes(streamResponse.status)) {
          const errorData = await streamResponse.json().catch(() => ({}));
          throw new Error(errorData?.error?.message || errorData?.detail || `Request failed (${streamResponse.status}).`);
        }
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data?.error?.message || data?.detail || `Request failed (${response.status}).`);
        }
        state.messages.push({
          id: newId(), role: "assistant", content: data.answer || "The service returned an empty answer.",
          profile: usedProfile, response: {
            chart_facts_used: data.chart_facts_used || [],
            retrieved_passages: data.retrieved_passages || [],
            model: data.model,
            generation_mode: data.generation_mode,
          }, created_at: new Date().toISOString(),
        });
      }
      clearActiveRequest();
      showStatus(data?.generation_status === "incomplete" ? "This answer may be incomplete." : "");
    } catch (error) {
      clearActiveRequest();
      if (streamingMessageId) {
        state.messages = state.messages.filter((message) => message.id !== streamingMessageId);
      }
      state.messages.push({
        id: newId(), role: "assistant", status: "error",
        content: `I couldn’t complete that request. ${error.message || "Check that Astro RAG and Chart Engine are running, then try again."}`,
        retry_question: cleanQuestion, response_model: responseModel,
        profile: usedProfile, created_at: new Date().toISOString(),
      });
      showStatus("Your question is still in the conversation. You can retry it when the service is available.", "error");
    } finally {
      clearActiveRequest();
      updateSending(false);
      persist();
      renderTranscript();
      elements.question.focus();
    }
  }

  function clearActiveRequest() {
    if (!activeRequest) return;
    clearInterval(activeRequest.progressTimer);
    activeRequest = null;
  }

  function resizeComposer() {
    elements.question.style.height = "auto";
    elements.question.style.height = `${Math.min(elements.question.scrollHeight, 180)}px`;
  }

  function openProfileDialog(isEditing) {
    editing = isEditing;
    opener = isEditing ? document.activeElement : null;
    elements.profileError.textContent = "";
    elements.dialogTitle.textContent = isEditing ? "Update birth details" : "Start with your birth details";
    elements.profileSave.textContent = isEditing ? "Save changes" : "Save birth details";
    elements.dialogClose.hidden = !isEditing;
    elements.profileCancel.hidden = !isEditing;
    elements.profileForm.reset();
    fillProfileForm(isEditing ? state.profile : null);
    if (!elements.dialog.open) elements.dialog.showModal();
    if (elements.mapDetails.open) setTimeout(() => initMap(), 80);
    (isEditing ? elements.placeSearch : $("#birth-date")).focus({ preventScroll: true });
  }

  function closeProfileDialog() {
    if (!editing) return;
    elements.dialog.close();
    opener?.focus?.();
    editing = false;
  }

  function fillProfileForm(profile) {
    if (!profile) {
      locationMode = "geocoded";
      elements.selectedPlace.textContent = "No location selected yet";
      elements.latitude.value = "";
      elements.longitude.value = "";
      elements.placeSearch.value = "";
      updateMapMarker(null);
      return;
    }
    $("#birth-date").value = profile.birth_date;
    $("#birth-time").value = profile.birth_time;
    elements.latitude.value = profile.location.latitude;
    elements.longitude.value = profile.location.longitude;
    elements.placeSearch.value = profile.location.display_name || "";
    elements.selectedPlace.textContent = profile.location.display_name || `${profile.location.latitude}, ${profile.location.longitude}`;
    locationMode = profile.location.mode || "geocoded";
    updateMapMarker([Number(profile.location.latitude), Number(profile.location.longitude)]);
  }

  function coordinatesChanged(mode = "map_click") {
    const lat = Number(elements.latitude.value);
    const lon = Number(elements.longitude.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
    locationMode = mode;
    const label = `Coordinates ${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    elements.selectedPlace.textContent = label;
    if (map) updateMapMarker([lat, lon]);
    if (mode !== "geocoded") elements.placeSearch.value = "";
  }

  function updateMapMarker(coords) {
    if (!map || !coords) return;
    if (!mapMarker) {
      mapMarker = L.marker(coords, { draggable: true }).addTo(map);
      mapMarker.on("dragend", () => {
        const point = mapMarker.getLatLng();
        elements.latitude.value = point.lat.toFixed(6);
        elements.longitude.value = point.lng.toFixed(6);
        coordinatesChanged("map_click");
      });
    } else mapMarker.setLatLng(coords);
    map.setView(coords, Math.max(map.getZoom(), 9));
  }

  function initMap() {
    if (map) { setTimeout(() => map.invalidateSize(), 80); return; }
    if (!window.L) {
      elements.mapFallback.hidden = false;
      return;
    }
    map = L.map("birth-map", { scrollWheelZoom: false }).setView([20, 0], 2);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
    }).addTo(map);
    map.on("click", (event) => {
      const { lat, lng } = event.latlng;
      elements.latitude.value = lat.toFixed(6);
      elements.longitude.value = lng.toFixed(6);
      updateMapMarker([lat, lng]);
      coordinatesChanged("map_click");
    });
    const lat = Number(elements.latitude.value), lon = Number(elements.longitude.value);
    if (Number.isFinite(lat) && Number.isFinite(lon) && elements.latitude.value && elements.longitude.value) {
      updateMapMarker([lat, lon]);
    }
    setTimeout(() => map.invalidateSize(), 100);
  }

  async function searchLocation() {
    const query = elements.placeSearch.value.trim();
    if (query.length < 2) {
      elements.placeResults.hidden = false;
      elements.placeResults.textContent = "Enter at least two characters to search.";
      elements.placeSearch.focus();
      return;
    }
    elements.searchPlace.disabled = true;
    elements.searchPlace.textContent = "Searching…";
    elements.placeResults.hidden = false;
    elements.placeResults.textContent = "Searching for places…";
    try {
      const response = await fetch("/location/geocode", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 5 }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data?.error?.message || data?.detail || "Location search failed.");
      const results = data.results || [];
      elements.placeResults.replaceChildren();
      if (!results.length) {
        elements.placeResults.textContent = "No matching places found. Try a nearby city or a more specific address.";
      } else {
        results.forEach((result, index) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "place-result";
          button.setAttribute("role", "option");
          button.textContent = result.display_name;
          button.addEventListener("click", () => selectLocation(result));
          elements.placeResults.append(button);
        });
      }
    } catch (error) {
      elements.placeResults.replaceChildren();
      elements.placeResults.textContent = `${error.message} You can also select a point on the map or enter coordinates.`;
    } finally {
      elements.searchPlace.disabled = false;
      elements.searchPlace.textContent = "Search";
    }
  }

  function selectLocation(result) {
    elements.latitude.value = Number(result.latitude).toFixed(6);
    elements.longitude.value = Number(result.longitude).toFixed(6);
    elements.selectedPlace.textContent = result.display_name;
    elements.placeSearch.value = result.display_name;
    elements.placeResults.hidden = true;
    locationMode = "geocoded";
    if (map) updateMapMarker([Number(result.latitude), Number(result.longitude)]);
    else if (elements.mapDetails.open) initMap();
  }

  function saveProfile(event) {
    event.preventDefault();
    elements.profileError.textContent = "";
    const form = elements.profileForm;
    if (!form.reportValidity()) return;
    const birthDate = $("#birth-date").value;
    const birthTime = $("#birth-time").value;
    const date = new Date(`${birthDate}T00:00:00`);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    if (date > today) { elements.profileError.textContent = "Date of birth cannot be in the future."; return; }
    const latitude = Number(elements.latitude.value);
    const longitude = Number(elements.longitude.value);
    if (locationMode === "unresolved") {
      elements.profileError.textContent = "Choose a search result, or select a point on the map before saving.";
      return;
    }
    if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
      elements.profileError.textContent = "Choose a valid birthplace from search, the map, or coordinates.";
      return;
    }
    const same = state.profile && state.profile.birth_date === birthDate && state.profile.birth_time === birthTime &&
      Number(state.profile.location.latitude) === latitude && Number(state.profile.location.longitude) === longitude;
    const profile = {
      id: same ? state.profile.id : newId(),
      birth_date: birthDate,
      birth_time: birthTime,
      location: {
        mode: locationMode,
        latitude,
        longitude,
        display_name: locationMode === "geocoded" ? (elements.placeSearch.value.trim() || elements.selectedPlace.textContent) : elements.selectedPlace.textContent,
      },
    };
    state.profile = profile;
    persist();
    renderProfile();
    elements.shell.hidden = false;
    elements.dialog.close();
    editing = false;
    renderTranscript();
    $("#question-input").focus();
  }

  function settingsChanged() {
    state.settings = {
      ...state.settings,
      ayanamsa: $("#ayanamsa").value,
      house_system: $("#house-system").value,
      node_type: $("#node-type").value,
      strategy: $("#retrieval-strategy").value,
      top_k: Number($("#top-k").value),
      response_model: $("#response-model").value,
      theme: $("#theme-mode").value,
    };
    persist();
    applyTheme();
  }

  function applySettingsToControls() {
    $("#ayanamsa").value = state.settings.ayanamsa;
    $("#house-system").value = state.settings.house_system;
    $("#node-type").value = state.settings.node_type;
    $("#retrieval-strategy").value = state.settings.strategy;
    $("#top-k").value = String(state.settings.top_k);
    $("#response-model").value = state.settings.response_model;
    $("#theme-mode").value = state.settings.theme;
  }

  function openSidebar() {
    elements.sidebar.classList.add("sidebar-open");
    elements.chatPanel.inert = true;
    elements.mobileScrim.hidden = false;
    elements.mobileMenu.setAttribute("aria-expanded", "true");
    elements.mobileMenu.setAttribute("aria-label", "Close sidebar");
    $("#edit-profile").focus();
  }

  function closeSidebar() {
    elements.sidebar.classList.remove("sidebar-open");
    elements.chatPanel.inert = false;
    elements.mobileScrim.hidden = true;
    elements.mobileMenu.setAttribute("aria-expanded", "false");
    elements.mobileMenu.setAttribute("aria-label", "Open sidebar");
    elements.mobileMenu.focus();
  }

  function initialize() {
    applyTheme();
    applySettingsToControls();
    elements.shell.hidden = !state.profile;
    if (state.profile) {
      renderProfile();
      renderTranscript();
      elements.dialogClose.hidden = false;
      elements.profileCancel.hidden = false;
    } else {
      elements.dialogClose.hidden = true;
      elements.profileCancel.hidden = true;
      openProfileDialog(false);
    }

    elements.profileForm.addEventListener("submit", saveProfile);
    elements.dialogClose.addEventListener("click", closeProfileDialog);
    elements.profileCancel.addEventListener("click", closeProfileDialog);
    elements.dialog.addEventListener("cancel", (event) => {
      if (!editing) event.preventDefault();
    });
    elements.dialog.addEventListener("click", (event) => {
      if (event.target === elements.dialog && editing) closeProfileDialog();
    });
    elements.dialog.addEventListener("close", () => {
      if (!state.profile) setTimeout(() => elements.dialog.showModal(), 0);
    });
    elements.mapDetails.addEventListener("toggle", () => { if (elements.mapDetails.open) initMap(); });
    elements.searchPlace.addEventListener("click", searchLocation);
    elements.placeSearch.addEventListener("input", () => {
      if (elements.placeSearch.value.trim() !== elements.selectedPlace.textContent.trim()) locationMode = "unresolved";
    });
    elements.placeSearch.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); searchLocation(); }
    });
    elements.latitude.addEventListener("change", () => coordinatesChanged("map_click"));
    elements.longitude.addEventListener("change", () => coordinatesChanged("map_click"));
    $("#edit-profile").addEventListener("click", () => openProfileDialog(true));
    elements.composer.addEventListener("submit", (event) => {
      event.preventDefault(); sendQuestion(elements.question.value);
    });
    elements.question.addEventListener("input", resizeComposer);
    elements.question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.composer.requestSubmit(); }
    });
    ["#ayanamsa", "#house-system", "#node-type", "#retrieval-strategy", "#top-k", "#response-model", "#theme-mode"]
      .forEach((selector) => $(selector).addEventListener("change", settingsChanged));
    elements.clearData.addEventListener("click", () => {
      if (!window.confirm("Clear your saved birth profile, settings, and chat from this browser?")) return;
      if (elements.sidebar.classList.contains("sidebar-open")) closeSidebar();
      localStorage.removeItem(STORAGE_KEY);
      state = emptyState();
      applySettingsToControls();
      elements.shell.hidden = true;
      elements.dialogTitle.textContent = "Start with your birth details";
      showStatus("");
      openProfileDialog(false);
    });
    elements.mobileMenu.addEventListener("click", () => elements.sidebar.classList.contains("sidebar-open") ? closeSidebar() : openSidebar());
    elements.mobileScrim.addEventListener("click", closeSidebar);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && elements.sidebar.classList.contains("sidebar-open")) closeSidebar();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
      if (state.settings.theme === "system") applyTheme();
    });
  }

  document.addEventListener("DOMContentLoaded", initialize);
})();
