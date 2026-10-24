# Astro RAG Chat UI

This document describes the current browser interface implemented in
`app/static/index.html`, `app/static/styles.css`, and `app/static/app.js`. It
replaces the earlier Ollama marketing-site design reference, which described
pages and components that are not part of Astro RAG.

## Product surface

Astro RAG is a responsive, single-page chart-question chat interface served by
FastAPI. The interface is focused on three tasks:

1. Collect a birth date, local birth time, and a resolved birthplace.
2. Let the user ask questions using a selected answer provider and retrieval
   settings.
3. Present the answer with expandable chart facts and source passages so the
   generated interpretation can be inspected.

The UI presents astrology as a traditional interpretive framework, not as
certainty or a replacement for professional advice.

## Layout and interaction

### First use and birth profile

Without a saved profile, the user starts in a dialog to enter date and time and
select a birthplace. Location can be selected through place search, the
optional Leaflet map, or valid latitude/longitude coordinates. The dialog
validates required fields and location bounds before saving.

After setup, the chat shell displays the active birth profile in a sidebar.
The profile can be edited; the profile snapshot associated with an existing
message is retained when the active profile changes.

### Chat

The main panel contains a sticky heading, scrollable transcript, and question
composer. Enter sends a question and Shift+Enter inserts a newline. Loading,
error, and incomplete-answer states are communicated through the live status
region. The transcript renders assistant answers and lets users inspect
returned chart facts and retrieved evidence.

### Advanced settings

The collapsible sidebar section exposes:

- Ayanamsa, house system, and lunar node type.
- Retrieval strategy and the number of source passages.
- Answer model: GPT-5-mini, local Astro Qwen, or Astro Qwen on Intel NPU.
- System, light, or dark appearance.
- A control to clear saved browser profile and conversation data.

The NPU provider requires its prepared OpenVINO model and runtime. Local Qwen
and NPU requests default to deterministic rules-based question-to-chart-fact
analysis for repeatability; GPT-5-mini uses the configured intent mode. API
callers may explicitly set `intent_mode` to override the local-provider default.

Questions are submitted through `POST /ask/stream`. The transcript displays
visible answer deltas as they arrive and uses the final event for the validated
response and its chart-fact/source details. If the server does not provide SSE,
the browser falls back to the JSON `POST /ask` endpoint.
The server applies its chart-claim repair to the completed answer before
returning `/ask` or the stream's final `done` event. Earlier streaming deltas
may contain text that is corrected in that final event.

### Responsive behavior

On narrow screens the sidebar becomes a slide-out panel controlled by a menu
button and scrim. The profile dialog, form fields, composer, and transcript
adapt to small viewport widths. Reduced-motion preferences are respected.

## Visual system

The visual language is quiet and reading-focused: a neutral canvas, thin
dividers, rounded controls, restrained status colors, and limited ornament.
Light and dark palettes are defined with CSS custom properties in
`app/static/styles.css`; system appearance is the default, with explicit light
and dark overrides available in settings. Typography uses system sans-serif
for body copy and a system rounded face for display headings.

The interface uses responsive cards for suggested questions, a constrained
transcript width, visible keyboard focus rings, semantic form labels, and
ARIA-live regions for status updates and transcript additions.

## Browser persistence and privacy

The versioned `astro-rag-ui-v1` local-storage record contains the active
profile, settings, theme, and transcript. It is browser/device-local and has a
clear-data control; it is not account storage or cross-device synchronization.
Requests are sent to Astro RAG when the user submits a question. Place search
uses the Astro RAG geocoding route and may be forwarded to the configured
Chart Engine/geocoding provider. Do not treat browser-local storage as
encrypted or as a substitute for device security.

## Source of truth

For behavior and copy, use the static HTML and JavaScript. For colors, spacing,
responsive breakpoints, and reduced-motion styling, use the CSS. Update this
document when those implemented surfaces change; do not assume a marketing
page or pricing UI exists.
