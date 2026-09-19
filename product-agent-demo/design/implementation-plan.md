# TrustLens live product-analysis workbench

Selected visual: `reference.png` (1487 × 1058). Current scope: desktop UI derived from the combined concept; functional source annotations, three selectable Agents, inspector, pan/zoom, source upload, batch execution, report export, and live API/retrieval configuration. The bottom mock playback strip and all fake product/evidence content were removed. Uploaded images run as isolated items; each item shares recognition/retrieval context with three parallel qwen3.8-max Agent requests.

1. Match reference layout and assets, including its two photo boards, lower collaboration cluster, and full-height agent inspector. Do not reintroduce the bottom mock playback strip.
2. Implement semantic interactive components. Persist nonsecret agent configuration; keep API keys in page memory. Live mode sends only uploaded images and actual returned evidence.
3. Support multiple uploaded posts in one batch. Every image gets an independent recognition/search/three-Agent run; reports, errors, source lists, and exports remain keyed to the image.
4. Implement drawer transition, annotation focus, evidence flow and playback with reduced-motion support.
5. Run meaningful state/integration tests; browser-test core controls and multi-image upload. Capture at desktop width and verify the batch button, configuration validation, narrow viewport, and reduced motion.
6. Deliver preview, screenshot and honest design QA record. A static reference cannot specify exact motion timing; use explicit 240 ms drawer / 180 ms focus / event-driven evidence motion.

The existing FastAPI project is the chosen implementation target; no new framework or starter is needed.
