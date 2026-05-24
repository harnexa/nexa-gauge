# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.16](https://github.com/harnexa/nexa-gauge/compare/v0.1.15...v0.1.16) (2026-05-24)


### Features

* GEval scoring modes (likert/binary) and optional reasoning ([#57](https://github.com/harnexa/nexa-gauge/issues/57)) ([5435308](https://github.com/harnexa/nexa-gauge/commit/5435308d1655e2c67477e9016d809d7f8fd566d6))

## [0.1.15](https://github.com/harnexa/nexa-gauge/compare/v0.1.14...v0.1.15) (2026-05-24)


### Refactoring

* promote input/output to canonical field names ([#55](https://github.com/harnexa/nexa-gauge/issues/55)) ([dd937cd](https://github.com/harnexa/nexa-gauge/commit/dd937cd37d8e33582733535bb869baf9493e2609))

## [0.1.14](https://github.com/harnexa/nexa-gauge/compare/v0.1.13...v0.1.14) (2026-05-23)


### Bug Fixes

* self-hosted model cache disambiguation, litellm 1.85, relevance prompt ([#53](https://github.com/harnexa/nexa-gauge/issues/53)) ([2e550cc](https://github.com/harnexa/nexa-gauge/commit/2e550cc1863dd76bd2df5b7f84deb5d1de417839))

## [0.1.13](https://github.com/harnexa/nexa-gauge/compare/v0.1.12...v0.1.13) (2026-05-17)


### Features

* user-defined record transforms for arbitrary dataset schemas ([#48](https://github.com/harnexa/nexa-gauge/issues/48)) ([300142f](https://github.com/harnexa/nexa-gauge/commit/300142fc6f07820b8cb20f92eb9e8e435145e02a))

## [0.1.12](https://github.com/harnexa/nexa-gauge/compare/v0.1.11...v0.1.12) (2026-05-16)


### Bug Fixes

* eliminate cache write races and duplicate work under high concurrency ([#46](https://github.com/harnexa/nexa-gauge/issues/46)) ([762937e](https://github.com/harnexa/nexa-gauge/commit/762937ecdcc96ad4fa8e34e801ca75c02380d4c0))

## [0.1.11](https://github.com/harnexa/nexa-gauge/compare/v0.1.10...v0.1.11) (2026-05-13)


### Features

* route LLM calls through OpenAI-compatible self-hosted endpoints ([#44](https://github.com/harnexa/nexa-gauge/issues/44)) ([4feb5cf](https://github.com/harnexa/nexa-gauge/commit/4feb5cfd0d4cb0a7bc42bea5e250fd0fdd07358c))

## [0.1.10](https://github.com/harnexa/nexa-gauge/compare/v0.1.9...v0.1.10) (2026-05-08)


### Features

* Hugging Face dataset support and --field column mapping ([#40](https://github.com/harnexa/nexa-gauge/issues/40)) ([3fbc960](https://github.com/harnexa/nexa-gauge/commit/3fbc960b9ce5af6bc8c98fc6d0ba9035b9a10c7e))

## [0.1.9](https://github.com/harnexa/nexa-gauge/compare/v0.1.8...v0.1.9) (2026-04-30)


### Bump

* lite llm version ([#38](https://github.com/harnexa/nexa-gauge/issues/38)) ([e071987](https://github.com/harnexa/nexa-gauge/commit/e0719877ded48fdd1752639cd09b877ada91954a))
* version update ([46a2f71](https://github.com/harnexa/nexa-gauge/commit/46a2f71228928fec3a1857a8f9aca2a0d3a773f4))

## [0.1.8](https://github.com/harnexa/nexa-gauge/compare/v0.1.7...v0.1.8) (2026-04-30)


### Refactoring

* remove legacy --model CLI alias and provider-qualify pricing ([#36](https://github.com/harnexa/nexa-gauge/issues/36)) ([a2c92fa](https://github.com/harnexa/nexa-gauge/commit/a2c92fa5cc15929540b4d77afaf141d32ad25986))

## [0.1.7](https://github.com/harnexa/nexa-gauge/compare/v0.1.6...v0.1.7) (2026-04-30)


### Features

* refiner node, topology-driven graph, and CLI strategy selection ([#34](https://github.com/harnexa/nexa-gauge/issues/34)) ([ceda9eb](https://github.com/harnexa/nexa-gauge/commit/ceda9ebe9e546e1768ec2e799a63e6de4955e2ff))

## [0.1.6](https://github.com/harnexa/nexa-gauge/compare/v0.1.5...v0.1.6) (2026-04-29)


### Dependencies

* **deps:** bump amannn/action-semantic-pull-request from 5 to 6 ([#32](https://github.com/harnexa/nexa-gauge/issues/32)) ([e5330d3](https://github.com/harnexa/nexa-gauge/commit/e5330d38bd857f4ae3957a4dd0da06a88f68424f))

## [0.1.5](https://github.com/harnexa/nexa-gauge/compare/v0.1.4...v0.1.5) (2026-04-29)


### Bug Fixes

* depandabot cli and bump keywords ([#30](https://github.com/harnexa/nexa-gauge/issues/30)) ([b0de278](https://github.com/harnexa/nexa-gauge/commit/b0de27804cc5dbcdceb72378958541f19fcb40d4))

## [0.1.4](https://github.com/harnexa/nexa-gauge/compare/v0.1.3...v0.1.4) (2026-04-27)


### Features

* **eval summary:** Update summary for evaluation ([#23](https://github.com/harnexa/nexa-gauge/issues/23)) ([1dc5abd](https://github.com/harnexa/nexa-gauge/commit/1dc5abdc4b3e428f19714a93544cf62ef25da94a))

## [0.1.3](https://github.com/harnexa/nexa-gauge/compare/v0.1.2...v0.1.3) (2026-04-23)


### Bug Fixes

* **release:** trigger release ([e98f88f](https://github.com/harnexa/nexa-gauge/commit/e98f88fc52fa45491d27eb385415659f60a3c2b1))
* **release:** trigger release ([8a08c4d](https://github.com/harnexa/nexa-gauge/commit/8a08c4d772a0e7e2c85d1522844273a5133b2e10))
* **release:** use PAT for release-please action ([#20](https://github.com/harnexa/nexa-gauge/issues/20)) ([f0c80e4](https://github.com/harnexa/nexa-gauge/commit/f0c80e49d251254d6187e3becc8c50b8cb4fa774))

## [Unreleased]

### Added
- Root package metadata and wheel/sdist build configuration for `nexa-gauge`.
- Top-level governance files: `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- Publish workflow for tagged PyPI releases.
- Dependabot config for dependency and GitHub Actions updates.

### Changed
- README refreshed with pip install and release workflow guidance.
- Setup and Makefile messaging updated to reflect CLI-first current architecture.

### Fixed
- Cache store writes now use writer-unique temp paths before atomic replace, preventing concurrent same-key write collisions on shared `.tmp` filenames.
- Runner now coalesces concurrent same-key step execution with in-process single-flight, reducing duplicate LLM calls and duplicate cache writes.
