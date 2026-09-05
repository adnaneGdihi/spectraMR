.PHONY: format test test-all cov-unit cov-full test-coverage test-missing clean train predict eda eda-dry-run help install-topology env-show check-deps check-deps-imports test-precommit test-pr test-nightly test-release test-release-suite test-mutation diagnostics diagnostics-fast skill-health reachable dev-health gate

# -----------------------------------------------------------------------------
# Environment-variable loading.
# `.env` (gitignored) overrides shell-inherited values; `.env.example` lists
# every variable the framework reads.
#
# `.env` uses bash syntax (`export VAR=$(cmd)`, `$OTHER_VAR` refs) — it must
# be *sourced by bash*, not `include`d as Make syntax. `include .env` used to
# parse it directly: Make's own `$(pwd)`/`$VAR` expansion (not bash's) turned
# `export PATH=$BART_TOOLBOX_PATH:$PATH` into the literal path
# "ART_TOOLBOX_PATH:ATH" (single-char Make variable refs `$B`/`$P`, both
# undefined), silently breaking PATH — and therefore every bare-name command
# (`bash`, `python`, `date`, ...) — for every recipe in this file. Sourcing in
# a real bash subshell and re-injecting only the resulting KEY=VALUE pairs
# resolves `$(pwd)` / `$VAR` correctly regardless of what they reference.
# -----------------------------------------------------------------------------
ifneq (,$(wildcard ./.env))
DOTENV_PAIRS := $(shell bash -c 'set -a; source ./.env >/dev/null 2>&1; set +a; \
	for v in $$(sed -En "s/^[[:space:]]*export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p; t; s/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p" .env); do \
		printf "%s=%s\n" "$$v" "$${!v}"; \
	done')
$(foreach pair,$(DOTENV_PAIRS),$(eval $(word 1,$(subst =, ,$(pair))) := $(word 2,$(subst =, ,$(pair)))))
export $(foreach pair,$(DOTENV_PAIRS),$(word 1,$(subst =, ,$(pair))))
endif

help:
	@echo "spectraMR Development Commands"
	@echo "============================"
	@echo "make format      - Run code formatting (ruff check --fix + ruff format)"
	@echo "make test        - Run test suite"
	@echo "make clean       - Remove pycache and temp files"
	@echo "make train       - Train with default config (override: CONFIG=path/to.yaml; loads .env if present)"
	@echo "make env-show    - Print every framework env var + its current value"
	@echo "make check-deps         - Verify declared deps are installed & version-correct (EXTRAS=mri,viz)"
	@echo "make check-deps-imports - As above, plus import each (catches installed-but-unimportable)"
	@echo "make eda-dry-run - Dataset EDA: cards + coverage only (no voxel load)"
	@echo "make eda         - Dataset EDA: full per-dataset figures from data/manifests + databases/"
	@echo "make eda-quick   - Dataset EDA with a small sample budget (2/dataset)"
	@echo "make eda-dataset - Dataset EDA for a subset (DATASETS='id1 id2 ...')"
	@echo "make gate        - Run the blocking PR lane locally (PASS/FAIL/UNRUNNABLE)"
	@echo "make diagnostics      - Refresh diagnostics bundles (md summary + fresh forensics png)"
	@echo "make diagnostics-fast - Refresh diagnostics bundles, logs/audit/metrics only (no forensics)"

# Activate .venv when it exists (local dev) and stay silent when it does not (CI).
# A hosted runner has no .venv, and an unconditional `. .venv/bin/activate &&` kills
# the recipe on its first line -- `make test-nightly` never reached pytest.
VENV := [ -f .venv/bin/activate ] && . .venv/bin/activate || true;

# ruff format is the formatter SSOT: it is what .pre-commit-config.yaml runs and what
# the pr-required lint gate checks against. black used to be invoked here despite
# being declared in no extra of pyproject.toml, so the two fought over every file a
# commit touched.
format:
	ruff check --fix src/ tests/ || true
	ruff format src/ tests/

test:
	python -m pytest tests/unit/ -v --tb=short

test-all:
	python -m pytest tests/ -v --tb=short

# cov-unit: fast local loop — only the wave dirs actively expanded by the coverage plan (minutes, not hours)
cov-unit:
	@$(VENV) python -m pytest \
		tests/physics/ tests/metrics/ \
		tests/unit/physics/ tests/unit/metrics/ tests/unit/losses/ \
		tests/unit/config/ tests/unit/transforms/ tests/unit/registration/ \
		--cov=src/spectramr --cov-branch \
		--cov-report=term-missing:skip-covered \
		--cov-report=xml:tests_experiments/coverage_local.xml \
		--cov-fail-under=0 \
		-q

# cov-full: comprehensive suite (slow, cluster/CI use)
cov-full:
	@$(VENV) python -m pytest tests/ --cov=src/spectramr --cov-branch \
		--cov-report=xml:tests_experiments/coverage_local.xml \
		-q

test-coverage:
	python -m pytest --cov=src/spectramr --cov-branch \
	       --cov-report=html:htmlcov \
	       --cov-report=term-missing:skip-covered \
	       --cov-report=xml:coverage.xml \
	       -m "not slow and not e2e and not gpu and not integration" \
	       --maxfail=20 -q tests/
	@echo ""
	@python scripts/coverage/print_per_layer.py coverage.xml || true

test-missing:
	@python scripts/coverage/missing_test_files.py --top 30

# ---------------------------------------------------------------------------
# CI lane targets  (mirrors the 4-lane marker matrix in
#   docs/superpowers/plans/2026-05-22-unified-test-suite-master-plan.md)
# ---------------------------------------------------------------------------

# pre-commit lane — <60s, every push.
# Excludes: gpu, slow, fuzz, benchmark, convergence, differential.
test-precommit:
	@$(VENV) python -m pytest \
		-m "not gpu and not slow and not fuzz and not benchmark and not convergence and not differential" \
		--cov=src/spectramr --cov-branch \
		--cov-report=xml:tests_experiments/coverage_precommit.xml \
		--cov-report=term-missing:skip-covered \
		-q tests/

# pull-request lane — adds gpu + convergence; excludes fuzz/benchmark.
# <15 min; requires a CUDA runner for gpu-marked tests.
# TODO: point runs-on at a CUDA runner when invoking from CI (currently
#       inherits the caller's runner, which may not have a GPU).
test-pr:
	@$(VENV) python -m pytest \
		-m "not fuzz and not benchmark" \
		--cov=src/spectramr --cov-branch \
		--cov-report=xml:tests_experiments/coverage_pr.xml \
		--cov-report=term-missing:skip-covered \
		-q tests/

# nightly lane — full suite minus benchmark; fuzz with a FIXED hypothesis seed.
# <2h; cluster/GPU runner.
test-nightly:
	@$(VENV) python -m pytest \
		-m "not benchmark" \
		--hypothesis-seed=0 \
		--cov=src/spectramr --cov-branch \
		--cov-report=xml:tests_experiments/coverage_nightly.xml \
		--cov-report=term-missing:skip-covered \
		-q tests/

# release lane — everything + mutation + benchmark.
# ≤8h; requires CUDA runner and mutmut/atheris installed.
#
# Split into two halves so CI runs them as CONCURRENT jobs (test-release.yml).
# `test-release` keeps the sequential meaning for a local run. Nothing in
# test-mutation reads the suite's result, so the ordering was never a dependency
# -- only a convenience -- and on CI it was pure serial dead time. The pytest
# invocation stays in ONE place (non-negotiable 17): the workflow calls this
# target, it does not restate the command.
test-release-suite:
	@$(VENV) python -m pytest \
		--timeout=900 \
		--hypothesis-seed=0 \
		--cov=src/spectramr --cov-branch \
		--cov-report=xml:tests_experiments/coverage_release.xml \
		--cov-report=term-missing:skip-covered \
		-q tests/

# `--timeout=900` -- a per-test ceiling, not a session one.
#
# pytest-timeout is ALREADY a hard dependency, added so the 16 files carrying
# `@pytest.mark.timeout(...)` are honoured. Nothing ever set a DEFAULT, so a test
# with no marker can block forever and burn the entire 480-minute job budget with
# no failure to read. #1284 is exactly that shape: tests/unit/pipelines hangs on a
# network call, order-dependent, so the test it lands on moves. This converts a
# silent 8-hour stall into one named red test in 15 minutes. A per-test
# `@pytest.mark.timeout(...)` still overrides it.
#
# Measured on tests/unit/physics, 2026-09-04: serial 17.76s vs 18.44s with the
# flag, identical 4 failed / 662 passed / 2 skipped. Confirmed against
# `--timeout-method=thread` too (18.24s, same set), so the default signal method
# is not doing anything surprising here.
#
# NOT parallelised -- deliberately, and this is the part to re-check before
# changing it. `-n auto --dist loadfile` looks free (pytest-xdist is already a
# dependency and pr-advisory's impacted-tests job already uses it) and is not:
# it CHANGES THE FAILURE SET at low worker counts. Measured over
# tests/unit/physics + tests/unit/core (3512 tests), same tree, same commit:
#
#     serial   197s   16 failed   <- the truth
#     -n 2     142s   43 failed
#     -n 4     144s   31 failed   <- what a 4-core hosted runner would report
#     -n 8     166s   27 failed
#     -n 16    214s   16 failed   <- matches serial exactly
#
# The extra failures are all registry-contract tests under
# tests/unit/core/metrics/ asserting a metric's declared `needs`, e.g.
# `assert () == ('sibling_contrasts',)` -- the tuple comes back EMPTY when the
# modules are split across processes, so registration content depends on which
# test files share a worker. Widening to 16 hides it again, which is why a
# 24-core laptop reports a clean run and a hosted runner would not. Note also
# there is no wall-clock case for it here: 1.37x at n=4, and SLOWER than serial
# at n=16, because every worker re-imports torch and the model registry.
# OMP_NUM_THREADS=1 does not change either number (154s, 30 failed).
# Fix the isolation defect first; then this is a one-flag change.

test-release: test-release-suite
	$(MAKE) test-mutation

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# The default names a config that ships. `experiments/active/dummy_gan.yaml` was
# the default and is NOT in the published tree, so `make train` failed on a fresh
# clone with "no such file" -- the first command a new reader runs.
CONFIG ?= experiments/templates/comprehensive_config_template.yaml
train:
	python -m spectramr.cli train --config $(CONFIG)

# INPUT has no default that is true anywhere: `data/test/` existed in neither tree
# (`data/` holds dataset payloads and is gitignored). Refuse rather than invent one
# -- an inference run against a silently-wrong directory is worse than a stopped
# one, and `predict` cannot guess what the reader wants reconstructed.
INPUT ?=
predict:
	@test -n "$(INPUT)" || { echo "make predict needs INPUT=<dir-or-file>, e.g. make predict INPUT=path/to/kspace/"; exit 2; }
	python -m spectramr.cli predict --config $(CONFIG) --model $(MODEL) --input $(INPUT)

# Where `predict` looks for weights. Was hardcoded to checkpoints/best.pt, which
# the training loop only writes once a run has completed.
MODEL ?= checkpoints/best.pt

benchmark:
	python -m spectramr.cli benchmark --suite standard

# EDA Framework targets.
#
# `run_eda.sh` is a research-tree script and is NOT in the published export, so
# these four targets are advertised by `make help` in a tree that cannot run them.
# The guard makes that a STATED absence rather than a bare "No such file or
# directory": the reader learns the feature is not part of this distribution
# instead of suspecting a broken checkout. Absent is a state to report, never one
# to infer (non-negotiable 18).
EDA := @test -x ./run_eda.sh || { echo "run_eda.sh is not part of the published spectraMR tree -- dataset EDA ships with the research repository only."; exit 2; }; ./run_eda.sh

eda-dry-run:
	@echo "Running dataset EDA dry-run (cards + coverage only)..."
	$(EDA) --dry-run

eda:
	@echo "Running full dataset EDA..."
	$(EDA)

eda-quick:
	@echo "Running quick dataset EDA (2 samples/dataset)..."
	$(EDA) --samples-per-dataset 2

eda-dataset:
	@echo "Running dataset EDA for a subset: make eda-dataset DATASETS='fastmri_knee_singlecoil m4raw_multicoil_train_kspace'"
	$(EDA) --datasets $(DATASETS)

eda-clean:
	@echo "Cleaning EDA results..."
	rm -rf experiments/results

# Diagnostics bundles: tests_experiments/diagnostics (local dispatch) +
# <cluster>_diagnostics (downloaded cluster tree, if present). `diagnostics`
# always re-renders forensics (contact-sheet PNGs) so the md+png are current;
# `diagnostics-fast` skips the (slower) image pass and only refreshes
# logs/audit/run_summary/validation_metrics evidence. See
# docs/validation_image_audit.rst#diagnostics_targetable_tree.
#
# Invoked by its own executable path (shebang-resolved by the kernel), NOT
# `bash <path>` — the .env-loading block above mangles Make's exported PATH
# for recipe shells (a pre-existing bug: `include .env` parses .env's bash
# `export VAR=$(pwd)` syntax as Make syntax, where `$(pwd)`/`$VAR` mean
# something different, corrupting PATH to "ART_TOOLBOX_PATH:ATH"), so a bare
# `bash` lookup fails inside a recipe even though PATH looks fine outside `make`.
diagnostics:
	./scripts/ci/refresh_diagnostics.sh

# --- developer-loop health checks -------------------------------------------
# Both are cheap, local, and exit non-zero on a finding, so they compose:
# `make dev-health` is the pre-PR sweep.

# Skill files rot silently: nothing fails when a skill teaches a path that moved.
# Checks routes, `file.py:NNN` citations, repo paths, retired spellings, and
# trigger-phrase collisions across every SKILL.md it can see.
# `--also-user` adds ~/.claude/skills; `--extra-root` is needed from a worktree,
# where `.claude` being gitignored means only force-added skills are present.
# From a worktree, add the primary checkout's skills, e.g.
#   make skill-health SKILL_ROOTS="--extra-root /path/to/spectramr/.claude/skills"
SKILL_ROOTS ?=
skill-health:
	@test -f scripts/maintenance/check_skill_health.py || { echo "check_skill_health.py is not part of the published spectraMR tree -- skill health ships with the research repository only."; exit 2; }
	@$(PYTHON) scripts/maintenance/check_skill_health.py --also-user $(SKILL_ROOTS)

# Registered is not reachable. Probes each registry twice in cold subprocesses --
# documented entry point vs a full module walk -- and reports names that only
# appear after the walk. Those cannot be resolved from a YAML config.
# Slower (spawns interpreters); not part of the fast lane.
# The probes inherit THIS interpreter, so they need one with spectramr installed.
# From a worktree (no local .venv) pass the primary checkout's:
#   make reachable PYTHON=/path/to/spectramr/.venv/bin/python
reachable:
	@$(PYTHON) scripts/maintenance/prove_reachable.py --audit

# The pre-PR sweep. Both halves take their own passthrough, so from a worktree:
#   make dev-health PYTHON=/path/to/spectramr/.venv/bin/python \
#       SKILL_ROOTS="--extra-root /path/to/spectramr/.claude/skills"
dev-health: skill-health reachable

# The blocking PR lane, run by hand. Actions is disabled on the private research
# repository -- not on the published one, where this lane fires on every PR -- so
# `.github/workflows/pr-required.yml` describes a lane that never fires -- 8 jobs, 13
# guard scripts and the architecture fitness functions with no execution path. This is
# that path. It PARSES the workflow rather than restating it, so a job added there shows
# up here with no edit (non-negotiable 17), and it reports PASS / FAIL / UNRUNNABLE
# without ever folding the third into the first (non-negotiable 18).
#
#   make gate                       # the whole lane
#   make gate GATE_ARGS="--list"    # what it derived, without running it
#   make gate GATE_ARGS="--jobs guards,lint-diff --quiet"
#
# Exit 1 = something failed. Exit 2 = something could not run (a tool is absent); pass
# --allow-unrunnable to state that you accept that gap rather than silently inheriting it.
GATE_ARGS ?=
gate:
	@$(PYTHON) scripts/ci/run_required_locally.py $(GATE_ARGS)

diagnostics-fast:
	./scripts/ci/refresh_diagnostics.sh --no-forensics

# Install cubical persistent homology + Wasserstein-2 backends
# (gudhi, POT) used by the GeoMamba-ULF topology losses.
# Heavy and platform-specific — kept as an opt-in extra.
#
# NOT available on linux/aarch64: gudhi publishes no aarch64 wheel and no sdist,
# so it is marked off that platform in pyproject.toml and this target installs
# POT alone there. The loss then raises at construction and says so by name
# rather than pointing back at this command.
install-topology:
	pip install -e ".[topology]"

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

env-show:
	@PYTHONPATH=src $(PYTHON) scripts/release/print_env.py

# Verify the declared dependency set (pyproject SSOT) is installed & version-correct.
# `check-deps` = metadata only (fast); `check-deps-imports` also imports each
# (catches installed-but-unimportable, e.g. torchmetrics under hf-hub>=1.0).
check-deps:
	@$(PYTHON) scripts/verify/verify_dependencies.py $(if $(EXTRAS),--extras $(EXTRAS),)

check-deps-imports:
	@$(PYTHON) scripts/verify/verify_dependencies.py --import-check $(if $(EXTRAS),--extras $(EXTRAS),)

# ---------------------------------------------------------------------------
# Fuzz / mutation targets
# ---------------------------------------------------------------------------

# Coverage-guided config_health_checker fuzzer (requires atheris).
# Seeds from experiments/inprogress/**/*.yaml; crashes saved in fuzz-corpus/.
# Usage: make fuzz-audit-ladder
#        make fuzz-audit-ladder FUZZ_RUNS=100000
FUZZ_RUNS ?= -1
fuzz-audit-ladder:
	@echo "=== fuzz-audit-ladder: atheris fuzzer for ConfigHealthChecker ==="
	@if $(VENV) python -c "import atheris" 2>/dev/null; then \
		$(VENV) \
		python tests/fuzz/audit_ladder_fuzz/atheris_runner.py \
			-runs=$(FUZZ_RUNS) \
			-artifact_prefix=fuzz-corpus/audit_ladder_crash_; \
	else \
		echo "WARNING: atheris not installed — skipping fuzz-audit-ladder."; \
		echo "Install with: pip install atheris"; \
	fi

# Mutation testing for Tier-1 physics + config_health_checker (requires mutmut).
# Runs mutmut against the four high-value target files and prints a results summary.
# Usage: make test-mutation
test-mutation:
	@echo "=== test-mutation: mutmut for physics + health checker ==="
	@if $(VENV) python -c "import mutmut" 2>/dev/null; then \
		$(VENV) \
		mutmut run \
			--paths-to-mutate \
				src/spectramr/infrastructure/physics/fft_ops.py,\
				src/spectramr/infrastructure/physics/data_consistency.py,\
				src/spectramr/infrastructure/physics/coil_sensitivity.py,\
				src/spectramr/infrastructure/validation/config_health_checker.py \
			--tests-dir tests/unit/physics/ \
			--runner "python -m pytest -x -q --tb=no" && \
		mutmut results; \
	else \
		echo "WARNING: mutmut not installed — skipping test-mutation."; \
		echo "Install with: pip install mutmut"; \
	fi
