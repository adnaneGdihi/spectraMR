"""Conformal Calibration Strategy (PR-3 / H1 + ULF-PR-20..24).

This strategy does NOT train a model. It runs the relevant
calibration/certification primitive (``ConformalCalibrator``,
``CalibratedHallucinationDetector``, ``PathologyRecallCertificate``,
``PACBayesCertificate``, ``DiceRiskCertificate``) on the held-out
calibration split, and writes the resulting JSON artefact to
``certification.<name>.output_artefact``.

The model to certify is the generator wired onto ``self.env``. For the
``dice_risk`` certificate this MUST be a trained network: the branch loads
``certification.dice_risk.checkpoint_path`` into the generator and RAISES if
it is unset — certifying a randomly-initialised model is scientifically
meaningless (pitfall #20). The other certificate branches assume the operator
restored weights via the trainer's ``--resume`` path.

The strategy is dispatched by the ``training.strategy_class`` field
(directly importable path) rather than by ``training_mode``; it
therefore plays nicely with the existing factory.

When multiple ``certification.*`` blocks have ``enabled=True``, every
matching certificate is computed in one pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.checkpointing.restore import weights
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.models.blocks.kspace_geometry_encoding import support_forward_kwargs

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


def _batch_size(batch: Any) -> int:
    """Number of images a calibration batch carries (batch axis of its input)."""
    x, _ = _calibration_input_target(batch)
    return len(x)


def _select_images(batch: Any, keep: list[int]) -> Any:
    """Keep the images at batch-axis positions ``keep`` from a dict or tuple batch.

    Tensors and per-image lists whose length is the batch size are indexed; any other
    field (scalars, shared metadata) is carried through unchanged.
    """
    n = _batch_size(batch)
    index = torch.as_tensor(keep, dtype=torch.long)

    def pick(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n:
            return value[index]
        if isinstance(value, (list, tuple)) and len(value) == n:
            return type(value)(value[i] for i in keep)
        return value

    if isinstance(batch, dict):
        return {k: pick(v) for k, v in batch.items()}
    return type(batch)(pick(v) for v in batch)


def _calibration_input_target(batch: Any) -> tuple[Any, Any]:
    """Unpack a calibration batch to ``(model_input, target)`` for the runner.

    Handles dict batches (``input`` / ``target``) and 2-tuples. The runner runs
    the model on ``model_input`` itself, so this returns the INPUT, not a
    prediction (distinct from ``_unpack_calibration_pair``, which returns a
    prediction for the dice/CHD paths). The runner's default unpacker only
    accepts 2-tuples, so dict batches would otherwise raise.
    """
    if isinstance(batch, dict):
        return batch.get("input"), batch.get("target")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(
        "conformal calibration batch must be a dict with input/target or a "
        f"(input, target) tuple; got {type(batch).__name__}"
    )


def _count_images(batches: list[Any]) -> int:
    return sum(_batch_size(b) for b in batches)


def _cap_images(batches: list[Any], n_requested: int) -> tuple[list[Any], int]:
    """Keep exactly ``n_requested`` images from ``batches``; raise when there are fewer."""
    available = _count_images(batches)
    if available < n_requested:
        raise ValueError(
            f"n_calibration={n_requested} but the calibration half holds {available} "
            "images; lower n_calibration or enlarge the calibration cohort."
        )
    kept: list[Any] = []
    left = n_requested
    for batch in batches:
        if left == 0:
            break
        size = _batch_size(batch)
        if size <= left:
            kept.append(batch)
            left -= size
        else:
            kept.append(_select_images(batch, list(range(left))))
            left = 0
    return kept, n_requested


def _batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move the tensor fields of a dict or tuple batch to ``device``."""
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return type(batch)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch)


def _lesion_recall_inputs(model: Any, batches: list[Any], classes: list[str]) -> dict:
    """Per-class ``(empirical_recall, n_positives)``. No lesion detector exists (#3041)."""
    raise NotImplementedError(
        "pathology_recall needs a lesion detector that turns reconstructions into "
        f"empirical_recall for {classes}; none is implemented, so the certificate is not "
        "computable (#3041)."
    )


def _pac_bayes_inputs(model: Any, batches: list[Any]) -> tuple[float, float, int]:
    """``(empirical_risk, KL(Q||P), n)``. No KL(Q||P) estimator exists (#3041)."""
    raise NotImplementedError(
        "pac_bayes needs a KL(Q||P) estimator for the posterior; none is implemented, so the "
        "certificate is not computable (#3041)."
    )


class ConformalCalibrationStrategy(ReconstructionTrainingStrategy):
    """Calibration-only strategy. Does not perform gradient updates.

    Subclassing ``ReconstructionTrainingStrategy`` lets us reuse the loss
    plumbing, FFT transformer, and DC layer without re-implementing
    them, even though we never call ``optimizer.step()``.
    """

    #: Canonical opt-in certificate keys, in evaluation order. Single source of
    #: truth for both ``__init__`` and ``_collect_enabled_certs`` so a test can
    #: drive the SAME collection logic the strategy uses (no copied loop).
    _CERT_KEYS: tuple[str, ...] = (
        "conformal",
        "chd",
        "pathology_recall",
        "pac_bayes",
        "dice_risk",
        "badge",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # All certificates are opt-in — collect those flagged enabled.
        cert = getattr(self.config, "certification", None)
        self._enabled_certs: list[str] = self._collect_enabled_certs(cert)
        logger.info(
            "ConformalCalibrationStrategy initialised. Enabled certificates: %s",
            self._enabled_certs or "<none — strategy is a no-op>",
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field + contrast into ``batch_context``.

        The pipeline's gated validation seam forwards a batch key ONLY when its name
        literally appears in this signature (``_VALIDATION_FORWARD_FIELDS`` in
        ``pipelines/train.py``). Without this override the seam dropped both keys, so
        ``_prepare_generator_inputs`` resolved ``field_strength = None`` and called the
        renderer bare — and ``AnatomyFieldRenderer.forward`` takes ``field_strength`` as
        a REQUIRED keyword-only argument. Every validation batch raised TypeError, the
        loop hit "Validation produced zero successful batches", and the run died before
        ``_maybe_run_calibration`` ever ran: the b17 Dice-risk RCPS certificate, which is
        this arm's entire deliverable, was never written. Mirrors the override its
        CrossFieldTranslationStrategy sibling already carries.
        """
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    @classmethod
    def _collect_enabled_certs(cls, cert: Any) -> list[str]:
        """Return the enabled certificate keys from a ``certification`` config.

        Extracted so the unit test exercises the real collection logic instead
        of a hand-copied duplicate (the copied-loop test could not detect a
        certificate being dropped from ``_CERT_KEYS``)."""
        if cert is None:
            return []
        return [
            letter
            for letter in cls._CERT_KEYS
            if getattr(cert, letter, None) is not None
            and getattr(getattr(cert, letter), "enabled", False)
        ]

    # Override training step — calibration takes no gradient steps.
    @accepts_step_io
    def train_step(
        self, batch: Any, epoch: int = 0, step: int = 0, **kwargs: Any
    ) -> list:  # pragma: no cover - exercised in integration tests
        """No-op train step (no optimizer steps); certificates run in
        ``run_calibration``. Returns an EMPTY step-config list — the old dict
        return hit KeyError('optimizer') in Trainer.execute_step."""
        return []

    def run_calibration(self, dataloader: Any) -> dict[str, dict[str, Any]]:
        """Iterate the calibration loader once and emit the requested certificates.

        Returns a dict mapping certificate-letter to the report dict.
        Side-effect: writes each report to its declared ``output_artefact``.
        """
        cert_cfg = getattr(self.config, "certification", None)
        if cert_cfg is None:
            return {}

        # Lazy-import primitives so the strategy class is still importable when
        # downstream models / packages are missing.
        from spectramr.infrastructure.calibration import (
            CalibratedHallucinationDetector,
            ConformalCalibrationRunner,
            ConformalCalibrator,
            PACBayesCertificate,
            PathologyRecallCertificate,
            resolve_calibration_score,
        )
        from spectramr.infrastructure.reporting.validation_badge import ValidationBadge

        reports: dict[str, dict[str, Any]] = {}
        model = getattr(self.env, "generator", None)
        if model is None:
            logger.warning("No generator on context — cannot run calibration.")
            return reports

        # Materialise the cohort once: every certificate re-iterates it. Calibration
        # and test are split by subject (no subject on both sides), seeded from run.seed.
        batches: list[Any] = list(dataloader)
        calib_all: list[Any] = []
        test_all: list[Any] = []
        split_info: dict[str, Any] = {}
        if {"conformal", "chd"} & set(self._enabled_certs):
            calib_all, test_all, split_info = self._subject_split(batches, int(self.config.run.seed))

        # Conformal / R1 — score function from the calibration-score registry.
        if "conformal" in self._enabled_certs:
            cc = cert_cfg.conformal
            raw_score = resolve_calibration_score(cc.score_fn)
            score_kwargs: dict[str, Any] = {}
            if getattr(cc, "marker_basis_path", None) is not None:
                score_kwargs["marker_basis_path"] = cc.marker_basis_path

            def _bound_score(pred, target, _fn=raw_score, _kw=score_kwargs):
                return _fn(pred, target, **_kw)

            calibrator = ConformalCalibrator(
                score_fn=_bound_score,
                alpha=cc.alpha,
                foreground_threshold=cc.foreground_threshold,
            )
            runner = ConformalCalibrationRunner(
                model=model,
                calibrator=calibrator,
                device=self.device,
                unpack_batch=_calibration_input_target,
            )
            calib_batches, n_calib_images = _cap_images(calib_all, cc.n_calibration)
            report = runner.run(calib_batches, test_all)
            payload = {
                "quantile": report.quantile,
                "alpha": report.alpha,
                "empirical_coverage": report.empirical_coverage,
                "avg_set_size": report.avg_set_size,
                "score_fn": cc.score_fn,
                "foreground_threshold": cc.foreground_threshold,
                "n_calibration_requested": cc.n_calibration,
                "n_calibration_images": n_calib_images,
                "n_calibration_pixels": report.n_calibration,
                "n_test_images": _count_images(test_all),
                **split_info,
            }
            reports["r1_cdfc"] = payload
            self._write_artefact(cc.output_artefact, payload)

        # CHD / R2 — threshold fit on calibration pixels (taken as non-hallucinated, H=0),
        # FPR measured on the disjoint test half.
        if "chd" in self._enabled_certs:
            ch = cert_cfg.chd
            calib_batches, n_calib_images = _cap_images(calib_all, ch.n_calibration)
            cal_scores = self._foreground_scores(calib_batches, model, ch.foreground_threshold)
            test_scores = self._foreground_scores(test_all, model, ch.foreground_threshold)
            detector = CalibratedHallucinationDetector(beta=ch.beta, delta=ch.delta).fit(
                cal_scores
            )
            payload = {
                "beta": ch.beta,
                "delta": ch.delta,
                "threshold": detector.threshold,
                "fpr_upper_bound": detector.fpr_upper_bound,
                "dkw_slack": detector.fpr_upper_bound - ch.beta,
                "bound_is_tight": detector.bound_is_tight,
                "empirical_fpr": detector.empirical_fpr(test_scores),
                "score_fn": "absolute_residual",
                "foreground_threshold": ch.foreground_threshold,
                "n_calibration_requested": ch.n_calibration,
                "n_calibration_images": n_calib_images,
                "n_calibration_pixels": detector.n_calibration,
                "n_test_images": _count_images(test_all),
                **split_info,
            }
            reports["r2_chd"] = payload
            self._write_artefact(ch.output_artefact, payload)

        # Pathology recall / R4 — per class, Hoeffding lower bound on lesion recall.
        # The lesion detector that yields empirical_recall does not exist; the config
        # schema rejects enabled=true, and the input helper refuses if reached anyway.
        if "pathology_recall" in self._enabled_certs:
            prc = cert_cfg.pathology_recall
            recalls = _lesion_recall_inputs(model, batches, prc.classes)
            per_class: dict[str, dict[str, Any]] = {}
            for cls in prc.classes:
                empirical_recall, n_positives = recalls[cls]
                cert = PathologyRecallCertificate(
                    pathology_class=cls, target_recall=prc.target_recall, delta=prc.delta
                )
                per_class[cls] = cert.certify(empirical_recall, n_positives).to_dict()
            payload = {
                "classes": per_class,
                "is_certified": all(r["is_certified"] for r in per_class.values()),
            }
            reports["r4_prc"] = payload
            self._write_artefact(prc.output_artefact, payload)

        # PAC-Bayes / R5 — McAllester bound. KL(Q||P) has no estimator; the config schema
        # rejects enabled=true, and the input helper refuses if reached anyway.
        if "pac_bayes" in self._enabled_certs:
            pac = cert_cfg.pac_bayes
            risk, kl, n = _pac_bayes_inputs(model, batches)
            cert = PACBayesCertificate(
                delta=pac.delta,
                loss_bound_M=pac.loss_bound_M,
                tv_tolerance=pac.tv_tolerance_tau,
            )
            payload = cert.certify(risk, kl, n).to_dict()
            reports["r5_pac_bayes"] = payload
            self._write_artefact(pac.output_artefact, payload)

        # Dice-risk RCPS / B-1.7 — segment pred+target over the calibration cohort,
        # certify a Hoeffding upper bound on the expected Dice-risk (1 - Dice).
        if "dice_risk" in self._enabled_certs:
            import torch

            from spectramr.core.metrics.quantitative.segmentation import (
                dice_score,
                get_segmenter,
            )
            from spectramr.infrastructure.calibration import DiceRiskCertificate

            dr = cert_cfg.dice_risk
            # ANTI-#20: a Dice-risk certificate over an untrained model is
            # meaningless. Require an explicit checkpoint and load it into the
            # generator before certifying — never silently certify random weights.
            ckpt_path = getattr(dr, "checkpoint_path", None)
            if not ckpt_path:
                raise ValueError(
                    "certification.dice_risk.checkpoint_path is required: certifying "
                    "the Dice-risk of an uninitialised/untrained synthesiser is "
                    "scientifically meaningless (pitfall #20). Point it at the trained "
                    "synthesis arm's <output_dir>/checkpoints/best.<format>."
                )
            self._load_generator_checkpoint(model, ckpt_path)
            segmenter = get_segmenter(dr.segmenter_backend, n_classes=dr.n_classes)
            unit = str(getattr(dr, "independence_unit", "subject"))
            risks: list[float] = []
            per_unit: dict[str, list[float]] = {}
            was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                for batch in batches:
                    pred, target = self._unpack_calibration_pair(batch, model)
                    if pred is None or target is None:
                        continue
                    seg_p = segmenter.segment(pred)
                    seg_t = segmenter.segment(target)
                    d = dice_score(seg_p, seg_t, segmenter.n_classes)
                    batch_risks = (1.0 - d).reshape(-1).tolist()
                    risks.extend(batch_risks)
                    if unit == "subject":
                        for sid, risk in zip(
                            self._subject_ids(batch, len(batch_risks)), batch_risks, strict=True
                        ):
                            per_unit.setdefault(sid, []).append(risk)
            if was_training and hasattr(model, "train"):
                model.train()
            # One Hoeffding draw per INDEPENDENT unit (#1707): on a slice-served,
            # co-registered cohort the slices of one subject are not independent,
            # so ``n`` counts subjects and each draw is that subject's mean risk.
            if unit == "subject":
                draws = [sum(v) / len(v) for v in per_unit.values()]
            else:
                draws = risks
            n = len(draws)
            mean_risk = (sum(draws) / n) if n else 1.0
            report = DiceRiskCertificate(alpha=dr.alpha, delta=dr.delta).certify(
                mean_risk,
                max(n, 1),
                independence_unit=unit,
                n_samples_scored=len(risks),
            )
            payload = report.to_dict()
            # Provenance (#15): record WHICH checkpoint was certified so the verdict
            # is attributable and reproducible.
            payload["checkpoint_path"] = str(ckpt_path)
            payload["segmenter_backend"] = dr.segmenter_backend
            reports["r6_dice_risk"] = payload
            self._write_artefact(dr.output_artefact, payload)

        # V.3 — bundles the upstream JSONs into a single ValidationBadge artefact.
        if "badge" in self._enabled_certs:
            badge = ValidationBadge.from_certificate_paths(
                cert_cfg.badge.upstream_certificate_artefacts,
                required_gates=cert_cfg.badge.gates,
            )
            reports["v3_badge"] = badge.to_dict()
            self._write_artefact(cert_cfg.badge.output_path, reports["v3_badge"])

        return reports

    def _subject_split(self, batches: list[Any], seed: int) -> tuple[list[Any], list[Any], dict]:
        """Split the cohort into disjoint calibration and test halves by subject.

        Subjects are shuffled with ``seed`` and the first half calibrates. Every image
        of a subject lands on one side, so the test half is exchangeable with, but
        never the same subject as, the calibration half.
        """
        from spectramr.data.batch_types import read_batch_field

        if any(read_batch_field(b, "subject_id") is None for b in batches):
            raise ValueError(
                "conformal/CHD calibration splits by subject and needs batch['subject_id'] on "
                "the calibration (validation) loader; the M4Raw k-space val loader must emit it."
            )
        per_batch = [self._subject_ids(b, _batch_size(b)) for b in batches]
        subjects = sorted({sid for ids in per_batch for sid in ids})
        if len(subjects) < 2:
            raise ValueError(
                "conformal calibration splits by subject and needs at least 2 subjects; "
                f"the cohort has {len(subjects)}."
            )
        order = torch.randperm(len(subjects), generator=torch.Generator().manual_seed(seed))
        n_cal = len(subjects) // 2
        cal_subjects = {subjects[i] for i in order[:n_cal].tolist()}
        calib: list[Any] = []
        test: list[Any] = []
        for batch, ids in zip(batches, per_batch, strict=True):
            cal_idx = [i for i, sid in enumerate(ids) if sid in cal_subjects]
            test_idx = [i for i, sid in enumerate(ids) if sid not in cal_subjects]
            if cal_idx:
                calib.append(_select_images(batch, cal_idx))
            if test_idx:
                test.append(_select_images(batch, test_idx))
        info = {
            "split_unit": "subject",
            "split_seed": seed,
            "subjects_calibration_pool": n_cal,
            "subjects_test": len(subjects) - n_cal,
        }
        return calib, test, info

    def _foreground_scores(self, batches: list[Any], model: Any, threshold: float | None) -> Any:
        """Absolute-residual scores of the foreground pixels over ``batches``."""
        from spectramr.infrastructure.calibration.conformal import foreground_mask

        if hasattr(model, "eval"):
            model.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in batches:
                on_device = _batch_to_device(batch, self.device)
                pred, target = self._unpack_calibration_pair(on_device, model)
                if pred is None:
                    raise ValueError("CHD calibration batch carries no input/target pair.")
                residual = (pred - target).abs()
                mask = foreground_mask(target, threshold)
                parts.append(residual[mask] if mask is not None else residual.flatten())
        return torch.cat(parts)

    @staticmethod
    def _subject_ids(batch: Any, count: int) -> list[str]:
        """One subject id per scored image, from ``batch['subject_id']`` (raises when absent).

        The independence unit of the certificate is the subject; a batch that
        carries no id cannot be grouped, and counting its images as draws is the
        inflation #1707 describes, so it is refused rather than guessed.
        """
        from spectramr.data.batch_types import read_batch_field

        ids = read_batch_field(batch, "subject_id")
        if ids is None:
            raise ValueError(
                "certification.dice_risk.independence_unit='subject' needs batch['subject_id'] "
                "(the mrixfields and nifti_paired datasets emit it). Declare independence_unit: "
                "'sample' only for a cohort whose scored images are independent draws."
            )
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(i) for i in (ids.tolist() if hasattr(ids, "tolist") else list(ids))]
        if len(ids) != count:
            raise ValueError(
                f"batch['subject_id'] has {len(ids)} entries for {count} scored images."
            )
        return ids

    @staticmethod
    def _unpack_calibration_pair(batch: Any, model: Any) -> tuple[Any, Any]:
        """Return ``(prediction, target)`` from a calibration batch (defensive).

        Accepts a dict batch (``input``/``target``) or a 2-tuple ``(input, target)``;
        runs ``model(input)`` and unwraps tuple/dict outputs. Returns ``(None, None)``
        when the batch shape is unrecognised (skipped, not silently mis-scored).
        """
        if isinstance(batch, dict):
            x = batch.get("input")
            target = batch.get("target")
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, target = batch[0], batch[1]
        else:
            return None, None
        if x is None or target is None:
            return None, None
        # Field-conditioned generators (AnatomyFieldRenderer) REQUIRE a field_strength
        # kwarg; detect it by signature (no except-TypeError retry, which would mask a
        # real error) and pass the batch's target field (or source as fallback).
        import inspect

        fwd = getattr(unwrap_model(model), "forward", model)
        fwd_kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(fwd).parameters
        except (TypeError, ValueError):
            params = {}
        if "field_strength" in params:
            fs = None
            if isinstance(batch, dict):
                fs = batch.get("field_strength_target", batch.get("field_strength"))
            if fs is not None:
                fwd_kwargs["field_strength"] = fs
        # A geometry-plane generator raises without ``mask=``; the batch holds only the
        # zero-filled measurement, whose non-zero support is the observation mask.
        if isinstance(x, torch.Tensor):
            fwd_kwargs.update(support_forward_kwargs(unwrap_model(model), x))
        pred = model(x, **fwd_kwargs)
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, dict):
            pred = pred.get("reconstruction", next(iter(pred.values())))
        return pred, target

    @staticmethod
    def _load_generator_checkpoint(model: Any, path: str) -> None:
        """Load a trained generator state-dict into ``model`` in place.

        Reads the generator part of a RunState, strictly. RAISES on a missing file,
        a file with no generator part, a partial state, or a model with no
        ``load_state_dict`` — never silently leaves the model at its random init
        (pitfall #9/#20).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"dice_risk checkpoint not found: {path}. Train the synthesis arm "
                "first (its <output_dir>/checkpoints/best.<format>) before certifying."
            )
        if not hasattr(model, "load_state_dict"):
            raise TypeError(
                f"Cannot load a checkpoint into {type(model).__name__}: no "
                "load_state_dict. The generator must be an nn.Module to be certified."
            )
        weights(p, "generator", model, policy="exact")
        logger.info("dice_risk: loaded synthesiser checkpoint %s", path)

    @staticmethod
    def _write_artefact(path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote calibration artefact to %s", out)


class ValidationBadgeStrategy(ConformalCalibrationStrategy):
    """Alias subclass — exists so YAMLs that point at a separate
    ``validation_badge_strategy.ValidationBadgeStrategy`` class still
    resolve. Behaviour is identical to ``ConformalCalibrationStrategy``;
    the badge is rolled in via the same ``run_calibration`` call.
    """
