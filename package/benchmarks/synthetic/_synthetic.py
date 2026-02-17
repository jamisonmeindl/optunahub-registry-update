from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from typing import Any

import numpy as np
import optuna
import optunahub

from .augmentation import SyntheticAugmenter


class KernelType(Enum):
    RBF = auto()
    MATERN32 = auto()
    MATERN52 = auto()
    RATIONAL_QUADRATIC = auto()
    EXPONENTIAL = auto()
    PERIODIC = auto()
    LINEAR = auto()
    POLYNOMIAL = auto()


_BBOB_NOISE_VARIANTS = (
    "gaussian_moderate",
    "gaussian_severe",
    "uniform_moderate",
    "uniform_severe",
    "cauchy_moderate",
    "cauchy_severe",
)


def _derive_seed(master_seed: int, *tags: Any) -> int:
    h = hashlib.blake2s(digest_size=8)
    h.update(str(int(master_seed)).encode())
    for tag in tags:
        h.update(b"|")
        h.update(str(tag).encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def _coerce_optional_bool(value: Any, option_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(f"{option_name} must be a boolean when provided.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{option_name} must be a boolean when provided.")


def _normalize_noise_variant(value: Any) -> str:
    if value is None:
        return "gaussian_moderate"
    variant = str(value).strip().lower().replace("-", "_")
    aliases = {
        "gauss_moderate": "gaussian_moderate",
        "gauss_severe": "gaussian_severe",
    }
    variant = aliases.get(variant, variant)
    if variant not in _BBOB_NOISE_VARIANTS:
        raise ValueError(
            f"noise_variant must be one of {list(_BBOB_NOISE_VARIANTS)}, got {value!r}."
        )
    return variant


@dataclass
class KernelSpec:
    kind: KernelType
    variance: float
    lengthscale: np.ndarray
    alpha: float | None = None
    period: float | None = None
    degree: int | None = None
    offset: float | None = None


@dataclass
class _ObjectiveSurface:
    kernel_specs: list[KernelSpec]
    solved: np.ndarray
    global_y_min: float
    global_y_max: float
    task: str
    obj_scale: float
    obj_offset: float
    y_flip: float
    global_min: float | None = None
    global_max: float | None = None
    global_optimum: float | None = None

def _param_config_to_space(
    param_config: dict[str, dict[str, Any]]
) -> dict[str, optuna.distributions.BaseDistribution]:
    space: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, cfg in param_config.items():
        ptype = cfg["type"]
        if ptype == "Continuous":
            low, high = cfg["range"]
            space[name] = optuna.distributions.FloatDistribution(
                float(low), float(high), log=bool(cfg.get("log", False))
            )
        elif ptype == "Integer":
            low, high = cfg["range"]
            space[name] = optuna.distributions.IntDistribution(
                int(low), int(high), log=bool(cfg.get("log", False))
            )
        elif ptype == "Categorical":
            space[name] = optuna.distributions.CategoricalDistribution(list(cfg["categories"]))
        elif ptype == "Boolean":
            space[name] = optuna.distributions.CategoricalDistribution(list(cfg["categories"]))
        else:
            raise ValueError(f"Unsupported param type '{ptype}' for '{name}'.")
    return space


def _scaled_squared_distance(
    X: np.ndarray, Y: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    lengthscale = np.asarray(lengthscale, dtype=float)
    while lengthscale.ndim < 1:
        lengthscale = np.asarray([lengthscale])
    if lengthscale.shape[0] == 1 and X.shape[1] != 1:
        lengthscale = np.full((X.shape[1],), lengthscale.item())
    X_scaled = X / lengthscale
    Y_scaled = Y / lengthscale
    diff = X_scaled[:, None, :] - Y_scaled[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def _kernel_value(spec: KernelSpec, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    dist_sq = _scaled_squared_distance(X, Y, spec.lengthscale)
    if spec.kind == KernelType.RBF:
        return spec.variance * np.exp(-0.5 * dist_sq)
    if spec.kind == KernelType.MATERN32:
        d = np.sqrt(np.maximum(dist_sq, 1e-12))
        return spec.variance * (1.0 + np.sqrt(3.0) * d) * np.exp(-np.sqrt(3.0) * d)
    if spec.kind == KernelType.MATERN52:
        d = np.sqrt(np.maximum(dist_sq, 1e-12))
        return spec.variance * (1.0 + np.sqrt(5.0) * d + 5.0 / 3.0 * d ** 2) * np.exp(
            -np.sqrt(5.0) * d
        )
    if spec.kind == KernelType.RATIONAL_QUADRATIC:
        alpha = spec.alpha or 1.0
        return spec.variance * (1.0 + 0.5 * dist_sq / alpha) ** (-alpha)
    if spec.kind == KernelType.EXPONENTIAL:
        d = np.sqrt(np.maximum(dist_sq, 1e-12))
        return spec.variance * np.exp(-d)
    if spec.kind == KernelType.PERIODIC:
        period = float(spec.period or 1.0)
        d = np.sqrt(np.maximum(dist_sq, 1e-12))
        return spec.variance * np.exp(-2.0 * (np.sin(np.pi * d / period) ** 2))
    if spec.kind == KernelType.LINEAR:
        scale = np.asarray(spec.lengthscale, dtype=float)
        if scale.ndim == 0:
            scale = np.asarray([scale])
        if scale.shape[0] == 1 and X.shape[1] != 1:
            scale = np.full((X.shape[1],), scale.item())
        Xs = X / scale
        Ys = Y / scale
        return spec.variance * (Xs @ Ys.T)
    if spec.kind == KernelType.POLYNOMIAL:
        scale = np.asarray(spec.lengthscale, dtype=float)
        if scale.ndim == 0:
            scale = np.asarray([scale])
        if scale.shape[0] == 1 and X.shape[1] != 1:
            scale = np.full((X.shape[1],), scale.item())
        Xs = X / scale
        Ys = Y / scale
        degree = int(spec.degree or 2)
        offset = float(spec.offset or 1.0)
        return spec.variance * (Xs @ Ys.T + offset) ** degree
    raise ValueError(f"Unknown kernel type: {spec.kind}")


class GPFunctionEnv:
    def __init__(
        self,
        dim: int,
        seed: int = 0,
        data_type: str = "Mixed",
        options: dict[str, Any] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim
        self.seed = 0 if seed is None else int(seed)
        self.user_seed = self.seed
        self.data_type = data_type
        self.title = "GPFunction Benchmark"
        self.description = f"Gaussian-process synthetic surface ({self.dim}D, {self.data_type})"
        self.eval_info = {
            "name": "Objective Value",
            "description": "Synthetic GP evaluation",
            "unit": "unknown",
        }
        self.eval_range = None
        opts = options or {}
        raw_num_objectives = 1
        if opts.get("num_objectives") is not None:
            raw_num_objectives = int(opts["num_objectives"])
        self.seed_tag = str(opts.get("seed_tag", "default"))
        self.seed = int(
            _derive_seed(
                self.seed,
                "landscape",
                self.dim,
                self.data_type,
                raw_num_objectives,
                self.seed_tag,
            )
        )
        self.rng = np.random.default_rng(self.seed)
        self.noise_rng = np.random.default_rng(_derive_seed(self.seed, "noise", self.seed_tag))
        self.failure_rng = np.random.default_rng(_derive_seed(self.seed, "failure", self.seed_tag))
        self.augmentation_rng = np.random.default_rng(
            _derive_seed(self.seed, "augmentation", self.seed_tag)
        )
        self.param_config = self._generate_random_param_config(dim, data_type)
        self.use_post_scaling = bool(options.get("use_post_scaling", True)) if options else True
        self.use_ard = bool(options.get("use_ard", True)) if options else True
        self.num_objectives = self._resolve_num_objectives(options)
        self.N = max(
            1,
            sum(
                1 if v["type"] in ("Integer", "Continuous") else len(v["categories"])
                for v in self.param_config.values()
            ),
        )
        self.num_samples = int(self.rng.integers(self.N, 25 * self.N))
        self.X_initial_raw = self.sample(self.num_samples)
        self.X_initial_encoded = self.encode(self.X_initial_raw)
        self._build_surfaces()
        self.global_y_min = min(surface.global_y_min for surface in self._surfaces)
        self.global_y_max = max(surface.global_y_max for surface in self._surfaces)
        self.global_min = None
        self.global_max = None
        self.global_optimum = None
        self.task = self._surfaces[0].task if self._surfaces else "minimize"
        self._augmenter: SyntheticAugmenter | None = None
        self._force_adds_noise = _coerce_optional_bool(opts.get("adds_noise"), "adds_noise")
        self._force_has_failure_regions = _coerce_optional_bool(
            opts.get("has_failure_regions"), "has_failure_regions"
        )
        self._force_does_augmentations = _coerce_optional_bool(
            opts.get("does_augmentations"), "does_augmentations"
        )
        noise_requested = (
            self._force_adds_noise is True
            or any(key in opts for key in ("noise_level", "noise_variant"))
        )
        self.noise_level = float(options.get("noise_level", 0.02)) if options else 0.02
        raw_noise_variant = opts.get("noise_variant")
        if raw_noise_variant is None and self._force_adds_noise:
            raw_noise_variant = self.noise_rng.choice(_BBOB_NOISE_VARIANTS)
        self.noise_variant = _normalize_noise_variant(raw_noise_variant)
        if self._force_adds_noise is None and not noise_requested:
            self.noise_active = False
        elif self._force_adds_noise is None:
            self.noise_active = True
        else:
            self.noise_active = self._force_adds_noise
        self.failure_value = None
        self.failure_worst_quantile = float(opts.get("failure_worst_quantile", 0.15))
        self.failure_worst_quantile = float(np.clip(self.failure_worst_quantile, 0.0, 0.95))
        self.failure_worst_mode = str(opts.get("failure_worst_mode", "quantile_mask")).strip().lower()
        if self.failure_worst_mode not in {"quantile_mask", "anchors"}:
            raise ValueError("failure_worst_mode must be either 'quantile_mask' or 'anchors'.")
        self.failure_worst_probe_count = int(
            opts.get("failure_worst_probe_count", max(256, 80 * self.dim))
        )
        self.failure_worst_probe_count = max(64, self.failure_worst_probe_count)
        self.failure_worst_anchor_fraction = float(opts.get("failure_worst_anchor_fraction", 0.5))
        self.failure_worst_anchor_fraction = float(
            np.clip(self.failure_worst_anchor_fraction, 0.1, 2.0)
        )
        self.failure_worst_radius_scale = float(opts.get("failure_worst_radius_scale", 0.22))
        self.failure_worst_radius_scale = float(np.clip(self.failure_worst_radius_scale, 0.05, 0.8))
        has_failure_shape_options = any(
            key in opts
            for key in (
                "failure_regions",
                "failure_region_density",
                "failure_region_scale",
                "failure_worst_quantile",
                "failure_worst_mode",
                "failure_worst_probe_count",
                "failure_worst_anchor_fraction",
                "failure_worst_radius_scale",
            )
        )
        failure_requested = (
            self._force_has_failure_regions is True
            or has_failure_shape_options
        )
        if options and "failure_regions" in options:
            self.failure_regions = int(options["failure_regions"])
        else:
            region_scale = float(options.get("failure_region_density", 1.0)) if options else 1.0
            region_scale = max(0.0, region_scale)
            self.failure_regions = max(2, int(round(self.dim * region_scale)))
        if not failure_requested and self._force_has_failure_regions is None:
            self.failure_regions = 0
        if self._force_has_failure_regions and self.failure_regions <= 0:
            self.failure_regions = 1
        self.failure_region_scale = float(options.get("failure_region_scale", 0.5)) if options else 0.5
        self.failure_as_constraint = bool(opts.get("failure_as_constraint", True))
        if self._force_has_failure_regions is None:
            self.failure_active = self.failure_regions > 0 and failure_requested
        else:
            self.failure_active = self.failure_regions > 0 and self._force_has_failure_regions
        self._failure_shapes: list[dict[str, Any]] | None = None
        self.last_failure_mask: np.ndarray | None = None
        if self.failure_active:
            self._init_failures()
        self._init_augmentation(options)
        if options and options.get("find_opt", False):
            self.find_global_optimum()

    def _resolve_num_objectives(self, options: dict[str, Any] | None) -> int:
        if not options:
            return 1
        raw = options.get("num_objectives", 1)
        num = 1 if raw is None else int(raw)
        if num < 1:
            raise ValueError("num_objectives must be >= 1.")
        return num

    def _build_surfaces(self) -> None:
        surfaces: list[_ObjectiveSurface] = []
        int64_max = np.iinfo(np.int64).max
        for _ in range(self.num_objectives):
            seed = int(self.rng.integers(0, int64_max, dtype=np.int64))
            surface_rng = np.random.default_rng(seed)
            surfaces.append(self._create_surface(surface_rng))
        self._surfaces = surfaces

    def _stabilize_covariance(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = 0.5 * (K + K.T)
        eye = np.eye(K.shape[0])
        jitter = 1e-6
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(K + jitter * eye)
                return K + jitter * eye, chol
            except np.linalg.LinAlgError:
                jitter *= 10.0
        eigvals = np.linalg.eigvalsh(K)
        min_eig = float(np.min(eigvals))
        if min_eig < 0:
            jitter = max(jitter, -min_eig + 1e-6)
        K = K + jitter * eye
        chol = np.linalg.cholesky(K)
        return K, chol

    def _create_surface(self, rng: np.random.Generator) -> _ObjectiveSurface:
        max_kernels = int(rng.integers(1, 3))
        variances = [
            float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
            for _ in range(max_kernels)
        ]
        lengthscales = [self._sample_lengthscale(rng) for _ in range(max_kernels)]
        kernel_specs: list[KernelSpec] = []
        for idx in range(max_kernels):
            ktype = rng.choice(list(KernelType))
            variance = variances[min(idx, len(variances) - 1)]
            lengthscale = lengthscales[min(idx, len(lengthscales) - 1)]
            alpha = (
                float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
                if ktype == KernelType.RATIONAL_QUADRATIC
                else None
            )
            period = None
            degree = None
            offset = None
            if ktype == KernelType.PERIODIC:
                period = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
            if ktype == KernelType.POLYNOMIAL:
                degree = int(rng.integers(2, 5))
                offset = float(rng.uniform(0.5, 2.0))
            kernel_specs.append(
                KernelSpec(
                    kind=ktype,
                    variance=variance,
                    lengthscale=lengthscale,
                    alpha=alpha,
                    period=period,
                    degree=degree,
                    offset=offset,
                )
            )

        K_initial = self._kernel_matrix(
            self.X_initial_encoded, self.X_initial_encoded, kernel_specs
        )
        K_initial, chol = self._stabilize_covariance(K_initial)
        y_sampled_initial = chol @ rng.standard_normal(self.num_samples)
        global_y_min = float(np.min(y_sampled_initial))
        global_y_max = float(np.max(y_sampled_initial))
        task = rng.choice(["minimize", "maximize"])
        try:
            solved = np.linalg.solve(K_initial, y_sampled_initial)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(K_initial, y_sampled_initial, rcond=None)[0]
        surface = _ObjectiveSurface(
            kernel_specs=kernel_specs,
            solved=solved,
            global_y_min=global_y_min,
            global_y_max=global_y_max,
            task=task,
            obj_scale=1.0,
            obj_offset=0.0,
            y_flip=1.0,
        )
        self._init_post_scaling(global_y_min, global_y_max, surface, rng)
        return surface

    def _init_post_scaling(
        self,
        y_min: float,
        y_max: float,
        surface: _ObjectiveSurface,
        rng: np.random.Generator,
    ) -> None:
        """
        Sets up scaling to map raw GP outputs to realistic objective ranges.
        Updates the tracked bounds on the supplied surface.
        """
        profiles = [
            "large_error", "fixed_interval", "standard_loss", "classification", "log_likelihood",
            "bound_fraction", "tiny_loss", "game_score",
            "percent_change", "latency_ms", "energy_ev", "db_signal",
            "throughput", "price", "rating", "random_positive",
            "random_mixed", "random_offset"
        ]
        weights = [
            0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02
        ]
        weights = [w / sum(weights) for w in weights]
        profile = rng.choice(profiles, p=weights)
        span = max(1e-12, y_max - y_min)
        surface.y_flip = rng.choice([-1.0, 1.0])

        if profile == "fixed_interval":
            ranges = [
                (0.0, 1.0), (0.0, 10.0), (0.0, 100.0),
                (-1.0, 1.0), (0.0, 255.0), (-100.0, 100.0)
            ]
            target_min, target_max = ranges[rng.choice(len(ranges))]
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "standard_loss":
            target_min = rng.uniform(0.1, 2.0)
            target_max = target_min + rng.uniform(0.5, 5.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "classification":
            if rng.random() < 0.5:
                target_min, target_max = rng.uniform(0.05, 0.4), rng.uniform(0.6, 1.0)
            else:
                target_min, target_max = rng.uniform(5.0, 40.0), rng.uniform(60.0, 100)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "log_likelihood":
            magnitude = rng.uniform(100, 10000)
            target_min = magnitude
            target_max = magnitude + rng.uniform(100, 10000)
            surface.obj_scale = (target_max - target_min) / span
            if rng.random() < 0.5:
                surface.obj_offset = target_min - (y_min * surface.obj_scale)
            else:
                target_min_neg, target_max_neg = -target_max, -magnitude
                surface.obj_scale = (target_max_neg - target_min_neg) / span
                surface.obj_offset = target_min_neg - (y_min * surface.obj_scale)
            surface.y_flip = rng.choice([-1.0, 1.0])

        elif profile == "large_error":
            lower_exp = rng.uniform(2, 6)
            target_min = 10 ** lower_exp
            target_max = target_min + (target_min * rng.uniform(2.0, 10.0))
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "bound_fraction":
            sub_span = rng.uniform(0.1, 0.4)
            target_min = rng.uniform(0.0, 1.0 - sub_span)
            target_max = target_min + sub_span
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "tiny_loss":
            log_min = rng.uniform(-6, -4)
            target_min = 10 ** log_min
            target_max = target_min * rng.uniform(1.5, 10.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "game_score":
            target_min = 0.0
            target_max = rng.uniform(1000.0, 50000.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "percent_change":
            limit = rng.uniform(1.0, 10.0)
            target_min, target_max = -limit, limit
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "latency_ms":
            target_min = rng.uniform(10.0, 200.0)
            target_max = target_min + rng.uniform(50.0, 800.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "energy_ev":
            magnitude = rng.uniform(500.0, 10000.0)
            width = rng.uniform(100.0, magnitude * 0.5)
            target_max, target_min = -magnitude + width, -magnitude
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)
            surface.y_flip = 1.0

        elif profile == "db_signal":
            target_min = rng.choice([-10.0, -20.0, -40.0, 0.0])
            target_max = target_min + rng.uniform(30.0, 80.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "throughput":
            target_min = rng.uniform(500.0, 5000.0)
            target_max = target_min + rng.uniform(5000.0, 45000.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "price":
            target_min = rng.uniform(10.0, 100.0)
            target_max = target_min + rng.uniform(50.0, 900.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "rating":
            limit = rng.choice([5.0, 10.0])
            target_min = 1.0
            target_max = limit
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "random_positive":
            target_min = rng.uniform(0.0, 1000.0)
            target_max = target_min + rng.uniform(1.0, 1000.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "random_mixed":
            target_min = rng.uniform(-1000.0, -1.0)
            target_max = rng.uniform(1.0, 1000.0)
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        elif profile == "random_offset":
            base = rng.choice([-10000.0, 10000.0]) * rng.uniform(0.5, 5.0)
            width = rng.uniform(1.0, 20.0)
            target_min = base
            target_max = base + width
            surface.obj_scale = (target_max - target_min) / span
            surface.obj_offset = target_min - (y_min * surface.obj_scale)

        v1 = self._post(np.array([y_min]), surface)[0]
        v2 = self._post(np.array([y_max]), surface)[0]
        surface.global_y_min = float(min(v1, v2))
        surface.global_y_max = float(max(v1, v2))

    def _sample_lengthscale(self, rng: np.random.Generator) -> np.ndarray:
        if self.use_ard:
            return np.exp(rng.uniform(np.log(0.05), np.log(1.0), size=self.N))
        value = float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
        return np.full(self.N, value)

    def _create_random_kernels(self) -> list[KernelSpec]:
        num_kernels = int(self.rng.integers(1, self.max_kernels + 1))
        kernels: list[KernelSpec] = []
        for idx in range(num_kernels):
            ktype = self.rng.choice(list(KernelType))
            variance = self.variances[min(idx, len(self.variances) - 1)]
            lengthscale = self.lengthscales[min(idx, len(self.lengthscales) - 1)]
            alpha = (
                float(np.exp(self.rng.uniform(np.log(0.1), np.log(2.0))))
                if ktype == KernelType.RATIONAL_QUADRATIC
                else None
            )
            kernels.append(KernelSpec(kind=ktype, variance=variance, lengthscale=lengthscale, alpha=alpha))
        return kernels

    def _kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_specs: list[KernelSpec],
    ) -> np.ndarray:
        total = np.zeros((X.shape[0], Y.shape[0]))
        for spec in kernel_specs:
            total += _kernel_value(spec, X, Y)
        return total


    def _generate_random_param_config(
        self, num_params: int, data_type: str
    ) -> dict[str, dict[str, Any]]:
        param_config: dict[str, dict[str, Any]] = {}
        if data_type not in ("Mixed", "Continuous"):
            raise ValueError(f"Unsupported data_type: {data_type}")
        
        # In real datasets, Continuous params are usually more common than others.
        if data_type == "Mixed":
            # Weighted choice: mostly continuous/int, fewer boolean/categorical
            param_types = ["Continuous", "Integer", "Categorical", "Boolean"]
            p_weights = [0.3, 0.3, 0.3, 0.1]
        else:
            param_types = ["Continuous"]
            p_weights = [1.0]

        for i in range(num_params):
            p_type = str(self.rng.choice(param_types, p=p_weights))
            entry = self._reasonable_param_entry(p_type)
            param_config[f"param_{i}"] = entry
        return param_config

    def _reasonable_param_entry(self, p_type: str) -> dict[str, Any]:
        """
        Generates parameter ranges based on an extensive set of real-world ML archetypes.
        """
        entry: dict[str, Any] = {"type": p_type}
        
        if p_type == "Boolean":
            entry["categories"] = [False, True]
            
        elif p_type == "Continuous":
            # Extended options list
            options = [
                "learning_rate", "unit_fraction", "amplitude", "decay", 
                "large_magnitude", "angle", "temperature", "epsilon"
            ]
            # Weights adjusted to keep common params (LR, Fraction) frequent
            weights = [0.25, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
            profile = self.rng.choice(options, p=weights)
            
            if profile == "learning_rate":
                # 1e-6 to 1e-1
                lower_exp = self.rng.uniform(-6, -2)
                width = self.rng.uniform(1, 4)
                low = float(10 ** lower_exp)
                high = float(10 ** (lower_exp + width))
                entry["log"] = True
                entry["range"] = [low, high]
                
            elif profile == "unit_fraction":
                # 0.0 to 1.0
                low = float(self.rng.uniform(0.0, 0.5))
                high = float(self.rng.uniform(low + 0.1, 1.0))
                entry["log"] = False
                entry["range"] = [low, high]
                
            elif profile == "decay":
                # 0.85 to 0.999...
                low = float(self.rng.uniform(0.85, 0.95))
                epsilon = float(10 ** self.rng.uniform(-4, -2))
                high = 1.0 - epsilon
                entry["log"] = False
                entry["range"] = [low, high]

            elif profile == "large_magnitude":
                # 1 to 10000+
                lower_exp = self.rng.uniform(0, 2)
                width = self.rng.uniform(2, 4)
                low = float(10 ** lower_exp)
                high = float(10 ** (lower_exp + width))
                entry["log"] = True
                entry["range"] = [low, high]

            elif profile == "angle":
                # 0 to 180/360 or -180 to 180
                limit = float(self.rng.choice([3.14159, 6.28318, 180.0, 360.0]))
                if self.rng.random() < 0.5:
                    entry["range"] = [0.0, limit]
                else:
                    entry["range"] = [-limit/2, limit/2]
                entry["log"] = False

            elif profile == "temperature":
                # NEW: Sampling temperature (e.g. 0.1 to 2.0)
                low = float(self.rng.uniform(0.01, 0.5))
                high = float(self.rng.uniform(1.0, 5.0))
                entry["log"] = False # Linear is common for temperature
                entry["range"] = [low, high]

            elif profile == "epsilon":
                # NEW: Stability constant (1e-8 to 1e-4)
                lower_exp = self.rng.uniform(-9, -7)
                upper_exp = self.rng.uniform(-5, -3)
                low = float(10 ** lower_exp)
                high = float(10 ** upper_exp)
                entry["log"] = True
                entry["range"] = [low, high]

            else: # "amplitude"
                # Generic weights
                if self.rng.random() < 0.5:
                    limit = float(self.rng.uniform(1.0, 10.0))
                    entry["range"] = [-limit, limit]
                else:
                    low = float(self.rng.uniform(0.0, 1.0))
                    high = float(self.rng.uniform(10.0, 100.0))
                    entry["range"] = [low, high]
                entry["log"] = False

        elif p_type == "Integer":
            # Extended options list
            options = [
                "capacity", "structure", "counts", "binary_int", 
                "large_vocab", "offset", "frequency", "kernel_size"
            ]
            weights = [0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05]
            profile = self.rng.choice(options, p=weights)
            
            if profile == "capacity":
                # Powers of 2
                base = 2
                min_exp = int(self.rng.integers(4, 6))
                max_exp = int(self.rng.integers(min_exp + 2, 11))
                low = int(base ** min_exp)
                high = int(base ** max_exp)
                entry["log"] = True
                entry["range"] = [low, high]
                
            elif profile == "structure":
                # Layers / Depth
                low = int(self.rng.integers(1, 5))
                high = int(self.rng.integers(low + 2, 25))
                entry["log"] = False
                entry["range"] = [low, high]
                
            elif profile == "binary_int":
                # 0/1 switch
                entry["range"] = [0, 1]
                entry["log"] = False
                
            elif profile == "large_vocab":
                # 1k to 100k
                low = int(self.rng.integers(1000, 5000))
                high = int(self.rng.integers(10000, 100000))
                entry["log"] = True
                entry["range"] = [low, high]
                
            elif profile == "offset":
                # Small +/- integer
                limit = int(self.rng.integers(5, 20))
                entry["range"] = [-limit, limit]
                entry["log"] = False

            elif profile == "frequency":
                # NEW: Logging/Checkpoint steps (e.g., 100 to 1000)
                # These are usually linear and start higher than 0
                low = int(self.rng.choice([10, 50, 100, 500]))
                high = low * int(self.rng.integers(5, 20)) # e.g. 500 * 10 = 5000
                entry["log"] = False
                entry["range"] = [low, high]

            elif profile == "kernel_size":
                # NEW: Very small integers (1 to 11) for convolutions
                low = int(self.rng.choice([1, 3]))
                high = int(self.rng.choice([5, 7, 9, 11]))
                entry["log"] = False
                entry["range"] = [low, high]

            else: # "counts"
                low = int(self.rng.integers(10, 50))
                high = low + int(self.rng.integers(50, 500))
                entry["log"] = bool(self.rng.random() < 0.5)
                entry["range"] = [low, high]

        else: # Categorical
            if bool(self.rng.random() < 0.9):
                num_categories = int(self.rng.integers(3, 11))
            else:
                num_categories = int(self.rng.integers(11, 51))
            prefix = str(self.rng.choice(["opt", "model", "act", "choice", "category"]))
            entry["categories"] = [f"{prefix}_{j}" for j in range(num_categories)]
            
        return entry

    def sample(self, num_samples=1) -> np.ndarray:
        samples = []
        for _ in range(int(num_samples)):
            sample = {}
            for key, value in self.param_config.items():
                if value["type"] == "Continuous":
                    low, high = value["range"]
                    if value.get("log"):
                        sample[key] = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
                    else:
                        sample[key] = float(self.rng.uniform(low, high))
                elif value["type"] == "Integer":
                    low, high = value["range"]
                    if value.get("log"):
                        log_low = np.log(low)
                        log_high = np.log(high)
                        raw = np.exp(self.rng.uniform(log_low, log_high))
                        clamped = int(np.clip(int(round(raw)), int(low), int(high)))
                        sample[key] = clamped
                    else:
                        sample[key] = int(self.rng.integers(int(low), int(high) + 1))
                elif value["type"] == "Boolean":
                    sample[key] = bool(self.rng.choice(value["categories"]))
                else:
                    sample[key] = self.rng.choice(value["categories"])
            samples.append(sample)
        return np.array(samples, dtype=object)

    def _normalize_samples(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = arr.reshape(-1)
        normalized: list[Any] = []
        for entry in arr:
            if isinstance(entry, np.ndarray) and entry.shape == ():
                try:
                    entry = entry.item()
                except ValueError:
                    pass
            normalized.append(entry)
        return np.array(normalized, dtype=object)

    def encode(self, X: np.ndarray) -> np.ndarray:
        encoded_parts: list[list[float]] = []
        normalized = self._normalize_samples(X)
        for x_i in normalized:
            encoded_part: list[float] = []
            for key, value in self.param_config.items():
                input_val = x_i[key]
                if value["type"] in ["Continuous", "Integer"]:
                    low, high = value["range"]
                    denom = float(high) - float(low)
                    scaled = 0.0 if denom == 0 else 2.0 * (float(input_val) - float(low)) / denom - 1.0
                    encoded_part.append(scaled)
                else:
                    cats = value["categories"]
                    cat_idx = cats.index(input_val)
                    ohe = np.zeros(len(cats), dtype=float)
                    ohe[cat_idx] = 1.0
                    encoded_part.extend(ohe.tolist())
            encoded_parts.append(encoded_part)
        return np.asarray(encoded_parts, dtype=float)

    def _post(self, y: np.ndarray, surface: _ObjectiveSurface) -> np.ndarray:
        if not self.use_post_scaling:
            return np.asarray(y, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        y_arr = surface.y_flip * y_arr
        y_arr = (y_arr * surface.obj_scale) + surface.obj_offset
        return y_arr

    def _evaluate_baseline(self, x: np.ndarray) -> np.ndarray:
        samples = self._normalize_samples(x)
        outputs: list[np.ndarray] = []
        for surface in self._surfaces:
            raw = self._predict_mean(samples, surface)
            outputs.append(self._post(raw, surface))
        if not outputs:
            return np.empty((samples.shape[0], 0))
        return np.column_stack(outputs)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self._augmenter is None:
            y = self._evaluate_baseline(x)
        else:
            y = self._augmenter.evaluate(x)
        y = self._maybe_add_noise(y)
        y, mask = self._apply_failures(x, y, apply_failure_value=not self.failure_as_constraint)
        self.last_failure_mask = mask
        return y

    def _maybe_add_noise(self, y: np.ndarray) -> np.ndarray:
        if not self.noise_active:
            return y
        base_scale = float(max(np.ptp(y), np.max(np.abs(y)), 1.0))
        variant = self.noise_variant
        if variant.startswith("gaussian_"):
            factor = 0.02 if variant.endswith("moderate") else 0.20
            sigma = base_scale * factor
            noise = self.noise_rng.normal(0.0, sigma, size=y.shape)
        elif variant.startswith("uniform_"):
            factor = 0.02 if variant.endswith("moderate") else 0.20
            sigma = base_scale * factor
            half_width = np.sqrt(3.0) * sigma
            noise = self.noise_rng.uniform(-half_width, half_width, size=y.shape)
        else:
            factor = 0.01 if variant.endswith("moderate") else 0.10
            gamma = base_scale * factor
            noise = self.noise_rng.standard_cauchy(size=y.shape) * gamma
            clip_k = 10.0 if variant.endswith("moderate") else 50.0
            noise = np.clip(noise, -clip_k * base_scale, clip_k * base_scale)
        return y + noise * max(self.noise_level / 0.02, 0.0)

    def _init_failures(self) -> None:
        # Scale radii per-dimension in encoded space (continuous/int in [-1, 1], one-hot in [0, 1]).
        scales = np.ones((self.N,), dtype=float)
        numeric_indices: list[int] = []
        categorical_groups: list[np.ndarray] = []
        dim_index = 0
        for _, spec in self.param_config.items():
            if spec["type"] in ("Continuous", "Integer"):
                scales[dim_index] = 2.0
                numeric_indices.append(dim_index)
                dim_index += 1
            else:
                count = len(spec["categories"])
                scales[dim_index : dim_index + count] = 1.0
                categorical_groups.append(np.arange(dim_index, dim_index + count, dtype=int))
                dim_index += count
        shapes: list[dict[str, Any]] = []

        def rand_unit(d: int) -> np.ndarray:
            v = self.failure_rng.normal(0, 1, size=(d,))
            norm = np.linalg.norm(v)
            return v if norm < 1e-12 else v / norm

        def sample_center() -> np.ndarray:
            center = np.zeros((self.N,), dtype=float)
            if numeric_indices:
                base = self.failure_rng.uniform(-0.7, 0.7, size=(len(numeric_indices),))
                edge_bias = self.failure_rng.random(len(numeric_indices)) < 0.70
                if np.any(edge_bias):
                    edge_mag = 0.80 + 0.20 * self.failure_rng.beta(0.8, 3.0, size=(int(edge_bias.sum()),))
                    edge_sign = self.failure_rng.choice([-1.0, 1.0], size=(int(edge_bias.sum()),))
                    base[edge_bias] = edge_sign * edge_mag
                center[np.asarray(numeric_indices, dtype=int)] = np.clip(base, -1.0, 1.0)
            for group in categorical_groups:
                chosen = int(self.failure_rng.choice(group))
                center[group] = 0.0
                center[chosen] = 1.0
            return center

        def scale_cap() -> float:
            return max(0.08, min(float(self.failure_region_scale), 0.45))

        def build_worst_quantile_shape() -> dict[str, Any] | None:
            if self.failure_worst_quantile <= 0.0:
                return None
            probe_count = max(self.failure_worst_probe_count, 16 * self.failure_regions)
            X_probe = self.sample(probe_count)
            Y_probe = np.asarray(self._evaluate_baseline(X_probe), dtype=float)
            if Y_probe.ndim == 1:
                Y_probe = Y_probe.reshape(-1, 1)
            if Y_probe.shape[0] == 0:
                return None
            badness = np.zeros((Y_probe.shape[0],), dtype=float)
            for obj_idx, surface in enumerate(self._surfaces):
                if obj_idx >= Y_probe.shape[1]:
                    break
                vals = np.asarray(Y_probe[:, obj_idx], dtype=float)
                if surface.task.lower() == "maximize":
                    vals = -vals
                lo = float(np.nanpercentile(vals, 5.0))
                hi = float(np.nanpercentile(vals, 95.0))
                span = max(hi - lo, 1e-12)
                badness += (vals - lo) / span
            cutoff = float(np.nanquantile(badness, 1.0 - self.failure_worst_quantile))
            tail_idx = np.where(badness >= cutoff)[0]
            if tail_idx.size == 0:
                return None

            if self.failure_worst_mode == "anchors":
                Z_tail = self.encode(X_probe[tail_idx]).astype(float)
                if Z_tail.shape[0] == 0:
                    return None
                edge_weights = np.ones((Z_tail.shape[0],), dtype=float)
                if numeric_indices:
                    numeric_tail = np.abs(Z_tail[:, np.asarray(numeric_indices, dtype=int)])
                    edge_strength = np.max(numeric_tail, axis=1)
                    edge_weights = np.clip(edge_strength, 1e-6, None) ** 2
                edge_weights = edge_weights / np.sum(edge_weights)

                anchor_count = max(1, int(round(self.failure_regions * self.failure_worst_anchor_fraction)))
                anchor_count = min(anchor_count, Z_tail.shape[0])
                chosen = self.failure_rng.choice(
                    Z_tail.shape[0], size=anchor_count, replace=False, p=edge_weights
                )
                anchors = Z_tail[np.asarray(chosen, dtype=int), :]
                radius = self.failure_worst_radius_scale * scales
                jitter = self.failure_rng.uniform(0.8, 1.25, size=(self.N,))
                radii = np.maximum(radius * jitter, 0.03)
                return {
                    "kind": "worst_quantile_region",
                    "mode": "anchors",
                    "anchors": anchors,
                    "radii": radii,
                }

            signs: list[float] = []
            lowers: list[float] = []
            spans: list[float] = []
            for obj_idx, surface in enumerate(self._surfaces):
                if obj_idx >= Y_probe.shape[1]:
                    break
                sign = -1.0 if surface.task.lower() == "maximize" else 1.0
                vals = sign * np.asarray(Y_probe[:, obj_idx], dtype=float)
                lo = float(np.nanpercentile(vals, 5.0))
                hi = float(np.nanpercentile(vals, 95.0))
                span = max(hi - lo, 1e-12)
                signs.append(sign)
                lowers.append(lo)
                spans.append(span)
            return {
                "kind": "worst_quantile_region",
                "mode": "quantile_mask",
                "signs": np.asarray(signs, dtype=float),
                "lowers": np.asarray(lowers, dtype=float),
                "spans": np.asarray(spans, dtype=float),
                "cutoff": cutoff,
            }

        shape_types: list[str] = ["ellipsoid"]
        if numeric_indices:
            shape_types.extend(["boundary_strip", "axis_band"])
        if len(numeric_indices) >= 2:
            shape_types.append("interaction_band")
        if categorical_groups and (numeric_indices or len(categorical_groups) >= 2):
            shape_types.append("categorical_gate")

        for _ in range(self.failure_regions):
            kind = str(self.failure_rng.choice(shape_types))
            if kind == "ellipsoid":
                base = float(self.failure_rng.uniform(0.08, scale_cap()))
                anis = self.failure_rng.uniform(0.6, 1.6, size=(self.N,))
                radii = base * anis * scales
                shapes.append(
                    {
                        "kind": "ellipsoid",
                        "center": sample_center(),
                        "radii": radii,
                    }
                )
            elif kind == "axis_band":
                axis = int(self.failure_rng.choice(numeric_indices))
                center = float(
                    float(self.failure_rng.choice([-1.0, 1.0]))
                    * (0.65 + 0.35 * float(self.failure_rng.beta(1.0, 2.8)))
                )
                width = float(self.failure_rng.uniform(0.04, max(0.08, 0.6 * scale_cap())))
                shapes.append(
                    {
                        "kind": "axis_band",
                        "axis": axis,
                        "center": center,
                        "width": width,
                    }
                )
            elif kind == "boundary_strip":
                axis = int(self.failure_rng.choice(numeric_indices))
                side = float(self.failure_rng.choice([-1.0, 1.0]))
                edge = side * float(self.failure_rng.uniform(0.82, 0.99))
                width = float(self.failure_rng.uniform(0.04, max(0.06, 0.55 * scale_cap())))
                anchor_axis = None
                anchor_center = None
                anchor_radius = None
                if len(numeric_indices) >= 2 and self.failure_rng.random() < 0.7:
                    choices = [i for i in numeric_indices if i != axis]
                    anchor_axis = int(self.failure_rng.choice(choices))
                    anchor_center = float(self.failure_rng.uniform(-0.6, 0.6))
                    anchor_radius = float(self.failure_rng.uniform(0.2, 0.8))
                shapes.append(
                    {
                        "kind": "boundary_strip",
                        "axis": axis,
                        "side": side,
                        "edge": edge,
                        "width": width,
                        "anchor_axis": anchor_axis,
                        "anchor_center": anchor_center,
                        "anchor_radius": anchor_radius,
                    }
                )
            elif kind == "interaction_band":
                a, b = self.failure_rng.choice(np.asarray(numeric_indices, dtype=int), size=2, replace=False)
                center_ab = self.failure_rng.uniform(-0.6, 0.6, size=(2,))
                edge_axis = int(self.failure_rng.integers(0, 2))
                center_ab[edge_axis] = float(self.failure_rng.choice([-1.0, 1.0])) * (
                    0.70 + 0.25 * float(self.failure_rng.beta(1.0, 2.8))
                )
                tangent = rand_unit(2)
                offset = float(self.failure_rng.uniform(-0.2, 0.2))
                width = float(self.failure_rng.uniform(0.05, max(0.08, 0.55 * scale_cap())))
                span = float(self.failure_rng.uniform(0.35, 1.0))
                shapes.append(
                    {
                        "kind": "interaction_band",
                        "axis_a": int(a),
                        "axis_b": int(b),
                        "center_ab": center_ab,
                        "tangent": tangent,
                        "offset": offset,
                        "width": width,
                        "span": span,
                    }
                )
            elif kind == "categorical_gate":
                group_a = categorical_groups[int(self.failure_rng.integers(0, len(categorical_groups)))]
                cat_a = int(self.failure_rng.choice(group_a))
                cat_b = None
                if len(categorical_groups) > 1 and self.failure_rng.random() < 0.6:
                    others = [g for g in categorical_groups if not np.array_equal(g, group_a)]
                    group_b = others[int(self.failure_rng.integers(0, len(others)))]
                    cat_b = int(self.failure_rng.choice(group_b))
                numeric_gate = None
                if numeric_indices and self.failure_rng.random() < 0.75:
                    gate_axis = int(self.failure_rng.choice(numeric_indices))
                    gate_side = float(self.failure_rng.choice([-1.0, 1.0]))
                    gate_threshold = gate_side * float(self.failure_rng.uniform(0.65, 0.95))
                    numeric_gate = (gate_axis, gate_side, gate_threshold)
                if cat_b is None and numeric_gate is None and numeric_indices:
                    gate_axis = int(self.failure_rng.choice(numeric_indices))
                    gate_side = float(self.failure_rng.choice([-1.0, 1.0]))
                    gate_threshold = gate_side * float(self.failure_rng.uniform(0.70, 0.98))
                    numeric_gate = (gate_axis, gate_side, gate_threshold)
                shapes.append(
                    {
                        "kind": "categorical_gate",
                        "cat_a": cat_a,
                        "cat_b": cat_b,
                        "numeric_gate": numeric_gate,
                    }
                )
            else:
                # Backward-compatible generic band fallback for unexpected types.
                normal = rand_unit(self.N)
                edge_val = float(self.failure_rng.uniform(-0.9, 0.9))
                width = float(self.failure_rng.uniform(0.03, max(0.05, 0.5 * scale_cap())))
                shapes.append(
                    {
                        "kind": "band",
                        "normal": normal,
                        "offset": edge_val,
                        "width": width,
                    }
                )

        worst_shape = build_worst_quantile_shape()
        if worst_shape is not None:
            # Always include this coupled-to-worst region in addition to all sampled regions.
            shapes.append(worst_shape)

        self._failure_shapes = shapes

    def _resolve_failure_value(self, y: np.ndarray) -> float | np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        tasks = [surface.task.lower() for surface in self._surfaces]
        if len(tasks) == 1:
            task = tasks[0] if tasks else "minimize"
            if task == "maximize":
                if self.global_min is not None:
                    return np.asarray(self.global_min, dtype=float)
                return np.nanmin(y_arr, axis=0)
            if self.global_max is not None:
                return np.asarray(self.global_max, dtype=float)
            return np.nanmax(y_arr, axis=0)

        values: list[float] = []
        for idx, task in enumerate(tasks):
            gmin = None
            gmax = None
            if self.global_min is not None:
                gmin = self.global_min[idx] if isinstance(self.global_min, tuple) else self.global_min
            if self.global_max is not None:
                gmax = self.global_max[idx] if isinstance(self.global_max, tuple) else self.global_max
            if task == "maximize":
                values.append(float(gmin) if gmin is not None else float(np.nanmin(y_arr[:, idx])))
            else:
                values.append(float(gmax) if gmax is not None else float(np.nanmax(y_arr[:, idx])))
        return np.asarray(values, dtype=float)

    def _failure_mask(self, x: np.ndarray) -> np.ndarray:
        y_dummy = np.zeros((self._normalize_samples(x).shape[0], 1), dtype=float)
        _, mask = self._apply_failures(x, y_dummy, apply_failure_value=False)
        return mask

    def _apply_failures(
        self, x: np.ndarray, y: np.ndarray, apply_failure_value: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.failure_active or not self._failure_shapes:
            mask = np.zeros((y.shape[0],), dtype=bool)
            return y, mask

        Z = self.encode(x).astype(float)
        mask = np.zeros((Z.shape[0],), dtype=bool)
        for shape in self._failure_shapes:
            kind = shape["kind"]
            if kind == "ellipsoid":
                deltas = Z - shape["center"][None, :]
                normed = deltas / (shape["radii"][None, :] + 1e-12)
                d2 = np.sum(normed * normed, axis=1)
                mask |= d2 <= 1.0
            elif kind == "axis_band":
                axis = int(shape["axis"])
                center = float(shape["center"])
                width = float(shape["width"])
                mask |= np.abs(Z[:, axis] - center) <= width
            elif kind == "boundary_strip":
                axis = int(shape["axis"])
                side = float(shape["side"])
                edge = float(shape["edge"])
                width = float(shape["width"])
                boundary_hit = side * (Z[:, axis] - edge) >= -width
                anchor_axis = shape.get("anchor_axis")
                if anchor_axis is not None:
                    anchor_center = float(shape["anchor_center"])
                    anchor_radius = float(shape["anchor_radius"])
                    local = np.abs(Z[:, int(anchor_axis)] - anchor_center) <= anchor_radius
                    boundary_hit &= local
                mask |= boundary_hit
            elif kind == "interaction_band":
                a = int(shape["axis_a"])
                b = int(shape["axis_b"])
                center_ab = np.asarray(shape["center_ab"], dtype=float)
                tangent = np.asarray(shape["tangent"], dtype=float)
                offset = float(shape["offset"])
                width = float(shape["width"])
                span = float(shape["span"])
                local = np.column_stack([Z[:, a], Z[:, b]]) - center_ab[None, :]
                dist_to_band = np.abs(local @ tangent - offset)
                radial = np.sqrt(np.sum(local * local, axis=1))
                mask |= (dist_to_band <= width) & (radial <= span)
            elif kind == "categorical_gate":
                gate = Z[:, int(shape["cat_a"])] > 0.5
                cat_b = shape.get("cat_b")
                if cat_b is not None:
                    gate &= Z[:, int(cat_b)] > 0.5
                numeric_gate = shape.get("numeric_gate")
                if numeric_gate is not None:
                    axis, side, threshold = numeric_gate
                    gate &= float(side) * (Z[:, int(axis)] - float(threshold)) >= 0.0
                mask |= gate
            elif kind == "worst_quantile_region":
                mode = str(shape.get("mode", "anchors"))
                if mode == "anchors":
                    anchors = np.asarray(shape["anchors"], dtype=float)
                    radii = np.asarray(shape["radii"], dtype=float)
                    deltas = Z[:, None, :] - anchors[None, :, :]
                    normed = deltas / (radii[None, None, :] + 1e-12)
                    d2 = np.sum(normed * normed, axis=2)
                    mask |= np.any(d2 <= 1.0, axis=1)
                else:
                    y_arr = np.asarray(y, dtype=float)
                    if y_arr.ndim == 1:
                        y_arr = y_arr.reshape(-1, 1)
                    signs = np.asarray(shape["signs"], dtype=float)
                    lowers = np.asarray(shape["lowers"], dtype=float)
                    spans = np.asarray(shape["spans"], dtype=float)
                    cutoff = float(shape["cutoff"])
                    if signs.size > 0 and y_arr.shape[1] > 0:
                        used = min(int(signs.size), int(y_arr.shape[1]))
                        badness = np.zeros((y_arr.shape[0],), dtype=float)
                        for j in range(used):
                            badness += ((signs[j] * y_arr[:, j]) - lowers[j]) / max(spans[j], 1e-12)
                        mask |= badness >= cutoff
            elif kind == "band":
                proj = Z @ shape["normal"]
                mask |= np.abs(proj - shape["offset"]) <= shape["width"]
            elif kind == "wedge":
                proj = Z @ shape["normal"]
                if shape["threshold"] >= 0:
                    mask |= proj >= shape["threshold"]
                else:
                    mask |= proj <= shape["threshold"]
            else:
                a = shape["axis_a"]
                b = shape["axis_b"]
                freq = shape["freq"]
                phase = shape["phase"]
                if self.N == 1:
                    wave = np.sign(np.sin(freq * Z[:, a] + phase))
                    mask |= wave > 0
                else:
                    wa = np.sign(np.sin(freq * Z[:, a] + phase))
                    wb = np.sign(np.sin(freq * Z[:, b] - phase))
                    mask |= (wa * wb) > 0

        y_out = np.asarray(y, dtype=float).copy()
        if apply_failure_value:
            failure_value = self._resolve_failure_value(y_out)
            if y_out.ndim == 1:
                y_out[mask] = float(failure_value)
            else:
                fail_vec = np.asarray(failure_value, dtype=float).ravel()
                if fail_vec.size == 1:
                    y_out[mask, :] = float(fail_vec.item())
                else:
                    y_out[mask, :] = fail_vec[None, :]
        return y_out, mask

    def evaluate_with_failures(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = self.evaluate(x)
        mask = self.last_failure_mask
        if mask is None:
            mask = np.zeros((y.shape[0],), dtype=bool)
        return y, mask

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        mask = self._failure_mask(x)
        # Match Optuna constrained convention: <= 0 feasible, > 0 infeasible.
        return np.where(mask, 1.0, -1.0).reshape(-1, 1)

    def _init_augmentation(self, options: dict[str, Any] | None) -> None:
        opts = options or {}
        augmentation_requested = (
            self._force_does_augmentations is True
            or any(
                key in opts
                for key in ("augment_options", "augment_sample_for_scale")
            )
        )
        if self._force_does_augmentations is None and not augmentation_requested:
            return
        if self._force_does_augmentations is False:
            return

        augment_options: dict[str, Any] = {}
        if isinstance(opts.get("augment_options"), dict):
            augment_options.update(opts["augment_options"])
        sample_for_scale = int(opts.get("augment_sample_for_scale", 256))

        self._augmenter = SyntheticAugmenter(
            self,
            sample_for_scale=sample_for_scale,
            options=augment_options,
        )

    def _predict_mean(self, x: np.ndarray, surface: _ObjectiveSurface) -> np.ndarray:
        x = self._normalize_samples(x)
        B = x.shape[0]
        MAX_BLOCK = 4096
        if B > MAX_BLOCK:
            parts = []
            for s in range(0, B, MAX_BLOCK):
                e = min(B, s + MAX_BLOCK)
                parts.append(self._predict_mean(x[s:e], surface))
            return np.concatenate(parts)
        x_encoded = self.encode(x)
        K_new_initial = self._kernel_matrix(
            x_encoded, self.X_initial_encoded, surface.kernel_specs
        )
        return K_new_initial @ surface.solved

    def name(self) -> str:
        tags: list[str] = []
        if self._augmenter is not None:
            tags.append("aug")
        if self.noise_active:
            tags.append(f"noise")
        if self.failure_active:
            tags.append("fail")
        tag_str = f"_{'-'.join(tags)}" if tags else ""
        return f"GPFunction_{self.data_type}_{self.dim}D_{self.user_seed}{tag_str}"

    def find_global_optimum(self, n_samples=None) -> None:
        if n_samples is None:
            n_samples = int(min(20000, max(2000, 600 * self.N)))
        batch_size = 1024
        best_min = [np.inf] * self.num_objectives
        best_max = [-np.inf] * self.num_objectives
        for idx in range(0, n_samples, batch_size):
            batch = min(batch_size, n_samples - idx)
            Xb = self.sample(batch)
            yb = np.asarray(self.evaluate(Xb), dtype=float)
            for obj_idx in range(self.num_objectives):
                obj_vals = yb[:, obj_idx]
                mask = np.isfinite(obj_vals)
                if not mask.any():
                    continue
                valid = obj_vals[mask]
                current_min = float(np.nanmin(valid))
                current_max = float(np.nanmax(valid))
                if current_min < best_min[obj_idx]:
                    best_min[obj_idx] = current_min
                if current_max > best_max[obj_idx]:
                    best_max[obj_idx] = current_max
        global_min = [None if value == np.inf else value for value in best_min]
        global_max = [None if value == -np.inf else value for value in best_max]
        global_optimum: list[float | None] = []
        for idx, surface in enumerate(self._surfaces):
            task = surface.task.lower()
            candidate = global_min[idx] if task != "maximize" else global_max[idx]
            fallback = global_max[idx] if global_max[idx] is not None else global_min[idx]
            optimum = candidate if candidate is not None else fallback
            global_optimum.append(optimum)
        self.global_min = tuple(global_min)
        self.global_max = tuple(global_max)
        self.global_optimum = tuple(global_optimum)

    @property
    def surfaces(self) -> list[_ObjectiveSurface]:
        return self._surfaces


class Problem(optunahub.benchmarks.ConstrainedMixin, optunahub.benchmarks.BaseProblem):
    def __init__(
        self,
        dim: int,
        seed: int = 0,
        data_type: str = "Mixed",
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        env_kwargs: dict[str, Any] = {
            "dim": dim,
            "seed": seed,
            "data_type": data_type,
            "options": options,
        }
        env_kwargs.update(kwargs)
        self._env = GPFunctionEnv(**env_kwargs)
        self._search_space = _param_config_to_space(self._env.param_config)

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._search_space.copy()

    @property
    def num_objectives(self) -> int:
        return self._env.num_objectives

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [
            optuna.study.StudyDirection.MAXIMIZE
            if surface.task.lower() == "maximize"
            else optuna.study.StudyDirection.MINIMIZE
            for surface in self._env.surfaces
        ]

    def evaluate(self, params: dict[str, Any]) -> float | tuple[float, ...]:
        ordered = {name: params[name] for name in self._search_space}
        values = self._env.evaluate(np.array([ordered], dtype=object))
        flattened = np.asarray(values, dtype=float).ravel()
        if flattened.size == 1:
            return float(flattened[0])
        return tuple(float(value) for value in flattened)

    def evaluate_with_failures(
        self, params: dict[str, Any]
    ) -> tuple[float | tuple[float, ...], bool]:
        ordered = {name: params[name] for name in self._search_space}
        values, mask = self._env.evaluate_with_failures(np.array([ordered], dtype=object))
        flattened = np.asarray(values, dtype=float).ravel()
        failure = bool(mask[0]) if mask.size > 0 else False
        if flattened.size == 1:
            return float(flattened[0]), failure
        return tuple(float(value) for value in flattened), failure

    def evaluate_constraints(self, params: dict[str, Any]) -> list[float]:
        ordered = {name: params[name] for name in self._search_space}
        constraints = self._env.evaluate_constraints(np.array([ordered], dtype=object))
        return [float(v) for v in np.asarray(constraints, dtype=float).ravel()]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
