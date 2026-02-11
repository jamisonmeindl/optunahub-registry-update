from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class SyntheticAugmenter:
    def __init__(
        self,
        base_env: Any,
        sample_for_scale: int = 256,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.base = base_env
        self.dim = base_env.dim
        self.seed = base_env.seed
        self.user_seed = getattr(base_env, "user_seed", self.seed)
        self.data_type = getattr(base_env, "data_type", "Mixed")
        self.param_config = base_env.param_config
        self.task = base_env.task
        self.title = f"{getattr(base_env, 'title', base_env.name())} (augmented)"
        self.description = getattr(base_env, "description", "")
        self.eval_info = getattr(
            base_env,
            "eval_info",
            {"name": "Objective Value", "description": "Synthetic GP augmentation", "unit": "unknown"},
        )
        self.eval_range = getattr(base_env, "eval_range", None)
        self.rng = getattr(base_env, "augmentation_rng", base_env.rng)
        self.use_post_scaling = getattr(base_env, "use_post_scaling", True)
        self.N = max(
            1,
            sum(
                1 if v["type"] in ("Integer", "Continuous") else len(v["categories"])
                for v in self.param_config.values()
            ),
        )
        self.sample_for_scale = sample_for_scale
        self.base_scale = self.rng.uniform(0.1, 2.0) * self._estimate_scale(sample_for_scale)

        self.K = max(3, min(12, 2 + self.N))
        self._init_shared_warp()

        opts = options or {}
        self._choose_augments(opts)
        self._init_augment_params(opts)
        self._init_range(opts)

    def _estimate_scale(self, n: int) -> float:
        Xs = self.base.sample(n)
        ys = self.base._evaluate_baseline(Xs).astype(float).ravel()
        std = float(np.std(ys))
        rng = float(np.ptp(ys))
        return max(std, rng / 6.0)

    def _init_shared_warp(self) -> None:
        def rand_unit(d: int) -> np.ndarray:
            v = self.rng.normal(0, 1, size=(d,))
            norm = np.linalg.norm(v)
            return v if norm < 1e-12 else v / norm

        self.W = np.stack([rand_unit(self.N) for _ in range(self.K)], axis=0)
        self.bias = self.rng.uniform(-0.8, 0.8, size=(self.K,))
        self.warp_A = self.rng.normal(0, 0.8, size=(self.K, self.K))
        u, s, vT = np.linalg.svd(self.warp_A, full_matrices=False)
        s = np.clip(s, 0.3, 1.0)
        self.warp_A = (u * s) @ vT
        self.warp_b = self.rng.uniform(-0.5, 0.5, size=(self.K,))

    def _choose_augments(self, options: dict[str, Any]) -> None:
        all_aug_names = [
            "gate",
            "steps",
            "plate",
            "mosaic",
            "hard_spots",
            "axis_steps",
            "voronoi",
            "kinks",
            "microtex",
            "funnel",
            "polymix",
            "bimlp",
            "rat",
            "parity",
            "tube",
            "moe",
            "shear",
            "global_drift",
            "global_bowl",
        ]
        kmin, kmax = map(int, options.get("aug_k", (1, 2)))
        if options.get("aug_names"):
            chosen = [n for n in options["aug_names"] if n in all_aug_names]
        else:
            count = int(self.rng.integers(kmin, kmax + 1))
            count = min(max(1, count), len(all_aug_names))
            chosen = list(self.rng.choice(all_aug_names, size=count, replace=False))
        self._chosen = set(chosen)
        for name in all_aug_names:
            setattr(self, f"use_{name}", False)
        for name in chosen:
            setattr(self, f"use_{name}", True)

    def _init_augment_params(self, options: dict[str, Any]) -> None:
        def rand_unit(d: int) -> np.ndarray:
            v = self.rng.normal(0, 1, size=(d,))
            norm = np.linalg.norm(v)
            return v if norm < 1e-12 else v / norm

        if self.use_gate:
            self.gate_idx = int(self.rng.integers(0, self.K))
            self.gate_tau = float(self.rng.uniform(-0.5, 0.5))
            self.gate_level = float(self.base_scale * self.rng.uniform(-20.0, 20.0))

        if self.use_steps:
            self.stairs_count = int(self.rng.integers(1, min(5, self.K) + 1))
            stair_sel = self.rng.choice(self.K, size=self.stairs_count, replace=False)
            self.stair_idx = stair_sel
            self.stair_tau = self.rng.uniform(-0.7, 0.7, size=self.stairs_count)
            self.stair_jump = (
                self.base_scale
                * self.rng.uniform(0.2, 1.2, size=self.stairs_count)
                * (2 * (self.rng.random(self.stairs_count) < 0.5) - 1)
            )

        if self.use_plate:
            self.plate_count = int(self.rng.integers(1, min(3, self.K) + 1))
            plate_sel = self.rng.choice(self.K, size=self.plate_count, replace=False)
            self.plate_idx = plate_sel
            self.plate_bins = self.rng.integers(3, 9, size=self.plate_count)
            self.plate_gain = (
                self.base_scale
                * self.rng.uniform(0.3, 2.0, size=self.plate_count)
                * (2 * (self.rng.random(self.plate_count) < 0.5) - 1)
            )

        if self.use_mosaic:
            self.mosaic_axes = self.rng.choice(
                self.K, size=int(self.rng.integers(2, min(4, self.K) + 1)), replace=False
            )
            self.mosaic_bins = self.rng.integers(4, 12, size=self.mosaic_axes.shape[0])
            self.mosaic_table_size = int(self.rng.integers(16, 129))
            self.mosaic_levels = self.base_scale * self.rng.uniform(-0.6, 0.6, size=(self.mosaic_table_size,))
            self.mosaic_hash = self.rng.integers(3, 97, size=self.mosaic_axes.shape[0])

        if self.use_hard_spots:
            self.spot_K = int(self.rng.integers(2, max(3, self.N + 2)))
            self.spot_centers = self.rng.uniform(-0.8, 0.8, size=(self.spot_K, self.N))
            self.spot_shape = np.exp(
                self.rng.uniform(np.log(0.4), np.log(2.2), size=(self.spot_K, self.N))
            )
            self.spot_r2 = np.exp(self.rng.uniform(np.log(0.01), np.log(0.06), size=(self.spot_K,)))
            self.spot_height = self.base_scale * self.rng.uniform(-20.0, 20.0, size=(self.spot_K,))
            self.spot_base = self.base_scale * self.rng.uniform(-2.0, 2.0)

        if self.use_axis_steps:
            self.axis_steps_count = int(self.rng.integers(1, min(4, self.N) + 1))
            self.axis_steps_axes = self.rng.choice(self.N, size=self.axis_steps_count, replace=False)
            self.axis_steps_tau = self.rng.uniform(-0.6, 0.6, size=self.axis_steps_count)
            self.axis_steps_jump = (
                self.base_scale
                * self.rng.uniform(0.4, 1.6, size=self.axis_steps_count)
                * (2 * (self.rng.random(self.axis_steps_count) < 0.5) - 1)
            )

        if self.use_voronoi:
            self.voronoi_K = int(self.rng.integers(3, 9))
            self.voronoi_centers = self.rng.uniform(-1.0, 1.0, size=(self.voronoi_K, self.N))
            self.voronoi_levels = self.base_scale * self.rng.uniform(-2.0, 2.0, size=(self.voronoi_K,))

        if self.use_kinks:
            self.kinks_count = int(self.rng.integers(1, min(4, self.K) + 1))
            kink_sel = self.rng.choice(self.K, size=self.kinks_count, replace=False)
            self.kink_idx = kink_sel
            self.kink_tau = self.rng.uniform(-0.5, 0.5, size=self.kinks_count)
            self.kink_gain = (
                self.base_scale
                * self.rng.uniform(0.5, 1.0, size=self.kinks_count)
                * (2 * (self.rng.random(self.kinks_count) < 0.5) - 1)
            )
            self.kink_gamma = self.rng.uniform(1.0, 2.5, size=self.kinks_count)

        if self.use_microtex:
            a = rand_unit(self.K)
            b = rand_unit(self.K)
            b = b - a * (a @ b)
            b /= (np.linalg.norm(b) + 1e-12)
            self.tex_A = np.stack([a, b], axis=1)
            self.tex_freqs = self.rng.uniform(6, 18, size=(2,))
            self.tex_amp = 0.5 * self.base_scale * self.rng.uniform(0.5, 5.0)
            self.tex_win_axis = int(self.rng.integers(0, self.K))
            t0, t1 = np.sort(self.rng.uniform(-0.7, 0.7, size=(2,)))
            self.tex_t0, self.tex_t1 = float(t0), float(t1)
            self.tex_win_sharp = self.rng.uniform(10.0, 25.0)

        if self.use_funnel:
            self.funnel_axis = rand_unit(self.N)
            self.funnel_center = self.rng.uniform(-0.3, 0.3)
            self.funnel_width0 = self.rng.uniform(0.15, 0.6)
            self.funnel_width1 = self.rng.uniform(self.funnel_width0, self.funnel_width0 * 4.0)
            self.funnel_gain = self.base_scale * self.rng.uniform(0.25, 0.9)

        if self.use_polymix:
            self.pm_rank_q = int(self.rng.integers(1, min(3, self.N) + 1))
            self.pm_rank_c = int(self.rng.integers(0, 2))
            self.pm_u = np.stack([rand_unit(self.N) for _ in range(self.pm_rank_q)], axis=0)
            self.pm_v = np.stack([rand_unit(self.N) for _ in range(self.pm_rank_q)], axis=0)
            self.pm_beta = self.rng.uniform(0.6, 2.0, size=(self.pm_rank_q,))
            self.pm_wq = self.base_scale * self.rng.uniform(-1.0, 1.0, size=(self.pm_rank_q,))
            if self.pm_rank_c > 0:
                self.pm_a = rand_unit(self.N)
                self.pm_b = rand_unit(self.N)
                self.pm_c = rand_unit(self.N)
                self.pm_wc = self.base_scale * self.rng.uniform(-0.8, 0.8)

        if self.use_bimlp:
            r = int(self.rng.integers(2, min(6, self.N) + 1))
            h = int(self.rng.integers(4, 9))
            self.bu = self.rng.normal(0, 1 / np.sqrt(self.N), size=(self.N, r))
            self.bv = self.rng.normal(0, 1 / np.sqrt(self.N), size=(self.N, r))
            self.bW1 = self.rng.normal(0, 1 / np.sqrt(2 * r), size=(h, 2 * r))
            self.bW2 = self.rng.normal(0, 1 / np.sqrt(h), size=(1, h))
            self.bgain = self.base_scale * self.rng.uniform(0.2, 0.8)

        if self.use_rat:
            self.rc_u = rand_unit(self.N)
            self.rc_v = rand_unit(self.N)
            self.rc_eps = 1e-3 * np.exp(self.rng.uniform(-4.0, 1.5))
            self.rc_beta = self.rng.uniform(0.5, 2.5)
            self.rc_gain = self.base_scale * self.rng.uniform(-1.0, 1.0)

        if self.use_parity:
            m = int(self.rng.integers(2, min(6, self.N) + 1))
            self.pw = np.stack([rand_unit(self.N) for _ in range(m)], axis=0)
            self.pfreq = self.rng.uniform(1.0, 5.0, size=(m,))
            self.pphase = self.rng.uniform(-np.pi, np.pi, size=(m,))
            self.pgain = self.base_scale * self.rng.uniform(-0.6, 0.6)

        if self.use_tube:
            u1 = rand_unit(self.N)
            v_raw = rand_unit(self.N)
            v_ortho = v_raw - u1 * (u1 @ v_raw)
            n2 = np.linalg.norm(v_ortho)
            if n2 < 1e-12:
                axis = int(np.argmin(np.abs(u1)))
                e = np.zeros(self.N, dtype=float)
                e[axis] = 1.0
                v_ortho = e - u1 * (u1 @ e)
                n2 = np.linalg.norm(v_ortho) + 1e-12
            u2 = v_ortho / n2
            self.tube_A = np.stack([u1, u2], axis=1)
            self.tube_w = self.rng.uniform(0.5, 2.0, size=(2,))
            self.tube_phi = self.rng.uniform(-np.pi, np.pi, size=(2,))
            self.tube_R = self.rng.uniform(0.2, 0.8)
            self.tube_width = self.rng.uniform(0.05, 0.25)
            self.tube_gain = self.base_scale * self.rng.uniform(0.2, 1.2)

        if self.use_moe:
            self.moe_M = int(self.rng.integers(2, 5))
            self.moe_gate_W = self.rng.normal(0, 1 / np.sqrt(self.K), size=(self.moe_M, self.K))
            self.moe_gate_b = self.rng.uniform(-0.5, 0.5, size=(self.moe_M,))
            self.moe_kind = self.rng.integers(0, 3, size=(self.moe_M,))
            self.moe_V = self.rng.normal(0, 1 / np.sqrt(self.K), size=(self.moe_M, self.K))
            self.moe_alpha = self.rng.uniform(0.5, 2.0, size=(self.moe_M,))
            self.moe_amp = self.base_scale * self.rng.uniform(0.2, 0.8, size=(self.moe_M,))

        if self.use_shear:
            self.sw_a = rand_unit(self.N)
            self.sw_b = rand_unit(self.N)
            self.sw_c = rand_unit(self.N)
            self.sw_gamma = self.rng.uniform(0.8, 2.5)
            self.sw_alpha = self.rng.uniform(0.1, 0.6)

        if self.use_global_drift:
            self.gd_dir = self.rng.normal(0, 1, size=(self.N,))
            self.gd_dir /= (np.linalg.norm(self.gd_dir) + 1e-12)
            self.gd_bias = float(self.base_scale * self.rng.uniform(-0.5, 0.5))
            self.gd_lin = float(self.base_scale * self.rng.uniform(-10.0, 10.0))
            self.gd_beta = float(self.rng.uniform(0.8, 2.5))
            self.gd_c = float(self.rng.uniform(-0.2, 0.2))

        if self.use_global_bowl:
            self.gb_center = self.rng.uniform(-0.2, 0.2, size=(self.N,))
            R, _ = np.linalg.qr(self.rng.normal(size=(self.N, self.N)))
            eig = np.exp(self.rng.uniform(np.log(0.05), np.log(0.5), size=(self.N,)))
            self.gb_M = (R * eig) @ R.T
            self.gb_gain = float(self.base_scale * self.rng.uniform(-10.0, 10.0))

    def _init_range(self, options: dict[str, Any]) -> None:
        self.eval_range = getattr(self.base, "eval_range", None)
        self._range_scale = 1.0
        self._range_offset = 0.0
        if self.eval_range is not None and len(self.eval_range) == 2:
            lo, hi = float(self.eval_range[0]), float(self.eval_range[1])
            Xs = self.base.sample(self.sample_for_scale)
            ys = self._evaluate_raw(Xs)
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))
            denom = y_max - y_min
            if denom == 0.0:
                self._range_scale = 1.0
                self._range_offset = 0.5 * (lo + hi)
            else:
                self._range_scale = (hi - lo) / denom
                self._range_offset = lo - self._range_scale * y_min
            self._range_clip_low = min(lo, hi)
            self._range_clip_high = max(lo, hi)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = self._evaluate_raw(x)
        if self.eval_range is not None and len(self.eval_range) == 2:
            y = y * self._range_scale + self._range_offset
            y = np.clip(y, self._range_clip_low, self._range_clip_high)
        return y

    def _evaluate_raw(self, x: np.ndarray) -> np.ndarray:
        y_base = np.asarray(self.base._evaluate_baseline(x), dtype=float)
        if y_base.ndim == 1:
            y_base = y_base.reshape(-1, 1)
        y = y_base[:, 0].copy()
        if self.N == 0:
            return y_base
        Z = self.base.encode(x).astype(float)
        P0 = Z @ self.W.T + self.bias
        Pw = np.tanh(P0 @ self.warp_A.T + self.warp_b)

        if self.use_gate:
            s = Pw[:, self.gate_idx]
            mask = s >= self.gate_tau
            alt = np.full_like(y, self.gate_level, dtype=float)
            y = np.where(mask, alt, y)

        if self.use_steps:
            for k, tau, jump in zip(self.stair_idx, self.stair_tau, self.stair_jump):
                s = Pw[:, k] - tau
                H = (s >= 0.0).astype(float)
                y += jump * H

        if self.use_plate:
            for k, bins, gain in zip(self.plate_idx, self.plate_bins, self.plate_gain):
                xk = Pw[:, k]
                idx = np.floor((np.clip(xk, -1.0, 1.0) + 1.0) * 0.5 * int(bins)).astype(int)
                idx = np.clip(idx, 0, int(bins) - 1)
                centers = np.linspace(-1.0, 1.0, int(bins))
                snap = centers[idx]
                y += gain * snap

        if self.use_mosaic:
            codes = np.zeros(Pw.shape[0], dtype=np.int64)
            for ax, bins, h in zip(self.mosaic_axes, self.mosaic_bins, self.mosaic_hash):
                xk = Pw[:, ax]
                idx = np.floor((np.clip(xk, -1.0, 1.0) + 1.0) * 0.5 * int(bins)).astype(np.int64)
                idx = np.clip(idx, 0, int(bins) - 1)
                codes = codes + int(h) * idx
            j = (codes % int(self.mosaic_table_size)).astype(np.int64)
            y += self.mosaic_levels[j]

        if self.use_hard_spots:
            DZ = Z[:, None, :] - self.spot_centers[None, :, :]
            d2 = np.sum(self.spot_shape[None, :, :] * DZ**2, axis=2)
            mask = (d2 <= self.spot_r2[None, :]).astype(float)
            y += self.spot_base + (mask * self.spot_height[None, :]).sum(axis=1)

        if self.use_axis_steps:
            for ax, tau, jump in zip(self.axis_steps_axes, self.axis_steps_tau, self.axis_steps_jump):
                s = Z[:, ax] - tau
                H = (s >= 0.0).astype(float)
                y += jump * H

        if self.use_voronoi:
            DZ = Z[:, None, :] - self.voronoi_centers[None, :, :]
            d2 = np.sum(DZ * DZ, axis=2)
            idx = np.argmin(d2, axis=1)
            y += self.voronoi_levels[idx]

        if self.use_kinks:
            for k, tau, gain, gamma in zip(self.kink_idx, self.kink_tau, self.kink_gain, self.kink_gamma):
                u = Pw[:, k] - tau
                odd_bend = np.sign(u) * (np.abs(u) ** gamma)
                hinge = np.maximum(u, 0.0)
                y += gain * (0.4 * odd_bend + 0.6 * hinge)

        if self.use_microtex:
            U2 = Pw @ self.tex_A
            wave = np.sin(self.tex_freqs[0] * U2[:, 0]) * np.cos(self.tex_freqs[1] * U2[:, 1])
            t = Pw[:, self.tex_win_axis]
            win = _sigmoid_stable(self.tex_win_sharp * (t - self.tex_t0)) * _sigmoid_stable(
                self.tex_win_sharp * (self.tex_t1 - t)
            )
            y += self.tex_amp * win * wave

        if self.use_funnel:
            t = Z @ self.funnel_axis - self.funnel_center
            u = 0.5 * (1.0 + np.tanh(t))
            width = self.funnel_width0 + (self.funnel_width1 - self.funnel_width0) * u
            tproj = np.outer(t, self.funnel_axis)
            orth = Z - tproj
            orth2 = np.sum(orth * orth, axis=1)
            y += self.funnel_gain * (orth2 / (width**2 + 1e-9))

        if self.use_polymix:
            if self.pm_rank_q > 0:
                Uz = Z @ self.pm_u.T
                Vz = Z @ self.pm_v.T
                quad = np.tanh(self.pm_beta[None, :] * (Uz * Vz))
                y += (quad * self.pm_wq[None, :]).sum(axis=1)
            if self.pm_rank_c > 0:
                a = Z @ self.pm_a
                b = Z @ self.pm_b
                c = Z @ self.pm_c
                y += self.pm_wc * (a * b * c)

        if self.use_bimlp:
            U = Z @ self.bu
            V = Z @ self.bv
            BIL = U * V
            H = np.tanh((BIL @ self.bW1[:, :BIL.shape[1]].T) + (np.concatenate([U, V], axis=1) @ self.bW1.T))
            f = (H @ self.bW2.T).ravel()
            y += self.bgain * np.tanh(f)

        if self.use_rat:
            u = Z @ self.rc_u
            v = Z @ self.rc_v
            y += self.rc_gain * np.tanh(self.rc_beta * ((u * u) / (self.rc_eps + v * v)))

        if self.use_parity:
            S = np.sign(np.sin(self.pfreq[None, :] * (Z @ self.pw.T) + self.pphase[None, :]))
            parity = np.prod(S, axis=1)
            y += self.pgain * np.tanh(parity)

        if self.use_tube:
            U2 = Z @ self.tube_A
            theta = np.arctan2(U2[:, 1], U2[:, 0] + 1e-12)
            R = self.tube_R * (
                1
                + 0.2 * np.sin(self.tube_w[0] * theta + self.tube_phi[0])
                + 0.2 * np.cos(self.tube_w[1] * theta + self.tube_phi[1])
            )
            r = np.sqrt(U2[:, 0] ** 2 + U2[:, 1] ** 2) + 1e-12
            dist = np.abs(r - R)
            tube = np.exp(-0.5 * (dist / (self.tube_width + 1e-9)) ** 2)
            y += self.tube_gain * tube

        if self.use_moe:
            G = Pw @ self.moe_gate_W.T + self.moe_gate_b[None, :]
            Wg = self._softmax(G, axis=1)
            Vp = Pw @ self.moe_V.T
            outs = np.zeros_like(Wg)
            for m in range(self.moe_M):
                kind = int(self.moe_kind[m])
                if kind == 0:
                    outs[:, m] = np.sin(self.moe_alpha[m] * Vp[:, m])
                elif kind == 1:
                    outs[:, m] = np.tanh(self.moe_alpha[m] * (Vp[:, m] ** 2 - 0.5))
                else:
                    outs[:, m] = np.exp(-0.5 * (self.moe_alpha[m] * Vp[:, m]) ** 2)
            y += (Wg * (self.moe_amp[None, :] * outs)).sum(axis=1)

        if self.use_shear:
            a = Z @ self.sw_a
            b = Z @ self.sw_b
            c = Z @ self.sw_c
            amp = self.sw_alpha * np.tanh(self.sw_gamma * a)
            y += self.base_scale * amp * (b * c)

        if getattr(self, "use_global_drift", False):
            t = Z @ self.gd_dir
            y += self.gd_bias + self.gd_lin * t + 0.5 * self.gd_lin * np.tanh(self.gd_beta * (t - self.gd_c))

        if getattr(self, "use_global_bowl", False):
            DZ = Z - self.gb_center[None, :]
            bowl = 0.5 * np.einsum("bi,ij,bj->b", DZ, self.gb_M, DZ, optimize=True)
            y += self.gb_gain * bowl

        y_out = y_base.copy()
        y_out[:, 0] = y
        return y_out

    def _softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        X = X - np.max(X, axis=axis, keepdims=True)
        e = np.exp(X)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)
