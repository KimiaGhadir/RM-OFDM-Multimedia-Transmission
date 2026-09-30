#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 wsccg.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr
import pmt
import os
import json


def build_matrix(delay, v, power, f_c, f_s, N):
    A = np.zeros((N, N), dtype=complex)
    # 对每条路径进行处理
    for d, speed, p in zip(delay, v, power):
        # 计算多普勒频移
        # 将速度从km/h转换为m/s
        v_ms = speed * (1000 / 3600)
        # 计算多普勒频率
        f_d = v_ms * (f_c / 3e8)

        # 生成时间序列
        t_seq = np.arange(N) / f_s

        # 计算多普勒相位
        doppler_phase_seq = 2 * np.pi * f_d * t_seq

        # 生成路径响应（考虑延迟和功率）
        path_response = p * np.exp(1j * doppler_phase_seq)

        # 将路径响应添加到对应对角线
        if d == 0:
            # 主对角线
            A += np.diag(path_response)
        elif d < N:
            # 上对角线（延迟为正）
            A += np.diag(path_response[:N - d], -d)

    return A


class OAMP_Equalizer(gr.sync_block):
    def __init__(
            self,
            block_size,
            v_n,
            it,
            mod_info,
            seed,
            occupied_carriers="((),)",
            fft_len=64,
            output_is_shifted="True",
            channel_type="None",
            mod_type="RM",
            detector="OAMP"
    ):
        gr.sync_block.__init__(
            self,
            name="OAMP_Equalizer",
            in_sig=[
                (np.complex64, block_size),  # y
            ],  # x (optional)
            out_sig=[
                (np.complex64, block_size),  # u_le
                (np.complex64, block_size),  # u_nle_p
            ]
        )
        self.block_size = block_size  # 一个向量的大小
        self.v_n = v_n  # 噪声方差
        self.it = it  # 算法迭代次数
        self.mod_info = mod_info  # 调制方式
        self.fft_len = fft_len  # 原OFDMfft长度
        self.thres_0 = 1e-6
        self.occupied_carriers = occupied_carriers
        self.channel_type = channel_type
        self.mod_type = mod_type
        self.detector = str(detector).upper()
        if self.detector not in ("OAMP", "LS"):
            raise ValueError("detector must be either 'OAMP' or 'LS'")

        # ====================================================
        # Mixed fixed-point experiment
        #
        # OAMP_FP_MODE:
        #   float   -> original behavior
        #   profile -> collect dynamic ranges only
        #   fixed   -> emulate fixed-point on selected vectors
        # ====================================================
        self.fp_mode = os.environ.get(
            "OAMP_FP_MODE", "float"
        ).strip().lower()

        if self.fp_mode not in ("float", "profile", "fixed"):
            raise ValueError(
                "OAMP_FP_MODE must be float, profile, or fixed"
            )

        self.fp_word_length = int(
            os.environ.get("OAMP_FP_W", "16")
        )

        self.fp_profile_every = int(
            os.environ.get("OAMP_FP_PROFILE_EVERY", "20")
        )

        self.fp_range_file = os.environ.get(
            "OAMP_FP_RANGE_FILE",
            "/tmp/oamp_fp_ranges.json"
        )

        self.fp_config_file = os.environ.get(
            "OAMP_FP_CONFIG",
            "/tmp/oamp_fp_config.json"
        )

        self.fp_ranges = {}
        self.fp_saturation = {}
        self.fp_calls = 0

        self.fp_frac = {}
        self.fp_word_lengths = {}

        # Block floating-point support
        self.fp_bfp_names = set()
        self.fp_bfp_exp_min = {}
        self.fp_bfp_exp_max = {}

        if self.fp_mode == "fixed":
            if not os.path.exists(self.fp_config_file):
                raise RuntimeError(
                    "Fixed-point configuration not found: "
                    + self.fp_config_file
                )

            with open(self.fp_config_file, "r") as f:
                cfg = json.load(f)

            self.fp_frac = cfg["fractional_bits"]
            self.fp_word_lengths = cfg.get(
                "word_lengths", {}
            )

            self.fp_bfp_names = set(
                cfg.get("block_float_names", [])
            )

            print(
                "[OAMP_FP] fixed mode, W=",
                self.fp_word_length,
                " config=",
                self.fp_config_file,
                flush=True
            )

        # Generate permutation index from seed
        self.rng = np.random.default_rng(seed)
        if mod_type == "RM":
            self.index_ev = self.rng.permutation(self.block_size)  # 生成随机序列
        elif mod_type in ("comb", "comb_alt", "comb_swap"):
            self.index_ev = self.rng.permutation(self.block_size // 2)  # 生成随机序列
        # Setup for tag processing
        self.key_chan_taps = pmt.intern(
            "ofdm_sync_chan_taps")  # pmt.intern("ofdm_sync_chan_taps")：将字符串类型的"ofdm_sync_chan_taps"转换为pmt类型
        self.dia = np.ones(block_size, dtype=np.complex64)  # 生成全为1的长为block_size的一维向量

        # Parse raw parameters
        self.output_is_shifted = output_is_shifted

        # 对输入的占用子载波列表进行一定处理
        self.process_occupied_carriers()

        # 设置标签传播策略
        self.set_tag_propagation_policy(gr.TPP_DONT)
        # propagate tags to both outputs
        ''' 信道矩阵构建'''
        # 创建动态多径矩阵
        if mod_type in ("comb", "comb_alt", "comb_swap"):
            N = self.block_size // 2
        else:
            N = self.block_size
        f_s = 480e3  # 采样频率
        f_c = 4e9  # 载波频率
        # matrix 1
        delay_1 = [0, 10]
        v_1 = [100, 100]  # km/h
        p_1 = [1, 0.1]
        A_1 = build_matrix(delay_1, v_1, p_1, f_c, f_s, N)
        D_1 = np.linalg.svd(A_1, compute_uv=False)
        # matrix 2
        delay_2 = [0, 10]
        v_2 = [100, 100]  # km/h
        p_2 = [0.7, 0.7]
        A_2 = build_matrix(delay_2, v_2, p_2, f_c, f_s, N)
        D_2 = np.linalg.svd(A_2, compute_uv=False)
        if mod_type in ("comb", "comb_alt", "comb_swap"):
            self.D_1 = np.concatenate((D_1, D_1))
            self.D_2 = np.concatenate((D_2, D_2))
        else:
            self.D_1 = D_1
            self.D_2 = D_2

    # ========================================================
    # Fixed-point helper functions
    # ========================================================

    def _fp_track(self, name, x):
        """Track maximum absolute real/imag component."""

        if self.fp_mode == "float":
            return

        a = np.asarray(x)

        if a.size == 0:
            return

        max_abs = float(
            max(
                np.max(np.abs(np.real(a))),
                np.max(np.abs(np.imag(a)))
            )
        )

        old = self.fp_ranges.get(name, 0.0)

        if max_abs > old:
            self.fp_ranges[name] = max_abs


    def _qfix(self, name, x):
        """
        Emulate signed fixed-point quantization.

        MATLAB/Python calculations are still floating-point internally;
        this function emulates finite word length, rounding and saturation.
        """

        self._fp_track(name, x)

        if self.fp_mode != "fixed":
            return x

        if (
            name not in self.fp_frac
            and name not in self.fp_bfp_names
        ):
            # Parameter intentionally left floating-point.
            return x

        W = int(
            self.fp_word_lengths.get(
                name,
                self.fp_word_length
            )
        )

        # ====================================================
        # Block Floating-Point quantization
        # One shared power-of-two exponent per vector/block.
        # ====================================================
        if name in self.fp_bfp_names:

            arr = np.asarray(x)

            max_abs = float(
                max(
                    np.max(np.abs(np.real(arr))),
                    np.max(np.abs(np.imag(arr)))
                )
            )

            if max_abs == 0.0:
                return x

            code_min = -(2 ** (W - 1))
            code_max = (2 ** (W - 1)) - 1

            exponent = int(
                np.ceil(
                    np.log2(
                        max_abs / max(code_max, 1)
                    )
                )
            )

            step = float(2.0 ** exponent)

            qr = np.round(
                np.real(arr) / step
            )

            sat_r = (
                (qr < code_min)
                | (qr > code_max)
            )

            qr = np.clip(
                qr,
                code_min,
                code_max
            )

            if np.iscomplexobj(arr):

                qi = np.round(
                    np.imag(arr) / step
                )

                sat_i = (
                    (qi < code_min)
                    | (qi > code_max)
                )

                qi = np.clip(
                    qi,
                    code_min,
                    code_max
                )

                result = (
                    qr + 1j * qi
                ) * step

                sat_count = int(
                    np.count_nonzero(sat_r)
                    + np.count_nonzero(sat_i)
                )

            else:

                result = qr * step

                sat_count = int(
                    np.count_nonzero(sat_r)
                )

            self.fp_saturation[name] = (
                self.fp_saturation.get(name, 0)
                + sat_count
            )

            self.fp_bfp_exp_min[name] = min(
                self.fp_bfp_exp_min.get(
                    name,
                    exponent
                ),
                exponent
            )

            self.fp_bfp_exp_max[name] = max(
                self.fp_bfp_exp_max.get(
                    name,
                    exponent
                ),
                exponent
            )

            return result

        # ====================================================
        # Conventional static fixed-point quantization
        # ====================================================

        F = int(self.fp_frac[name])

        scale = float(2 ** F)

        min_val = -(2 ** (W - 1)) / scale
        max_val = ((2 ** (W - 1)) - 1) / scale

        arr = np.asarray(x)

        xr = np.round(np.real(arr) * scale) / scale

        sat_r = (xr < min_val) | (xr > max_val)

        xr = np.clip(xr, min_val, max_val)

        if np.iscomplexobj(arr):
            xi = np.round(np.imag(arr) * scale) / scale

            sat_i = (xi < min_val) | (xi > max_val)

            xi = np.clip(xi, min_val, max_val)

            result = xr + 1j * xi

            sat_count = int(
                np.count_nonzero(sat_r)
                + np.count_nonzero(sat_i)
            )

        else:
            result = xr
            sat_count = int(np.count_nonzero(sat_r))

        self.fp_saturation[name] = (
            self.fp_saturation.get(name, 0)
            + sat_count
        )

        return result


    def _fp_tick(self):
        """Periodically save/print accumulated range information."""

        if self.fp_mode == "float":
            return

        self.fp_calls += 1

        if self.fp_calls % self.fp_profile_every != 0:
            return

        payload = {
            "mode": self.fp_mode,
            "word_length": self.fp_word_length,
            "word_lengths": self.fp_word_lengths,
            "block_float_names": sorted(
                self.fp_bfp_names
            ),
            "bfp_exponent_min": self.fp_bfp_exp_min,
            "bfp_exponent_max": self.fp_bfp_exp_max,
            "calls": self.fp_calls,
            "ranges": self.fp_ranges,
            "saturation": self.fp_saturation,
        }

        try:
            with open(self.fp_range_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(
                "[OAMP_FP] could not write range file:",
                e,
                flush=True
            )

        range_text = " ".join(
            f"{k}={v:.6g}"
            for k, v in sorted(self.fp_ranges.items())
        )

        sat_total = sum(self.fp_saturation.values())

        print(
            f"[OAMP_FP] calls={self.fp_calls} "
            f"sat={sat_total} "
            + range_text,
            flush=True
        )


    def process_occupied_carriers(self):
        """Process occupied carriers configuration"""
        if not self.occupied_carriers or not any(self.occupied_carriers):
            raise ValueError("Occupied carriers must be of type list of lists,e.g. [[],]")
        # 创建处理后载波分配的副本
        self.used_carriers = [row[:] for row in self.occupied_carriers]
        for i in range(len(self.used_carriers)):  # 遍历各个载波分配列表
            for j in range(len(self.used_carriers[i])):  # 遍历列表内元素
                # 处理负索引
                if self.used_carriers[i][j] < 0:
                    self.used_carriers[i][j] += self.fft_len
                else:
                    self.used_carriers[i][j] = self.used_carriers[i][j]
                # 边界检查
                if self.used_carriers[i][j] >= self.fft_len or self.used_carriers[i][j] < 0:
                    raise ValueError(f"Carrier index {self.used_carriers[i][j]} out of bounds [0,{self.fft_len - 1}]")
                # 处理频移
                if self.output_is_shifted:
                    self.used_carriers[i][j] = (self.used_carriers[i][j] + self.fft_len // 2) % self.fft_len

    def extract_valid_subcarriers(self, tags):

        # If the channel estimate is shifted, unshift it
        # Process tags to get channel estimate
        for tag in tags:
            if pmt.eqv(tag.key, self.key_chan_taps):
                # Extract and process channel taps
                full_taps = pmt.to_python(tag.value)
                break
        valid_taps = np.array([])
        for i in range(len(self.used_carriers)):  # 遍历各个载波分配列表
            valid_taps = np.concatenate((valid_taps, np.take(full_taps, self.used_carriers[i], mode='clip')), axis=0)
        num_use = len(valid_taps)
        full_repeat = self.block_size // num_use  # 整除计算需重复的次数
        remainder = self.block_size % num_use  # 计算余数
        result = np.tile(valid_taps, full_repeat)  # tile:重复函数
        if (remainder > 0):  # 若有余数
            result = np.concatenate((result, valid_taps[:remainder]), axis=0)
        # if self.output_is_shifted:
        # full_taps = np.roll(full_taps, self.fft_len // 2)
        return result

    def A_times_x(self, x, dia, block_size):
        """A*x = DPFx.

        FFT itself is retained in floating point for the first
        mixed fixed-point implementation. Its input/output are
        quantized when fixed mode is enabled.
        """

        x = self._qfix("u", x)

        x_f = np.fft.fft(x) / np.sqrt(block_size)

        x_f = self._qfix("fft_out", x_f)

        x_f_permuted = x_f[self.index_ev]

        out = dia * x_f_permuted

        return self._qfix("A_times_x", out)


    def AH_times_x(self, x, dia, block_size):
        """A^H*x.

        IFFT remains floating point; its surrounding vectors are
        quantized.
        """

        x = self._qfix("AH_input", x)

        tmp = np.zeros(
            block_size,
            dtype=np.complex128
        )

        tmp[self.index_ev] = (
            np.conj(dia) * x
        )

        tmp = self._qfix("AH_mult", tmp)

        out = (
            np.fft.ifft(tmp)
            * np.sqrt(block_size)
        )

        return self._qfix("ifft_out", out)

    def Orth(self, u_post, v_post, u_pri, v_pri):
        """Orthogonalization step.

        Variance and divisions intentionally remain floating point.
        Only the vector result is quantized.
        """

        v_pri = max(v_pri, 1e-10)
        v_post = max(v_post, 1e-10)

        v_orth = (
            1.0
            / (1.0 / v_post - 1.0 / v_pri)
        )

        u_orth = v_orth * (
            u_post / v_post
            - u_pri / v_pri
        )

        u_orth = self._qfix(
            "u_orth",
            u_orth
        )

        return u_orth, max(v_orth, 1e-10)

    def LE_OAMP(self, u, v, dia, y, block_size, noise_var=None):
        """Mixed fixed/floating OAMP linear estimator.

        Kept floating point:
            v, v_n, rho, Dia, D, v_post

        Quantized in fixed mode:
            u, y, dia, residual, FFT/IFFT surrounding vectors,
            tmp2, tmp3 and u_post
        """

        u = self._qfix("u", u)
        y = self._qfix("y", y)
        dia = self._qfix("dia", dia)

        # Critical scalar/vector coefficients stay floating point
        if noise_var is None:
            noise_var = self.v_n

        rho = noise_var / max(v, 1e-10)

        Dia = np.abs(dia) ** 2

        D = 1.0 / (Dia + rho)

        self._fp_track("rho", rho)
        self._fp_track("D", D)

        # Residual calculation
        Ax = self.A_times_x(
            u,
            dia,
            block_size
        )

        residual = y - Ax

        residual = self._qfix(
            "residual",
            residual
        )

        tmp2 = D * residual

        tmp2 = self._qfix(
            "tmp2",
            tmp2
        )

        tmp3 = self.AH_times_x(
            tmp2,
            dia,
            block_size
        )

        tmp3 = self._qfix(
            "tmp3",
            tmp3
        )

        u_post = u + tmp3

        u_post = self._qfix(
            "u_post",
            u_post
        )

        v_post = (
            v
            - v * np.sum(Dia * D)
            / block_size
        )

        self._fp_track("v_post", v_post)

        return u_post, max(v_post, 1e-10)

    def denoiser(self, r, v, mod_type):
        """MMSE denoiser for various modulations"""
        EXP_B = 50  # Maximum exponent boundary

        if mod_type == 'BPSK':
            # Handle complex input by taking real part and adjusting variance
            if np.iscomplexobj(r):
                r = np.real(r)
                v = v / 2

            # Limit the range of d to avoid overflow
            d = -2 * r / v
            d = np.clip(d, -EXP_B, EXP_B)  # 限制数据范围在(-EXP_B, EXP_B)之间

            # Compute posterior probabilities
            p_1 = 1 / (1 + np.exp(d))
            u_p = 2 * p_1 - 1

            # Compute posterior variance
            v_p = np.mean(1 - u_p ** 2)
            return u_p.astype(complex), v_p

        elif mod_type == 'QPSK':
            # Process real and imaginary parts separately
            u1, v1 = self.denoiser_bpsk(np.sqrt(2) * np.real(r), v)
            u2, v2 = self.denoiser_bpsk(np.sqrt(2) * np.imag(r), v)

            # Combine results
            u_p = (u1 + 1j * u2) / np.sqrt(2)
            v_p = (v1 + v2) / 2
            return u_p, v_p

        elif mod_type == '16QAM':
            # Normalized 16QAM constellation points
            X = np.array([-3, -1, 1, 3]) / np.sqrt(10)
            P = np.array([0.25, 0.25, 0.25, 0.25])

            # Process real and imaginary parts separately
            u1, v1 = self.denoiser_rd(np.real(r), v / 2, X, P, EXP_B)
            u2, v2 = self.denoiser_rd(np.imag(r), v / 2, X, P, EXP_B)

            # Combine results
            u_p = u1 + 1j * u2
            v_p = v1 + v2
            return u_p, v_p

        elif mod_type == '64QAM':
            # Normalized 64QAM constellation points
            X = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
            P = np.array([0.125] * 8)

            # Process real and imaginary parts separately
            u1, v1 = self.denoiser_rd(np.real(r), v / 2, X, P, EXP_B)
            u2, v2 = self.denoiser_rd(np.imag(r), v / 2, X, P, EXP_B)

            # Combine results
            u_p = u1 + 1j * u2
            v_p = v1 + v2
            return u_p, v_p

        else:
            # Default: treat as all-zero distortion (pass-through)
            return r.copy(), v

    def denoiser_bpsk(self, r, v):
        """Helper function for BPSK denoising"""
        EXP_B = 50
        d = -2 * r / v
        d = np.clip(d, -EXP_B, EXP_B)
        p_1 = 1 / (1 + np.exp(d))
        u_p = 2 * p_1 - 1
        v_p = np.mean(1 - u_p ** 2)
        return u_p, v_p

    def denoiser_rd(self, r, v, X, P, EXP_B):
        """
        Real-discrete MMSE denoiser.
        Python implementation matched to MATLAB Demod_RD.
        """
        r = np.asarray(r, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float).reshape(-1)

        N = len(r)
        n = len(X)

        p_p = np.zeros((N, n), dtype=float)
        X2 = X ** 2

        for ii in range(n):
            x_i = X[ii]

            tmp = x_i ** 2 - X2

            d = (
                2.0 * r[:, None] * (X[None, :] - x_i)
                + tmp[None, :]
            ) / (2.0 * v)

            d = np.clip(d, -EXP_B, EXP_B)

            denominator = np.sum(
                P[None, :] * np.exp(d),
                axis=1
            )

            p_p[:, ii] = P[ii] / denominator

        u_p = np.sum(
            p_p * X[None, :],
            axis=1
        )

        v_p_vec = (
            np.sum(p_p * X2[None, :], axis=1)
            - u_p ** 2
        )

        v_p = np.mean(v_p_vec)

        return u_p, v_p
        
    def detect_rm_oamp(self, y, dia, block_size):
        """Detect RM using OAMP with MATLAB-like stopping."""
        
        # GNU Radio QPSK has RMS symbol amplitude 2,
        # while the internal OAMP QPSK model is normalized to unit power.
        # Absorb this factor into the effective channel model.
        if self.mod_info == "QPSK":
            dia = 2.0 * dia

        # ----------------------------------------------------
        # Runtime normalization for GNU Radio OAMP.
        #
        #   y'     = y / S
        #   dia'   = dia / S
        #   v_n'   = v_n / S^2
        #
        # S should preferably be a power of two so that a
        # hardware implementation can realize it using shifts.
        #
        # Kept separate from OAMP_INPUT_SCALE used by the
        # standalone benchmark, to avoid accidental double scaling.
        # ----------------------------------------------------
        fp_mode = os.environ.get(
            "OAMP_FP_MODE",
            "float"
        ).lower()

        runtime_scale = 1.0

        if fp_mode in ("profile", "fixed"):
            runtime_scale = float(
                os.environ.get(
                    "OAMP_RUNTIME_SCALE",
                    "1"
                )
            )

        if runtime_scale <= 0.0:
            raise ValueError(
                "OAMP_RUNTIME_SCALE must be positive"
            )

        if runtime_scale != 1.0:
            y = np.asarray(y) / runtime_scale
            dia = np.asarray(dia) / runtime_scale

        noise_var_le = (
            self.v_n
            / (runtime_scale * runtime_scale)
        )

        # Input/channel quantization for the mixed fixed-point path.
        y = self._qfix("y", y)
        dia = self._qfix("dia", dia)

        u_nle = np.zeros(
            block_size,
            dtype=np.complex64
        )

        u_nle = self._qfix(
            "u",
            u_nle
        )

        v_nle = 1.0

        u_le = u_nle.copy()
        u_nle_p = u_nle.copy()

        thres = 1e-7

        for i in range(self.it):

            # Linear estimator
            u_le_p, v_le_p = self.LE_OAMP(
                u_nle,
                v_nle,
                dia,
                y,
                block_size,
                noise_var=noise_var_le
            )

            # Check whether LE -> Orth would produce invalid variance
            denom_le = (1.0 / v_le_p) - (1.0 / v_nle)

            if denom_le <= 0 or not np.isfinite(denom_le):
                break

            u_le, v_le = self.Orth(
                u_le_p, v_le_p, u_nle, v_nle
            )

            # Nonlinear estimator
            u_nle_p, v_nle_p = self.denoiser(
                u_le, v_le, self.mod_info
            )

            # Nonlinear estimator itself remains floating point.
            # Its output is quantized for the next fixed-point stage.
            u_nle_p = self._qfix(
                "u_nle_p",
                u_nle_p
            )

            self._fp_track(
                "v_nle_p",
                v_nle_p
            )

            # Stopping threshold
            if v_nle_p < thres:
                break

            # Check NLE -> Orth before actually doing it
            denom_nle = (1.0 / v_nle_p) - (1.0 / v_le)

            if denom_nle <= 0 or not np.isfinite(denom_nle):
                break

            u_nle, v_nle = self.Orth(
                u_nle_p, v_nle_p, u_le, v_le
            )

        self._fp_tick()

        return (
            u_le.astype(np.complex64),
            u_nle_p.astype(np.complex64)
        )
        
    def detect_rm_ls(self, y, dia, block_size):
        """Detect the RM pre-coded part using an LS / pseudo-inverse solution.

        The RM model used by this block is
            y = diag(dia) * P * F * x + n,
        where F is the unitary FFT and P is the permutation generated from
        the shared random seed.  Since P and F are unitary, the LS solution
        can be evaluated efficiently without explicitly building the matrix.
        """
        inv_dia = np.zeros(block_size, dtype=np.complex64)
        valid = np.abs(dia) > 1e-10
        inv_dia[valid] = 1.0 / dia[valid]

        # Undo the channel first.
        freq_permuted = y * inv_dia

        # Undo the RM permutation.
        freq_unpermuted = np.zeros(block_size, dtype=np.complex64)
        freq_unpermuted[self.index_ev] = freq_permuted

        # Undo the unitary FFT used by Random_Modulation.
        x_hat = np.fft.ifft(freq_unpermuted) * np.sqrt(block_size)
        return x_hat.astype(np.complex64)

    def work(self, input_items, output_items):
        # Extract inputs
        y = input_items[0][0]  # Received signal
        
        # 获取H矩阵
        tags = self.get_tags_in_window(0, 0, 1)
        self.dia = self.extract_valid_subcarriers(tags)

        if self.channel_type == "Static_Multipath_slight":
            self.dia = self.dia * self.D_1

        elif self.channel_type == "Static_Multipath_serious":
            self.dia = self.dia * self.D_2

        else:
            self.dia = self.dia

        if self.mod_type == "RM":
            if self.detector == "OAMP":
                u_view, u_detect = self.detect_rm_oamp(y, self.dia, self.block_size)
            else:  # LS
                u_detect = self.detect_rm_ls(y, self.dia, self.block_size)
                u_view = u_detect

            output_items[0][0] = u_view
            output_items[1][0] = u_detect

        elif self.mod_type == "OFDM":
            # The coded/OFDM part keeps the existing LS one-tap detector.
            u_nle_ofdm = np.zeros(self.block_size, dtype=np.complex64)
            valid = np.abs(self.dia) > 1e-10
            u_nle_ofdm[valid] = y[valid] / self.dia[valid]
            output_items[0][0] = u_nle_ofdm
            output_items[1][0] = u_nle_ofdm

        elif self.mod_type == "comb":
            half_block_size = self.block_size // 2
            y_rm = y[:half_block_size]
            dia_rm = self.dia[:half_block_size]
            y_ofdm = y[half_block_size:]
            dia_ofdm = self.dia[half_block_size:]

            # RM pre-coded half: selectable OAMP or LS detector.
            if self.detector == "OAMP":
                rm_view, rm_detect = self.detect_rm_oamp(y_rm, dia_rm, half_block_size)
            else:  # LS
                rm_detect = self.detect_rm_ls(y_rm, dia_rm, half_block_size)
                rm_view = rm_detect

            # Coded/OFDM half: keep the existing LS detector.
            ofdm_detect = np.zeros(half_block_size, dtype=np.complex64)
            valid = np.abs(dia_ofdm) > 1e-10
            ofdm_detect[valid] = y_ofdm[valid] / dia_ofdm[valid]

            output_items[0][0] = np.concatenate((rm_view, ofdm_detect))
            output_items[1][0] = np.concatenate((rm_detect, ofdm_detect))

        # Propagate all tags to output
        # self.propagate_tags(0, 0, len(output_items[0]), tags)
        
        elif self.mod_type == "comb_alt":
            half_block_size = self.block_size // 2

            # Undo alternating arrangement
            y_rm = y[0::2]
            y_ofdm = y[1::2]

            dia_rm = self.dia[0::2]
            dia_ofdm = self.dia[1::2]

            # RM part: OAMP or LS
            if self.detector == "OAMP":
                rm_view, rm_detect = self.detect_rm_oamp(
                    y_rm, dia_rm, half_block_size
                )
            else:
                rm_detect = self.detect_rm_ls(
                    y_rm, dia_rm, half_block_size
                )
                rm_view = rm_detect

            # OFDM part: LS
            ofdm_detect = np.zeros(
                half_block_size, dtype=np.complex64
            )
            valid = np.abs(dia_ofdm) > 1e-10
            ofdm_detect[valid] = (
                y_ofdm[valid] / dia_ofdm[valid]
            )

            # Restore logical [RM][OFDM] order
            output_items[0][0] = np.concatenate(
                (rm_view, ofdm_detect)
            )
            output_items[1][0] = np.concatenate(
                (rm_detect, ofdm_detect)
            )
        
        elif self.mod_type == "comb_swap":
            half_block_size = self.block_size // 2

            # Physical layout in comb_swap is:
            # [OFDM][RM]
            y_ofdm = y[:half_block_size]
            dia_ofdm = self.dia[:half_block_size]

            y_rm = y[half_block_size:]
            dia_rm = self.dia[half_block_size:]

            # RM detection is exactly the same as in comb
            if self.detector == "OAMP":
                rm_view, rm_detect = self.detect_rm_oamp(
                    y_rm, dia_rm, half_block_size
                )
            else:
                rm_detect = self.detect_rm_ls(
                    y_rm, dia_rm, half_block_size
                )
                rm_view = rm_detect

            # OFDM still uses LS
            ofdm_detect = np.zeros(
                half_block_size, dtype=np.complex64
            )
            valid = np.abs(dia_ofdm) > 1e-10
            ofdm_detect[valid] = (
                y_ofdm[valid] / dia_ofdm[valid]
            )

            # Important:
            # restore the original logical order [RM][OFDM]
            # before sending data to decoder / BER path.
            output_items[0][0] = np.concatenate(
                (rm_view, ofdm_detect)
            )
            output_items[1][0] = np.concatenate(
                (rm_detect, ofdm_detect)
            )
            
        # Manually forward input tags to output 1 (BER/decoder path)
        for tag in tags:
            self.add_item_tag(1, tag.offset, tag.key, tag.value)

        return 1
