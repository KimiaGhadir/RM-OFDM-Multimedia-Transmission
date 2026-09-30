#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 wsccg.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
import pmt
from gnuradio import gr


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


class Random_Modulation(gr.sync_block):
    """
    Vectorized Random Modulation block
    Processes entire packets at once using vector input/output
    """

    def __init__(self, header_length=0,
                 total_length=0, seed=0,
                 len_tag_key="packet_len",
                 channel_type="None",
                 mod_type="RM"):
        # 确保向量长度是整数
        total_length = int(total_length)  # 表示每一个向量的长度

        gr.sync_block.__init__(
            self,
            name="Random_Modulation",
            in_sig=[(np.complex64, total_length)],  # 向量输入
            out_sig=[(np.complex64, total_length)]  # 向量输出
        )
        self.header_length = header_length
        self.total_length = total_length
        self.payload_len = total_length - header_length
        self.seed = seed
        self.len_tag_key = len_tag_key
        self.channel_type = channel_type
        self.mod_type = mod_type
        self.rng = np.random.default_rng(seed)  # 基于随机数种子seed构建的随机数生成器
        if mod_type == "RM":
            self.indices = self.rng.permutation(self.payload_len)
        elif mod_type in ("comb", "comb_alt", "comb_swap"):
            self.indices = self.rng.permutation(self.payload_len // 2)
        # 设置标签传播策略
        self.set_tag_propagation_policy(gr.TPP_DONT)  # 不自动传播标签
        self.tag_key_pmt = pmt.intern(len_tag_key) if len_tag_key else pmt.PMT_NIL  # 将标签类型转换为pmt符号类型
        ''' 信道矩阵构建'''
        # 创建动态多径矩阵
        if mod_type in ("comb", "comb_alt", "comb_swap"):
            N = self.payload_len // 2
        else:
            N = self.payload_len
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

    def work(self, input_items, output_items):
        in0 = input_items[0]  # 输入向量
        out0 = output_items[0]  # 输出向量

        # 处理每个输入向量（每个向量是一个完整的数据包）
        for i in range(len(in0)):  # len(in0)表示输入的向量个数
            packet = in0[i]  # 获取第i个向量
            # 验证头部长度
            if self.header_length > self.total_length:
                print(f"错误: 头部长度({self.header_length})大于向量长度({self.total_length})")
                out0[i][:] = packet
                continue
            header = packet[:self.header_length]
            payload = packet[self.header_length:]
            if self.mod_type == "RM":
                # 对载荷进行FFT
                fft_result = np.fft.fft(payload) / np.sqrt(self.payload_len)

                # 生成随机索引并打乱

                shuffled = fft_result[self.indices]

                # 组合头部和打乱后的载荷
                # processed = np.concatenate((header, shuffled))
                # 复制到输出
                data_part = shuffled
            elif self.mod_type == "OFDM":
                data_part = payload
            elif self.mod_type == "comb":
                # 提取头部和载荷
                half_len = self.payload_len // 2
                payload_rm = payload[:half_len]
                payload_ofdm = payload[half_len:]
                fft_result = np.fft.fft(payload_rm) / np.sqrt(half_len)
                shuffled = fft_result[self.indices]
                # 复制到输出
                data_part = np.concatenate((shuffled, payload_ofdm))
            elif self.mod_type == "comb_alt":
                half_len = self.payload_len // 2

                payload_rm = payload[:half_len]
                payload_ofdm = payload[half_len:]

                fft_result = np.fft.fft(payload_rm) / np.sqrt(half_len)
                shuffled = fft_result[self.indices]

                data_part = np.empty(self.payload_len, dtype=np.complex64)
                data_part[0::2] = shuffled
                data_part[1::2] = payload_ofdm
            
            elif self.mod_type == "comb_swap":
                half_len = self.payload_len // 2

                payload_rm = payload[:half_len]
                payload_ofdm = payload[half_len:]

                fft_result = np.fft.fft(payload_rm) / np.sqrt(half_len)
                shuffled = fft_result[self.indices]

                data_part = np.concatenate((payload_ofdm, shuffled))
            else:
                data_part = payload
            '''add channel effect'''
            if self.channel_type == "Static_Multipath_slight":
                data_part = data_part * self.D_1
            elif self.channel_type == "Static_Multipath_serious":
                data_part = data_part * self.D_2
            else:
                data_part = data_part

            out0[i][:] = np.concatenate((header, data_part))
            # 添加输出标签
            tag_value = pmt.from_long(self.total_length)  # 将total_length转换为pmt的long型数据
            self.add_item_tag(
                0,  # 表明添加标签的输出端口号
                self.nitems_written(0) + i,  # 添加标签的绝对位置，nitems_written(0)表示已经输出的最后一个变量的绝对位置
                self.tag_key_pmt,  # 标签名称
                tag_value  # 标签对应的值
            )

        return len(in0)
