#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 wsccg.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class data_separation(gr.sync_block):
    """
    docstring for block data_separation
    """
    def __init__(self, vector_length,mod_type="RM"):
        gr.sync_block.__init__(self,
            name="data_separation",
            in_sig=[(np.complex64, vector_length)],
            out_sig=[(np.complex64, vector_length),
                     (np.complex64, vector_length)
                     ])
        self.vector_length = vector_length
        self.mod_type = mod_type

    def work(self, input_items, output_items):
        in0 = input_items[0][0]
        if self.mod_type=="RM":
           output_items[0][0] = in0
           output_items[1][0] = np.zeros(self.vector_length)
        elif self.mod_type=="OFDM":
           output_items[0][0] = np.zeros(self.vector_length)
           output_items[1][0] = in0
        elif self.mod_type in ("comb", "comb_alt", "comb_swap"):
           output_items[0][0] = np.concatenate((in0[:self.vector_length//2],in0[:self.vector_length//2]))
           output_items[1][0] = np.concatenate((in0[self.vector_length//2:],in0[self.vector_length//2:]))
        return 1
