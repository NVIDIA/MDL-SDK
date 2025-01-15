/******************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#ifndef MDL_COMPILER_STDMODULE_ENUMS_H
#define MDL_COMPILER_STDMODULE_ENUMS_H

// This file replicates the enums from the standard modules for use in C++ code.

namespace mi {
namespace mdl {

namespace df {

enum scatter_mode {
    scatter_reflect,
    scatter_transmit,
    scatter_reflect_transmit
};

}

namespace state {

enum coordinate_space {
    coordinate_internal,
    coordinate_object,
    coordinate_world
};

}

namespace tex {

enum gamma_mode {
    gamma_default,
    gamma_linear,
    gamma_srgb
};

enum wrap_mode {
    wrap_clamp,
    wrap_repeat,
    wrap_mirrored_repeat,
    wrap_clip
};

}

} // namespace mdl
} // namespace mi

#endif // MDL_COMPILER_STDMODULE_ENUMS_H
