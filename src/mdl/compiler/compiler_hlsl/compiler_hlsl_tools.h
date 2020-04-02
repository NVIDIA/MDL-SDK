/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_TOOLS_H
#define MDL_COMPILER_HLSL_TOOLS_H 1

#include <mdl/compiler/compilercore/compilercore_tools.h>

#include "compiler_hlsl_compiler.h"
#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_printers.h"

namespace mi {
namespace mdl {

template<>
inline hlsl::Compilation_unit const *impl_cast(hlsl::ICompilation_unit const *u) {
    return static_cast<hlsl::Compilation_unit const *>(u);
}

template<>
inline hlsl::Compilation_unit *impl_cast(hlsl::ICompilation_unit *u) {
    return static_cast<hlsl::Compilation_unit *>(u);
}

template<>
inline hlsl::Compiler &impl_cast(hlsl::ICompiler &u) {
    return static_cast<hlsl::Compiler &>(u);
}

template<>
inline hlsl::Compiler *impl_cast(hlsl::ICompiler *u) {
    return static_cast<hlsl::Compiler *>(u);
}

template<>
inline hlsl::Printer *impl_cast(hlsl::IPrinter *u) {
    return static_cast<hlsl::Printer *>(u);
}

}  // mdl
}  // mi

#endif
