/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"


#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>

#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/compiler/compilercore/compilercore_assert.h"
#include "mdl/compiler/compilercore/compilercore_errors.h"

#include "generator_jit_llvm.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 8050)
 #include "glue_libdevice_20.h"
 #include "glue_libdevice_30.h"
 #include "glue_libdevice_35.h"
#else // unified since CUDA 9.0
 #include "glue_libdevice.h"
#endif

namespace mi {
namespace mdl {

// Get libdevice as LLVM bitcode.
unsigned char const *LLVM_code_generator::get_libdevice(
    size_t    &size,
    unsigned  &min_ptx_version)
{
#if defined(CUDA_VERSION) && (CUDA_VERSION < 8050)
    // no special restriction
    min_ptx_version = 0;
    // select the right libdevice version, see
    // http://docs.nvidia.com/cuda/libdevice-users-guide/basic-usage.html#linking-with-libdevice
    if (arch > 37) {
        size = dimension_of(glue_bitcode_30);
        return glue_bitcode_30;
    } else if (arch >= 35) {
        size = dimension_of(glue_bitcode_35);
        return glue_bitcode_35;
    } else if (arch >= 31) {
        size = dimension_of(glue_bitcode_20);
        return glue_bitcode_20;
    } else if (arch == 30) {
        size = dimension_of(glue_bitcode_30);
        return glue_bitcode_30;
    } else if (arch >= 20) {
        size = dimension_of(glue_bitcode_20);
        return glue_bitcode_20;
    } else {
        MDL_ASSERT(!"Unsupported architecture for libdevice");
        return NULL;
    }
#else
    // unified since CUDA 9.0
    size            = dimension_of(glue_bitcode);
    // the CUDA 9.0 libdevice contains inline asm that requires PTX 4.0
    min_ptx_version = 40;
    return glue_bitcode;
#endif
}

// Load libdevice.
std::unique_ptr<llvm::Module> LLVM_code_generator::load_libdevice(
    llvm::LLVMContext &llvm_context,
    unsigned          &min_ptx_version)
{
    size_t size = 0;

    unsigned char const *data = get_libdevice(size, min_ptx_version);

    std::unique_ptr<llvm::MemoryBuffer> mem(llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef((char const *)data, size),
        "libdevice",
        /*RequiresNullTerminator=*/ false));
    auto module = llvm::parseBitcodeFile(*mem.get(), llvm_context);
    if (!module) {
        error(PARSING_LIBDEVICE_MODULE_FAILED, Error_params(get_allocator()));
        MDL_ASSERT(!"Parsing libdevice failed");
        return nullptr;
    }
    return std::move(module.get());
}

}  // mdl
}  // mi

