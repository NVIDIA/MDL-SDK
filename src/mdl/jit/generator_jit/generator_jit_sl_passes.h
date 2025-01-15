/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_SL_PASSES_H
#define MDL_GENERATOR_JIT_SL_PASSES_H 1

namespace mi {
namespace mdl {
class LLVM_code_generator;
} // mdl
} // mi

namespace llvm {
class Pass;

namespace sl {

/// Creates a pass that removes any pointer selects.
/// Basically, it transforms
///
///   *(cond ? ptr1 : ptr2)
///
/// into
///
///   cond ? *ptr1 : *ptr2
///
/// This operation is safe in general, because the single Load/Store
/// is replaced by an if.
llvm::Pass *createHandlePointerSelectsPass();

/// Creates a pass that removes any memory PHIs where all input values are the same or undefined.
llvm::Pass *createRemovePointerPHIsPass();

/// Creates a pass that replaces some constructs that are not directly supported by GLSL,
/// in particular and and or on boolean vector.
llvm::Pass *createRemoveBoolVectorOPsPass(
    mi::mdl::LLVM_code_generator &code_gen);

}  // sl
}  // llvm

#endif // MDL_GENERATOR_JIT_SL_PASSES_H
