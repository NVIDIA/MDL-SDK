/******************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>

#include <base/system/stlext/i_stlext_restore.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>

#include "generator_jit_code_printer.h"
#include "generator_jit_generated_code.h"

namespace mi {
namespace mdl {

// Print the code to the given printer.
void JIT_code_printer::print(Printer *printer, mi::base::IInterface const *code) const
{
    mi::base::Handle<IGenerated_code_executable const> code_jit(
        code->get_interface<IGenerated_code_executable>());

    if (!code_jit.is_valid_interface())
        return;

    Generated_code_jit const *code_llvm =
        static_cast<Generated_code_jit const *>(code_jit.get());

    MI::STLEXT::Store<Printer *> tmp(m_printer, printer);

    llvm::Module const *llvm_module = code_llvm->get_llvm_module();
    if (llvm_module != NULL) {
        std::string llvm_asm;
        llvm::raw_string_ostream s(llvm_asm);

        llvm_module->print(s, /*AssemblyAnnotationWriter=*/NULL);

        printer->print(llvm_asm.c_str());
    } else {
        printer->print(code_llvm->get_ptx_code());
    }
}

} // mdl
} // mi
