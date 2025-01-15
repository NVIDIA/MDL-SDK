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
 ******************************************************************************/

#ifndef _MDLTLC_
#define _MDLTLC_ 1

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>

#include "mdltlc_compiler.h"

class Compilation_unit;

/// The MDLTL command line compiler application.
class Mdltlc
{
public:

    //! Constructor.
    ///
    /// \param program_name  The name of the command line application.
    Mdltlc(char const *program_name);

    //! Run the application.
    ///
    /// \param  argc    The argument count.
    /// \param  argv    The argument values.
    ///
    /// \returns    EXIT_SUCCESS on success, EXIT_FAILURE otherwise.
    int run(int argc, char *argv[]);

private:
    /// Prints usage.
    void usage();

    /// Compile all the given files in order. Increase err_count by
    /// the total number of errors encountered.
    void compile(Compiler_options &comp_options, unsigned& err_count);

    mi::base::Handle<Compiler> create_compiler();

private:

    /// The program name.
    const char *m_program;

    /// The MDL compiler interface. Instantiated when starting
    /// compilation.
    mi::base::Handle<mi::mdl::IMDL> m_imdl;
};

#endif
