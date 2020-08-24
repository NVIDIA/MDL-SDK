/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_CHECKER_H
#define MDL_COMPILERCORE_CHECKER_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_printers.h>
#include "compilercore_visitor.h"

namespace mi {
namespace mdl {

class Module;
class Value_factory;
class IType;
class MDL;
class Generated_code_dag;

/// Base class for code checkers.
class Code_checker
{
protected:
    /// Constructor.
    ///
    /// \param verbose  if true, write a verbose output to stderr
    /// \param printer  the printer for writing error messages, takes ownership
    explicit Code_checker(bool verbose, IPrinter *printer);

    /// Check a value factory for soundness.
    ///
    /// \param factory  the value factory to check
    void check_factory(Value_factory const *factory);

    /// Check a given value for soundness.
    ///
    /// \param value   the value to check
    void check_value(IValue const *value);

    /// Check a given type for soundness.
    ///
    /// \param type   the type to check
    void check_type(IType const *type);

    /// Report an error.
    ///
    /// \param msg  the error message
    void report(char const *msg);

    /// Get the error count.
    size_t get_error_count() const { return m_error_count; }

protected:
    /// If true, verbose mode is activated
    bool const m_verbose;

private:
    /// The current error count.
    size_t m_error_count;

protected:
    /// The printer for error reports.
    mi::base::Handle<IPrinter> m_printer;
};

/// A checker for Modules.
class Module_checker : protected Code_checker
{
    typedef Code_checker Base;
public:
    /// Check a module.
    ///
    /// \param compiler  the MDL compiler (that owns the module)
    /// \param module    the module to check
    /// \param verbose   if true, write a verbose output the stderr
    ///
    /// \return true on success
    static bool check(MDL const *compiler, Module const *module, bool verbose);

private:
    /// Constructor.
    ///
    /// \param verbose  if true, write a verbose output to stderr
    /// \param printer  the printer for writing error messages, takes ownership
    explicit Module_checker(bool verbose, IPrinter *printer);

private:
};

}  // mdl
}  // mi

#endif  // MDL_COMPILERCORE_CHECKER_H
