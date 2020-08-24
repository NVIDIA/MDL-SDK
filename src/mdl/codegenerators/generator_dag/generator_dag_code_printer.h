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
#ifndef MDL_GENERATOR_DAG_CODE_PRINTER
#define MDL_GENERATOR_DAG_CODE_PRINTER 1

#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_printers.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_printers.h"

namespace mi {
namespace mdl {

///
/// A printer for DAG code.
///
class DAG_code_printer : public Allocator_interface_implement<IPrinter_interface>
{
    typedef Allocator_interface_implement<IPrinter_interface> Base;
public:
    /// Constructor.
    explicit DAG_code_printer(IAllocator *alloc)
    : Base(alloc)
    , m_printer(NULL)
    {
    }

    /// Print the code to the given printer.
    void print(Printer *printer, mi::base::IInterface const *code) const MDL_FINAL;

private:
    /// Print a DAG IR node inside a material or function definition.
    ///
    /// \param depth            The indentation depth.
    /// \param dag              The generated code dag this expression belongs to.
    /// \param def_index        The index of the definition this expression belongs to.
    ///                         Material indices are positive, function indices are negative,
    ///                         the original function index is recovered by -(def_index+1).
    /// \param node             The DAG IR node to print.
    ///
    void print_exp(
        int                       depth,
        IGenerated_code_dag const *dag,
        size_t                    def_index,
        DAG_node const            *node) const;

    /// Print the semantics if known as a comment.
    ///
    /// \param sema  the semantics
    void print_sema(IDefinition::Semantics sema) const;

    /// Print indentation.
    ///
    /// \param depth  the indentation depth
    void indent(int depth) const    { return m_printer->indent(depth); }

    /// Push current color to color stack and switch color.
    void push_color(ISyntax_coloring::Syntax_elements se) const {
        return m_printer->push_color(se);
    }

    /// Pop the color stack.
    void pop_color() const {
        return m_printer->pop_color();
    }

    template <typename T>
    void print(T s) const { return m_printer->print(s); }

    /// Print a keyword.
    void keyword(const char *w) const;

    /// Print only the last part of an absolute MDL name.
    void print_short(char const *abs_name) const;

    /// Print a type in MDL syntax.
    void print_mdl_type(IType const *type, bool full = false) const;

    /// Print all types of the code dag.
    void print_types(IGenerated_code_dag const *code_dag) const;

    /// Print all constants of the code dag.
    void print_constants(IGenerated_code_dag const *code_dag) const;

    /// Print all functions of the code dag.
    void print_functions(IGenerated_code_dag const *code_dag) const;

    /// Print all materials of the code dag.
    void print_materials(IGenerated_code_dag const *code_dag) const;

    /// Print all annotations of the code dag.
    void print_annotations(IGenerated_code_dag const *code_dag) const;

private:
    // Temporary, the used printer.
    mutable Printer *m_printer;
};

///
/// A printer for a Material Instance.
///
class Material_instance_printer : public Allocator_interface_implement<IPrinter_interface>
{
    typedef Allocator_interface_implement<IPrinter_interface> Base;
public:
    /// Constructor.
    explicit Material_instance_printer(mi::base::IAllocator *alloc)
    : Base(alloc)
    , m_printer(NULL)
    {
    }

    /// Print the interface to the given printer.
    void print(Printer *printer, mi::base::IInterface const *inst) const MDL_FINAL;

private:
    /// Print a DAG IR node inside a material instance.
    ///
    /// \param depth            The indentation depth.
    /// \param instance         The instance this expression belongs to.
    /// \param node             The IR node to print.
    ///
    void print_exp(
        int                                           depth,
        IGenerated_code_dag::IMaterial_instance const *instance,
        DAG_node const                                *node) const;

    void indent(int depth) const    { return m_printer->indent(depth); }

    template <typename T>
    void print(T s) const { return m_printer->print(s); }

    /// Set the given color
    void color(ISyntax_coloring::Syntax_elements c) const { return m_printer->color(c); }

    /// Print a keyword.
    void keyword(const char *w) const;

private:
    // Temporary, the used printer.
    mutable Printer *m_printer;
};

} // mdl
} // mi

#endif  // MDL_GENERATOR_DAG_CODE_PRINTER
