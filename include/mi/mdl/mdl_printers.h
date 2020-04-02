/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_printers.h
/// \brief Interfaces for printing various MDL entities to streams
#ifndef MDL_PRINTERS_H
#define MDL_PRINTERS_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_streams.h>

namespace mi {
namespace mdl {

// forwards
class ISymbol;
class ISimple_name;
class IQualified_name;
class IType_name;
class IType;
class IValue;
class IValue_resource;
class IExpression;
class IStatement;
class IDeclaration;
class IModule;
class IMessage;
class IAnnotation;
class IAnnotation_enable_if;
class IAnnotation_block;
class IArgument;
class IGenerated_code;
class ISemantic_version;

/// An interface for setting colors for syntax highlighting.
class ISyntax_coloring : public
    mi::base::Interface_declare<0x6cd5460e,0xad1c,0x413e,0x9d,0xaf,0x73,0x46,0x1c,0xd6,0x1a,0x56,
    mi::base::IInterface>
{
public:
    /// Syntax element classes.
    enum Syntax_elements {
        C_DEFAULT = -1,      ///< Default.
        C_KEYWORD,           ///< Keyword.
        C_TYPE,              ///< Types.
        C_LITERAL,           ///< Literals
        C_ANNOTATION,        ///< Annotations.
        C_ENTITY,            ///< MDL entities.
        C_ERROR,             ///< (Syntax) Errors.
        C_COMMENT,           ///< Comments.
        C_LAST = C_COMMENT
    };

    /// Set the color for a particular syntax element.
    ///
    /// \param element             The syntax element for which to set the color.
    /// \param foreground_color    The foreground color of the syntax element.
    /// \param foreground_bold     Flag to indicate if the foreground of the syntax element
    ///                            should be rendered in bold font/higher intensity.
    /// \param background_color    The background color of the syntax element.
    /// \param background_bold     Flag to indicate if the background of the syntax element
    ///                            should be rendered in bold font/higher intensity.
    virtual void set_color(
        Syntax_elements               element,
        IOutput_stream_colored::Color foreground_color,
        bool                          foreground_bold = false,
        IOutput_stream_colored::Color background_color = IOutput_stream_colored::DEFAULT,
        bool                          background_bold = false) = 0;
};

/// A printer.
///
/// Pretty-print various data types in MDL format.
/// Note that a Printer is not an MDL backend, i.e. the generated output might not
/// be (and is not in various cases) valid MDL code. Printers are more a debug tool.
/// Use the IMDL_exporter for writing valid MDL code. However, Printers can handle
/// broken MDL code (i.e. code containing syntax and semantic errors), whicle the
/// IMDL_exporter can only handle valid MDL modules.
/// Additionally the printer supports output of several MDL AST constructs which is
/// not possible with the IMDL_exporter.
class IPrinter : public
    mi::base::Interface_declare<0x4e36eb44,0xaad5,0x46e1,0x8e,0x54,0x2b,0x21,0x16,0xe2,0x34,0x20,
    ISyntax_coloring>
{
public:
    /// Indent output.
    ///
    /// \param depth      The depth to which to indent.
    virtual void indent(int depth) = 0;

    /// Print a C-string.
    ///
    /// \param string     The C-string to print.
    virtual void print(char const *string) = 0;

    /// Print a character.
    ///
    /// \param c          The character to print.
    virtual void print(char c) = 0;

    /// Print a boolean.
    ///
    /// \param b          The boolean to print.
    virtual void print(bool b) = 0;

    /// Print a long integer.
    ///
    /// \param n          The long integer to print.
    virtual void print(long n) = 0;

    /// Print a double.
    ///
    /// \param d          The double to print.
    virtual void print(double d) = 0;

    /// Print a symbol.
    ///
    /// \param sym        The symbol to print.
    virtual void print(ISymbol const *sym) = 0;

    /// Print a simple name.
    ///
    /// \param name       The name to print.
    virtual void print(ISimple_name const *name) = 0;

    /// Print a qualified name.
    ///
    /// \param name       The name to print.
    virtual void print(IQualified_name const *name) = 0;

    /// Print a type name.
    ///
    /// \param name       The name to print.
    virtual void print(IType_name const *name) = 0;

    /// Print a type.
    ///
    /// \param type       The type to print.
    virtual void print(IType const *type) = 0;

    /// Print a value.
    ///
    /// \param value      The value to print.
    virtual void print(IValue const *value) = 0;

    /// Print an expression.
    ///
    /// \param expr        The expression to print.
    virtual void print(IExpression const *expr) = 0;

    /// Print a statement.
    ///
    /// \param stmt        The statement to print.
    virtual void print(IStatement const *stmt) = 0;

    /// Print a declaration.
    ///
    /// \param decl        The declaration to print.
    virtual void print(IDeclaration const *decl) = 0;

    /// Print an annotation.
    ///
    /// \param anno        The annotation to print.
    virtual void print(IAnnotation const *anno) = 0;

    /// Print an enable_if annotation.
    ///
    /// \param anno        The enable_if annotation to print.
    virtual void print(IAnnotation_enable_if const *anno) = 0;

    /// Print an annotation block.
    ///
    /// \param anno        The annotation block to print.
    virtual void print(IAnnotation_block const *anno) = 0;

    /// Print an argument.
    ///
    /// \param arg         The argument to print.
    virtual void print(IArgument const *arg) = 0;

    /// Print a whole MDL module.
    ///
    /// \param module      The module to print.
    virtual void print(IModule const *module) = 0;

    /// Print a message.
    ///
    /// \param message        The message to print.
    /// \param include_notes  If true, include notes.
    virtual void print(
        IMessage const *message,
        bool           include_notes = false) = 0;

    /// Print a semantic version.
    ///
    /// \param sem_ver        The semantic version to print.
    virtual void print(ISemantic_version const *sem_ver) = 0;

    /// Print generated code.
    ///
    /// \param  code        The code to print.
    virtual void print(IGenerated_code const *code) = 0;

    /// Print a general interface.
    ///
    /// \param  iface The interface to print.
    virtual void print(mi::base::IInterface const *iface) = 0;

    /// Flush output.
    virtual void flush() = 0;

    /// Enable color output.
    ///
    /// \param  enable   true if output should be colored.
    virtual void enable_color(bool enable = true) = 0;

    /// Enable/disable positions output.
    ///
    /// If positions output is enabled, every statement and declaration is preceded
    /// by a comment telling the position where it was declared.
    ///
    /// \param  enable   true if position output should be enabled
    virtual void show_positions(bool enable = true) = 0;

    /// Enable/disable all type modifiers.
    ///
    /// The MDL type system supports more type modifiers then used in MDL
    /// languages. If disabled (default), ignore those extra modifiers and
    /// print the type unmodified, if enabled print extra modifiers as comments.
    ///
    /// \param  enable   true if extra modifiers should be printed
    virtual void show_extra_modifiers(bool enable = true) = 0;

    /// Enable/disable MDL language levels.
    ///
    /// The MDL compiler assigns two MDL language level to every entity.
    /// if enabled print these levels as comments.
    ///
    /// \param  enable   true if MDL language levels should be printed
    virtual void show_mdl_versions(bool enable = true) = 0;

    /// Enable/disable MDL resource table comments.
    ///
    /// The MDL compiler computes for every module a resource table.
    /// if enabled print this table as comments.
    ///
    /// \param  enable   true if the resource table should be printed
    virtual void show_resource_table(bool enable = true) = 0;

    /// Enable/disable MDL function hash table comments.
    ///
    /// The MDL compiler computes for some modules a function hash table.
    /// if enabled print this table as comments.
    ///
    /// \param  enable   true if the function hash table should be printed
    virtual void show_function_hash_table(bool enable = true) = 0;
};

/// Resource callback for the MDL exporter.
class IMDL_exporter_resource_callback
{
public:
    /// Retrieve the "resource name" of an MDL resource value.
    ///
    /// \param  v                            a resource value
    /// \param support_strict_relative_path  if true, the resource name allows strict relative path
    ///
    /// \return The MDL name of this resource value.
    virtual char const *get_resource_name(
        IValue_resource const *v,
        bool                  support_strict_relative_path) = 0;
};

/// The MDL exporter.
///
/// Exports a valid MDL module into MDL syntax.
class IMDL_exporter : public
    mi::base::Interface_declare<0x4fe47e94,0x71ce,0x4eb7,0xa3,0x89,0x98,0x92,0x2b,0xef,0x48,0xfa,
    ISyntax_coloring>
{
public:
    /// Export a module in MDL syntax to an output stream.
    ///
    /// \param stream       The output stream.
    /// \param module       The module to export.
    /// \param resource_cb  If non-NULL this resource callback is used to retrieve the name of
    ///                     resource values.
    virtual void export_module(
        IOutput_stream                  *stream,
        IModule const                   *module,
        IMDL_exporter_resource_callback *resource_cb) = 0;

    /// Enable color output.
    ///
    /// \param  enable   true if output should be colored.
    virtual void enable_color(bool enable = true) = 0;
};

}  // mdl
}  // mi

#endif
