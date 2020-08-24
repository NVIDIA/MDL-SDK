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

#ifndef MDL_COMPILERCORE_PRINTERS_H
#define MDL_COMPILERCORE_PRINTERS_H 1

#include <mi/base/handle.h>
#include <mi/base/iallocator.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_messages.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

class IOutput_stream;
class Printer;
class Sema_printer;

///
/// An interface allowing to delegate printing of an interface to the interface itself.
///
class IPrinter_interface : public
    mi::base::Interface_declare<0x0f802d754,0xf042,0x4915,0x8b,0xe2,0x2f,0x74,0x6a,0xa0,0x84,0xcf,
                                mi::base::IInterface>
{
public:
    /// Print the interface to the given printer.
    virtual void print(Printer *printer, mi::base::IInterface const *code) const = 0;
};

/// Entries in the color table.
struct Color_entry {
    /// Default constructor.
    Color_entry()
    : fg_color(IOutput_stream_colored::DEFAULT)
    , bg_color(IOutput_stream_colored::DEFAULT)
    , fg_bold(false)
    , bg_bold(false)
    {}

    /// Constructor.
    Color_entry(
        IOutput_stream_colored::Color fg_color,
        bool                          fg_bold,
        IOutput_stream_colored::Color bg_color,
        bool                          bg_bold)
    : fg_color(fg_color)
    , bg_color(bg_color)
    , fg_bold(fg_bold)
    , bg_bold(bg_bold)
    {}

    /// The foreground color.
    IOutput_stream_colored::Color fg_color;

    /// The background color.
    IOutput_stream_colored::Color bg_color;

    /// Flag to indicate if the foreground should be rendered in bold font.
    bool                          fg_bold;

    /// Flag to indicate if the background should be rendered in bold font.
    bool                          bg_bold;
};

/// A color table, containing colors for all colored entities.
class Color_table {
public:
    /// Set the default colors.
    void set_default_colors()
    {
        typedef IOutput_stream_colored Color;
        typedef ISyntax_coloring       Sc;

        entries[Sc::C_KEYWORD] =    Color_entry(Color::YELLOW,  true,  Color::DEFAULT, false);
        entries[Sc::C_TYPE] =       Color_entry(Color::GREEN,   true,  Color::DEFAULT, false);
        entries[Sc::C_LITERAL] =    Color_entry(Color::MAGENTA, true,  Color::DEFAULT, false);
        entries[Sc::C_ANNOTATION] = Color_entry(Color::CYAN,    true,  Color::DEFAULT, false);
        entries[Sc::C_ENTITY] =     Color_entry(Color::CYAN,    false, Color::DEFAULT, false);
        entries[Sc::C_ERROR] =      Color_entry(Color::WHITE,   false, Color::RED,     true);
        entries[Sc::C_COMMENT] =    Color_entry(Color::GREEN,   false, Color::DEFAULT, false);
    }

    Color_entry &operator[](ISyntax_coloring::Syntax_elements index) //-V302
    {
        return entries[index];
    }

private:
    Color_entry entries[ISyntax_coloring::C_LAST + 1];
};

///
/// Mixin class for ISyntax_coloring implementation.
///
template <typename Interface>
class Syntax_coloring : public Allocator_interface_implement<Interface> {
    typedef Allocator_interface_implement<Interface> Base;
public:
    /// Set the color for a particular syntax element.
    /// \param  element             The syntax element for which to set the color.
    /// \param  foreground_color    The foreground color of the syntax element.
    /// \param  foreground_bold     Flag to indicate if the foreground of the syntax element
    ///                             should be rendered in bold font.
    /// \param  background_color    The background color of the syntax element.
    /// \param  background_bold     Flag to indicate if the background of the syntax element
    ///                             should be rendered in bold font.
    void set_color(
        ISyntax_coloring::Syntax_elements element,
        IOutput_stream_colored::Color     foreground_color,
        bool                              foreground_bold = false,
        IOutput_stream_colored::Color     background_color = IOutput_stream_colored::DEFAULT,
        bool                              background_bold = false)
    {
        m_color_table[element] = Color_entry(
            foreground_color,
            foreground_bold,
            background_color,
            background_bold);
    }

protected:
    /// Constructor.
    explicit Syntax_coloring(IAllocator *alloc)
    : Base(alloc)
    {
        m_color_table.set_default_colors();
    }

protected:
    /// The color table, containing colors for all colored entities.
    Color_table m_color_table;
};

///
/// Implementation of the Pretty-printer for MDL ASTs.
///
/// \note: This printer prints AST information only and does NOT use semantic
/// informations. Use it primary for debugging.
///
class Printer : public Syntax_coloring<IPrinter>
{
    typedef Syntax_coloring<IPrinter> Base;
public:

    /// Indent output.
    ///  \param  depth   The depth to which to indent.
    void indent(int depth) MDL_FINAL;

    /// Print string.
    ///  \param  string  The string to print.
    void print(char const *string) MDL_FINAL;

    /// Print character.
    ///  \param  c  The character to print.
    void print(char c) MDL_FINAL;

    /// Print boolean.
    ///  \param  b  The boolean to print.
    void print(bool b) MDL_FINAL;

    /// Print long.
    ///  \param  n  The integer to print.
    void print(long n) MDL_FINAL;

    /// Print double.
    ///  \param  d  The double to print.
    void print(double d) MDL_FINAL;

    /// Print a symbol.
    /// \param  sym    The symbol to print.
    void print(ISymbol const *sym) MDL_FINAL;

    /// Print simple name.
    /// \param  name    The name to print.
    void print(ISimple_name const *name) MDL_OVERRIDE;

    /// Print qualified name.
    /// \param  name    The name to print.
    void print(IQualified_name const *name) MDL_OVERRIDE;

    /// Print type name.
    /// \param  name    The name to print.
    void print(IType_name const *name) MDL_FINAL;

    /// Print type.
    /// \param  type    The type to print.
    void print(IType const *type) MDL_FINAL;

    /// Print value.
    /// \param  value    The value to print.
    void print(IValue const *value) MDL_FINAL;

    /// Print expression.
    /// \param  expr    The expression to print.
    void print(IExpression const *expr) MDL_FINAL;

    /// Print statement.
    /// \param  stmt   The statement to print.
    void print(IStatement const *stmt) MDL_FINAL;

    /// Print declaration.
    /// \param  decl    The declaration to print.
    void print(IDeclaration const *decl) MDL_FINAL;

    /// Print qualifier.
    /// \param qualifier    The qualifier to print.
    virtual void print(Qualifier qualifier);

    /// Print parameter.
    /// \param parameter    The parameter to print.
    virtual void print(IParameter const *parameter);

    /// Print annotation.
    /// \param  anno    The annotation to print.
    void print(IAnnotation const *anno) MDL_FINAL;

    /// Print enable_if annotation.
    /// \param  anno    The annotation to print.
    void print(IAnnotation_enable_if const *anno) MDL_OVERRIDE;

    /// Print annotation block.
    /// \param  anno    The annotation block to print.
    void print(IAnnotation_block const *anno) MDL_FINAL;

    /// Print an argument.
    /// \param  arg    The argument to print.
    void print(IArgument const *arg) MDL_FINAL;

    /// Print module.
    /// \param  module  The module to print.
    void print(IModule const *module) MDL_OVERRIDE;

    /// Print message.
    /// \param  message        The message to print.
    /// \param  include_nodes  If true, include notes.
    void print(IMessage const *message, bool include_nodes) MDL_FINAL;

    /// Print semantic version.
    /// \param  sem_ver        The semantic version to print.
    void print(ISemantic_version const *sem_ver) MDL_FINAL;

    /// Print generated code.
    /// \param  code The generated code to print.
    void print(IGenerated_code const *code) MDL_FINAL;

    /// Print a general interface.
    /// \param  iface The interface to print.
    void print(mi::base::IInterface const *iface) MDL_FINAL;

    /// Flush output.
    void flush() MDL_FINAL;

    /// Enable color output.
    void enable_color(bool enable = true) MDL_FINAL;

    /// Enable/disable positions output.
    ///
    /// If locations output are enabled, every statement and declaration is preceded
    /// by a comment telling the position where it was declared.
    ///
    /// \param  enable   true if position output should be enabled
    void show_positions(bool enable = true) MDL_FINAL;

    /// Enable/disable all type modifiers.
    ///
    /// The MDL type system supports more type modifiers then used in MDL
    /// languages. If disabled (default), ignore those extra modifiers and
    /// print the type unmodified, if enabled print extra modifiers as comments.
    ///
    /// \param  enable   true if extra modifiers should be printed
    void show_extra_modifiers(bool enable = true) MDL_FINAL;

    /// Enable/disable MDL language levels.
    ///
    /// The MDL compiler assigns two MDL language level to every entity.
    /// if enabled print these levels as comments.
    ///
    /// \param  enable   true if MDL language levels should be printed
    void show_mdl_versions(bool enable = true) MDL_FINAL;

    /// Enable/disable MDL resource table comments.
    ///
    /// The MDL compiler computes for every module a resource table.
    /// if enabled print this table as comments.
    ///
    /// \param  enable   true if the resource table should be printed
    void show_resource_table(bool enable = true) MDL_FINAL;

    /// Enable/disable MDL function hash table comments.
    ///
    /// The MDL compiler computes for some modules a function hash table.
    /// if enabled print this table as comments.
    ///
    /// \param  enable   true if the function hash table should be printed
    void show_function_hash_table(bool enable = true) MDL_FINAL;

    // ------------------- non-interface methods -------------------

    /// Print namespace.
    /// \param  ns    The namespace to print.
    void print_namespace(IQualified_name const *name);

    /// Prints an annotation block.
    ///
    /// \param block   the annotation block or NULL
    /// \param prefix  if non-NULL, print this in front of an non-empty block
    virtual void print_anno_block(
        IAnnotation_block const *block,
        char const              *prefix);

    /// Prints a resource value.
    ///
    /// \param res   the resource value
    virtual void print_resource(IValue_resource const *res);

    /// Print an utf8 string with escapes.
    void print_utf8(char const *utf8_string, bool escapes);

    /// Format print.
    ///
    /// \param format  standard printf like format string
    ///
    /// \note: Beware: this function uses an internal (limited) buffer,
    ///        do not print prints with it
    void printf(char const *format, ...);

    /// Prints a newline and do indentation.
    void nl();

    /// Set the given color.
    void color(ISyntax_coloring::Syntax_elements code);

    /// Set the given color and push it on the color stack.
    void push_color(ISyntax_coloring::Syntax_elements code);

    /// Remove one entry from the color stack.
    void pop_color();

    /// Print a keyword.
    void keyword(char const *w);

    /// Print a literal.
    void literal(char const *w);

    /// Print a type with current color.
    ///
    /// \param type  the type to print
    /// \param name  if type is a function type, print name instead of * if != NULL
    virtual void print_type(IType const *type, ISymbol const *name = NULL);

    /// Print a type prefix (i.e. only the package name).
    ///
    /// \param e_type  the enum type to print
    virtual void print_type_prefix(IType_enum const *e_type);

    /// Returns true if a variable declaration of kind T v(a); can be rewritten as T v = a;
    virtual bool can_rewite_constructor_init(IExpression const * init);

    /// Print a type part.
    void typepart(char const *w);

    /// Returns the priority of an operator.
    int get_priority(int op) const;

    /// Prints a statement.
    void print(IStatement const *stmt, bool is_toplevel);

    /// Print declaration.
    void print(IDeclaration const *decl, bool is_toplevel);

    /// Print expression.
    /// \param  exp       The expression to print.
    /// \param  priority  The priority of the parent.
    virtual void print(IExpression const *exp, int priority);

    /// Print an argument.
    /// \param  arg           The argument to print.
    /// \param  priority      The priority of the parent expression.
    /// \param  ignore_named  If true, ignore named argument prefix
    void print(IArgument const *arg, int priority, bool ignore_named = false);

    /// Prints the position of the given statement as a comment.
    void print_position(IStatement const *stmt);

    /// Prints the position of the given declaration as a comment.
    void print_position(IDeclaration const *decl);

    /// Prints the MDL language levels for the given definition.
    void print_mdl_versions(IDefinition const *idef, bool insert = false);

public:
    /// Constructor.
    explicit Printer(IAllocator *alloc, IOutput_stream *stream);

    /// Set the string used to quote string values, default is "\"".
    ///
    /// \param quote  new quote string
    void set_string_quote(char const *quote);

protected:
    /// Destructor.
    ~Printer() MDL_OVERRIDE;

    /// Print a message.
    /// \param message   The message.
    /// \param sev       The severity for the message.
    void print_message(IMessage const *message, IMessage::Severity sev);

private:
    // Copy constructor not implemented.
    Printer(Printer const &) MDL_DELETED_FUNCTION;
    // Assignment operator not implemented.
    Printer &operator=(Printer const &) MDL_DELETED_FUNCTION;

protected:
    /// The version for printing.
    unsigned m_version;

    /// The current indentation depth;
    int m_indent;

    /// The priority map
    int m_priority_map[IExpression::OK_LAST + 1];

    /// If set, allow color output
    bool m_color_output;

    /// if set, positions are shown.
    bool m_show_positions;

    /// if set, print all modifiers, not only MDL uniform and varying
    bool m_show_extra_modifiers;

    /// if set, print all MDL language levels
    bool m_show_mdl_versions;

    /// If set, prints the resource table as comment.
    bool m_show_res_table;

    /// If set, prints the function hash table as comment.
    bool m_show_func_hashes;

    /// The output stream to write to.
    mi::base::Handle<IOutput_stream> m_ostr;

    /// The colored output stream to write to.
    mi::base::Handle<IOutput_stream_colored> m_c_ostr;

    typedef stack<ISyntax_coloring::Syntax_elements>::Type Syntax_elements_stack;

    /// The color stack.
    Syntax_elements_stack m_color_stack;

    /// The quote string.
    string m_string_quote;
};

///
/// Implementation of the MDL module export.
///
class MDL_exporter : public Syntax_coloring<IMDL_exporter>
{
    typedef Syntax_coloring<IMDL_exporter> Base;
public:
    /// Export a module in MDL syntax to an output stream.
    ///
    /// \param stream       The output stream.
    /// \param module       The module to export.
    /// \param resource_cb  If non-NULL this resource callback is used to retrieve the name of
    ///                     resource values.
    void export_module(
        IOutput_stream                  *stream,
        IModule const                   *module,
        IMDL_exporter_resource_callback *resource_cb) MDL_FINAL;

    /// Enable color output.
    ///
    /// \param  enable   true if output should be colored.
    void enable_color(bool enable = true) MDL_FINAL;

public:
    /// Constructor.
    ///
    /// \param alloc   the allocator
    explicit MDL_exporter(IAllocator *alloc);

    /// Destructor.
    ~MDL_exporter() MDL_FINAL;

private:
    // Copy constructor not implemented.
    MDL_exporter(MDL_exporter const &);
    // Assignment operator not implemented.
    MDL_exporter &operator=(MDL_exporter const &);

private:
    /// The builder.
    Allocator_builder m_builder;

    /// True, if color output is enabled.
    bool m_color_output;
};

}  // mdl
}  // mi

#endif
