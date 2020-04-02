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

#ifndef MDL_COMPILER_HLSL_PRINTERS_H
#define MDL_COMPILER_HLSL_PRINTERS_H 1

#include <mi/mdl/mdl_printers.h>
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_exprs.h"
#include "compiler_hlsl_messages.h"

namespace mi {
namespace mdl {
namespace hlsl {

class ICompilation_unit;
class Compilation_unit;
class Declaration;
class Definition;
class Type_name;
class Type_qualifier;
class Init_declarator;
class Name;
class Array_specifier;
class Instance_name;
class Stmt;
class Symbol;
class Type;
class Value;
class HLSL_attribute;

/// A printer.
/// Pretty-print various data types in MDL format.
class IPrinter : public
    mi::base::Interface_declare<0x2a31be3f,0x6da2,0x4d75,0x95,0xb6,0x3b,0xc7,0x42,0xee,0x7e,0x81,
    mi::mdl::ISyntax_coloring>
{
public:
    /// Indent output.
    ///  \param  depth      The depth to which to indent.
    virtual void indent(int depth) = 0;

    /// Print string.
    ///  \param  string     The string to print.
    virtual void print(char const *string) = 0;

    /// Print character.
    ///  \param  c          The character to print.
    virtual void print(char c) = 0;

    /// Print boolean.
    ///  \param  b          The boolean to print.
    virtual void print(bool b) = 0;

    /// Print 32bit integer.
    ///  \param  n          The integer to print.
    virtual void print(int32_t n) = 0;

    // Print 32bit unsigned integer.
    ///  \param  n          The unsigned to print.
    virtual void print(uint32_t n) = 0;

    /// Print 64bit integer.
    ///  \param  n          The integer to print.
    virtual void print(int64_t n) = 0;

    // Print 64bit unsigned integer.
    ///  \param  n          The unsigned to print.
    virtual void print(uint64_t n) = 0;

    /// Print double.
    ///  \param  d          The double to print.
    virtual void print(double d) = 0;

    /// Print a symbol.
    /// \param  sym         The symbol to print.
    virtual void print(Symbol *sym) = 0;

    /// Print type.
    /// \param  type        The type to print.
    virtual void print(Type *type) = 0;

    /// Print value.
    /// \param  value       The value to print.
    virtual void print(Value *value) = 0;

    /// Print expression.
    /// \param  expr        The expression to print.
    virtual void print(Expr const *expr) = 0;

    /// Print statement.
    /// \param  stmt        The statement to print.
    virtual void print(Stmt const *stmt) = 0;

    /// Print declaration.
    /// \param  decl        The declaration to print.
    virtual void print(Declaration const *decl) = 0;

    /// Print a definition.
    /// \param def          The definition to print.
    virtual void print(Definition const *def) = 0;

    /// Print a type name.
    /// \param  tn        The type name to print.
    virtual void print(Type_name const *tn) = 0;

    /// Print a type qualifier.
    /// \param  tq        The type qualifier to print.
    virtual void print(Type_qualifier const *tq) = 0;

    /// Print an init declarator.
    /// \param  init       The init declarator to print.
    virtual void print(Init_declarator const *init) = 0;

    /// Print a name.
    /// \param  name      The name to print.
    virtual void print(Name const *name) = 0;

    /// Print an array specifier.
    /// \param  spec      The array specifier to print.
    virtual void print(Array_specifier const *spec) = 0;

    /// Print an instance name.
    /// \param  spec      The instance name to print.
    virtual void print(Instance_name const *name) = 0;

    /// Print compilation unit.
    /// \param  unit      The unit to print.
    virtual void print(ICompilation_unit const *unit) = 0;

    /// Print message.
    /// \param  message        The message to print.
    /// \param  include_notes  If true, include notes.
    virtual void print(Message const *message, bool include_notes = false) = 0;

    /// Flush output.
    virtual void flush() = 0;

    /// Enable color output.
    ///
    /// \param  enable   true if output should be colored.
    virtual void enable_color(bool enable = true) = 0;

    /// Enable location printing using #line directives.
    ///
    /// \param  enable  if true, AST locations will be dumped as #line directives for statements
    virtual void enable_locations(bool enable = true) = 0;
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

    Color_entry &operator[](ISyntax_coloring::Syntax_elements index)
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
/// Implementation of the Pretty-printer for HLSL ASTs.
///
/// \note: This printer prints AST information only and does NOT use semantic
/// informations. Use it primary for debugging.
///
class Printer : public Syntax_coloring<IPrinter>
{
    typedef Syntax_coloring<IPrinter> Base;
    friend class mi::mdl::Allocator_builder;
public:

    /// Indent output.
    ///  \param  depth      The depth to which to indent.
    void indent(int depth) HLSL_FINAL;

    /// Print string.
    ///  \param  string     The string to print.
    void print(const char *string) HLSL_FINAL;

    /// Print character.
    ///  \param  c          The character to print.
    void print(char c) HLSL_FINAL;

    /// Print boolean.
    ///  \param  b          The boolean to print.
    void print(bool b) HLSL_FINAL;

    /// Print 32bit integer.
    ///  \param  n          The integer to print.
    void print(int32_t n) HLSL_FINAL;

    // Print 32bit unsigned integer.
    ///  \param  n          The unsigned to print.
    void print(uint32_t n) HLSL_FINAL;

    /// Print 64bit integer.
    ///  \param  n          The integer to print.
    void print(int64_t n) HLSL_FINAL;

    // Print 64bit unsigned integer.
    ///  \param  n          The unsigned to print.
    void print(uint64_t n) HLSL_FINAL;

    /// Print double.
    ///  \param  d          The double to print.
    void print(double d) HLSL_FINAL;

    /// Print a symbol.
    /// \param  sym         The symbol to print.
    void print(Symbol *sym) HLSL_FINAL;

    /// Print type.
    /// \param  type        The type to print.
    void print(Type *type) HLSL_FINAL;

    /// Print value.
    /// \param  value       The value to print.
    void print(Value *value) HLSL_FINAL;

    /// Print expression.
    /// \param  expr        The expression to print.
    void print(Expr const *expr) HLSL_FINAL;

    /// Print statement.
    /// \param  stmt        The statement to print.
    void print(Stmt const *stmt) HLSL_FINAL;

    /// Print declaration.
    /// \param  decl        The declaration to print.
    void print(Declaration const *decl) HLSL_FINAL;

    /// Print a definition.
    /// \param def          The definition to print.
    void print(Definition const *def) MDL_FINAL;

    /// Print a type name.
    /// \param  tn        The type name to print.
    void print(Type_name const *tn) HLSL_FINAL;

    /// Print a type qualifier.
    /// \param  tq        The type qualifier to print.
    void print(Type_qualifier const *tq) HLSL_FINAL;

    /// Print an init declarator.
    /// \param  init       The init declarator to print.
    void print(Init_declarator const *init) HLSL_FINAL;

    /// Print a name.
    /// \param  name      The name to print.
    void print(Name const *name) HLSL_FINAL;

    /// Print an array specifier.
    /// \param  spec      The array specifier to print.
    void print(Array_specifier const *spec) HLSL_FINAL;

    /// Print an instance name.
    /// \param  spec      The instance name to print.
    void print(Instance_name const *name) HLSL_FINAL;

    /// Print compilation unit.
    /// \param  unit      The unit to print.
    void print(ICompilation_unit  const *unit) HLSL_FINAL;

    /// Print message.
    /// \param  message        The message to print.
    /// \param  include_notes  If true, include notes.
    void print(Message const *message, bool include_notes = false) HLSL_FINAL;

    /// Flush output.
    void flush() HLSL_FINAL;

    /// Enable color output.
    ///
    /// \param  enable   true if output should be colored.
    void enable_color(bool enable = true) HLSL_FINAL;

    /// Enable location printing using #line directives.
    ///
    /// \param  enable  if true, AST locations will be dumped as #line directives for statements
    void enable_locations(bool enable = true) HLSL_FINAL;

    // ---------------- non-interface ----------------

    /// Print a statement.
    void print_condition(Stmt const *stmt);

    /// Print a declaration.
    ///
    /// \param decl        The declaration to print.
    /// \param embedded    If true, this declaration is embedded, else at top-level
    void print_decl(
        Declaration const *decl,
        bool              embedded);

    /// Print an attribute.
    ///
    /// \param attr   The attribute to print.
    void print_attr(
        HLSL_attribute const *attr);

    /// Prints a newline and do indentation.
    void nl(unsigned count = 1);

    /// Un-Indent output.
    ///  \param  depth      The depth to which to un-indent.
    void un_indent(int depth);

private:
    /// Constructor.
    explicit Printer(IAllocator *alloc, IOutput_stream *stream);

protected:
    /// Print expression.
    /// \param  expr        The expression to print.
    /// \param  priority    The operator priority of the caller.
    void print(
        Expr const *expr,
        int        priority);

    /// Print a location.
    ///
    /// \param location  the location to print
    void print_location(Location const &loc);

    /// Print a message.
    /// \param message   The message.
    /// \param sev       The severity for the message.
    void print_message(Message const *message, Message::Severity sev);

    /// Format print.
    void printf(char const *format, ...);

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

    /// Print a type part.
    void typepart(char const *w);

    /// Returns the priority of an operator.
    int get_priority(int op) const;

    /// Print a type with current color.
    void print_type(Type *type, Symbol *sym = NULL);

    /// Print a value.
    ///
    /// \param value           the value
    void print_value(Value *value);

private:
    // Copy constructor not implemented.
    Printer(Printer const &) HLSL_DELETED_FUNCTION;
    // Assignment operator not implemented.
    Printer &operator=(Printer const &) HLSL_DELETED_FUNCTION;

private:
    /// The current indentation depth;
    unsigned m_indent;

    /// The priority map
    int m_priority_map[Expr::OK_LAST + 1];

    /// If set, allow color output
    bool m_color_output;

    /// True if locations should be dumped as line directives
    bool m_enable_loc;

    /// The last file ID that was used for a #line directive.
    unsigned m_last_file_id;

    /// The output stream to write to.
    mi::base::Handle<IOutput_stream> m_ostr;

    /// The colored output stream to write to.
    mi::base::Handle<IOutput_stream_colored> m_c_ostr;

    /// The color stack.
    std::stack<ISyntax_coloring::Syntax_elements> m_color_stack;

    /// The current compilation unit if any.
    Compilation_unit const *m_curr_unit;
};

}  // hlsl
}  // mdl
}  // mi

#endif
