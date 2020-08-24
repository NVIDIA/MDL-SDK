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

#ifndef MDL_COMPILERCORE_MANGLE_H
#define MDL_COMPILERCORE_MANGLE_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_types.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

class IDefinition;
class ISymbol;
class Function_instance;
class Array_instance;

/// Name mangling for MDL functions, similar to C++ Itanium ABI.
class MDL_name_mangler {
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param out    the resulting string
    MDL_name_mangler(IAllocator *alloc, string &out);

    /// Mangle an MDL entity (function/non-function).
    ///
    /// \param prefix  the package prefix, '::' used as separator or NULL
    /// \param def     the entity to mangle
    bool mangle(char const *prefix, IDefinition const *def);

    /// Mangle an MDL function instance.
    ///
    /// \param prefix  the package prefix, '::' used as separator or NULL
    /// \param inst    the instance to mangle
    bool mangle(char const *prefix, Function_instance const &inst);

    /// Mangle an MDL entity (non-function only).
    ///
    /// \param prefix       the package prefix, '::' used as separator or NULL
    /// \param name         the (non-absolute) name of the entity
    bool mangle_name(char const *prefix, char const *name);

    /// Demangle a C++ name to an MDL function name for use with resolvers.
    ///
    /// \param mangled_name      mangled C++ name to demangle. Does not need to be null-terminated.
    /// \param mangled_name_len  the length of the mangled C++ name
    ///
    /// \returns  true, if demangling was successful
    bool demangle(char const *mangled_name, size_t mangled_name_len);

    /// Ignore uniform/varying modifiers.
    void ignore_uniform_varying();

private:
    /// Mangle a function declaration.
    ///
    /// \param prefix   namespace prefix to add
    /// \param def      the definition of the function
    /// \param inst     if non-NULL, a "template" instance of the function
    bool mangle_function_decl(
        char const              *prefix,
        IDefinition const       *def,
        Function_instance const *inst);

    void mangle_function_encoding(
        char const              *prefix,
        IDefinition const       *def,
        Function_instance const *inst);

    void mangle_name(
        char const              *prefix,
        char const              *name,
        Function_instance const *inst);

    void mangle_type_name(ISymbol const *sym);
    void mangle_unqualified_name(char const *name, Function_instance const *inst);
    void mangle_source_name(char const *name);
    void mangle_local_name(IDefinition const *def);

    void mangle_nested_name(
        char const              *prefix,
        char const              *name,
        Function_instance const *inst);

    void mangle_prefix(char const *prefix);
    void mangle_type_qualifiers(IType::Modifiers mod);
    void mangle_type(IType const *type);
    void mangle_type(IType_atomic const *type);
    void mangle_type(IType_function const *type);

    void mangle_bare_function_type(
        IType_function const    *type,
        bool                    MangleReturnType,
        Function_instance const *inst);

    void mangle_type(IType_struct const *type);
    void mangle_type(IType_enum const *type);
    void mangle_type(IType_array const *type);
    void mangle_array_type(IType const *type, int size);
    void mangle_type(IType_vector const *type);
    void mangle_type(IType_matrix const *type);
    void mangle_type(IType_color const *type);
    void mangle_type(IType_reference const *type);
    void mangle_deferred_size(IType_array_size const *size);
    void mangle_template_argument_list(Function_instance const &inst);
    void mangle_template_argument(Array_instance const &ai);

    /// Parse "\<len\>\<name\>" from strptr and append the name to given string object,
    ///
    /// \param strptr   the string to parse. The referenced pointer will be advanced to after
    ///                 the name.
    /// \param endptr   points to the end of the string
    /// \param ref_str  the string object to which the name will be appended
    ///
    /// \returns  false if parsing failed
    bool demangle_name(char const *&strptr, char const *endptr, string &ref_str);

    /// Parse the sequence id for a substitution.
    ///
    /// \param strptr   the string to parse. The referenced pointer will be advanced to after
    ///                 the trailing '_'.
    /// \param endptr   points to the end of the string
    ///
    /// \returns the parsed sequence id or ~0 on error.
    unsigned long parse_sequence_id(char const *&strptr, char const *endptr);

private:
    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The generated string.
    string &m_out;

    /// If set, ignore uniform/varying type specifier.
    bool m_ignore_uniform_varying;
};


/// String output stream.
class String_output_stream : public Allocator_interface_implement<IOutput_stream>
{
    typedef Allocator_interface_implement<IOutput_stream> Base;
public:
    /// Constructor.
    String_output_stream(IAllocator *alloc)
    : Base(alloc)
    , m_string("", alloc)
    {
    }

    /// Write a char to the stream.
    void write_char(char c) MDL_FINAL { m_string += c; }

    /// Write a string to the stream.
    void write(char const *string) MDL_FINAL { m_string += string; }

    /// Flush stream.
    void flush() MDL_FINAL { }

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL {
        size_t l = m_string.size();
        if (l > 0 && m_string[l - 1] == c) {
            m_string.erase(m_string.begin() + l - 1);
            return true;
        }
        return false;
    }

    /// Get string value.
    string const &get_buffer() const { return m_string; }

    /// Clear string.
    void clear() { m_string.clear(); }

private:
    // Destructor.
    ~String_output_stream() MDL_FINAL {}

private:
    string m_string;
};

/// Helper class to handling name printing.
class Name_printer
{
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param mdl    the MDL compiler interface
    Name_printer(IAllocator *alloc, IMDL *mdl)
    : m_printer_sos(Allocator_builder(alloc).create<String_output_stream>(alloc))
    , m_printer(mdl->create_printer(m_printer_sos.get()))
    {
    }

    /// Print a type.
    void print(IType const *type) { m_printer->print(type); }

    /// Print a string.
    void print(char const *s) { m_printer->print(s); }

    /// Print a character.
    void print(char c) { m_printer->print(c); }

    /// Print a size_t.
    void print(size_t s) { m_printer->print(long(s)); }

    /// Print a symbol.
    void print(ISymbol const *sym) { m_printer->print(sym); }

    /// Get the current printed line and clear the buffer.
    string get_line() { string s(m_printer_sos->get_buffer()); m_printer_sos->clear(); return s; }

private:
    /// The string output stream the printer prints to.
    mi::base::Handle<String_output_stream> m_printer_sos;

    /// A printer printing to m_printer_sos.
    mi::base::Handle<IPrinter> m_printer;
};

/// Helper class for mangling names using the DAG style.
class DAG_mangler {
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler interface
    DAG_mangler(IAllocator *alloc, IMDL *compiler)
    : m_printer(alloc, compiler)
    , m_alloc(alloc)
    {
    }

    /// Mangle a definition.
    ///
    /// \param def                     the definition to mangle
    /// \param module_name             the absolute module name of this definition
    /// \param with_signature_suffix   indicates whether to include the signature suffix
    string mangle(IDefinition const *def, char const *module_name, bool with_signature_suffix = true);

    /// Mangle a definition.
    ///
    /// \param def                     the definition to mangle
    /// \param owner                   the owner module of this definition
    /// \param with_signature_suffix   indicates whether to include the signature suffix
    string mangle(IDefinition const *def, IModule const *owner, bool with_signature_suffix = true);

    /// Mangle a struct type.
    ///
    /// \param type         the type to mangle
    /// \param module_name  the absolute module name of this type
    string mangle(IType const *type, char const *module_name);

    /// Mangle a struct type.
    ///
    /// \param type         the type to mangle
    /// \param owner        the owner module of this type
    string mangle(IType const *type, IModule const *owner);

    /// Mangle a paramater type.
    ///
    /// \param type         the type to mangle
    string mangle_parameter_type(IType const *type);

    /// Get the name printer.
    Name_printer &get_printer() { return m_printer; }

private:
    /// Get the allocator.
    IAllocator *get_allocator() const  { return m_alloc; }

    /// Check if the name for the given definition must get a signature suffix.
    bool need_signature_suffix(IDefinition const *def) const;

private:
    /// The printer for names.
    Name_printer m_printer;

    /// The allocator.
    IAllocator *m_alloc;
};

}  // mdl
}  // mi

#endif
