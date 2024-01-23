/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_ERRORS_H
#define MDL_COMPILER_GLSL_ERRORS_H 1

#include "compiler_glsl_version.h"

namespace mi {
namespace mdl {
namespace glsl {

class Definition;
class Type;
class Symbol;
class Value;
class IPrinter;

/// Errors issued by the GLSL compiler.
enum GLSL_compiler_error {
    PREVIOUS_DEFINITION,

    SYNTAX_ERROR = 100,
    UNDETERMINED_COMMENT,
    UNDETERMINED_STRING,
    ILLEGAL_BYTE_ORDER_MARK_IGNORED,
    FILEID_NOT_A_STRING_CONSTANT,
    VERSION_DIRECTIVE_MUST_BE_FIRST,
    VERSION_DIRECTIVE_ALREADY_SEEN,
    VERSION_IS_NOT_NUMBER,
    LINE_IS_NOT_A_NUMBER,
    FILEID_NOT_A_NUMBER,
    BAD_PROFILE_NAME,
    PROFILE_NOT_ALLOWED_BELOW_150,
    VERSION_REQUIRES_ES_PROFILE,
    UNSUPPORTED_GLSLANG_VERSION,
    MISSING_EXTENSION_NAME,
    MISSING_COLON_AFTER_EXTENSION_NAME,
    MISSING_EXTENSION_BEHAVIOR,
    UNSUPPORTED_EXTENSION_BEHAVIOR,
    WRONG_BEHAVIOR_FOR_ALL_EXTENSION,
    UNSUPPORTED_EXTENSION,
    MISSING_MACRO_NAME,
    EMPTY_PRAGMA,
    RESERVED_STDGL_PRAGMA,
    MALFORMED_PRAGMA,
    ERROR_PRAGMA,
    WARNING_PRAGMA,
    WRONG_MACRO_NAME,
    PREDEFINED_NAME_DEFINE,
    RESERVED_NAME_DEFINE,
    RESERVED_NAME_WITH_TWO_UNDERSCORES,
    MACRO_REDEFINED,
    EXTRA_TOKENS_AT_DIRECTIVE_END,
    MISSING_RIGHT_PARA_IN_MACRO_PARAMETERS,
    MACRO_PARAMETERS_MUST_BE_COMMA_SEPARATED,
    MAY_NOT_APPEAR_IN_MACRO_PARAMETERS,
    DUPLICATE_MACRO_PARAMETER,
    UNTERMINATED_MACRO_ARGUMENT,
    WRONG_MACRO_ARGUMENT_COUNT,
    CONCAT_CANNOT_APPEAR_AT_BORDER,
    PASTING_DOES_NOT_GIVE_VALID_TOKEN,
    ELSE_WITHOUT_IF,
    ELIF_WITHOUT_IF,
    ENDIF_WITHOUT_IF,
    UNKNOWN_ESCAPE_SEQUENCE,
    OCTAL_ESCAPE_SEQUENCE_OVERFLOW,
    HEX_ESCAPE_SEQUENCE_OVERFLOW,
    INCOMPLETE_UNICODE_NAME,
    RESERVED_KEYWORD,
    CONSTANT_TOO_BIG,
    NON_STRICT_UNSIGNED_KEYWORD,
    NON_STRICT_EMPTY_DECLARATION,
    ENT_REDECLARATION,
    TYPE_REDECLARATION,
    PARAMETER_REDECLARATION,
    REDECLARATION_OF_DIFFERENT_KIND,
    UNKNOWN_IDENTIFIER,
    UNKNOWN_TYPE,
    UNKNOWN_MEMBER,
    NOT_A_TYPE_NAME,
    PARAMETER_SHADOWED,
    INCLUDE_NEEDS_FILE,
    INCLUDE_MISSING_TERMINAL,
    CANNOT_OPEN_INCLUDE,

    MAX_ERROR_NUM = CANNOT_OPEN_INCLUDE,
    INTERNAL_COMPILER_ERROR = 999,
};

/// Get the error template for the given code.
/// \param err  the error code
const char *get_error_template(GLSL_compiler_error err);

///
/// Helper type for safe message inserts.
///
class Error_params {
public:
    /// Supported directions.
    enum Direction { DIR_LEFT, DIR_RIGHT };

    /// Kinds of error parameters.
    enum Kind {
        EK_TYPE,
        EK_SYMBOL,
        EK_INTEGER,
        EK_STRING,
        EK_VALUE,
        EK_DEFINITION,
    };

private:
    struct Entry {
        Kind kind;

        union U {
            Type             *type;
            Symbol           *sym;
            int              i;
            char const       *string;
            Value            *val;
            Definition const *def;
        } u;
    };

public:

    /// Get the kind of given argument
    ///
    /// \param index  the argument index
    Kind get_kind(size_t index) const;

    /// Return the number of arguments.
    size_t get_arg_count() const { return m_args.size(); }

    /// Add a type argument.
    ///
    /// \param type  the type
    Error_params &add(Type *type);

    /// Return the type argument of given index.
    ///
    /// \param index  the argument index
    Type *get_type_arg(size_t index) const;

    /// Add a symbol argument.
    ///
    /// \param sym  the symbol
    Error_params &add(Symbol *sym);

    /// Return the type argument of given index.
    ///
    /// \param index  the argument index
    Symbol *get_symbol_arg(size_t index) const;

    /// Add an integer argument.
    ///
    /// \param v  the integer
    Error_params &add(int v);

    /// Return the integer argument of given index.
    ///
    /// \param index  the argument index
    int get_int_arg(size_t index) const;

    /// Add a GLSLang profile argument (and convert to string).
    ///
    /// \param profile  the profile
    Error_params &add(GLSLang_profile profile);


    /// Add a string argument.
    ///
    /// \param s  the string
    Error_params &add(char const *s);

    /// Return the string argument of given index.
    ///
    /// \param index  the argument index
    char const *get_string_arg(size_t index) const;

    /// Add a value argument.
    ///
    /// \param val  the value
    Error_params &add(Value *val);

    /// Return the value argument of given index.
    ///
    /// \param index  the argument index
    Value *get_value_arg(size_t index) const;

    /// Add a function signature.
    ///
    /// \param def  a definition of a function of constructor
    Error_params &add_signature(Definition const *def);

    /// Return the signature argument of given index.
    ///
    /// \param index  the argument index
    Definition const *get_signature_arg(size_t index) const;

public:
    /// Constructor.
    ///
    /// \param alloc  an Allocator interface
    explicit Error_params(IAllocator *alloc);

    /// Constructor.
    ///
    /// \param o   any object supporting a get_allocator() method
    template<typename T>
    explicit Error_params(T const &o)
    : m_args(o.get_allocator())
    {
    }

private:
    /// List of all collected argument.
    vector<Entry>::Type m_args;
};

/// Format and print a message.
///
/// \param code     the error code
/// \param params   error parameter/inserts
/// \param printer  a printer that will be used to print the message
void print_error_message(
    GLSL_compiler_error code,
    Error_params const  &params,
    IPrinter            *printer);


}  // glsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_GLSL_ERRORS_H

