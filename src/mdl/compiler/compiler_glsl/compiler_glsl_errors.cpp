/******************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cctype>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

#include "compiler_glsl_analysis.h"
#include "compiler_glsl_assert.h"
#include "compiler_glsl_symbols.h"
#include "compiler_glsl_types.h"
#include "compiler_glsl_values.h"
#include "compiler_glsl_printers.h"
#include "compiler_glsl_errors.h"

namespace mi {
namespace mdl {
namespace glsl {

// Get the error template for the given code.
// FIXME: this must be generated
char const *get_error_template(GLSL_compiler_error err)
{
    switch (err) {
    case PREVIOUS_DEFINITION:
        return "'$0' previously defined here";

    case SYNTAX_ERROR:
        return "syntax error: $0";
    case UNDETERMINED_COMMENT:
        return "unexpected end of file found in comment";
    case UNDETERMINED_STRING:
        return "unexpected end of file found in string constant";
    case ILLEGAL_BYTE_ORDER_MARK_IGNORED:
        return "illegal byte order mark ignored";
    case FILEID_NOT_A_STRING_CONSTANT:
        return "fileid must be a string constant";
    case VERSION_DIRECTIVE_MUST_BE_FIRST:
        return "#version directive must be first";
    case VERSION_DIRECTIVE_ALREADY_SEEN:
        return "#version directive was already set";
    case VERSION_IS_NOT_NUMBER:
        return "#version must be followed by version number";
    case LINE_IS_NOT_A_NUMBER:
        return "#line must be followed by line number";
    case FILEID_NOT_A_NUMBER:
        return "#line num must be followed by a file id";
    case BAD_PROFILE_NAME:
        return "bad profile name; use es, core, or compatibility";
    case PROFILE_NOT_ALLOWED_BELOW_150:
        return "#version directive does not allow a profile for versions below 150";
    case VERSION_REQUIRES_ES_PROFILE:
        return "#version $0 requires the es profile";
    case UNSUPPORTED_GLSLANG_VERSION:
        return "unsupported #version $0 $1";
    case MISSING_EXTENSION_NAME:
        return "extension name missing after #extension";
    case MISSING_COLON_AFTER_EXTENSION_NAME:
        return "missing ':' after extension name";
    case MISSING_EXTENSION_BEHAVIOR:
        return "missing extension behavior";
    case UNSUPPORTED_EXTENSION_BEHAVIOR:
        return "unsupported extension behavior '$0'";
    case WRONG_BEHAVIOR_FOR_ALL_EXTENSION:
        return "extension 'all' cannot have 'require' or 'enable' behavior";
    case UNSUPPORTED_EXTENSION:
        return "extension '$0' is not supported";
    case MISSING_MACRO_NAME:
        return "no macro name given in #$0 directive";
    case EMPTY_PRAGMA:
        return "empty pragma directive";
    case RESERVED_STDGL_PRAGMA:
        return "reserved #pragma STDGL";
    case MALFORMED_PRAGMA:
        return "malformed #pragma $0: only '#pragma $0 (on|off)' is allowed";
    case ERROR_PRAGMA:
        return "#error$0";
    case WARNING_PRAGMA:
        return "#warning$0";
    case WRONG_MACRO_NAME:
        return "macro names must be identifiers";
    case PREDEFINED_NAME_DEFINE:
        return "$0: predefined names can't be $1";
    case RESERVED_NAME_DEFINE:
        return "$0: names beginning with 'GL_' can't be $1";
    case RESERVED_NAME_WITH_TWO_UNDERSCORES:
        return "$0: names containing consecutive underscores are reserved";
    case MACRO_REDEFINED:
        return "'$0' redefined";
    case EXTRA_TOKENS_AT_DIRECTIVE_END:
        return "extra tokens at end of #$0 directive";
    case MISSING_RIGHT_PARA_IN_MACRO_PARAMETERS:
        return "missing ')' in macro parameter list";
    case MACRO_PARAMETERS_MUST_BE_COMMA_SEPARATED:
        return "macro parameters must be comma-separated";
    case MAY_NOT_APPEAR_IN_MACRO_PARAMETERS:
        return "'$0' may not appear in macro parameter list";
    case DUPLICATE_MACRO_PARAMETER:
        return "duplicate macro parameter '$0'";
    case UNTERMINATED_MACRO_ARGUMENT:
        return "unterminated argument list invoking macro '$0'";
    case WRONG_MACRO_ARGUMENT_COUNT:
        return "macro '$0' passed $1 arguments, but takes just $2";
    case CONCAT_CANNOT_APPEAR_AT_BORDER:
        return "'##' cannot appear at either end of a macro expansion";
    case PASTING_DOES_NOT_GIVE_VALID_TOKEN:
        return "pasting '$0' and '$1' does not give a valid preprocessing token";
    case ELSE_WITHOUT_IF:
        return "#else without #if";
    case ELIF_WITHOUT_IF:
        return "#elif without #if";
    case ENDIF_WITHOUT_IF:
        return "#endif without #if";
    case UNKNOWN_ESCAPE_SEQUENCE:
        return "unknown escape sequence: '$0'";
    case OCTAL_ESCAPE_SEQUENCE_OVERFLOW:
        return "octal escape sequence out of range";
    case HEX_ESCAPE_SEQUENCE_OVERFLOW:
        return "hex escape sequence out of range";
    case INCOMPLETE_UNICODE_NAME:
        return "incomplete universal character name '$0'";
    case RESERVED_KEYWORD:
        return "Reserved keyword '$0' used";
    case CONSTANT_TOO_BIG:
        return "Constant too big";
    case NON_STRICT_UNSIGNED_KEYWORD:
        return "unsigned is a reserved keyword in strict GLSL";
    case NON_STRICT_EMPTY_DECLARATION:
        return "empty declarations are not allowed in strict GLSL";
    case ENT_REDECLARATION:
        return "redeclaration of '$0'";
    case TYPE_REDECLARATION:
        return "redeclaration of type '$0'";
    case PARAMETER_REDECLARATION:
        return "redeclaration of parameter '$0'";
    case REDECLARATION_OF_DIFFERENT_KIND:
        return "redeclaration of '$0' as a different kind of symbol";
    case UNKNOWN_IDENTIFIER:
        return "'$0' has not been declared";
    case UNKNOWN_TYPE:
        return "type '$0' has not been declared";
    case UNKNOWN_MEMBER:
        return "type '$0' has no member '$1'";
    case NOT_A_TYPE_NAME:
        return "'$0' does not name a type";
    case PARAMETER_SHADOWED:
        return "declaration shadows parameter '$0'";
    case INCLUDE_NEEDS_FILE:
        return "#include must be followed by a file designation";
    case INCLUDE_MISSING_TERMINAL:
        return "#include missing terminating > character";
    case CANNOT_OPEN_INCLUDE:
        return "Fatal: cannot open include file '$0'";

    // ------------------------------------------------------------- //
    case INTERNAL_COMPILER_ERROR:
        return "internal compiler error: $.";
    }
    return "";
}

// Constructor.
Error_params::Error_params(IAllocator *alloc)
: m_args(alloc)
{
}

// Get the kind of given argument
Error_params::Kind Error_params::get_kind(size_t index) const
{
    Entry const &e = m_args.at(index);

    return e.kind;
}

// Add a type argument.
Error_params &Error_params::add(Type *type)
{
    Entry e;
    e.kind   = EK_TYPE;
    e.u.type = type;

    m_args.push_back(e);
    return *this;
}

// Return the type argument of given index.
Type *Error_params::get_type_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    GLSL_ASSERT(e.kind == EK_TYPE);
    return e.u.type;
}

// Add a symbol argument.
Error_params &Error_params::add(Symbol *sym)
{
    Entry e;
    e.kind  = EK_SYMBOL;
    e.u.sym = sym;

    m_args.push_back(e);
    return *this;
}

// Return the type argument of given index.
Symbol *Error_params::get_symbol_arg(size_t index) const {
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_SYMBOL);
    return e.u.sym;
}

// Add an integer argument.
Error_params &Error_params::add(int v)
{
    Entry e;
    e.kind = EK_INTEGER;
    e.u.i  = v;

    m_args.push_back(e);
    return *this;
}

// Return the integer argument of given index.
int Error_params::get_int_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_INTEGER);
    return e.u.i;
}

// Add a GLSLang profile argument (and convert to string).
Error_params &Error_params::add(GLSLang_profile profile)
{
    // map to a string
    char const *s = "";
    switch (profile) {
    case GLSL_PROFILE_CORE:          s = "core"; break;
    case GLSL_PROFILE_COMPATIBILITY: s = "compatibility"; break;
    case GLSL_PROFILE_ES:            s = "es"; break;
    }
    return add(s);
}


// Add a string argument.
Error_params &Error_params::add(char const *s)
{
    Entry e;
    e.kind     = EK_STRING;
    e.u.string = s;

    m_args.push_back(e);
    return *this;
}

// Return the string argument of given index.
char const *Error_params::get_string_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_STRING);
    return e.u.string;
}

// Add a value argument.
Error_params &Error_params::add(Value *val)
{
    Entry e;
    e.kind  = EK_VALUE;
    e.u.val = val;

    m_args.push_back(e);
    return *this;
}

// Return the value argument of given index.
Value *Error_params::get_value_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_VALUE);
    return e.u.val;
}

// Add a function signature.
Error_params &Error_params::add_signature(Definition const *def)
{
    Entry e;
    e.kind  = EK_DEFINITION;
    e.u.def = def;

    m_args.push_back(e);
    return *this;
}

// Return the signature argument of given index.
Definition const *Error_params::get_signature_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_DEFINITION);
    return e.u.def;
}

// Helper for print_error_message().
static void print_error_param(
    Error_params const &params,
    size_t             idx,
    IPrinter           *printer)
{
    switch (params.get_kind(idx)) {
    case Error_params::EK_TYPE:
        {
            // type param
            Type *type = params.get_type_arg(idx);
            if (is<Type_struct>(type)) {
                printer->print("struct ");
            }
            printer->print(type);
        }
        break;
    case Error_params::EK_SYMBOL:
        // symbol param
        printer->print(params.get_symbol_arg(idx));
        break;
    case Error_params::EK_INTEGER:
        // integer
        printer->print(params.get_int_arg(idx));
        break;
    case Error_params::EK_STRING:
        // cstring param
        printer->print(params.get_string_arg(idx));
        break;
    case Error_params::EK_VALUE:
        // value param
        printer->print(params.get_value_arg(idx));
        break;
    case Error_params::EK_DEFINITION:
        {
            Definition const *def = params.get_signature_arg(idx);
            Definition::Kind kind = def->get_kind();

            if (kind == Definition::DK_MEMBER) {
                // print type_name::member_name
                Scope const *scope = def->get_def_scope();

                // use the scope name (== type name without scope)
                printer->print(scope->get_scope_name());

                printer->print("::");
                printer->print(def->get_symbol());
                break;
            } else if (kind == Definition::DK_FUNCTION) {
                Type_function *func_type = cast<Type_function>(def->get_type());

                Type *ret_type = func_type->get_return_type();
                printer->print(ret_type);
                printer->print(" ");

                printer->print(def->get_symbol());

                printer->print("(");
                for (size_t i = 0, n = func_type->get_parameter_count(); i < n; ++i) {
                    Type_function::Parameter *param = func_type->get_parameter(i);

                    if (i > 0)
                        printer->print(", ");
                    printer->print(param->get_type());
                }
                printer->print(")");
            } else {
                printer->print(def->get_symbol());
                break;
            }
        }
        break;

    default:
        MDL_ASSERT(!"Unsupported format kind");
        break;
    }
}

// Format a message.
void print_error_message(
    GLSL_compiler_error code,
    Error_params const  &params,
    IPrinter            *printer)
{
    size_t last_idx = (size_t)-1;
    if (char const *tmpl = get_error_template(code)) {
        for (char c = *tmpl; c != '\0'; c = *tmpl) {
            ++tmpl;
            if (c == '$') {
                if (*tmpl == '$') {
                    printer->print('$');
                    ++tmpl;
                    continue;
                } else if (*tmpl == '.') {
                    // just enroll all the rest

                    for (size_t idx = last_idx + 1; idx < params.get_arg_count(); ++idx) {
                        print_error_param(params, idx, printer);
                    }
                    last_idx = params.get_arg_count();
                    ++tmpl;
                    continue;
                }

                size_t idx = 0;

                while (isdigit(*tmpl)) {
                    idx = idx * 10 + *tmpl - '0';
                    ++tmpl;
                }

                if (idx >= params.get_arg_count()) {
                    GLSL_ASSERT(!"missing error parameter");
                    printer->print("???");
                    continue;
                }
                last_idx = idx;

                print_error_param(params, idx, printer);
            } else {
                printer->print(c);
            }
        }
    }
}

}  // glsl
}  // mdl
}  // mi


