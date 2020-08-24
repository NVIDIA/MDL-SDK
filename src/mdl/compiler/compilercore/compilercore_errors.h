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

#ifndef MDL_COMPILERCORE_ERRORS_H
#define MDL_COMPILERCORE_ERRORS_H 1

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_names.h>
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

class ISymbol;
class IType;
class IQualified_name;
class IExpression;
class Analysis;
class IDefinition;
class Definition;
class IPrinter;
class IValue_string;
class ISemantic_version;

/// Compiler errors.
enum Compilation_error {
    PREVIOUS_DEFINITION,
    PREVIOUSLY_USED,
    DEFINED_AT,
    DECLARED_AT,
    FIRST_DEFAULT,
    RECURSION_CHAIN,
    SELF_RECURSION,
    BOUND_IN_CONTEXT,
    ASSIGNED_AT_PARAM_NAME,
    ASSIGNED_AT_PARAM_POS,
    FUNCTION_CALLS_ARE_STATE_INDEPENDENT,
    WRITTEN_BUT_NOT_READ,
    LET_TEMPORARIES_ARE_RVALUES,
    SAME_CONDITION,
    IMPORT_ENUM_TYPE,
    DEFAULT_ARG_DECLARED_AT,
    CURR_DIR_FORBIDDEN_IN_RESOURCE,
    PARENT_DIR_FORBIDDEN_IN_RESOURCE,
    IMPORTED_WITHOUT_EXPORT,
    USES_FUTURE_FEATURE,
    TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
    PREVIOUSLY_ANNOTATED,
    ONLY_REFERNCED_INSIDE_DEFAULT_INITIALIZER,
    LOCATED_AT,
    POSSIBLE_MATCH,
    CANDIDATES_ARE,
    CANDIDATES_ARE_NEXT,
    POSSIBLE_ABSOLUTE_IMPORT,

    SYNTAX_ERROR = 100,
    ILLEGAL_BOM_IGNORED,
    UNDETERMINED_COMMENT,
    INTEGER_OVERFLOW,
    ENT_REDECLARATION,
    TYPE_REDECLARATION,
    PARAMETER_REDECLARATION,
    REDECLARATION_OF_DIFFERENT_KIND,
    UNKNOWN_PACKAGE_NAME,
    UNKNOWN_IDENTIFIER,
    UNKNOWN_TYPE,
    UNKNOWN_MEMBER,
    NON_CONSTANT_INITIALIZER,
    MISSING_CONSTANT_INITIALIZER,
    ANNONATION_NON_CONST_DEF_PARAM,
    FUNCTION_BODY_NOT_BLOCK,
    MATERIAL_BODY_NOT_EXPR,
    MATERIAL_PRESET_BODY_NOT_CALL,
    NOT_A_MATERIAL_NAME,
    MATERIAL_QUALIFIER_NOT_ALLOW,
    MODULE_NOT_FOUND,
    ERRONEOUS_IMPORT,
    CONSTRUCTOR_SHADOWED,
    PARAMETER_SHADOWED,
    UNKNOWN_EXPORT,
    ENT_NOT_EXPORTED,
    INDEX_OF_NON_ARRAY,
    ILLEGAL_INDEX_TYPE,
    CONDITION_BOOL_TYPE_REQUIRED,
    MISMATCHED_CONDITIONAL_EXPR_TYPES,
    ILLEGAL_SWITCH_CONTROL_TYPE,
    ILLEGAL_CASE_LABEL_TYPE,
    CANNOT_CONVERT_CASE_LABEL_TYPE,
    MULTIPLE_DEFAULT_LABEL,
    MULTIPLE_CASE_VALUE,
    MISSING_CASE_LABEL,
    CASE_LABEL_NOT_CONSTANT,
    DEFAULT_WITHOUT_CASE,
    INVALID_CONVERSION,
    INVALID_CONVERSION_RETURN,
    NOT_A_TYPE_NAME,
    REEXPORTING_QUALIFIED_NAMES_FORBIDDEN,
    FORBIDDED_GLOBAL_STAR,
    MODULE_NAME_GIVEN_NOT_ENTITY,
    FORBIDDEN_RETURN_TYPE,
    CALLED_OBJECT_NOT_A_FUNCTION,
    POS_ARGUMENT_AFTER_NAMED,
    NO_MATCHING_OVERLOAD,
    AMBIGUOUS_OVERLOAD,
    AMBIGUOUS_UNARY_OPERATOR,
    AMBIGUOUS_BINARY_OPERATOR,
    CANNOT_CONVERT_POS_ARG_TYPE,
    CANNOT_CONVERT_ARG_TYPE,
    CANNOT_CONVERT_POS_ARG_TYPE_FUNCTION,
    CANNOT_CONVERT_ARG_TYPE_FUNCTION,
    MISSING_PARAMETER,
    PARAMETER_PASSED_TWICE,
    UNKNOWN_NAMED_PARAMETER,
    ILLEGAL_LVALUE_ASSIGNMENT,
    OPERATOR_LEFT_ARGUMENT_ILLEGAL,
    OPERATOR_RIGHT_ARGUMENT_ILLEGAL,
    CONFLICTING_REDECLARATION,
    FUNCTION_REDEFINITION,
    FUNCTION_REDECLARED_DIFFERENT_RET_TYPE,
    REDECLARED_DIFFERENT_PARAM_NAME,
    DEFAULT_PARAM_REDEFINITION,
    FUNCTION_PROTOTYPE_USELESS,
    ENUMERATOR_VALUE_NOT_INT,
    ARRAY_SIZE_NON_CONST,
    ARRAY_SIZE_NOT_INTEGRAL,
    ARRAY_SIZE_NEGATIVE,
    ARRAY_INDEX_OUT_OF_RANGE,
    NO_ARRAY_CONVERSION,
    CANNOT_CONVERT_ARRAY_ELEMENT,
    MISSING_DEFAULT_ARGUMENT,
    MISSING_DEFAULT_FIELD_INITIALIZER,
    RECURSION_CALL_NOT_ALLOWED,
    CALL_OF_UNDEFINED_FUNCTION,
    ABSTRACT_ARRAY_RETURN_TYPE,
    ABSTRACT_ARRAY_INSIDE_STRUCT,
    FORBIDDEN_SELF_REFERENCE,
    UNUSED_VARIABLE,
    UNUSED_CONSTANT,
    UNUSED_PARAMETER,
    DIVISION_BY_ZERO_IN_CONSTANT_EXPR,
    NOT_A_ANNOTATION,
    NON_CONSTANT_ANNOTATION_ARG_POS,
    NON_CONSTANT_ANNOTATION_ARG_NAME,
    ANNOTATION_IGNORED,
    ENUM_VALUE_NOT_HANDLED_IN_SWITCH,
    SUGGEST_PARENTHESES_AROUND_ASSIGNMENT,
    WRONG_MATERIAL_PRESET,
    TOO_MANY_PRESET_PARAMS,
    CONFLICTING_PROTOTYPE,
    NAME_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER,
    POS_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER,
    EMPTY_CONTROL_STMT,
    UNREACHABLE_STATEMENT,
    BREAK_OUTSIDE_LOOP_OR_SWITCH,
    CONTINUE_OUTSIDE_LOOP,
    EFFECTLESS_STATEMENT,
    FUNCTION_MUST_RETURN_VALUE,
    WRONG_COMPARISON_BETWEEN_ENUMS,
    FUNCTION_OVERLOADING_ON_MODIFIER,
    FUNCTION_PREVIOUSLY_EXPORTED,
    FUNCTION_NOW_EXPORTED,
    FUNCTION_REDECLARED_DIFFERENT_QUALIFIER,
    FORBIDDEN_VARIABLE_TYPE,
    FORBIDDEN_CONSTANT_TYPE,
    IMPORT_SYMBOL_CLASH,
    OVERLOADS_MUST_BE_EXPORTED,
    LET_USED_OUTSIDE_MATERIAL,
    DEFAULT_NOT_EXPORTED,
    VARYING_CALL_FROM_UNIFORM_FUNC,
    UNIFORM_FUNCTION_DECLARED_WITH_VARYING_RESULT,
    RESULT_OF_UNIFORM_FUNCTION_IS_VARYING,
    NONVARYING_RESULT_OF_VARYING_FUNCTION,
    VARIABLE_DEPENDS_ON_AUTOTYPED_PARAMETER,
    VARIABLE_DEPENDS_ON_VARYING_VALUE,
    CANNOT_CONVERT_ARG_TYPE_PARAM_DEP,
    CANNOT_CONVERT_POS_ARG_TYPE_PARAM_DEP,
    CONSTANT_TOO_BIG,
    UNSUPPORTED_MDL_VERSION,
    FUNCTION_ANNOTATIONS_NOT_AT_PROTOTYPE,
    EXPORTED_FUNCTION_NOT_DEFINED,
    UNREFERENCED_FUNCTION_REMOVED,
    PARAMETER_ONLY_WRITTEN,
    USED_TYPE_NOT_EXPORTED,
    OPERATOR_EQUAL_WITHOUT_EFFECT,
    OPERATOR_WITHOUT_EFFECT,
    IDENTICAL_SUBEXPRESSION,
    IDENTICAL_THEN_ELSE,
    ARRAY_CONSTRUCTOR_ARGUMENT_MISMATCH,
    ABSTRACT_ARRAY_CONSTRUCTOR_ARGUMENT,
    IDENTICAL_IF_CONDITION,
    UNCONDITIONAL_EXIT_IN_LOOP,
    OPERATOR_IS_STRICT_IN_MATERIAL,
    IMPORT_ENUM_VALUE_SYMBOL_CLASH,
    CANNOT_CONVERT_DEFAULT_ARG_TYPE_FUNCTION,
    CANNOT_CONVERT_DEFAULT_ARG_TYPE,
    FORBIDDEN_FIELD_TYPE,
    RESOURCE_PARAMETER_NOT_UNIFORM,
    IGNORED_ON_NON_FUNCTION,
    IMPORT_LOOP,
    UNREFERENCED_MATERIAL_REMOVED,
    TOO_MANY_ARGUMENTS,
    ARRAY_CONSTRUCTOR_DOESNT_HAVE_NAMED_ARG,
    SEQUENCE_WITHOUT_EFFECT_INSIDE_MATERIAL,
    FORBIDDEN_ARRAY_TYPE,
    FORBIDDEN_NESTED_ARRAY,
    FORBIDDEN_FUNCTION_PARAMETER_TYPE,
    FORBIDDEN_MATERIAL_PARAMETER_TYPE,
    TERNARY_COND_NOT_UNIFORM,
    WRONG_ANNOTATION_RANGE_INTERVAL,
    DEFAULT_INITIALIZER_OUTSIDE_HARD_RANGE,
    TYPE_DOES_NOT_SUPPORT_RANGE,
    ADDITIONAL_ANNOTATION_IGNORED,
    ENTITY_DOES_NOT_SUPPORT_ANNOTATION,
    RETURN_VALUE_DOES_NOT_SUPPORT_ANNOTATION,
    ARGUMENT_OUTSIDE_HARD_RANGE,
    FIELD_VALUE_OUTSIDE_HARD_RANGE,
    SOFT_RANGE_OUTSIDE_HARD_RANGE,
    FUNCTION_PTR_USED,
    USED_VARIABLE_MARKED_UNUSED,
    USED_CONSTANT_MARKED_UNUSED,
    USED_PARAMETER_MARKED_UNUSED,
    USED_FUNCTION_MARKED_UNUSED,
    USED_MATERIAL_MARKED_UNUSED,
    ASSIGNMENT_INSIDE_MATERIAL,
    DIVISION_BY_ZERO,
    INCOMPLETE_ARRAY_SPECIFICATION,
    ARRAY_ASSIGNMENT_FORBIDDEN,
    RESOURCE_NAME_NOT_LITERAL,
    FORBIDDED_C_STYLE_CAST,
    EXTRA_PARENTHESIS_AROUND_FUNCTION_NAME,
    DEPRECATED_ENTITY,
    FORBIDDEN_QUALIFIER_ON_CONST_DECL,
    DOUBLE_TYPE_USED,
    MODULE_DOES_NOT_SUPPORT_ANNOTATION,
    MATERIAL_BODY_IS_NOT_A_MATERIAL,
    UNIFORM_RESULT_DEPENDS_ON_AUTOTYPED_PARAMETER,
    UNIFORM_RESULT_IS_VARYING,
    FUNCTION_PRESET_BODY_NOT_CALL,
    FUNCTION_PRESET_WRONG_RET_TYPE,
    RELATIVE_IMPORTS_NOT_SUPPORTED,
    EXPORTED_PRESET_OF_UNEXPORTED_ORIGIN,
    NON_EXISTANT_DEPENDENCY,
    DEPENDENCY_VERSION_MISSING,
    DEPENDENCY_VERSION_REQUIRED,
    NON_DIRECT_DEPENDENCY,
    NOT_A_MODULE_NAME,
    INVALID_DIRECTORY_NAME,
    INVALID_DIRECTORY_NAME_NESTING,
    STRICT_RELATIVE_PATH_IN_STRING_MODULE,
    FILE_PATH_CONSISTENCY,
    DEPRECATED_RESOURCE_PATH_USED,
    AMBIGUOUS_FILE_LOCATION,
    IMPORT_FROM_OLD_ARCHIVE_VERSION,
    UNABLE_TO_RESOLVE,
    INVALID_MDL_ARCHIVE_DETECTED,
    ARCHIVE_DEPENDENCY_MISSING,
    FORBIDDEN_MDL_VERSION_IN_ARCHIVE,
    TYPE_CONVERSION_MUST_BE_EXPLICIT,
    MISSING_RESOURCE,
    RESOURCE_OUTSIDE_ARCHIVE,
    IGNORED_ON_NON_PARAMETER,
    INVALID_THUMBNAIL_EXTENSION,
    CONDITION_NOT_LITERAL,
    INVALID_ENABLE_IF_CONDITION,
    DEPENDENCY_VERSION_MAJOR_MISMATCH,
    FORBIDDEN_BINARY_OP_IN_ENABLE_IF,
    FORBIDDEN_UNARY_OP_IN_ENABLE_IF,
    FORBIDDEN_TERNARY_OP_IN_ENABLE_IF,
    FORBIDDEN_CALL_IN_ENABLE_IF,
    FORBIDDEN_SELF_REF_IN_ENABLE_IF,
    FORBIDDEN_ENABLE_IF_TYPE_REF,
    ENABLE_IF_CONDITION_HAS_WARNINGS,
    FUNC_VARIANT_WITH_DEFERRED_ARRAY_RET_TYPE,
    ARCHIVE_CONFLICT,
    MATERIAL_PTR_USED,
    CAST_TYPES_UNRELATED,
    CAST_MISSING_ENUM_VALUE_SRC,
    CAST_MISSING_ENUM_VALUE_DST,
    CAST_STRUCT_FIELD_COUNT,
    CAST_STRUCT_FIELD_INCOMPATIBLE,
    CAST_ARRAY_ELEMENT_INCOMPATIBLE,
    CAST_ARRAY_DEFERRED_TO_IMM,
    CAST_ARRAY_IMM_TO_DEFERRED,
    CAST_ARRAY_DIFFERENT_LENGTH,
    FORBIDDEN_ANNOTATION_PARAMETER_TYPE,
    ANNOS_ON_ANNO_DECL_NOT_SUPPORTED,
    CONST_EXPR_ARGUMENT_REQUIRED,
    USING_ALIAS_REDECLARATION,
    USING_ALIAS_DECL_FORBIDDEN,
    PACKAGE_NAME_CONTAINS_FORBIDDEN_CHAR,
    ABSOLUTE_ALIAS_NOT_AT_BEGINNING,

    MAX_ERROR_NUM = ABSOLUTE_ALIAS_NOT_AT_BEGINNING,
    EXTERNAL_APPLICATION_ERROR = 998,
    INTERNAL_COMPILER_ERROR = 999,
};

/// Archiver tool errors.
enum Archiver_error {
    MANIFEST_MISSING,

    PACKAGE_NAME_INVALID = 100,
    EXTRA_FILES_FOUND,
    DIRECTORY_MISSING,
    ARCHIVE_ALREADY_EXIST,
    ARCHIVE_DOES_NOT_EXIST,
    CANT_OPEN_ARCHIVE,
    MEMORY_ALLOCATION,
    RENAME_FAILED,
    IO_ERROR,
    CRC_ERROR,
    CREATE_FILE_FAILED,
    MANIFEST_BROKEN,
    KEY_NULL_PARAMETERS,
    VALUE_MUST_BE_IN_TIME_FORMAT,
    VALUE_MUST_BE_IN_SEMA_VERSION_FORMAT,
    FORBIDDEN_KEY,
    SINGLE_VALUED_KEY,
    INACCESSIBLE_MDL_FILE,
    MDL_FILENAME_NOT_IDENTIFIER,
    FAILED_TO_OPEN_TEMPFILE,
    FAILED_TO_REMOVE,
    INVALID_MDL_ARCHIVE,
    INVALID_PASSWORD,
    READ_ONLY_ARCHIVE,
    ARCHIVE_DOES_NOT_CONTAIN_ENTRY,
    INVALID_MDL_ARCHIVE_NAME,
    EMPTY_ARCHIVE_CONTENT,
    EXTRA_FILES_IGNORED,
    MDR_INVALID_HEADER,
    MDR_INVALID_HEADER_VERSION,
    MDR_PRE_RELEASE_VERSION,

    INTERNAL_ARCHIVER_ERROR = 999,
};

/// Encapsulate tool errors.
enum Encapsulator_error {
    MDLE_FILE_ALREADY_EXIST = 100,
    MDLE_FILE_DOES_NOT_EXIST,
    MDLE_CANT_OPEN_FILE,
    MDLE_MEMORY_ALLOCATION,
    MDLE_RENAME_FAILED,
    MDLE_IO_ERROR,
    MDLE_CRC_ERROR,
    MDLE_FAILED_TO_OPEN_TEMPFILE,
    MDLE_INVALID_MDLE,
    MDLE_INVALID_PASSWORD,
    MDLE_DOES_NOT_CONTAIN_ENTRY,
    MDLE_INVALID_NAME,
    MDLE_INVALID_USER_FILE,
    MDLE_INVALID_RESOURCE,
    MDLE_CONTENT_FILE_INTEGRITY_FAIL,
    MDLE_INVALID_HEADER,
    MDLE_INVALID_HEADER_VERSION,
    MDLE_FAILED_TO_ADD_ZIP_COMMENT,
    MDLE_PRE_RELEASE_VERSION,

    MDLE_INTERNAL_ERROR = 999,
};

/// JIT backend errors.
enum Jit_backend_error {
    COMPILING_LLVM_CODE_FAILED = 100,
    LINKING_LIBDEVICE_FAILED,
    LINKING_LIBBSDF_FAILED,
    PARSING_LIBDEVICE_MODULE_FAILED,
    PARSING_LIBBSDF_MODULE_FAILED,
    PARSING_STATE_MODULE_FAILED,
    DEMANGLING_NAME_OF_EXTERNAL_FUNCTION_FAILED,
    WRONG_FUNCTYPE_FOR_MDL_RUNTIME_FUNCTION,
    LINKING_STATE_MODULE_FAILED,
    STATE_MODULE_FUNCTION_MISSING,
    WRONG_RETURN_TYPE_FOR_STATE_MODULE_FUNCTION,
    API_STRUCT_TYPE_MUST_BE_OPAQUE,
    GET_SYMBOL_FAILED,
    LINKING_LIBMDLRT_FAILED,
    PARSING_RENDERER_MODULE_FAILED,
    LINKING_RENDERER_MODULE_FAILED,

    INTERNAL_JIT_BACKEND_ERROR = 999,
};

/// Comparator errors.
enum Comparator_error {
    OTHER_DEFINED_AT,
    MISSING_STRUCT_MEMBER,
    DIFFERENT_STRUCT_MEMBER_TYPE,
    ADDED_STRUCT_MEMBER,
    MISSING_ENUM_VALUE,
    DIFFERENT_ENUM_VALUE,
    ADDED_ENUM_VALUE,

    TYPE_DOES_NOT_EXISTS = 100,
    TYPES_DIFFERENT,
    INCOMPATIBLE_STRUCT,
    INCOMPATIBLE_ENUM,
    DIFFERENT_MDL_VERSIONS,
    DIFFERENT_DEFAULT_ARGUMENT,
    CONSTANT_DOES_NOT_EXISTS,
    CONSTANT_OF_DIFFERENT_TYPE,
    CONSTANT_OF_DIFFERENT_VALUE,
    FUNCTION_DOES_NOT_EXISTS,
    FUNCTION_RET_TYPE_DIFFERENT,
    FUNCTION_PARAM_DELETED,
    FUNCTION_PARAM_DEF_ARG_DELETED,
    FUNCTION_PARAM_DEF_ARG_CHANGED,
    ANNOTATION_DOES_NOT_EXISTS,
    ANNOTATION_PARAM_DELETED,
    ANNOTATION_PARAM_DEF_ARG_DELETED,
    ANNOTATION_PARAM_DEF_ARG_CHANGED,
    SEMA_VERSION_IS_HIGHER,
    ARCHIVE_DOES_NOT_CONTAIN_MODULE,

    INTERNAL_COMPARATOR_ERROR = 999,
};

/// Module_transformer errors.
enum Transformer_error {
    SOURCE_MODULE_INVALID,
    INLINING_MODULE_FAILED,

    INTERNAL_TRANSFORMER_ERROR = 999
};

/// Get the error template for the given code.
///
/// \param code       the error code
/// \param msg_class  the message class for the error
char const *get_error_template(
    int  code,
    char msg_class);

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
        EK_ARRAY_TYPE,
        EK_SYMBOL,
        EK_INTEGER,
        EK_POS,
        EK_STRING,
        EK_DOT_STRING,
        EK_QUAL_NAME,
        EK_EXPR,
        EK_SEM_VER,
        EK_OPERATOR,
        EK_DIRECTION,
        EK_SIGNATURE,
        EK_SIGNATURE_NO_RT,
        EK_VALUE,
        EK_QUALIFIER,
        EK_OPT_MESSAGE,
        EK_PATH_PREFIX,
        EK_PATH_KIND,
        EK_ENTITY_KIND,
        EK_MODULE_ORIGIN,
        EK_MDL_VERSION,
        EK_CHAR
    };

    /// File path prefix.
    enum File_path_prefix {
        FPP_ABSOLUTE,
        FPP_WEAK_RELATIVE,
        FPP_STRICT_RELATIVE
    };

    /// Path kind.
    enum Path_kind {
        PK_CURRENT_SEARCH_PATH,
        PK_CURRENT_DIRECTORY
    };

    /// Entity kind.
    enum Entity_kind {
        EK_MODULE,
        EK_RESOURCE
    };

private:
    struct Entry {
        Kind kind;

        union U {
            struct {
                IType const         *type;
                bool                suppress_prefix;
            }                       type;
            struct {
                IType const         *e_type;
                int                 size;
            }                       a_type;
            ISymbol const           *sym;
            int                     i;
            int                     pos;
            char const              *string;
            struct {
                IQualified_name const *name;
                bool                  ignore_last;
            }                       qual_name;
            IExpression const       *expr;
            IExpression::Operator   op;
            Direction               dir;
            IDefinition const       *sig;
            IValue const            *val;
            Qualifier               qual;
            File_path_prefix        file_path_prefix;
            Path_kind               path_kind;
            Entity_kind             entity_kind;
            ISemantic_version const *sem_ver;
            IMDL::MDL_version       mdl_version;
            unsigned char           c;
        } u;
    };

public:

    /// Get the kind of given argument
    ///
    /// \param index  the argument index
    Kind get_kind(size_t index) const;

    /// Return the number of arguments.
    size_t get_arg_count() const { return m_args.size(); }

    /// Add a symbol argument.
    ///
    /// \param sym  the symbol
    Error_params &add(ISymbol const *sym);

    /// Return the type argument of given index.
    ///
    /// \param index  the argument index
    ISymbol const *get_symbol_arg(size_t index) const;

    /// Add a type argument.
    ///
    /// \param type     the type
    /// \param suppress_prefix  if true, do not write "struct/enum" if front
    Error_params &add(IType const *type, bool supress_prefix = false);

    /// Return the type argument of given index.
    ///
    /// \param[in]  index           the argument index
    /// \param[out] supress_prefix  if true, do not print the prefix
    IType const *get_type_arg(size_t index, bool &suppress_prefix) const;

    /// Add an array type argument.
    ///
    /// \param type  the type
    Error_params &add(IType const *e_type, int size);

    /// Return the array type size argument of given index.
    ///
    /// \param index  the argument index
    int get_type_size_arg(size_t index) const;

    /// Add an integer argument.
    ///
    /// \param v  the integer
    Error_params &add(int v);

    /// Return the integer argument of given index.
    ///
    /// \param index  the argument index
    int get_int_arg(size_t index) const;

    /// Add an operator kind argument.
    ///
    /// \param op  the operator
    Error_params &add(IExpression::Operator op);

    /// Add an operator kind argument.
    ///
    /// \param op  the operator
    Error_params &add(IExpression_binary::Operator op) { return add(IExpression::Operator(op)); }

    /// Add an operator kind argument.
    ///
    /// \param op  the operator
    Error_params &add(IExpression_unary::Operator op)  { return add(IExpression::Operator(op)); }

    /// Return the operator kind argument of given index.
    ///
    /// \param index  the argument index
    IExpression::Operator get_op_arg(size_t index) const;

    /// Add an direction argument.
    ///
    /// \param dir  the direction
    Error_params &add(Direction dir);

    // Return the direction argument of given index.
    ///
    /// \param index  the argument index
    Direction get_dir_arg(size_t index) const;

    /// Add a string argument.
    ///
    /// \param s  the string
    Error_params &add(char const *s);

    /// Add a string argument.
    ///
    /// \param s  the string
    Error_params &add(string const &s);

    /// Return the string argument of given index.
    ///
    /// \param index  the argument index
    char const *get_string_arg(size_t index) const;

    /// Add a qualified name argument.
    ///
    /// \param q            the qualified name
    /// \param ignore_last  if true, ignore the last component
    Error_params &add(IQualified_name const *q, bool ignore_last = false);

    /// Return the qualified name argument of given index.
    ///
    /// \param index        the argument index
    /// \param ignore_last  if set to true after return, the last component must be ignored
    IQualified_name const *get_qual_name_arg(size_t index, bool &ignore_last) const;

    /// Add an expression argument.
    ///
    /// \param expr  the expression
    Error_params &add(IExpression const *expr);

    /// Return the expression argument of given index.
    ///
    /// \param index  the argument index
    IExpression const *get_expr_arg(size_t index) const;

    /// Add a semantic version argument.
    ///
    /// \param ver  the version
    Error_params &add(ISemantic_version const *ver);

    /// Return the semantic version argument of given index.
    ///
    /// \param index  the argument index
    ISemantic_version const *get_sem_ver_arg(size_t index) const;

    /// Add a positional argument as a num word.
    ///
    /// \param pos  the position
    Error_params &add_numword(size_t pos);

    /// Return the positional argument of given index.
    ///
    /// \param index  the argument index
    int get_pos_arg(size_t index) const;

    /// Add a function signature (including return type).
    ///
    /// \param def  a definition of a function of constructor
    Error_params &add_signature(IDefinition const *def);

    /// Add a function signature (without return type).
    ///
    /// \param def  a definition of a function of constructor
    Error_params &add_signature_no_rt(IDefinition const *def);

    /// Return the signature argument of given index.
    ///
    /// \param index  the argument index
    IDefinition const *get_signature_arg(size_t index) const;

    /// Add a value argument.
    ///
    /// \param val  the value
    Error_params &add(IValue const *val);

    /// Return the value argument of given index.
    ///
    /// \param index  the argument index
    IValue const *get_value_arg(size_t index) const;

    /// Add a qualifier argument.
    ///
    /// \param q  the qualifier
    Error_params &add(Qualifier q);

    /// Return the qualifier argument of given index.
    ///
    /// \param index  the argument index
    Qualifier get_qualifier_arg(size_t index) const;

    /// Add a possible match.
    ///
    /// \param sym  the possible match for a misspelled symbol
    Error_params &add_possible_match(ISymbol const *sym);

    /// Return the possible match symbol if any.
    ISymbol const *get_possible_match() const { return m_possible_match; }

    /// Add an optional string value argument.
    ///
    /// \param val  the value
    Error_params &add_opt_message(IValue_string const *msg);

    /// Return the optional string value argument of given index.
    ///
    /// \param index  the argument index
    IValue_string const *get_opt_message_arg(size_t index) const;

    /// Add an absolute path prefix.
    Error_params &add_absolute_path_prefix();

    /// Add a weak relative path prefix.
    Error_params &add_weak_relative_path_prefix();

    /// Add a strict relative path prefix.
    Error_params &add_strict_relative_path_prefix();

    /// Add an MDL version.
    ///
    /// \param ver  the version
    Error_params &add_mdl_version(IMDL::MDL_version ver);

    /// Return the MDL version.
    ///
    /// \param index  the argument index
    IMDL::MDL_version get_mdl_version(size_t index) const;

    /// Return the path prefix.
    ///
    /// \param index  the argument index
    File_path_prefix get_path_prefix(size_t index) const;

    /// Add a "current search path" path kind.
    Error_params &add_current_search_path();

    /// Add a "current directory" path kind
    Error_params &add_current_directory();

    /// Add a module origin.
    char const *get_module_origin(size_t index) const;

    /// Add a module origin.
    Error_params &add_module_origin(char const *origin);

    /// Return the path kind.
    ///
    /// \param index  the argument index
    Path_kind get_path_kind(size_t index) const;

    /// Add an entity kind.
    Error_params &add_entity_kind(Entity_kind kind);

    /// Return the entity kind.
    ///
    /// \param index  the argument index
    Entity_kind get_entity_kind(size_t index) const;

    /// Add a dot (if the string is non-empty) and a string.
    Error_params &add_dot_string(char const *s);

    /// Return the dot string.
    ///
    /// \param index  the argument index
    char const *get_dot_string(size_t index) const;

    /// Add a character.
    Error_params &add_char(char c);

    /// Return the character.
    ///
    /// \param index  the argument index
    unsigned char get_char(size_t index) const;

    /// Add a definition kind (is converted into a string).
    Error_params &add_entity_kind(IDefinition::Kind kind);

    /// Add 'package' or 'archive'.
    Error_params &add_package_or_archive(bool is_package);

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

public:
    /// Constructor.
    ///
    /// \param ana  an Analysis pass
    explicit Error_params(Analysis const &ana);

    /// Constructor.
    ///
    /// \param alloc  an Allocator interface
    explicit Error_params(IAllocator *alloc);

private:
    /// The used allocator.
    IAllocator *m_alloc;

    /// List of all collected argument.
    vector<Entry>::Type m_args;

    /// If non-null, a possible match for a misspelled symbol.
    ISymbol const *m_possible_match;
};

/// Format and print a message.
///
/// \param code       the error code
/// \param msg_class  the message class
/// \param params     error parameter/inserts
/// \param printer    a printer that will be used to print the message
void print_error_message(
    int                code,
    char               msg_class,
    Error_params const &params,
    IPrinter           *printer);

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_ERRORS_H
