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
#include "pch.h"

#include <cctype>

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_printers.h>

#include "compilercore_errors.h"
#include "compilercore_assert.h"
#include "compilercore_tools.h"

namespace mi {
namespace mdl {

// FIXME: must be generated for every supported language
struct Numwords {
    size_t     num;
    char const *word;
};

static Numwords const en_words[] = {
    { 1, "first" },
    { 2, "second" },
    { 3, "third" },
    { 4, "fourth" },
    { 0, NULL }
};

// Get the error template for the given code.
// FIXME: this must be generated
char const *get_error_template(
    int  code,
    char msg_class)
{
    if (msg_class == 'C') {
        Compilation_error err = Compilation_error(code);
        switch (err) {
        case PREVIOUS_DEFINITION:
            return "'$0' previously defined here";
        case PREVIOUSLY_USED:
            return "previously used here";
        case DEFINED_AT:
            return "'$0' was defined here";
        case DECLARED_AT:
            return "'$0' was declared here";
        case FIRST_DEFAULT:
            return "this is the first default label";
        case RECURSION_CHAIN:
            return "'$0' called by '$1'";
        case SELF_RECURSION:
            return "'$0' is self recursive";
        case BOUND_IN_CONTEXT:
            return "parameter type '$0' is bound to '$1' in this context";
        case ASSIGNED_AT_PARAM_NAME:
            return "'$0' was assigned at parameter '$1'";
        case ASSIGNED_AT_PARAM_POS:
            return "'$0' was assigned at $1 parameter";
        case FUNCTION_CALLS_ARE_STATE_INDEPENDENT:
            return "all function calls inside this expression are state-independent";
        case WRITTEN_BUT_NOT_READ:
            return "'$0' is written to but never read";
        case LET_TEMPORARIES_ARE_RVALUES:
            return "let expression temporary '$0' is a rvalue";
        case SAME_CONDITION:
            return "same condition here";
        case IMPORT_ENUM_TYPE:
            return "'$0' is an enum value, import the enum type '$1' instead";
        case DEFAULT_ARG_DECLARED_AT:
            return "default argument '$0' was declared here";
        case CURR_DIR_FORBIDDEN_IN_RESOURCE:
            return "the current directory '.' cannot be used in a resource file path in MDL $0.$1";
        case PARENT_DIR_FORBIDDEN_IN_RESOURCE:
            return "the parent directory '..' cannot be used in a resource file path in MDL $0.$1";
        case IMPORTED_WITHOUT_EXPORT:
            return "'$0' was imported here but not exported";
        case USES_FUTURE_FEATURE:
            return "'$0' exists but cannot be imported, it uses features"
                   " not available in this version of MDL";
        case TYPE_CONTAINS_FORBIDDEN_SUBTYPE:
            return "'$0' contains forbidden type '$1'";
        case PREVIOUSLY_ANNOTATED:
            return "$0 was annotated previously here";
        case ONLY_REFERNCED_INSIDE_DEFAULT_INITIALIZER:
            return "'$0' is only referenced inside a parameter default initializer";
        case LOCATED_AT:
            return "located at '$0'";
        case POSSIBLE_MATCH:
            return "; did you mean '$0'?";
        case CANDIDATES_ARE:
            return "candidates are: $0";
        case CANDIDATES_ARE_NEXT:
            return "                $0";
        case POSSIBLE_ABSOLUTE_IMPORT:
            return "did you mean '$0'?";

        case SYNTAX_ERROR:
            return "syntax error: $0";
        case ILLEGAL_BOM_IGNORED:
            return "Illegal byte order mark at start of file ignored";
        case UNDETERMINED_COMMENT:
            return "unexpected end of file found in comment";
        case INTEGER_OVERFLOW:
            return "Integer constant overflow";
        case ENT_REDECLARATION:
            return "redeclaration of '$0'";
        case TYPE_REDECLARATION:
            return "redeclaration of type '$0'";
        case PARAMETER_REDECLARATION:
            return "redeclaration of parameter '$0'";
        case REDECLARATION_OF_DIFFERENT_KIND:
            return "redeclaration of '$0' as a different kind of symbol";
        case UNKNOWN_PACKAGE_NAME:
            return "'$0' is not a package or module name";
        case UNKNOWN_IDENTIFIER:
            return "'$0' has not been declared";
        case UNKNOWN_TYPE:
            return "type '$0' has not been declared";
        case UNKNOWN_MEMBER:
            return "type '$0' has no member '$1'";
        case NON_CONSTANT_INITIALIZER:
            return "'$0' initializer is not constant";
        case MISSING_CONSTANT_INITIALIZER:
            return "constant '$0' has no initializer";
        case ANNONATION_NON_CONST_DEF_PARAM:
            return "annotation default parameter '$0' must be a constant expression";
        case FUNCTION_BODY_NOT_BLOCK:
            return "body of a function definition must be a compound statement";
        case MATERIAL_BODY_NOT_EXPR:
            return "body of a material definition must be an expression";
        case MATERIAL_PRESET_BODY_NOT_CALL:
            return "body of a material preset must be an material instantiation";
        case NOT_A_MATERIAL_NAME:
            return "'$0' is not a material";
        case MATERIAL_QUALIFIER_NOT_ALLOW:
            return "qualifier '$0' not allowed in material definition";
        case MODULE_NOT_FOUND:
            return "could not find module '$0' in module path";
        case ERRONEOUS_IMPORT:
            return "Imported module '$0' contains errors";
        case CONSTRUCTOR_SHADOWED:
            return "field '$0 $1::$2' with same name as $3";
        case PARAMETER_SHADOWED:
            return "declaration shadows parameter '$0'";
        case UNKNOWN_EXPORT:
            return "module '$0' does not contain '$1'";
        case ENT_NOT_EXPORTED:
            return "'$0' is not exported from module '$1'";
        case INDEX_OF_NON_ARRAY:
            return "attempting to index a non-array type '$0'";
        case ILLEGAL_INDEX_TYPE:
            return "array subscript is not an integer but '$0'";
        case CONDITION_BOOL_TYPE_REQUIRED:
            return "could not convert condition expression from type '$0' to 'bool'";
        case MISMATCHED_CONDITIONAL_EXPR_TYPES:
            return "types in conditional expression do not match";
        case ILLEGAL_SWITCH_CONTROL_TYPE:
            return "switch quantity neither int nor enum type but '$0'";
        case ILLEGAL_CASE_LABEL_TYPE:
            return "case label neither int nor enum type but '$0'";
        case CANNOT_CONVERT_CASE_LABEL_TYPE:
            return "case label cannot be converted to type '$0'";
        case MULTIPLE_DEFAULT_LABEL:
            return "multiple default labels in one switch";
        case MULTIPLE_CASE_VALUE:
            return "case value '$0' already used";
        case MISSING_CASE_LABEL:
            return "unhandled switch case '$0'";
        case CASE_LABEL_NOT_CONSTANT:
            return "case label does not reduce to a constant";
        case DEFAULT_WITHOUT_CASE:
            return "switch statement contains 'default' but no 'case' labels";
        case INVALID_CONVERSION:
            return "invalid conversion from '$0' to '$1'";
        case INVALID_CONVERSION_RETURN:
            return "cannot convert '$0' to '$1' in return";
        case NOT_A_TYPE_NAME:
            return "'$0' does not name a type";
        case REEXPORTING_QUALIFIED_NAMES_FORBIDDEN:
            return "re-exporting qualified names is forbidden";
        case FORBIDDED_GLOBAL_STAR:
            return "Missing module name before star import '*'";
        case MODULE_NAME_GIVEN_NOT_ENTITY:
            return "'$0' names a module, not an entity, use '$0::*' to import a whole module";
        case FORBIDDEN_RETURN_TYPE:
            return "forbidden return type '$0' for function '$1'";
        case CALLED_OBJECT_NOT_A_FUNCTION:
            return "called object $0$1$2is not a function";
        case POS_ARGUMENT_AFTER_NAMED:
            return "positional argument $0 after named argument '$1' is forbidden";
        case NO_MATCHING_OVERLOAD:
            return "no overloaded version of function '$0' matches calling parameters";
        case AMBIGUOUS_OVERLOAD:
            return "ambiguous call to overloaded function: '$0'";
        case AMBIGUOUS_UNARY_OPERATOR:
            return "unary '$0' : no operator found which takes '$1' operand";
        case AMBIGUOUS_BINARY_OPERATOR:
            return "binary '$0' : no operator found which takes '$1' and '$2' operands";
        case CANNOT_CONVERT_POS_ARG_TYPE:
            return "'$0' : cannot convert $1 argument from '$2' to '$3'";
        case CANNOT_CONVERT_ARG_TYPE:
            return "'$0' : cannot convert argument '$1' from '$2' to '$3'";
        case CANNOT_CONVERT_POS_ARG_TYPE_FUNCTION:
            return "'$0' : cannot pass a function as $1 argument";
        case CANNOT_CONVERT_ARG_TYPE_FUNCTION:
            return "'$0' : cannot pass a function as argument '$1'";
        case MISSING_PARAMETER:
            return "'$0' : missing parameter '$1'";
        case PARAMETER_PASSED_TWICE:
            return "'$0' : parameter '$1' passed twice";
        case UNKNOWN_NAMED_PARAMETER:
            return "'$0' has no parameter named '$1'";
        case ILLEGAL_LVALUE_ASSIGNMENT:
            return "'$0' : $1 operand must be an l-value";
        case OPERATOR_LEFT_ARGUMENT_ILLEGAL:
            return "'$0' : illegal, left operand has type '$1'";
        case OPERATOR_RIGHT_ARGUMENT_ILLEGAL:
            return "'$0' : illegal, right operand has type '$1'";
        case CONFLICTING_REDECLARATION:
            return "declaration of '$0' conflicts with previous declaration of '$0'";
        case FUNCTION_REDEFINITION:
            return "redefinition of function: '$0'";
        case FUNCTION_REDECLARED_DIFFERENT_RET_TYPE:
            return "redeclaration of function '$0' with return type '$1', was '$2'";
        case REDECLARED_DIFFERENT_PARAM_NAME:
            return "redeclaration of '$0' with $1 parameter named '$2', was '$3'";
        case DEFAULT_PARAM_REDEFINITION:
            return "redefinition of $0 default parameter '$1' of '$2'";
        case FUNCTION_PROTOTYPE_USELESS:
            return "useless function prototype for '$0'";
        case ENUMERATOR_VALUE_NOT_INT:
            return "enumerator value for '$0' not integer constant";
        case ARRAY_SIZE_NON_CONST:
            return "size of array is not a constant expression";
        case ARRAY_SIZE_NOT_INTEGRAL:
            return "size of array has non-integral type '$0'";
        case ARRAY_SIZE_NEGATIVE:
            return "size of array is negative";
        case ARRAY_INDEX_OUT_OF_RANGE:
            return "array index out of range: $0 $1 $2";
        case NO_ARRAY_CONVERSION:
            return "cannot convert arrays from '$0' to '$1'";
        case CANNOT_CONVERT_ARRAY_ELEMENT:
            return "cannot convert $0 argument of array constructor from '$1' to '$2'";
        case MISSING_DEFAULT_ARGUMENT:
            return "default argument missing for parameter $0 of '$1'";
        case MISSING_DEFAULT_FIELD_INITIALIZER:
            return "default initializer missing for field '$0' of 'struct $1'";
        case RECURSION_CALL_NOT_ALLOWED:
            return "recursive function '$0' not allowed";
        case CALL_OF_UNDEFINED_FUNCTION:
            return "Called function '$0' is not defined";
        case ABSTRACT_ARRAY_RETURN_TYPE:
            return "abstract arrays cannot be defined in a functions return type";
        case ABSTRACT_ARRAY_INSIDE_STRUCT:
            return "forbidden abstract array type '$0' for member '$1'";
        case FORBIDDEN_SELF_REFERENCE:
            return "the initializer of '$0' references itself";
        case UNUSED_VARIABLE:
            return "unused variable '$0'";
        case UNUSED_CONSTANT:
            return "unused constant '$0'";
        case UNUSED_PARAMETER:
            return "unused parameter '$0'";
        case DIVISION_BY_ZERO_IN_CONSTANT_EXPR:
            return "division by zero in constant expression";
        case NOT_A_ANNOTATION:
            return "'$0' is not an annotation";
        case NON_CONSTANT_ANNOTATION_ARG_POS:
            return "$0 annotation parameter is not constant";
        case NON_CONSTANT_ANNOTATION_ARG_NAME:
            return "annotation parameter '$0' is not constant";
        case ANNOTATION_IGNORED:
            return "annotation '$0' will be ignored";
        case ENUM_VALUE_NOT_HANDLED_IN_SWITCH:
            return "enumeration value '$0' not handled in switch";
        case SUGGEST_PARENTHESES_AROUND_ASSIGNMENT:
            return "suggest parentheses around assignment used as truth value";
        case WRONG_MATERIAL_PRESET:
            return "material preset must be declared using 'material'";
        case TOO_MANY_PRESET_PARAMS:
            return "material $'0' has only $1 parameters";
        case CONFLICTING_PROTOTYPE:
            return "conflicting prototype for '$0'";
        case NAME_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER:
            return "argument '$0': value may depend on evaluation order, uses value of '$1'";
        case POS_ARGUMENT_VALUE_DEPENDS_ON_EVALUATION_ORDER:
            return "$0 argument: value may depend on evaluation order, uses value of '$1'";
        case EMPTY_CONTROL_STMT:
            return "empty $0 statement found; is this the intent?";
        case UNREACHABLE_STATEMENT:
            return "statement is unreachable";
        case BREAK_OUTSIDE_LOOP_OR_SWITCH:
            return "break statement outside of loop or switch";
        case CONTINUE_OUTSIDE_LOOP:
            return "continue statement outside of loop";
        case EFFECTLESS_STATEMENT:
            return "statement has no effect";
        case FUNCTION_MUST_RETURN_VALUE:
            return "function '$0' must return a value";
        case WRONG_COMPARISON_BETWEEN_ENUMS:
            return "comparison between '$0' and '$1'";
        case FUNCTION_OVERLOADING_ON_MODIFIER:
            return "'$0' signature differs on type modifiers only";
        case FUNCTION_PREVIOUSLY_EXPORTED:
            return "'$0' will be exported because was previously declared exported";
        case FUNCTION_NOW_EXPORTED:
            return "'$0' will be exported even if previously not declared exported";
        case FUNCTION_REDECLARED_DIFFERENT_QUALIFIER:
            return "redeclaration of function '$0' with modifier '$1', was '$2'";
        case FORBIDDEN_VARIABLE_TYPE:
            return "variables of type '$0' are forbidden";
        case FORBIDDEN_CONSTANT_TYPE:
            return "cannot create constant of type '$0'";
        case IMPORT_SYMBOL_CLASH:
            return "cannot import '$0', this name is already used by '$1' imported from '$2'";
        case OVERLOADS_MUST_BE_EXPORTED:
            return "inconsistent export of an overload set: '$0' not exported";
        case LET_USED_OUTSIDE_MATERIAL:
            return "let expressions are only allowed inside material declarations";
        case DEFAULT_NOT_EXPORTED:
            return "non-exported '$0' is used inside a default initializer of exported '$1'";
        case VARYING_CALL_FROM_UNIFORM_FUNC:
            return "forbidden varying call of '$0' from function declared uniform";
        case UNIFORM_FUNCTION_DECLARED_WITH_VARYING_RESULT:
            return "uniform function '$0' cannot return a varying value";
        case RESULT_OF_UNIFORM_FUNCTION_IS_VARYING:
            return "return value of uniform '$0' is varying";
        case NONVARYING_RESULT_OF_VARYING_FUNCTION:
            return "return value of varying '$0' is not varying, declare as uniform";
        case VARIABLE_DEPENDS_ON_AUTOTYPED_PARAMETER:
            return "cannot declare variable '$0' of type '$1',"
                   " its value depends on auto-typed parameters";
        case VARIABLE_DEPENDS_ON_VARYING_VALUE:
            return "cannot declare variable '$0' of type '$1',"
                " its value depends on varying values";
        case CANNOT_CONVERT_POS_ARG_TYPE_PARAM_DEP:
            return "'$0' : cannot convert $1 argument from '$2' to '$3': "
                "depends on parameter type";
        case CANNOT_CONVERT_ARG_TYPE_PARAM_DEP:
            return "'$0' : cannot convert argument '$1' from '$2' to '$3': "
                "depends on parameter type";
        case CONSTANT_TOO_BIG:
            return "Constant too big";
        case UNSUPPORTED_MDL_VERSION:
            return "Unsupported MDL version $0.$1";
        case FUNCTION_ANNOTATIONS_NOT_AT_PROTOTYPE:
            return "Function annotations must be placed at the first function prototype";
        case EXPORTED_FUNCTION_NOT_DEFINED:
            return "exported function '$0' not defined";
        case UNREFERENCED_FUNCTION_REMOVED:
            return "unreferenced local function '$0' has been removed";
        case PARAMETER_ONLY_WRITTEN:
            return "parameter '$0' is never read, only written";
        case USED_TYPE_NOT_EXPORTED:
            return "exported '$0' uses non-exported type '$1'";
        case OPERATOR_EQUAL_WITHOUT_EFFECT:
            return "operator '$0' has no effect, maybe '=' was indented?";
        case OPERATOR_WITHOUT_EFFECT:
            return "operator '$0' has no effect";
        case IDENTICAL_SUBEXPRESSION:
            return "identical subexpression to the left and to the right of the '$0' operator";
        case IDENTICAL_THEN_ELSE:
            return "the 'then' statement is equivalent to the 'else' statement";
        case ARRAY_CONSTRUCTOR_ARGUMENT_MISMATCH:
            return "array constructor of type '$0[$1]' must have zero or $2 arguments";
        case ABSTRACT_ARRAY_CONSTRUCTOR_ARGUMENT:
            return "abstract array constructors cannot have arguments";
        case IDENTICAL_IF_CONDITION:
            return "identical conditions in an if-cascade";
        case UNCONDITIONAL_EXIT_IN_LOOP:
            return "unconditional $0 statement inside a loop";
        case OPERATOR_IS_STRICT_IN_MATERIAL:
            return "operator $0 is strict inside a material";
        case IMPORT_ENUM_VALUE_SYMBOL_CLASH:
            return "cannot import enum value '$0' from import '$1', "
                   "this name is already used by '$2' imported from '$3'";
        case CANNOT_CONVERT_DEFAULT_ARG_TYPE_FUNCTION:
            return "'$0' : cannot pass a function as default argument '$1'";
        case CANNOT_CONVERT_DEFAULT_ARG_TYPE:
            return "'$0' : cannot convert default argument '$1' from '$2' to '$3'";
        case FORBIDDEN_FIELD_TYPE:
            return "structure fields of type '$0' are forbidden";
        case RESOURCE_PARAMETER_NOT_UNIFORM:
            return "parameter '$0' of type '$1' must be uniform";
        case IGNORED_ON_NON_FUNCTION:
            return "annotation '$0' will be ignored because '$1' is not a function";
        case IMPORT_LOOP:
            return "cannot import '$0', loop import detected";
        case UNREFERENCED_MATERIAL_REMOVED:
            return "unreferenced local material '$0' has been removed";
        case TOO_MANY_ARGUMENTS:
            return "too many arguments to '$0'";
        case ARRAY_CONSTRUCTOR_DOESNT_HAVE_NAMED_ARG:
            return "named argument '$0' not allowed in an array constructor";
        case SEQUENCE_WITHOUT_EFFECT_INSIDE_MATERIAL:
            return "sequence expression has no effect inside a material body";
        case FORBIDDEN_ARRAY_TYPE:
            return "arrays of type '$0' are forbidden";
        case FORBIDDEN_NESTED_ARRAY:
            return "nested arrays are forbidden";
        case FORBIDDEN_FUNCTION_PARAMETER_TYPE:
            return "function parameters of type '$0' are forbidden";
        case FORBIDDEN_MATERIAL_PARAMETER_TYPE:
            return "material parameters of type '$0' are forbidden";
        case TERNARY_COND_NOT_UNIFORM:
            return "conditional operator on type '$0' must have uniform condition";
        case WRONG_ANNOTATION_RANGE_INTERVAL:
            return "wrong range annotation: ($0, $1) is not a valid $2 interval";
        case DEFAULT_INITIALIZER_OUTSIDE_HARD_RANGE:
            return "default initializer '$0' is outside the given hard_range ($1, $2)";
        case TYPE_DOES_NOT_SUPPORT_RANGE:
            return "type '$0' cannot be annotated by a range of type '$1'";
        case ADDITIONAL_ANNOTATION_IGNORED:
            return "additional $0 ignored";
        case ENTITY_DOES_NOT_SUPPORT_ANNOTATION:
            return "'$0' cannot be annotated by a $1";
        case RETURN_VALUE_DOES_NOT_SUPPORT_ANNOTATION:
            return "return values cannot be annotated by a $0";
        case ARGUMENT_OUTSIDE_HARD_RANGE:
            return "argument '$0' is outside the given hard_range ($1, $2) for parameter '$3'";
        case FIELD_VALUE_OUTSIDE_HARD_RANGE:
            return "value '$0' is outside the given hard_range ($1, $2) for struct member '$3'";
        case SOFT_RANGE_OUTSIDE_HARD_RANGE:
            return "soft_range($0, $1) is outside the specified hard_range($2, $3)";
        case FUNCTION_PTR_USED:
            return "Missing parentheses on function call";
        case USED_VARIABLE_MARKED_UNUSED:
            return "used variable '$0' marked with anno::unused() annotation";
        case USED_CONSTANT_MARKED_UNUSED:
            return "used constant '$0' marked with anno::unused() annotation";
        case USED_PARAMETER_MARKED_UNUSED:
            return "used parameter '$0' marked with anno::unused() annotation";
        case USED_FUNCTION_MARKED_UNUSED:
            return "used function '$0' marked with anno::unused() annotation";
        case USED_MATERIAL_MARKED_UNUSED:
            return "used material '$0' marked with anno::unused() annotation";
        case ASSIGNMENT_INSIDE_MATERIAL:
            return "operator '$0' is forbidden inside a material";
        case DIVISION_BY_ZERO:
            return "division by zero";
        case INCOMPLETE_ARRAY_SPECIFICATION:
            return "forbidden incomplete array specification";
        case ARRAY_ASSIGNMENT_FORBIDDEN:
            return "assignment operator '$0' is forbidden on arrays";
        case RESOURCE_NAME_NOT_LITERAL:
            return "the name of a resource must be a string literal";
        case FORBIDDED_C_STYLE_CAST:
            return "forbidden C-style cast, use function style cast instead";
        case EXTRA_PARENTHESIS_AROUND_FUNCTION_NAME:
            return "extra parenthesis around function name '$0'";
        case DEPRECATED_ENTITY:
            return "'$0' is deprecated$1";
        case FORBIDDEN_QUALIFIER_ON_CONST_DECL:
            return "forbidden type qualifier '$0' on a constant declaration";
        case DOUBLE_TYPE_USED:
            return "double type is used";
        case MODULE_DOES_NOT_SUPPORT_ANNOTATION:
            return "modules cannot be annotated by a $0";
        case MATERIAL_BODY_IS_NOT_A_MATERIAL:
            return "material body is not a material expression but of type '$0'";
        case UNIFORM_RESULT_DEPENDS_ON_AUTOTYPED_PARAMETER:
            return "return value of '$0' depends on auto-typed parameters";
        case UNIFORM_RESULT_IS_VARYING:
            return "return value of '$0' is varying";
        case FUNCTION_PRESET_BODY_NOT_CALL:
            return "body of a function preset must be a function instantiation";
        case FUNCTION_PRESET_WRONG_RET_TYPE:
            return "function preset's return type must be '$1' not '$0'";
        case RELATIVE_IMPORTS_NOT_SUPPORTED:
            return "relative imports are not allowed in MDL $0.$1";
        case EXPORTED_PRESET_OF_UNEXPORTED_ORIGIN:
            return "cannot export variant '$0' of unexported '$1'";
        case NON_EXISTANT_DEPENDENCY:
            return "This module does not depend on the module '$0' at all.";
        case DEPENDENCY_VERSION_MISSING:
            return "this module requires version '$1.$2.$3$4' of module '$0'";
        case DEPENDENCY_VERSION_REQUIRED:
            return "this module requires version '$1.$2.$3$4' of module '$0', "
                   "but '$5.$6.$7$8' was found";
        case NON_DIRECT_DEPENDENCY:
            return "indirect dependency, because this module does not import module '$0'";
        case NOT_A_MODULE_NAME:
            return "'$0' is not a valid module name";
        case INVALID_DIRECTORY_NAME:
            return "$0 file path contains an invalid directory name '$1'";
        case INVALID_DIRECTORY_NAME_NESTING:
            return "$0 file path contains too many directory names '..'";
        case STRICT_RELATIVE_PATH_IN_STRING_MODULE:
            return "Strict relative file paths like '$0' are not supported in string-based modules";
        case FILE_PATH_CONSISTENCY:
            return "The file path '$0' is resolved to '$1' which is not in the $2 '$3'";
        case DEPRECATED_RESOURCE_PATH_USED:
            return "The file path '$0' is only found via the resource search paths, "
                "not via the module search paths. This is deprecated.";
        case AMBIGUOUS_FILE_LOCATION:
            return "The $0 '$1' is found at several places, this is ambiguous";
        case IMPORT_FROM_OLD_ARCHIVE_VERSION:
            return "Import from an old archive: version $0 required, but $1 found";
        case UNABLE_TO_RESOLVE:
            return "Unable to resolve file path '$0'$1";
        case INVALID_MDL_ARCHIVE_DETECTED:
            return "'$0' is an invalid MDL archive";
        case ARCHIVE_DEPENDENCY_MISSING:
            return "'$0' is imported from archive '$1', but has no dependency in archive '$2'";
        case FORBIDDEN_MDL_VERSION_IN_ARCHIVE:
            return "This module uses MDL version $0, but its owning archive is restricted to $1";
        case TYPE_CONVERSION_MUST_BE_EXPLICIT:
            return "conversion from '$0' to '$1' must be explicit";
        case MISSING_RESOURCE:
            return "Missing resource '$0' referenced";
        case RESOURCE_OUTSIDE_ARCHIVE:
            return "Referenced resource '$0' located outside the current archive";
        case IGNORED_ON_NON_PARAMETER:
            return "annotation '$0' will be ignored because '$1' is not a parameter";
        case INVALID_THUMBNAIL_EXTENSION:
            return "the thumbnail must point to a \".png\", \".jpg\" or \".jpeg\" file";
        case CONDITION_NOT_LITERAL:
            return "the condition must be a string literal";
        case INVALID_ENABLE_IF_CONDITION:
            return "invalid enable_if condition will be ignored";
        case DEPENDENCY_VERSION_MAJOR_MISMATCH:
            return "this module requires major version '$1' of module '$0', "
                "but '$2.$3.$4$5' was found";
        case FORBIDDEN_BINARY_OP_IN_ENABLE_IF:
            return "forbidden binary operator '$0' inside an enable_if";
        case FORBIDDEN_UNARY_OP_IN_ENABLE_IF:
            return "forbidden unary operator '$0' inside an enable_if";
        case FORBIDDEN_TERNARY_OP_IN_ENABLE_IF:
            return "forbidden conditional operator '?:' inside an enable_if";
        case FORBIDDEN_CALL_IN_ENABLE_IF:
            return "forbidden call to '$0' inside an enable_if";
        case FORBIDDEN_SELF_REF_IN_ENABLE_IF:
            return "forbidden self reference to '$0' inside an enable_if";
        case FORBIDDEN_ENABLE_IF_TYPE_REF:
            return "referenced $2 '$1' has forbidden type '$0' inside an enable_if";
        case ENABLE_IF_CONDITION_HAS_WARNINGS:
            return "enable_if condition contains warnings";
        case FUNC_VARIANT_WITH_DEFERRED_ARRAY_RET_TYPE:
            return "Cannot create function variant from function with deferred array type return";
        case ARCHIVE_CONFLICT:
            return "inside search path '$0': "
                   "archive '$1' conflicts with $2 '$3', both locations are ignored";
        case MATERIAL_PTR_USED:
            return "Missing parentheses on material instance";
        case CAST_TYPES_UNRELATED:
            return "Cannot cast '$0' into '$1$2', different type kinds";
        case CAST_MISSING_ENUM_VALUE_SRC:
            return "Cannot cast '$0' into '$1', missing enum value '$2 = $3' in source type";
        case CAST_MISSING_ENUM_VALUE_DST:
            return "Cannot cast '$0' into '$1', missing enum value '$2 = $3' in destination type";
        case CAST_STRUCT_FIELD_COUNT:
            return "Cannot cast '$0' into '$1', different number of fields";
        case CAST_STRUCT_FIELD_INCOMPATIBLE:
            return "Cannot cast '$0' into '$1', fields '$2' and '$3' are incompatible";
        case CAST_ARRAY_ELEMENT_INCOMPATIBLE:
            return "Cannot cast '$0' into '$1$2', element types are incompatible";
        case CAST_ARRAY_DEFERRED_TO_IMM:
            return "Cannot cast '$0' into '$1', deferred sized array cannot be casted to immediate";
        case CAST_ARRAY_IMM_TO_DEFERRED:
            return "Cannot cast '$0' into '$1', immediate sized array cannot be casted to deferred";
        case CAST_ARRAY_DIFFERENT_LENGTH:
            return "Cannot cast '$0' into '$1', different array size";
        case FORBIDDEN_ANNOTATION_PARAMETER_TYPE:
            return "annotation parameters of type '$0' are forbidden";
        case ANNOS_ON_ANNO_DECL_NOT_SUPPORTED:
            return "Annotations on annotation declarations are forbidden in MDL $0.$1 "
                "and will be ignored";
        case CONST_EXPR_ARGUMENT_REQUIRED:
            return "'$0' requires a const expression as first argument";
        case USING_ALIAS_REDECLARATION:
            return "redeclaration of using alias '$0'";
        case USING_ALIAS_DECL_FORBIDDEN:
            return "using alias declaration require MDL version 1.6 or later";
        case PACKAGE_NAME_CONTAINS_FORBIDDEN_CHAR:
            return "package name contains forbidden character '$0'";
        case ABSOLUTE_ALIAS_NOT_AT_BEGINNING:
            return "Alias name '$0' defines an absolute path, but is not at the beginning of a "
                "qualified name";

        // ------------------------------------------------------------- //
        case EXTERNAL_APPLICATION_ERROR:
            return "external application error: $.";
        case INTERNAL_COMPILER_ERROR:
            return "internal compiler error: $.";
        }
    } else if (msg_class == 'A') {
        Archiver_error err = Archiver_error(code);
        switch (err) {
        case MANIFEST_MISSING:
            return "MANIFEST is missing from the archive";
        case PACKAGE_NAME_INVALID:
            return "The package name '$0' is not a valid MDL identifier";
        case EXTRA_FILES_FOUND:
            return "Directory '$0' contains extra file/directory '$1'";
        case DIRECTORY_MISSING:
            return "Directory '$0' is missing";
        case ARCHIVE_ALREADY_EXIST:
            return "Archive '$0' already exist";
        case ARCHIVE_DOES_NOT_EXIST:
            return "Archive '$0' does not exist";
        case CANT_OPEN_ARCHIVE:
            return "Cannot open archive '$0'";
        case MEMORY_ALLOCATION:
            return "Required memory could not be allocated";
        case RENAME_FAILED:
            return "Renaming temporary file failed";
        case IO_ERROR:
            return "Input/Output error accessing archive '$0'";
        case CRC_ERROR:
            return "Broken archive '$0': CRC error";
        case CREATE_FILE_FAILED:
            return "Cannot create file '$0'";
        case MANIFEST_BROKEN:
            return "Invalid MDL archive";
        case KEY_NULL_PARAMETERS:
            return "Manifest keys and values must be non-NULL";
        case VALUE_MUST_BE_IN_TIME_FORMAT:
            return "The value of key '$0' must be in TIME format";
        case VALUE_MUST_BE_IN_SEMA_VERSION_FORMAT:
            return "The value of key '$0' must be in SEMANTIC VERSION format";
        case FORBIDDEN_KEY:
            return "Changing the value of key '$0' is forbidden";
        case SINGLE_VALUED_KEY:
            return "Key '$0' cannot have multiple values";
        case INACCESSIBLE_MDL_FILE:
            return "The directory '$0' does not form a valid MDL package, "
                   "MDL file '$1' cannot be compiled";
        case MDL_FILENAME_NOT_IDENTIFIER:
            return "Invalid MDL identifier for file '$1' in directory '$0'";
        case FAILED_TO_OPEN_TEMPFILE:
            return "Failure to create temporary file";
        case FAILED_TO_REMOVE:
            return "Failure to remove file";
        case INVALID_MDL_ARCHIVE:
            return "'$0' is an invalid MDL archive";
        case INVALID_PASSWORD:
            return "MDL archive could not be opened: invalid password";
        case READ_ONLY_ARCHIVE:
            return "MDL archive is read-only";
        case ARCHIVE_DOES_NOT_CONTAIN_ENTRY:
            return "MDL archive '$0' does not contain '$1'";
        case INVALID_MDL_ARCHIVE_NAME:
            return "Invalid MDL archive name '$0'";
        case EMPTY_ARCHIVE_CONTENT:
            return "Archive '$0' would be empty";
        case EXTRA_FILES_IGNORED:
            return "File/directory '$1' in directory '$0' will be ignored";
        case MDR_INVALID_HEADER:
            return "Header of MDL archive '$0' is invalid";
        case MDR_PRE_RELEASE_VERSION:
            return "Header of MDL archive '$0' contains a pre-release version";
        case MDR_INVALID_HEADER_VERSION:
            return "Header version of MDL archive'$0' is invalid";

        // ------------------------------------------------------------- //
        case INTERNAL_ARCHIVER_ERROR:
            return "internal archiver error: $.";
        }
    } else if (msg_class == 'E') {
        Encapsulator_error err = Encapsulator_error(code);
        switch (err) {
        case MDLE_FILE_ALREADY_EXIST:
            return "MDLE '$0' already exist";
        case MDLE_FILE_DOES_NOT_EXIST:
            return "MDLE '$0' does not exist";
        case MDLE_CANT_OPEN_FILE:
            return "Cannot open MDLE '$0'";
        case MDLE_MEMORY_ALLOCATION:
            return "Required memory could not be allocated";
        case MDLE_RENAME_FAILED:
            return "Renaming temporary file failed";
        case MDLE_IO_ERROR:
            return "Input/Output error accessing MDLE '$0'";
        case MDLE_CRC_ERROR:
            return "Broken MDLE '$0': CRC error";
        case MDLE_FAILED_TO_OPEN_TEMPFILE:
            return "Failure to create temporary file";
        case MDLE_INVALID_MDLE:
            return "'$0' is an invalid MDLE";
        case MDLE_INVALID_PASSWORD:
            return "MDLE could not be opened: invalid password";
        case MDLE_DOES_NOT_CONTAIN_ENTRY:
            return "MDLE '$0' does not contain '$1'";
        case MDLE_INVALID_NAME:
            return "Invalid MDLE name '$0'";
        case MDLE_INVALID_USER_FILE:
            return "User defined file '$1' for MDLE '$0' is invalid";
        case MDLE_INVALID_RESOURCE:
            return "Resource file defined file '$0' is invalid";
        case MDLE_CONTENT_FILE_INTEGRITY_FAIL:
            return "File '$1' in MDLE '$0' failed the MD5 check";
        case MDLE_INVALID_HEADER:
            return "Header of MDLE file '$0' is invalid";
        case MDLE_INVALID_HEADER_VERSION:
            return "Header version of MDLE file '$0' is invalid";
        case MDLE_PRE_RELEASE_VERSION:
            return "Header of MDLE file '$0' contains a pre-release version";
        case MDLE_FAILED_TO_ADD_ZIP_COMMENT:
            return "Filed to add zip comment to MDLE file '$0'";

        // ------------------------------------------------------------- //
        case MDLE_INTERNAL_ERROR:
            return "internal MDLE error: $.";
        }
    } else if (msg_class == 'J') {
        Jit_backend_error err = Jit_backend_error(code);
        switch (err) {
        case COMPILING_LLVM_CODE_FAILED:
            return "compiling LLVM code failed: $0";
        case LINKING_LIBDEVICE_FAILED:
            return "linking libdevice failed: $0";
        case LINKING_LIBBSDF_FAILED:
            return "linking libbsdf failed: $0";
        case PARSING_LIBDEVICE_MODULE_FAILED:
            return "parsing the libdevice module failed";
        case PARSING_LIBBSDF_MODULE_FAILED:
            return "parsing the libbsdf module failed";
        case PARSING_STATE_MODULE_FAILED:
            return "parsing the user-specified state module failed";
        case DEMANGLING_NAME_OF_EXTERNAL_FUNCTION_FAILED:
            return "demangling name of external function '$0' failed";
        case WRONG_FUNCTYPE_FOR_MDL_RUNTIME_FUNCTION:
            return "wrong function type for MDL runtime function '$0'";
        case LINKING_STATE_MODULE_FAILED:
            return "linking the user-specified state module failed: $0";
        case STATE_MODULE_FUNCTION_MISSING:
            return "user-specified state module is missing required function '$0'";
        case WRONG_RETURN_TYPE_FOR_STATE_MODULE_FUNCTION:
            return "function '$0' in user-specified state module has wrong return type";
        case API_STRUCT_TYPE_MUST_BE_OPAQUE:
            return "opaque API struct '$0' in user-specified state module may not be redefined";
        case GET_SYMBOL_FAILED:
            return "getting a symbol in the jit compiled code failed: $0";
        case LINKING_LIBMDLRT_FAILED:
            return "linking libmdlrt failed: $0";
        case PARSING_RENDERER_MODULE_FAILED:
            return "parsing the user-specified renderer module failed: $0";
        case LINKING_RENDERER_MODULE_FAILED:
            return "linking the user-specified renderer module failed: $0";

        // ------------------------------------------------------------- //
        case INTERNAL_JIT_BACKEND_ERROR:
            return "internal JIT backend error: $.";
        }
    } else if (msg_class == 'T') {
        Transformer_error err = Transformer_error(code);
        switch (err) {
        case SOURCE_MODULE_INVALID:
            return "invalid source modules cannot be inlined";
        case INLINING_MODULE_FAILED:
            return "The module $0 could not be inlined.";

        // ------------------------------------------------------------- //
        case INTERNAL_TRANSFORMER_ERROR:
            return "internal module transformer error: $.";
        }
    } else if (msg_class == 'V') {
        Comparator_error err = Comparator_error(code);
        switch (err) {
        case OTHER_DEFINED_AT:
            return "other '$0' was defined here";
        case MISSING_STRUCT_MEMBER:
            return "field '$0' was removed";
        case DIFFERENT_STRUCT_MEMBER_TYPE:
            return "field '$0' has type $1 but type $2 in $3";
        case ADDED_STRUCT_MEMBER:
            return "field '$0' was added";
        case MISSING_ENUM_VALUE:
            return "enum value '$0' was removed";
        case DIFFERENT_ENUM_VALUE:
            return "enum value '$0' has integer value $1 but $2 in $3";
        case ADDED_ENUM_VALUE:
            return "enum value '$0' was added";

        case TYPE_DOES_NOT_EXISTS:
            return "Type '$0' does not exists in $1";
        case TYPES_DIFFERENT:
            return "Types '$0' are of different kind";
        case INCOMPATIBLE_STRUCT:
            return "incompatible struct type $0";
        case INCOMPATIBLE_ENUM:
            return "incompatible enum type $0";
        case DIFFERENT_MDL_VERSIONS:
            return "modules using different MDL versions, mdl $0 != mdl $1";
        case DIFFERENT_DEFAULT_ARGUMENT:
            return "different default arguments on '$0'";
        case CONSTANT_DOES_NOT_EXISTS:
            return "Constant '$0' does not exists in $1";
        case CONSTANT_OF_DIFFERENT_TYPE:
            return "Constants '$0' are of different type, $1 != $2";
        case CONSTANT_OF_DIFFERENT_VALUE:
            return "Constants '$0' are of different values, $1 != $2";
        case FUNCTION_DOES_NOT_EXISTS:
            return "Function '$0' does not exists in $1";
        case FUNCTION_RET_TYPE_DIFFERENT:
            return "Functions '$0' have different return types, $1 != $2";
        case FUNCTION_PARAM_DELETED:
            return "Function '$0' has fewer parameters in $1";
        case FUNCTION_PARAM_DEF_ARG_DELETED:
            return "Parameter '$0' of function '$1' has no default argument in $2";
        case FUNCTION_PARAM_DEF_ARG_CHANGED:
            return "Parameter '$0' of function '$1' has a different default argument in $2";
        case ANNOTATION_DOES_NOT_EXISTS:
            return "'$0' does not exists in $1";
        case ANNOTATION_PARAM_DELETED:
            return "Annotation '$0' has fewer parameters in $1";
        case ANNOTATION_PARAM_DEF_ARG_DELETED:
            return "Parameter '$0' of '$1' has no default argument in $2";
        case ANNOTATION_PARAM_DEF_ARG_CHANGED:
            return "Parameter '$0' of '$1' has a different default argument in $2";
        case SEMA_VERSION_IS_HIGHER:
            return "The archive version of '$0' is higher then of '$1'";
        case ARCHIVE_DOES_NOT_CONTAIN_MODULE:
            return "Module '$0' was removed from archive";

        // ------------------------------------------------------------- //
        case INTERNAL_COMPARATOR_ERROR:
            return "internal comparator error: $.";
        }
    }
    return "";
}

// Get the kind of given argument
Error_params::Kind Error_params::get_kind(size_t index) const
{
    Entry const &e = m_args.at(index);

    return e.kind;
}

// Add a symbol argument.
Error_params &Error_params::add(ISymbol const *sym)
{
    Entry e;
    e.kind  = EK_SYMBOL;
    e.u.sym = sym;

    m_args.push_back(e);
    return *this;
}

// Return the type argument of given index.
ISymbol const *Error_params::get_symbol_arg(size_t index) const {
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_SYMBOL);
    return e.u.sym;
}

// Add a type argument.
Error_params &Error_params::add(IType const *type, bool suppress_prefix)
{
    Entry e;
    e.kind                   = EK_TYPE;
    e.u.type.type            = type;
    e.u.type.suppress_prefix = suppress_prefix;

    m_args.push_back(e);
    return *this;
}

// Return the type argument of given index.
IType const *Error_params::get_type_arg(size_t index, bool &suppress_prefix) const
{
    Entry const &e = m_args.at(index);

    if (e.kind == EK_ARRAY_TYPE)
        return e.u.a_type.e_type;
    MDL_ASSERT(e.kind == EK_TYPE);
    suppress_prefix = e.u.type.suppress_prefix;
    return e.u.type.type;
}

// Add an array type argument.
Error_params &Error_params::add(IType const *e_type, int size)
{
    Entry e;
    e.kind            = EK_ARRAY_TYPE;
    e.u.a_type.e_type = e_type;
    e.u.a_type.size   = size;

    m_args.push_back(e);
    return *this;
}

// Return the array type size argument of given index.
int Error_params::get_type_size_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_ARRAY_TYPE);
    return e.u.a_type.size;
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

// Add an operator kind argument.
Error_params &Error_params::add(IExpression::Operator op)
{
    Entry e;
    e.kind = EK_OPERATOR;
    e.u.op = op;

    m_args.push_back(e);
    return *this;
}

// Return the operator kind argument of given index.
IExpression::Operator Error_params::get_op_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_OPERATOR);
    return e.u.op;
}

// Add an direction argument.
Error_params &Error_params::add(Direction dir)
{
    Entry e;
    e.kind  = EK_DIRECTION;
    e.u.dir = dir;

    m_args.push_back(e);
    return *this;
}

// Return the direction argument of given index.
Error_params::Direction Error_params::get_dir_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_DIRECTION);
    return e.u.dir;
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

// Add a string argument.
Error_params &Error_params::add(string const &s)
{
    return add(s.c_str());
}

// Return the string argument of given index.
char const *Error_params::get_string_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_STRING);
    return e.u.string;
}

// Add a qualified name argument.
Error_params &Error_params::add(IQualified_name const *q, bool ignore_last)
{
    Entry e;
    e.kind                    = EK_QUAL_NAME;
    e.u.qual_name.name        = q;
    e.u.qual_name.ignore_last = ignore_last;

    m_args.push_back(e);
    return *this;
}

// Return the qualified name argument of given index.
IQualified_name const *Error_params::get_qual_name_arg(
    size_t index,
    bool   &ignore_last) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_QUAL_NAME);
    ignore_last = e.u.qual_name.ignore_last;
    return e.u.qual_name.name;
}

// Add an expression argument.
Error_params &Error_params::add(IExpression const *expr)
{
    Entry e;
    e.kind   = EK_EXPR;
    e.u.expr = expr;

    m_args.push_back(e);
    return *this;
}

// Return the expression argument of given index.
IExpression const *Error_params::get_expr_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_EXPR);
    return e.u.expr;
}

// Add a semantic version argument.
Error_params &Error_params::add(ISemantic_version const *ver)
{
    Entry e;
    e.kind      = EK_SEM_VER;
    e.u.sem_ver = ver;

    m_args.push_back(e);
    return *this;
}

// Return the semantic version argument of given index.
ISemantic_version const *Error_params::get_sem_ver_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_SEM_VER);
    return e.u.sem_ver;
}

// Add a positional argument as a num word.
Error_params &Error_params::add_numword(size_t pos)
{
    // FIXME: use language specific table
    Numwords const *words = en_words;

    for (size_t i = 0; words[i].word != NULL; ++i) {
        if (words[i].num == pos) {
            Entry e;
            e.kind     = EK_STRING;
            e.u.string = words[i].word;

            m_args.push_back(e);
            return *this;
        }
    }
    Entry e;
    e.kind  = EK_POS;
    e.u.pos = int(pos);

    m_args.push_back(e);
    return *this;
}

// Return the positional argument of given index.
int Error_params::get_pos_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_POS);
    return e.u.pos;
}

// Add a function signature (including return type).
Error_params &Error_params::add_signature(IDefinition const *def)
{
    Entry e;
    e.kind  = EK_SIGNATURE;
    e.u.sig = def;

    m_args.push_back(e);
    return *this;
}

// Add a function signature (without return type).
Error_params &Error_params::add_signature_no_rt(IDefinition const *def)
{
    Entry e;
    e.kind = EK_SIGNATURE_NO_RT;
    e.u.sig = def;

    m_args.push_back(e);
    return *this;
}

// Return the signature argument of given index.
IDefinition const *Error_params::get_signature_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_SIGNATURE || e.kind == EK_SIGNATURE_NO_RT);
    return e.u.sig;
}

// Add a value argument.
Error_params &Error_params::add(IValue const *val)
{
    Entry e;
    e.kind  = EK_VALUE;
    e.u.val = val;

    m_args.push_back(e);
    return *this;
}

// Return the value argument of given index.
IValue const *Error_params::get_value_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_VALUE);
    return e.u.val;
}

// Add a qualifier argument.
Error_params &Error_params::add(Qualifier q)
{
    Entry e;
    e.kind   = EK_QUALIFIER;
    e.u.qual = q;

    m_args.push_back(e);
    return *this;
}

// Return the qualifier argument of given index.
Qualifier Error_params::get_qualifier_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_QUALIFIER);
    return e.u.qual;
}

// Add a possible match.
Error_params &Error_params::add_possible_match(ISymbol const *sym)
{
    if (sym != NULL) {
        MDL_ASSERT(m_possible_match == NULL && "More then one possible match added");
        m_possible_match = sym;
    }
    return *this;
}


// Add an optional string value argument.
Error_params &Error_params::add_opt_message(IValue_string const *msg)
{
    Entry e;
    e.kind  = EK_OPT_MESSAGE;
    e.u.val = msg;

    m_args.push_back(e);
    return *this;
}

// Return the optional string value argument of given index.
IValue_string const *Error_params::get_opt_message_arg(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_OPT_MESSAGE);
    return cast<IValue_string>(e.u.val);
}

// Add an absolute path prefix.
Error_params &Error_params::add_absolute_path_prefix()
{
    Entry e;
    e.kind  = EK_PATH_PREFIX;
    e.u.file_path_prefix = FPP_ABSOLUTE;

    m_args.push_back(e);
    return *this;
}

// Add a weak relative path prefix.
Error_params &Error_params::add_weak_relative_path_prefix()
{
    Entry e;
    e.kind  = EK_PATH_PREFIX;
    e.u.file_path_prefix = FPP_WEAK_RELATIVE;

    m_args.push_back(e);
    return *this;
}

// Add a strict relative path prefix.
Error_params &Error_params::add_strict_relative_path_prefix()
{
    Entry e;
    e.kind  = EK_PATH_PREFIX;
    e.u.file_path_prefix = FPP_STRICT_RELATIVE;

    m_args.push_back(e);
    return *this;
}

// Add an MDL version.
Error_params &Error_params::add_mdl_version(IMDL::MDL_version ver)
{
    Entry e;
    e.kind  = EK_MDL_VERSION;
    e.u.mdl_version = ver;

    m_args.push_back(e);
    return *this;
}

// Return the MDL version.
IMDL::MDL_version Error_params::get_mdl_version(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_MDL_VERSION);
    return e.u.mdl_version;
}

// Return the path prefix.
Error_params::File_path_prefix Error_params::get_path_prefix(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_PATH_PREFIX);
    return e.u.file_path_prefix;
}

// Add a "current search path" path kind.
Error_params &Error_params::add_current_search_path()
{
    Entry e;
    e.kind  = EK_PATH_KIND;
    e.u.path_kind = PK_CURRENT_SEARCH_PATH;

    m_args.push_back(e);
    return *this;
}

// Add a "current directory" path kind
Error_params &Error_params::add_current_directory()
{
    Entry e;
    e.kind  = EK_PATH_KIND;
    e.u.path_kind = PK_CURRENT_DIRECTORY;

    m_args.push_back(e);
    return *this;
}

// Add a module origin.
char const *Error_params::get_module_origin(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_MODULE_ORIGIN);
    return e.u.string;
}

// Add a module origin.
Error_params &Error_params::add_module_origin(char const *origin)
{
    Entry e;
    e.kind  = EK_MODULE_ORIGIN;
    e.u.string = origin;

    m_args.push_back(e);
    return *this;
}

// Return the path kind.
Error_params::Path_kind Error_params::get_path_kind(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_PATH_KIND);
    return e.u.path_kind;
}

// Add an entity kind.
Error_params &Error_params::add_entity_kind(Entity_kind kind)
{
    Entry e;
    e.kind  = EK_ENTITY_KIND;
    e.u.entity_kind = kind;

    m_args.push_back(e);
    return *this;
}

// Return the entity kind.
Error_params::Entity_kind Error_params::get_entity_kind(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_ENTITY_KIND);
    return e.u.entity_kind;
}

// Add a dot (if the string is non-empty) and a string.
Error_params &Error_params::add_dot_string(char const *s)
{
    Entry e;
    e.kind = EK_DOT_STRING;
    e.u.string = s;

    m_args.push_back(e);
    return *this;
}

// Return the dot condition.
char const *Error_params::get_dot_string(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_DOT_STRING);
    return e.u.string;
}

// Add a character.
Error_params &Error_params::add_char(char c)
{
    Entry e;
    e.kind = EK_CHAR;
    e.u.c = c;

    m_args.push_back(e);
    return *this;
}

// Return the character.
unsigned char Error_params::get_char(size_t index) const
{
    Entry const &e = m_args.at(index);

    MDL_ASSERT(e.kind == EK_CHAR);
    return e.u.c;
}

// Add a definition kind (is converted into a string).
Error_params &Error_params::add_entity_kind(IDefinition::Kind kind)
{
    Entry e;
    e.kind = EK_STRING;

    switch (kind) {
    default:
        MDL_ASSERT(!"Unsupported definition kind");
        // fall through
    case IDefinition::DK_ERROR:
        // should not happen
        e.u.string = "<ERROR>";
        break;
    case IDefinition::DK_CONSTANT:
        e.u.string = "constant";
        break;
    case IDefinition::DK_ENUM_VALUE:
        e.u.string = "enum value";
        break;
    case IDefinition::DK_ANNOTATION:
        e.u.string = "annotation";
        break;
    case IDefinition::DK_TYPE:
        e.u.string = "type";
        break;
    case IDefinition::DK_FUNCTION:
        e.u.string = "function";
        break;
    case IDefinition::DK_VARIABLE:
        e.u.string = "variable";
        break;
    case IDefinition::DK_MEMBER:
        e.u.string = "field";
        break;
    case IDefinition::DK_CONSTRUCTOR:
        e.u.string = "constructor";
        break;
    case IDefinition::DK_PARAMETER:
        e.u.string = "parameter";
        break;
    case IDefinition::DK_ARRAY_SIZE:
        e.u.string = "array size";
        break;
    case IDefinition::DK_OPERATOR:
        e.u.string = "operator";
        break;
    case IDefinition::DK_NAMESPACE:
        e.u.string = "namespace";
        break;
    }

    m_args.push_back(e);
    return *this;
}

// Add 'package' or 'archive'.
Error_params &Error_params::add_package_or_archive(bool is_package)
{
    Entry e;
    e.kind = EK_STRING;

    if (is_package) {
        e.u.string = "package";
    } else {
        e.u.string = "archive";
    }

    m_args.push_back(e);
    return *this;
}

// Helper for format_msg().
static void print_error_param(
    Error_params const &params,
    size_t             idx,
    IPrinter           *printer)
{
    Error_params::Kind err_kind = params.get_kind(idx);
    switch (err_kind) {
    case Error_params::EK_TYPE:
        {
            // type param
            bool suppress_prefix = false;
            IType const *type = params.get_type_arg(idx, suppress_prefix);

            switch (type->get_kind()) {
            case IType::TK_STRUCT:
                {
                    IType_struct const *s_type = cast<IType_struct>(type);
                    if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL) {
                        printer->print("material");
                        return;
                    } else if (!suppress_prefix) {
                        printer->print("struct ");
                    }
                }
                break;
            case IType::TK_ENUM:
                if (!suppress_prefix)
                    printer->print("enum ");
                break;
            default:
                break;
            }
            printer->print(type);
        }
        break;
    case Error_params::EK_ARRAY_TYPE:
        {
            // type param
            bool suppress_prefix = false;
            IType const *type = params.get_type_arg(idx, suppress_prefix);

            switch (type->get_kind()) {
            case IType::TK_STRUCT:
                {
                    IType_struct const *s_type = cast<IType_struct>(type);
                    if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL) {
                        printer->print("material");
                        return;
                    }
                }
                break;
            default:
                break;
            }
            printer->print(type);
            printer->print('[');
            int size = params.get_type_size_arg(idx);
            if (size >= 0)
                printer->print(long(size));
            printer->print(']');
        }
        break;
    case Error_params::EK_SYMBOL:
        {
            // symbol param
            ISymbol const *sym = params.get_symbol_arg(idx);
            printer->print(sym);
        }
        break;
    case Error_params::EK_INTEGER:
        {
            // integer
            int v = params.get_int_arg(idx);
            printer->print(long(v));
        }
        break;
    case Error_params::EK_POS:
        {
            // position
            int pos = params.get_pos_arg(idx);
            printer->print(long(pos));
            printer->print(".");
        }
        break;
    case Error_params::EK_STRING:
        {
            // cstring param
            char const *s = params.get_string_arg(idx);
            printer->print(s);
        }
        break;
    case Error_params::EK_DOT_STRING:
        {
            // cstring param
            char const *s = params.get_dot_string(idx);
            if (s[0] != '\0')
                printer->print('.');
            printer->print(s);
        }
        break;
    case Error_params::EK_QUAL_NAME:
        {
            // qual_name param
            bool ignore_last = false;
            IQualified_name const *q = params.get_qual_name_arg(idx, ignore_last);

            bool add_scope = q->is_absolute();
            for (int i = 0, n = q->get_component_count() - (ignore_last ? 1 : 0); i < n; ++i) {
                if (add_scope)
                    printer->print("::");
                add_scope = true;
                printer->print(q->get_component(i));
            }
        }
        break;
    case Error_params::EK_EXPR:
        {
            // expression param
            IExpression const *e = params.get_expr_arg(idx);
            printer->print(e);
        }
        break;
    case Error_params::EK_SEM_VER:
        {
            // semantic version param
            ISemantic_version const *v = params.get_sem_ver_arg(idx);
            printer->print(v);
        }
        break;
    case Error_params::EK_OPERATOR:
        {
            // operator kind param
            IExpression::Operator op = params.get_op_arg(idx);
            char const *s = "<unknown>";
            switch (op) {

            // unary
            case IExpression::OK_BITWISE_COMPLEMENT:          s = "~"; break;
            case IExpression::OK_LOGICAL_NOT:                 s = "!"; break;
            case IExpression::OK_POSITIVE:                    s = "+"; break;
            case IExpression::OK_NEGATIVE:                    s = "-"; break;
            case IExpression::OK_PRE_INCREMENT:               s = "++"; break;
            case IExpression::OK_PRE_DECREMENT:               s = "--"; break;
            case IExpression::OK_POST_INCREMENT:              s = "++"; break;
            case IExpression::OK_POST_DECREMENT:              s = "--"; break;
            case IExpression::OK_CAST:                        s = "cast<>"; break;

            // binary
            case IExpression::OK_SELECT:                      s = "."; break;
            case IExpression::OK_ARRAY_INDEX:                 s = "[]"; break;
            case IExpression::OK_MULTIPLY:                    s = "*"; break;
            case IExpression::OK_DIVIDE:                      s = "/"; break;
            case IExpression::OK_MODULO:                      s = "%"; break;
            case IExpression::OK_PLUS:                        s = "+"; break;
            case IExpression::OK_MINUS:                       s = "-"; break;
            case IExpression::OK_SHIFT_LEFT:                  s = "<<"; break;
            case IExpression::OK_SHIFT_RIGHT:                 s = ">>"; break;
            case IExpression::OK_UNSIGNED_SHIFT_RIGHT:        s = ">>>"; break;
            case IExpression::OK_LESS:                        s = "<"; break;
            case IExpression::OK_LESS_OR_EQUAL:               s = "<="; break;
            case IExpression::OK_GREATER_OR_EQUAL:            s = ">"; break;
            case IExpression::OK_GREATER:                     s = ">="; break;
            case IExpression::OK_EQUAL:                       s = "=="; break;
            case IExpression::OK_NOT_EQUAL:                   s = "!="; break;
            case IExpression::OK_BITWISE_AND:                 s = "&"; break;
            case IExpression::OK_BITWISE_XOR:                 s = "^"; break;
            case IExpression::OK_BITWISE_OR:                  s = "|"; break;
            case IExpression::OK_LOGICAL_AND:                 s = "&&"; break;
            case IExpression::OK_LOGICAL_OR:                  s = "||"; break;
            case IExpression::OK_ASSIGN:                      s = "="; break;
            case IExpression::OK_MULTIPLY_ASSIGN:             s = "*="; break;
            case IExpression::OK_DIVIDE_ASSIGN:               s = "/="; break;
            case IExpression::OK_MODULO_ASSIGN:               s = "%="; break;
            case IExpression::OK_PLUS_ASSIGN:                 s = "+="; break;
            case IExpression::OK_MINUS_ASSIGN:                s = "-="; break;
            case IExpression::OK_SHIFT_LEFT_ASSIGN:           s = "<<="; break;
            case IExpression::OK_SHIFT_RIGHT_ASSIGN:          s = ">>="; break;
            case IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN: s = ">>>="; break;
            case IExpression::OK_BITWISE_OR_ASSIGN:           s = "|="; break;
            case IExpression::OK_BITWISE_XOR_ASSIGN:          s = "^="; break;
            case IExpression::OK_BITWISE_AND_ASSIGN:          s = "&="; break;
            case IExpression::OK_SEQUENCE:                    s = ","; break;

            // ternary
            case IExpression::OK_TERNARY:                     s = "?:"; break;

            // variadic
            case IExpression::OK_CALL:                        s = "()"; break;
            }
            printer->print(s);
        }
        break;
    case Error_params::EK_DIRECTION:
        printer->print(
            params.get_dir_arg(idx) == Error_params::DIR_LEFT ?
            // FIXME: get from language resource
            "left" : "right"
            );
        break;
    case Error_params::EK_SIGNATURE:
    case Error_params::EK_SIGNATURE_NO_RT:
        {
            IDefinition const *def = params.get_signature_arg(idx);
            IDefinition::Kind kind = def->get_kind();

            if (kind == IDefinition::DK_MEMBER) {
                // print type_name::member_name
                Definition const *f_def = impl_cast<Definition>(def);
                Scope const *scope = f_def->get_def_scope();

                if (scope->get_owner_definition()->has_flag(Definition::DEF_IS_IMPORTED)) {
                    // use full name for imported types
                    printer->print(scope->get_scope_type());
                } else {
                    // use the scope name (== type name without scope)
                    printer->print(scope->get_scope_name());
                }
                printer->print("::");
                printer->print(f_def->get_symbol());
                break;
            } else if (kind != IDefinition::DK_FUNCTION &&
                kind != IDefinition::DK_CONSTRUCTOR &&
                kind != IDefinition::DK_ANNOTATION) {
                printer->print(def->get_symbol());
                break;
            }

            IType_function const *func_type = cast<IType_function>(def->get_type());

            bool is_material = false;
            if (kind == Definition::DK_FUNCTION) {
                IType const *ret_type = func_type->get_return_type();
                if (IType_struct const *s_type = as<IType_struct>(ret_type))
                    if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL)
                        is_material = true;

                if (is_material || err_kind == Error_params::EK_SIGNATURE) {
                    printer->print(ret_type);
                    printer->print(" ");
                }
            } else if (kind == Definition::DK_ANNOTATION)
                printer->print("annotation ");

            printer->print(def->get_symbol());

            if (!is_material) {
                printer->print("(");
                for (int i = 0, n = func_type->get_parameter_count(); i < n; ++i) {
                    IType const   *p_type;
                    ISymbol const *p_sym;
                    func_type->get_parameter(i, p_type, p_sym);

                    if (i > 0)
                        printer->print(", ");
                    printer->print(p_type);
                }
                printer->print(")");
            }
        }
        break;
    case Error_params::EK_VALUE:
        // value param
        printer->print(params.get_value_arg(idx));
        break;
    case Error_params::EK_QUALIFIER:
        {
            // qualifier param
            char const *s = "none";
            switch (params.get_qualifier_arg(idx)) {
            case FQ_NONE:                   break;
            case FQ_VARYING: s = "varying"; break;
            case FQ_UNIFORM: s = "uniform"; break;
            }
            printer->print(s);
        }
        break;
    case Error_params::EK_OPT_MESSAGE:
        if (IValue_string const *msg = params.get_opt_message_arg(idx)) {
            printer->print(": ");
            printer->print(msg->get_value());
        }
        break;
    case Error_params::EK_PATH_PREFIX:
        {
            char const *s = "absolute";
            switch (params.get_path_prefix(idx)) {
            case Error_params::FPP_ABSOLUTE:
                s = "absolute"; break;
            case Error_params::FPP_WEAK_RELATIVE:
                s = "weak relative"; break;
            case Error_params::FPP_STRICT_RELATIVE:
                s = "strict relative"; break;
            }
            printer->print(s);
        }
        break;
    case Error_params::EK_PATH_KIND:
        {
            char const *s = "current search path";
            switch (params.get_path_kind(idx)) {
            case Error_params::PK_CURRENT_SEARCH_PATH:
                s = "current search path"; break;
            case Error_params::PK_CURRENT_DIRECTORY:
                s = "current working directory"; break;
            }
            printer->print(s);
        }
        break;
    case Error_params::EK_ENTITY_KIND:
        {
            char const *s =
                params.get_entity_kind(idx) == Error_params::EK_MODULE ? "module" : "resource";
            printer->print(s);
        }
        break;
    case Error_params::EK_MODULE_ORIGIN:
        {
            char const *s = params.get_module_origin(idx);
            if (s != NULL && s[0] != '\0') {
                printer->print(" in the module '");
                printer->print(s);
                printer->print("'");
            }
        }
        break;
    case Error_params::EK_MDL_VERSION:
        {
            IMDL::MDL_version ver = params.get_mdl_version(idx);
            char const *s = "1.0";
            switch (ver) {
            case IMDL::MDL_VERSION_1_0: s = "1.0"; break;
            case IMDL::MDL_VERSION_1_1: s = "1.1"; break;
            case IMDL::MDL_VERSION_1_2: s = "1.2"; break;
            case IMDL::MDL_VERSION_1_3: s = "1.3"; break;
            case IMDL::MDL_VERSION_1_4: s = "1.4"; break;
            case IMDL::MDL_VERSION_1_5: s = "1.5"; break;
            case IMDL::MDL_VERSION_1_6: s = "1.6"; break;
            case IMDL::MDL_VERSION_1_7: s = "1.7"; break;
            }
            printer->print(s);
        }
        break;
    case Error_params::EK_CHAR:
        {
            // char param
            unsigned char c = params.get_char(idx);
            if (c < 32 || c == 127) {
                printer->print("0x");
                printer->print(long(c));
            } else {
                printer->print(char(c));
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
    int                code,
    char               msg_class,
    Error_params const &params,
    IPrinter           *printer)
{
    size_t last_idx = (size_t)-1;
    if (char const *tmpl = get_error_template(code, msg_class)) {
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
                    printer->print("???");
                    continue;
                }
                last_idx = idx;

                print_error_param(params, idx, printer);
            } else {
                printer->print(c);
            }
        }
        if (ISymbol const *sym = params.get_possible_match()) {
            Error_params add_params(params.get_allocator());

            print_error_message(POSSIBLE_MATCH, msg_class, add_params.add(sym), printer);
        }
    }
}


}  // mdl
}  // mi
