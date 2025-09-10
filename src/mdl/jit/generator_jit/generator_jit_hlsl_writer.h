/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_HLSL_WRITER_H
#define MDL_GENERATOR_JIT_HLSL_WRITER_H 1

#include <unordered_map>
#include <vector>
#include <list>

#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_declarations.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_definitions.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_exprs.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_printers.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_stmts.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_types.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_type_cache.h>

#include <llvm/Pass.h>

#include "generator_jit_llvm.h"
#include "generator_jit_sl_function.h"
#include "generator_jit_sl_depgraph.h"
#include "generator_jit_sl_utils.h"
#include "generator_jit_type_map.h"

namespace mi {
namespace mdl {

// forward
class IOutput_stream;
class Messages_impl;
class Options_impl;

namespace hlsl {

// forward
class Compilation_unit;
class Declaration_struct;
class Decl_factory;
class Definition;
class Def_function;
class Def_param;
class Def_variable;
class Expr;
class Expr_binary;
class Expr_factory;
class ICompiler;
class Location;
class Name;
class Stmt;
class Stmt_compound;
class Stmt_factory;
class Symbol;
class Symbol_table;
class Type_factory;
class Type_name;
class Value;

/// Type traits for the HLSl AST.
struct HLSLAstTraits {
    typedef hlsl::ICompiler             ICompiler;
    typedef hlsl::IPrinter              IPrinter;
    typedef hlsl::Compilation_unit      Compilation_unit;
    typedef hlsl::Definition_table      Definition_table;
    typedef hlsl::Definition            Definition;
    typedef hlsl::Def_function          Def_function;
    typedef hlsl::Def_variable          Def_variable;
    typedef hlsl::Def_param             Def_param;
    typedef hlsl::Scope                 Scope;
    typedef hlsl::Symbol_table          Symbol_table;
    typedef hlsl::Symbol                Symbol;
    typedef hlsl::Type_factory          Type_factory;
    typedef hlsl::Type                  Type;
    typedef hlsl::Type_void             Type_void;
    typedef hlsl::Type_bool             Type_bool;
    typedef hlsl::Type_int              Type_int;
    typedef hlsl::Type_uint             Type_uint;
    typedef hlsl::Type_float            Type_float;
    typedef hlsl::Type_double           Type_double;
    typedef hlsl::Type_scalar           Type_scalar;
    typedef hlsl::Type_vector           Type_vector;
    typedef hlsl::Type_array            Type_array;
    typedef hlsl::Type_struct           Type_struct;
    typedef hlsl::Type_function         Type_function;
    typedef hlsl::Type_matrix           Type_matrix;
    typedef hlsl::Type_compound         Type_compound;
    typedef hlsl::Value_factory         Value_factory;
    typedef hlsl::Value                 Value;
    typedef hlsl::Value_scalar          Value_scalar;
    typedef hlsl::Value_fp              Value_fp;
    typedef hlsl::Value_vector          Value_vector;
    typedef hlsl::Value_matrix          Value_matrix;
    typedef hlsl::Value_array           Value_array;
    typedef hlsl::Name                  Name;
    typedef hlsl::Type_name             Type_name;
    typedef hlsl::Type_qualifier        Type_qualifier;
    typedef hlsl::Declaration           Declaration;
    typedef hlsl::Declaration_struct    Declaration_struct;
    typedef hlsl::Declaration_field     Declaration_field;
    typedef hlsl::Declaration_function  Declaration_function;
    typedef hlsl::Declaration_variable  Declaration_variable;
    typedef hlsl::Declaration_param     Declaration_param;
    typedef hlsl::Field_declarator      Field_declarator;
    typedef hlsl::Init_declarator       Init_declarator;
    typedef hlsl::Stmt_factory          Stmt_factory;
    typedef hlsl::Stmt                  Stmt;
    typedef hlsl::Stmt_compound         Stmt_compound;
    typedef hlsl::Stmt_expr             Stmt_expr;
    typedef hlsl::Expr_factory          Expr_factory;
    typedef hlsl::Expr                  Expr;
    typedef hlsl::Expr_binary           Expr_binary;
    typedef hlsl::Expr_unary            Expr_unary;
    typedef hlsl::Expr_ref              Expr_ref;
    typedef hlsl::Expr_call             Expr_call;
    typedef hlsl::Expr_compound         Expr_compound;
    typedef hlsl::Expr_literal          Expr_literal;
    typedef hlsl::Location              Location;
};

/// Base HLSL Writer pass.
class HLSLWriterBasePass : public llvm::ModulePass
{
    typedef llvm::ModulePass Base;

protected:
    /// Type traits for HLSL.
    typedef HLSLAstTraits TypeTraits;

    /// The zero location.
    static Location const zero_loc;

    /// The prototype language.
    static const IGenerated_code_executable::Prototype_language proto_lang;

protected:
    /// Constructor.
    ///
    /// \param pid                  the pass ID
    /// \param alloc                the allocator
    /// \param type_mapper          the type mapper
    /// \param options              backend options
    /// \param messages             backend messages
    /// \param enable_debug         true, if debug info should be generated
    /// \param enable_opt_remarks   true, if OptimizationRemarks should be enabled
    HLSLWriterBasePass(
        char                        &pid,
        mi::mdl::IAllocator         *alloc,
        Type_mapper const           &type_mapper,
        mi::mdl::Options_impl const &options,
        mi::mdl::Messages_impl      &messages,
        bool                        enable_debug,
        bool                        enable_opt_remarks);

    /// Return the name for this pass.
    llvm::StringRef getPassName() const final;

public:
    /// Set how to handle the noinline attribute.
    void set_noinline_mode(hlsl::IPrinter::Attribute_noinline_mode noinline_mode)
    {
        m_noinline_mode = noinline_mode;
    }

protected:
    /// Create a new compilation unit with the given name.
    ///
    /// \param compiler  the HLSL compiler
    /// \param name      the (file) name of the unit
    static Compilation_unit *create_compilation_unit(
        ICompiler *compiler,
        char const *name);

    /// Generate HLSL predefined entities into the definition table of a given unit.
    ///
    /// \param unit  a new (empty) compilation unit
    static void fillPredefinedEntities(
        Compilation_unit *unit);

    /// Fill type debug info from a module.
    void enter_type_debug_info(llvm::Module const &llvm_module)
    {
        return m_debug_types.enter_type_debug_info(llvm_module);
    }

    /// Find the debug type info for a given (composite) type name.
    ///
    /// \param name  an LLVM type name
    llvm::DICompositeType *find_composite_type_info(llvm::StringRef const &name) const
    {
        return m_debug_types.find_composite_type_info(name);
    }

    /// Find the subelement name if exists.
    llvm::DIType *find_subelement_type_info(
        llvm::DICompositeType *type_info,
        unsigned              field_index)
    {
        return m_debug_types.find_subelement_type_info(type_info, field_index);
    }

    /// Find the API debug type info for a given struct type.
    ///
    /// \param s_type  an LLVM struct type
    sl::DebugTypeHelper::API_type_info const *find_api_type_info(
        llvm::StructType *s_type) const;

    /// Add API type info.
    ///
    /// \param llvm_type_name  name of the API type in the LLVM-IR
    /// \param api_type_name   name of the API type in the target language
    /// \param fields          field names
    void add_api_type_info(
        char const              *llvm_type_name,
        char const              *api_type_name,
        Array_ref<char const *> fields)
    {
        return m_debug_types.add_api_type_info(llvm_type_name, api_type_name, fields);
    }

    /// Return the "inner" element type of array types.
    ///
    /// \param type  the type to process
    ///
    /// \return if type is an array type, return the inner element type, else type itself.
    static hlsl::Type *inner_element_type(hlsl::Type *type);

    /// Get an HLSL symbol for an LLVM string.
    hlsl::Symbol *get_sym(llvm::StringRef const &str);

    /// Get an HLSL symbol for a string.
    hlsl::Symbol *get_sym(char const *str);

    /// Get an unique HLSL symbol for an LLVM string and a template.
    ///
    /// \param str    an LLVM string (representing an LLVM symbol name)
    /// \param templ  if str contains invalid characters, use this as a template
    ///
    /// \return an unique HLSL symbol
    hlsl::Symbol *get_unique_sym(
        llvm::StringRef const &str,
        char const            *templ);

    /// Get an unique HLSL symbol for an LLVM string and a template.
    ///
    /// \param str    a string (representing a symbol name)
    /// \param templ  if str contains invalid characters, use this as a template
    ///
    /// \return an unique HLSL symbol
    hlsl::Symbol *get_unique_sym(
        char const *str,
        char const *templ);

    /// Get an HLSL name for the given location and symbol.
    ///
    /// \param loc   the location of the reference
    /// \param sym   the symbol of the referenced entity
    hlsl::Name *get_name(
        Location     loc,
        hlsl::Symbol *sym);

    /// Get an HLSL name for the given location and a C-string.
    ///
    /// \param loc   the location of the reference
    /// \param str   the name of the referenced entity
    hlsl::Name *get_name(
        Location   loc,
        char const *str);

    /// Get an HLSL name for the given location and LLVM string reference.
    ///
    /// \param loc   the location of the reference
    /// \param str   the name of the referenced entity
    hlsl::Name *get_name(
        Location              loc,
        llvm::StringRef const &str);

    /// Get an HLSL type name for an LLVM type.
    ///
    /// \param type  the LLVM type
    hlsl::Type_name *get_type_name(
        llvm::Type *type);

    /// Get an HLSL type name for an HLSL type.
    ///
    /// \param type  the HLSl type
    hlsl::Type_name *get_type_name(
        hlsl::Type *type);

    /// Get an HLSL type name for an HLSL symbol.
    ///
    /// \param sym  the HLSl type symbol
    hlsl::Type_name *get_type_name(
        hlsl::Symbol *sym);

    /// Add array specifier to a declaration if necessary.
    ///
    /// \param decl  the declaration
    /// \param type  the HLSL type
    template<typename Decl_type>
    void add_array_specifiers(
        Decl_type  *decl,
        hlsl::Type *type);

    /// Add parameter qualifier from a function type parameter at index.
    ///
    /// \param param_type_name   the type name of a parameter
    /// \param func_type         the HLSL function type
    /// \param index             the parameter index inside the function type
    static void add_param_qualifier(
        hlsl::Type_name     *param_type_name,
        hlsl::Type_function *func_type,
        size_t              index);

    /// Returns the return type if the first parameter of a function type is a output parameter.
    ///
    /// \param func_type  a HLSL function type
    static hlsl::Type *return_type_from_first_parameter(
        hlsl::Type_function *func_type)
    {
        hlsl::Type_function::Parameter *param = func_type->get_parameter(0);
        if (param->get_modifier() == hlsl::Type_function::Parameter::PM_OUT) {
            return param->get_type();
        }
        return nullptr;
    }

    /// Add a field to a struct declaration.
    ///
    /// \param decl_struct   the HLSl struct declaration
    /// \param type          the HLSL type of the field to be added
    /// \param sym           the symbol of the field to be added
    hlsl::Type_struct::Field add_struct_field(
        hlsl::Declaration_struct *decl_struct,
        hlsl::Type               *type,
        hlsl::Symbol             *sym);

    /// Add a field to an HLSL struct declaration.
    ///
    /// \param decl_struct   the HLSl struct declaration
    /// \param type          the HLSL type of the field to be added
    /// \param name          the name of the field to be added
    hlsl::Type_struct::Field add_struct_field(
        hlsl::Declaration_struct *decl_struct,
        hlsl::Type               *type,
        char const               *name);

    /// Create the HLSL resource data struct for the corresponding LLVM struct type.
    ///
    /// \param type  the LLVM type that represents the Res_data tuple
    hlsl::Type_struct *create_res_data_struct_type(
        llvm::StructType *type);

    /// Convert an LLVM struct type to an HLSL type.
    ///
    /// \param type  the LLVM struct type
    hlsl::Type *convert_struct_type(
        llvm::StructType *type);

    /// Convert an LLVM type to an HLSL type.
    ///
    /// \param type  the LLVM type
    hlsl::Type *convert_type(
        llvm::Type *type);

    /// Create the HLSL definition for a user defined LLVM function.
    ///
    /// \param func  an LLVM function
    hlsl::Def_function *create_definition(
        llvm::Function *func);

    /// Check if a given LLVM array type is the representation of the HLSL matrix type.
    ///
    /// \param type  an LLVM array type
    bool is_matrix_type(
        llvm::ArrayType *type) const;

    /// Get the name for a given vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    ///
    /// \param index  the vector index
    char const *get_vector_index_str(
        uint64_t index) const;

    /// Create a reference to an entity of the given name and type.
    ///
    /// \param type_name  the type name of the entity
    /// \param type       the HLSL type of the entity
    hlsl::Expr_ref *create_reference(
        hlsl::Type_name *type_name,
        hlsl::Type *type);

    /// Create a reference to an entity of the given symbol and HLSL type.
    ///
    /// \param sym        the symbol of the entity
    /// \param type       the HLSL type of the entity
    hlsl::Expr_ref *create_reference(
        hlsl::Symbol *sym,
        hlsl::Type *type);

    /// Create a reference to an entity of the given symbol and LLVM type.
    ///
    /// \param sym        the symbol of the entity
    /// \param type       the LLVM type of the entity
    hlsl::Expr_ref *create_reference(
        hlsl::Symbol *sym,
        llvm::Type *type);

    /// Create a reference to an entity.
    ///
    /// \param def       the definition the entity
    hlsl::Expr_ref *create_reference(
        hlsl::Definition *def);

    /// Create a new unary expression.
    ///
    /// \param loc    the location
    /// \param op     the operator
    /// \param arg    the argument of the operator
    ///
    /// \returns    the created expression
    hlsl::Expr *create_unary(
        hlsl::Location const       &loc,
        hlsl::Expr_unary::Operator op,
        hlsl::Expr                 *arg);

    /// Create a new binary expression.
    ///
    /// \param op     the operator
    /// \param left   the left argument of the operator
    /// \param right  the right argument of the operator
    ///
    /// \returns    the created expression
    hlsl::Expr *create_binary(
        hlsl::Expr_binary::Operator op,
        hlsl::Expr                  *left,
        hlsl::Expr                  *right);

    /// Create a call to a GLSL runtime function.
    ///
    /// \param loc    the location of the call
    /// \param func   the LLVM intrinsic function representing the runtime function
    /// \param args   converted arguments of the call
    hlsl::Expr *create_runtime_call(
        hlsl::Location const          &loc,
        llvm::Function                *func,
        Array_ref<hlsl::Expr *> const &args);

    /// Get the constructor for the given HLSL type.
    ///
    /// \param type  the type
    /// \param args  arguments to the constructor
    hlsl::Def_function *lookup_constructor(
        hlsl::Type                    *type,
        Array_ref<hlsl::Expr *> const &args) const;

    /// Creates a bitcast expression from float to int.
    ///
    /// \param dst  the destination type, either int or an int vector
    hlsl::Expr *create_float2int_bitcast(
        hlsl::Type *dst,
        hlsl::Expr *arg);

    /// Creates a bitcast expression from int to float.
    ///
    /// \param dst  the destination type, either int or an int vector
    hlsl::Expr *create_int2float_bitcast(
        hlsl::Type *dst,
        hlsl::Expr *arg);

    /// Create a type cast expression.
    ///
    /// \param dst  the destination type
    /// \param arg  the expression to cast
    hlsl::Expr *create_type_cast(
        hlsl::Type *dst,
        hlsl::Expr *arg);

    /// Returns true if a compound expression is allowed in the given context.
    ///
    /// \param is_global  true if we are inside a global initializer
    static bool compound_allowed(bool is_global) {
        // currently Slang supports compound expressions only as global initializers
        return is_global;
    }

    /// Creates an initializer.
    ///
    /// \param loc   the location
    /// \param type  the type
    /// \param args  arguments, number must match the type
    hlsl::Expr *create_initializer(
        hlsl::Location const          &loc,
        hlsl::Type                    *type,
        Array_ref<hlsl::Expr *> const &args);

    /// Set the type qualifier for a global constant in HLSL.
    ///
    /// \param tq  the type qualifier
    ///
    /// \note This sets static for HLSL
    static void set_global_constant_qualifier(Type_qualifier &tq);

    /// Convert a function type parameter qualifier to a AST parameter qualifier.
    ///
    /// \param param  a function type parameter
    static hlsl::Parameter_qualifier convert_type_modifier_to_param_qualifier(
        hlsl::Type_function::Parameter *param);

    /// Set the out parameter qualifier.
    ///
    /// \param param  a type name
    static void make_out_parameter(
        hlsl::Type_name *param);

    /// Convert the LLVM debug location (if any is attached to the given instruction)
    /// to an HLSL location.
    ///
    /// \param inst  an LLVM instruction
    hlsl::Location convert_location(
        llvm::Instruction *inst);

    /// Called for every function that is just a prototype in the original LLVM module.
    ///
    /// \param func        the LLVM function (declaration)
    hlsl::Def_function *create_prototype(
        llvm::Function &func);

    /// Return true if the user of an instruction requires its materialization
    ///
    /// \param user   the user instruction
    bool must_be_materialized(
        llvm::Instruction *user);

    /// Finalize the compilation unit and write it to the given output stream.
    ///
    /// \param M       the LLVM module
    /// \param code    the generated source code
    /// \param remaps  list of remapped entities
    void finalize(
        llvm::Module                                               &M,
        Generated_code_source                                      *code,
        list<std::pair<char const *, hlsl::Symbol *> >::Type const &remaps);

    /// HLSL has C-style type casts.
    static bool has_c_style_type_casts() { return true; }

    /// HLSL has the double type.
    static bool has_double_type() { return true; }

    /// Generates a new global static const variable to hold an LLVM value.
    hlsl::Definition *create_global_const(
        llvm::StringRef name, hlsl::Expr *c_expr);

    /// Mark the function with the noinline attribute.
    static void add_noinline_attribute(
        hlsl::Declaration_function *func)
    {
        func->set_attr_noinline(true);
    }

    /// Set whether the function is marked as an export.
    static void set_is_export(
        hlsl::Declaration_function *func,
        bool is_export)
    {
        func->set_is_export(is_export);
    }

    /// Dump the current AST.
    void dump_ast();

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    void error(int code);

protected:
    /// MDL allocator used for generating the HLSL AST.
    IAllocator *m_alloc;

    /// the Type mapper.
    Type_mapper const &m_type_mapper;

    /// The HLSL compiler.
    mi::base::Handle<Compiler> m_compiler;

    /// The HLSL compilation unit.
    mi::base::Handle<Compilation_unit> m_unit;

    /// The HLSL declaration factory of the compilation unit.
    Decl_factory &m_decl_factory;

    /// The HLSL expression factory of the compilation unit.
    Expr_factory &m_expr_factory;

    /// The HLSL statement factory of the compilation unit.
    Stmt_factory &m_stmt_factory;

    /// The HLSL type cache, holding all predefined hlsl types.
    Type_cache m_tc;

    /// The HLSL value factory of the compilation unit.
    Value_factory &m_value_factory;

    /// The HLSL symbol table of the compilation unit.
    Symbol_table &m_symbol_table;
 
    /// The HLSL definition table of the compilation unit.
    Definition_table &m_def_tab;

    typedef ptr_hash_map<llvm::Type, hlsl::Type *>::Type Type2type_map;

    /// The type cache mapping from LLVM to HLSL types.
    Type2type_map m_type_cache;

    /// Debug info on types.
    sl::DebugTypeHelper m_debug_types;

    typedef hash_map<string, unsigned, string_hash<string> >::Type Ref_fname_id_map;

    /// Referenced source files.
    Ref_fname_id_map m_ref_fnames;

    /// Backend messages.
    mi::mdl::Messages_impl &m_messages;

    typedef ptr_hash_set<hlsl::Declaration const>::Type Decl_set;

    /// The set of all API related declarations.
    Decl_set m_api_decls;

    /// ID used to create unique names.
    unsigned m_next_unique_name_id;

    /// How to handle the noinline attribute.
    hlsl::IPrinter::Attribute_noinline_mode m_noinline_mode;

    /// If true, use debug info.
    bool m_use_dbg;

    /// If true, enable OptimizationRemarks.
    bool m_opt_remarks;
};

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_HLSL_WRITER_H
