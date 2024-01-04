/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_GLSL_WRITER_H
#define MDL_GENERATOR_JIT_GLSL_WRITER_H 1

#include <unordered_map>
#include <vector>
#include <list>

#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_declarations.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_definitions.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_exprs.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_printers.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_stmts.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_types.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_type_cache.h>

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

namespace glsl {

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
struct GLSLAstTraits {
    typedef glsl::ICompiler             ICompiler;
    typedef glsl::IPrinter              IPrinter;
    typedef glsl::Compilation_unit      Compilation_unit;
    typedef glsl::Definition_table      Definition_table;
    typedef glsl::Definition            Definition;
    typedef glsl::Def_function          Def_function;
    typedef glsl::Def_variable          Def_variable;
    typedef glsl::Def_param             Def_param;
    typedef glsl::Scope                 Scope;
    typedef glsl::Symbol_table          Symbol_table;
    typedef glsl::Symbol                Symbol;
    typedef glsl::Type_factory          Type_factory;
    typedef glsl::Type                  Type;
    typedef glsl::Type_void             Type_void;
    typedef glsl::Type_bool             Type_bool;
    typedef glsl::Type_int              Type_int;
    typedef glsl::Type_uint             Type_uint;
    typedef glsl::Type_float            Type_float;
    typedef glsl::Type_double           Type_double;
    typedef glsl::Type_scalar           Type_scalar;
    typedef glsl::Type_vector           Type_vector;
    typedef glsl::Type_array            Type_array;
    typedef glsl::Type_struct           Type_struct;
    typedef glsl::Type_function         Type_function;
    typedef glsl::Type_matrix           Type_matrix;
    typedef glsl::Type_compound         Type_compound;
    typedef glsl::Value_factory         Value_factory;
    typedef glsl::Value                 Value;
    typedef glsl::Value_scalar          Value_scalar;
    typedef glsl::Value_fp              Value_fp;
    typedef glsl::Value_vector          Value_vector;
    typedef glsl::Value_matrix          Value_matrix;
    typedef glsl::Value_array           Value_array;
    typedef glsl::Name                  Name;
    typedef glsl::Type_name             Type_name;
    typedef glsl::Type_qualifier        Type_qualifier;
    typedef glsl::Declaration           Declaration;
    typedef glsl::Declaration_struct    Declaration_struct;
    typedef glsl::Declaration_field     Declaration_field;
    typedef glsl::Declaration_function  Declaration_function;
    typedef glsl::Declaration_variable  Declaration_variable;
    typedef glsl::Declaration_param     Declaration_param;
    typedef glsl::Field_declarator      Field_declarator;
    typedef glsl::Init_declarator       Init_declarator;
    typedef glsl::Stmt_factory          Stmt_factory;
    typedef glsl::Stmt                  Stmt;
    typedef glsl::Stmt_compound         Stmt_compound;
    typedef glsl::Stmt_expr             Stmt_expr;
    typedef glsl::Expr_factory          Expr_factory;
    typedef glsl::Expr                  Expr;
    typedef glsl::Expr_binary           Expr_binary;
    typedef glsl::Expr_unary            Expr_unary;
    typedef glsl::Expr_ref              Expr_ref;
    typedef glsl::Expr_call             Expr_call;
    typedef glsl::Expr_compound         Expr_compound;
    typedef glsl::Expr_literal          Expr_literal;
    typedef glsl::Location              Location;
};

/// Base GLSL Writer pass.
class GLSLWriterBasePass : public llvm::ModulePass
{
    typedef llvm::ModulePass Base;

protected:
    /// Type traits for GLSL.
    typedef GLSLAstTraits TypeTraits;

    /// Additional flags for runtime functions.
    enum Runtime_flag {
        RTF_NONE            =  0, ///< no conversion needed
        RTF_CONV_DOUBLE     =  1, ///< convert D => F for params, F => D for result
        RTF_CONV_VECTOR_ALL =  2, ///< if one argument is a vector, convert ALL to vector
        RTF_CONV_VECTOR_2   =  4, ///< convert parameter 2 to the vector type of param 0
        RTF_NORM_VEC_SCALAR =  8, ///< exchange param 0 and 1 if the first is a scalar,
                                  ///  the second is a vector
        RTF_ADD_0_1         = 16, ///< add two extra parameters: 0, 1
        RTF_MULT_1_LOG2_10  = 32, ///< multiply the result by 1.0/log2(10)
    };  // can be or'ed

    typedef unsigned Runtime_flags;

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
    /// \param options              the backend options
    /// \param messages             the backend messages
    /// \param enable_debug         true, if debug info should be generated
    /// \param enable_opt_remarks   true, if OptimizationRemarks should be enabled
    GLSLWriterBasePass(
        char                        &pid,
        mi::mdl::IAllocator         *alloc,
        Type_mapper const           &type_mapper,
        mi::mdl::Options_impl const &options,
        mi::mdl::Messages_impl      &messages,
        bool                        enable_debug,
        bool                        enable_opt_remarks);

    /// Return the name for this pass.
    llvm::StringRef getPassName() const final;

protected:
    /// Create a new (empty) compilation unit with the given name.
    ///
    /// \param compiler  the GLSL compiler
    /// \param name      the (file) name of the unit
    static Compilation_unit *create_compilation_unit(
        ICompiler  *compiler,
        char const *name);

    /// Set the GLSL target language version.
    bool set_target_version(
        unsigned major,
        unsigned minor);

    /// Parse extensions and set them.
    ///
    /// \param ctx   the GLSL context to set the extensions
    /// \param ext   a comma separated string of extension names
    /// \param eb    the extension behavior
    ///
    /// \return true on success
    bool set_extensions(
        glsl::GLSLang_context                     &ctx,
        char const                                *ext,
        glsl::GLSLang_context::Extension_behavior eb);

    /// Set the GLSL target context to the compilation unit.
    ///
    /// \param unit  the unit to modify
    ///
    /// Sets version, profile, extensions.
    void set_glsl_target_context(
        Compilation_unit *unit);

    /// Fill the predefined entities into the (at this point empty) compilation unit;
    void fill_predefined_entities();

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
    static glsl::Type *inner_element_type(glsl::Type *type);

    /// Get an GLSL symbol for an LLVM string.
    glsl::Symbol *get_sym(llvm::StringRef const &str);

    /// Get an GLSL symbol for a string.
    glsl::Symbol *get_sym(char const *str);

    /// Get an unique GLSL symbol for an LLVM string and a template.
    ///
    /// \param str    an LLVM string (representing an LLVM symbol name)
    /// \param templ  if str contains invalid characters, use this as a template
    ///
    /// \return an unique GLSL symbol
    glsl::Symbol *get_unique_sym(
        llvm::StringRef const &str,
        char const            *templ);

    /// Get an unique GLSL symbol for an LLVM string and a template.
    ///
    /// \param str    a string (representing a symbol name)
    /// \param templ  if str contains invalid characters, use this as a template
    ///
    /// \return an unique GLSL symbol
    glsl::Symbol *get_unique_sym(
        char const *str,
        char const *templ);

    /// Get an GLSL name for the given location and symbol.
    ///
    /// \param loc   the location of the reference
    /// \param sym   the symbol of the referenced entity
    glsl::Name *get_name(
        Location     loc,
        glsl::Symbol *sym);

    /// Get an GLSL name for the given location and a C-string.
    ///
    /// \param loc   the location of the reference
    /// \param str   the name of the referenced entity
    glsl::Name *get_name(
        Location   loc,
        char const *str);

    /// Get an GLSL name for the given location and LLVM string reference.
    ///
    /// \param loc   the location of the reference
    /// \param str   the name of the referenced entity
    glsl::Name *get_name(
        Location              loc,
        llvm::StringRef const &str);

    /// Get an GLSL type name for an LLVM type.
    ///
    /// \param type  the LLVM type
    glsl::Type_name *get_type_name(
        llvm::Type *type);

    /// Get an GLSL type name for an GLSL type.
    ///
    /// \param type  the HLSl type
    glsl::Type_name *get_type_name(
        glsl::Type *type);

    /// Get an GLSL type name for an GLSL symbol.
    ///
    /// \param sym  the HLSl type symbol
    glsl::Type_name *get_type_name(
        glsl::Symbol *sym);

    /// Add array specifier to a declaration if necessary.
    ///
    /// \param decl  the declaration
    /// \param type  the GLSL type
    template<typename Decl_type>
    void add_array_specifiers(
        Decl_type  *decl,
        glsl::Type *type);

    /// Add parameter qualifier from a function type parameter at index.
    ///
    /// \param param_type_name   the type name of a parameter
    /// \param func_type         the GLSL function type
    /// \param index             the parameter index inside the function type
    static void add_param_qualifier(
        glsl::Type_name     *param_type_name,
        glsl::Type_function *func_type,
        size_t              index);

    /// Returns the return type if the first parameter of a function type is a output parameter.
    ///
    /// \param func_type  a GLSL function type
    static glsl::Type *return_type_from_first_parameter(
        glsl::Type_function *func_type)
    {
        glsl::Type_function::Parameter *param = func_type->get_parameter(0);
        if (param->get_modifier() == glsl::Type_function::Parameter::PM_OUT) {
            return param->get_type();
        }
        return nullptr;
    }

    /// Add a field to a struct declaration.
    ///
    /// \param decl_struct   the HLSl struct declaration
    /// \param type          the GLSL type of the field to be added
    /// \param sym           the symbol of the field to be added
    glsl::Type_struct::Field add_struct_field(
        glsl::Declaration_struct *decl_struct,
        glsl::Type               *type,
        glsl::Symbol             *sym);

    /// Add a field to an GLSL struct declaration.
    ///
    /// \param decl_struct   the HLSl struct declaration
    /// \param type          the GLSL type of the field to be added
    /// \param name          the name of the field to be added
    glsl::Type_struct::Field add_struct_field(
        glsl::Declaration_struct *decl_struct,
        glsl::Type               *type,
        char const               *name);

    /// Create the GLSL resource data struct for the corresponding LLVM struct type.
    ///
    /// \param type  the LLVM type that represents the Res_data tuple
    glsl::Type_struct *create_res_data_struct_type(
        llvm::StructType *type);

    /// Convert an LLVM struct type to an GLSL type.
    ///
    /// \param type  the LLVM struct type
    glsl::Type *convert_struct_type(
        llvm::StructType *type);

    /// Type conversion flags.
    enum Type_conversion_flags {
        TCF_NO_DOUBLE = 0x0001,
    };

    /// Convert an LLVM type to an GLSL type.
    ///
    /// \param type  the LLVM type
    glsl::Type *convert_type(
        llvm::Type *type);

    /// Create the GLSL definition for a user defined LLVM function.
    ///
    /// \param func  an LLVM function
    glsl::Def_function *create_definition(
        llvm::Function *func);

    /// Check if a given LLVM array type is the representation of the GLSL matrix type.
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
    /// \param type       the GLSL type of the entity
    glsl::Expr_ref *create_reference(
        glsl::Type_name *type_name,
        glsl::Type *type);

    /// Create a reference to an entity of the given symbol and GLSL type.
    ///
    /// \param sym        the symbol of the entity
    /// \param type       the GLSL type of the entity
    glsl::Expr_ref *create_reference(
        glsl::Symbol *sym,
        glsl::Type *type);

    /// Create a reference to an entity of the given symbol and LLVM type.
    ///
    /// \param sym        the symbol of the entity
    /// \param type       the LLVM type of the entity
    glsl::Expr_ref *create_reference(
        glsl::Symbol *sym,
        llvm::Type   *type);

    /// Create a reference to an entity.
    ///
    /// \param def       the definition the entity
    glsl::Expr_ref *create_reference(
        glsl::Definition *def);

    /// Create a new unary expression.
    ///
    /// \param loc    the location
    /// \param op     the operator
    /// \param arg    the argument of the operator
    ///
    /// \returns    the created expression
    glsl::Expr *create_unary(
        glsl::Location const       &loc,
        glsl::Expr_unary::Operator op,
        glsl::Expr                 *arg);

    /// Create a new binary expression.
    ///
    /// \param op     the operator
    /// \param left   the left argument of the operator
    /// \param right  the right argument of the operator
    ///
    /// \returns    the created expression
    glsl::Expr *create_binary(
        glsl::Expr_binary::Operator op,
        glsl::Expr                  *left,
        glsl::Expr                  *right);

    /// Create a call to a GLSL runtime function.
    ///
    /// \param loc    the location of the call
    /// \param func   the LLVM intrinsic function representing the runtime function
    /// \param args   converted arguments of the call
    glsl::Expr *create_runtime_call(
        glsl::Location const          &loc,
        llvm::Function                *func,
        Array_ref<glsl::Expr *> const &args);

    /// Create a call to a GLSL runtime function.
    ///
    /// \param loc    the location of the call
    /// \param func   the name of the runtime function
    /// \param args   converted arguments of the call
    glsl::Expr *create_runtime_call(
        glsl::Location const          &loc,
        char const                    *func,
        Array_ref<glsl::Expr *> const &args);

    /// Create a type cast expression.
    ///
    /// \param dst  the destination type
    /// \param arg  the expression to cast
    glsl::Expr *create_type_cast(
        glsl::Type *dst,
        glsl::Expr *arg);

    /// Creates a bitcast expression from float to int.
    ///
    /// \param dst  the destination type, either int or an int vector
    /// \param arg  the expression to cast
    glsl::Expr *create_float2int_bitcast(
        glsl::Type *dst,
        glsl::Expr *arg);

    /// Creates a bitcast expression from int to float.
    ///
    /// \param dst  the destination type, either int or an int vector
    /// \param arg  the expression to cast
    glsl::Expr *create_int2float_bitcast(
        glsl::Type *dst,
        glsl::Expr *arg);

    /// Modify a double based argument type to a float based.
    ///
    /// \param arg_type  the type to modify
    glsl::Type *to_genType(glsl::Type *arg_type) const;

    /// Modify a float based argument type to a double based.
    ///
    /// \param arg_type  the type to modify
    glsl::Type *to_genDType(glsl::Type *arg_type) const;

    /// Find an (fully) matching overload.
    ///
    /// \param[in]  f_def  a function definition (tail of overload list)
    /// \param[in]  args   argument expressions
    /// \param[out] flags  runtime flags
    glsl::Def_function *find_overload(
        glsl::Def_function            *f_def,
        Array_ref<glsl::Expr *> const &args,
        Runtime_flags                 &flags) const;

    /// Get the constructor for the given GLSL type.
    ///
    /// \param type  the type
    /// \param args  arguments to the constructor
    glsl::Def_function *lookup_constructor(
        glsl::Type                    *type,
        Array_ref<glsl::Expr *> const &args) const;

    /// Get the runtime function for the given GLSL type.
    ///
    /// \param[in]  sym    the name of the function
    /// \param[in]  args   arguments to the function
    /// \param[out] flags  runtime flags
    glsl::Def_function *lookup_runtime(
        glsl::Symbol                  *sym,
        Array_ref<glsl::Expr *> const &args,
        Runtime_flags                 &flags) const;

    /// Get the runtime function for the given GLSL type.
    ///
    /// \param[in]  name   the name of the function
    /// \param[in]  args   arguments to the function
    /// \param[out] flags  runtime flags
    glsl::Def_function *lookup_runtime(
        char const                    *name,
        Array_ref<glsl::Expr *> const &args,
        Runtime_flags                 &flags) const;

    /// Returns true if a compound expression is allowed in the given context.
    ///
    /// \param is_global  true if we are inside a global initializer
    bool compound_allowed(
        bool is_global) const;

    /// Creates an initializer.
    ///
    /// \param loc   the location
    /// \param type  the type
    /// \param args  arguments, number must match the type
    glsl::Expr *create_initializer(
        glsl::Location const          &loc,
        glsl::Type                    *type,
        Array_ref<glsl::Expr *> const &args);

    /// Set the type qualifier for a global constant in GLSL.
    ///
    /// \param tq  the type qualifier
    ///
    /// \note This sets static for GLSL
    static void set_global_constant_qualifier(Type_qualifier &tq);

    /// Convert a function type parameter qualifier to a AST parameter qualifier.
    ///
    /// \param param  a function type parameter
    static glsl::Parameter_qualifier convert_type_modifier_to_param_qualifier(
        glsl::Type_function::Parameter *param);

    /// Set the out parameter qualifier.
    ///
    /// \param param  a type name
    static void make_out_parameter(
        glsl::Type_name *param);

    /// Convert the LLVM debug location (if any is attached to the given instruction)
    /// to an GLSL location.
    ///
    /// \param inst  an LLVM instruction
    glsl::Location convert_location(
        llvm::Instruction *inst);

    /// Add a JIT backend warning message to the messages.
    ///
    /// \param code    the code of the error message
    void warning(int code);

    /// Add a JIT backend error message to the messages.
    ///
    /// \param code    the code of the error message
    void error(int code);

    /// Called for every function that is just a prototype in the original LLVM module.
    ///
    /// \param func  the LLVM function (declaration)
    glsl::Def_function *create_prototype(
        llvm::Function &func);

    /// Return true if the user of an instruction requires its materialization.
    ///
    /// \param user   the user instruction
    bool must_be_materialized(
        llvm::Instruction *user);

    /// Convert a GLSL value into binary data.
    ///
    /// \param data    point to the binary data buffer
    /// \param value   the value to convert
    /// \param offset  current data offset
    ///
    /// \return current data offset
    size_t fill_binary_data(
        unsigned char *data,
        glsl::Value   *value,
        size_t        offset);

    /// Finalize the compilation unit and write it to the given output stream.
    ///
    /// \param M       the LLVM module
    /// \param code    the generated source code
    /// \param remaps  list of remapped entities
    void finalize(
        llvm::Module                                               &M,
        Generated_code_source                                      *code,
        list<std::pair<char const *, glsl::Symbol *> >::Type const &remaps);

    /// GLSL does not have C-style type casts.
    static bool has_c_style_type_casts() { return false; }

    /// Returns true if the double type exists in the current context.
    bool has_double_type() const { return m_ctx.has_double_type(); }

    /// Check if the constant data is too big for the code itself and should be moved to
    /// an uniform buffer.
    ///
    /// \param v  a value that should be added to the constant data
    ///
    /// \return true if this value should be moved to uniform initializers
    bool move_to_uniform(
        glsl::Value *v);

    /// Add an uniform initializer.
    ///
    /// \param def   the definition of the initializer
    /// \param init  the value of the initializer
    void add_uniform_initializer(
        glsl::Definition *def,
        glsl::Value      *init);

    /// Generates a new global static const variable to hold an LLVM value.
    glsl::Definition *create_global_const(
        llvm::StringRef name, glsl::Expr *c_expr);

    /// Mark the function with the noinline attribute.
    static void add_noinline_attribute(
        glsl::Declaration_function *func)
    {
        // unsupported yet
    }

    /// Dump the current AST.
    void dump_ast();

protected:
    /// MDL allocator used for generating the GLSL AST.
    IAllocator *m_alloc;

    /// the Type mapper.
    Type_mapper const &m_type_mapper;

    /// The GLSL compiler.
    mi::base::Handle<Compiler> m_compiler;

    /// The GLSL compilation unit.
    mi::base::Handle<Compilation_unit> m_unit;

    /// The GLSL declaration factory of the compilation unit.
    Decl_factory &m_decl_factory;

    /// The GLSL expression factory of the compilation unit.
    Expr_factory &m_expr_factory;

    /// The GLSL statement factory of the compilation unit.
    Stmt_factory &m_stmt_factory;

    /// The GLSL type cache, holding all predefined GLSL types.
    mutable Type_cache  m_tc;

    /// The GLSL value factory of the compilation unit.
    Value_factory &m_value_factory;

    /// The GLSL symbol table of the compilation unit.
    Symbol_table &m_symbol_table;
 
    /// The GLSL definition table of the compilation unit.
    Definition_table &m_def_tab;

    /// The current GLSLang context.
    glsl::GLSLang_context const &m_ctx;

    typedef ptr_hash_map<llvm::Type, glsl::Type *>::Type Type2type_map;

    /// The type cache mapping from LLVM to GLSL types.
    Type2type_map m_type_cache;

    /// Debug info on types.
    sl::DebugTypeHelper m_debug_types;

    typedef hash_map<string, unsigned, string_hash<string> >::Type Ref_fname_id_map;

    /// Referenced source files.
    Ref_fname_id_map m_ref_fnames;

    /// Backend messages.
    mi::mdl::Messages_impl &m_messages;

    typedef ptr_hash_set<glsl::Declaration const>::Type Decl_set;

    /// The set of all API related declarations.
    Decl_set m_api_decls;

    // An entry in the uniform initializer list.
    class Uniform_initializer {
    public:
        /// Constructor.
        Uniform_initializer(
            glsl::Definition *d,
            glsl::Value      *v)
        : def(d)
        , init(v)
        {
        }

        /// Get the definition.
        glsl::Definition *get_def() const { return def; }

        /// Get the initializer.
        glsl::Value *get_init() const { return init; }

    private:
        glsl::Definition *def;
        glsl::Value      *init;
    };

    typedef list<Uniform_initializer>::Type Uniform_initializers;

    /// The list of uniform initializers.
    Uniform_initializers m_uniform_inits;

    /// Maximum size of the constant data segment in the generated GLSL code.
    size_t m_max_const_segment_size;

    /// Current size of the constant data segment in the generated GLSL code.
    size_t m_curr_const_segment_size;

    /// Target version for this code generator.
    glsl::GLSLang_version m_glslang_version;

    /// Target language profile.
    unsigned m_glslang_profile;

    /// Enabled glsl extensions if any.
    char const *m_glsl_enabled_extentions;

    /// Required glsl extensions if any.
    char const *m_glsl_required_extentions;

    /// The uniform SSBO interface declaration once one is created.
    glsl::Declaration_interface *m_ssbo_decl;

    /// uniform SSBO name if any.
    char const *m_glsl_uniform_ssbo_name;

    /// The shader storage buffer object binding (if != ~0u).
    unsigned m_glsl_uniform_ssbo_binding;

    /// The shader storage buffer object set (if != ~0u).
    unsigned m_glsl_uniform_ssbo_set;

    /// ID used to create unique names.
    unsigned m_next_unique_name_id;

    /// If true, uniform initializers will be combined into one shader storage buffer object.
    bool m_place_uniform_inits_into_ssbo;

    /// If true, use debug info.
    bool m_use_dbg;
};

}  // glsl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_GLSL_WRITER_H
