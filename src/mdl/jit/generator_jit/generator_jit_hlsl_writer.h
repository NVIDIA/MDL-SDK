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

#ifndef MDL_GENERATOR_JIT_HLSL_WRITER_H
#define MDL_GENERATOR_JIT_HLSL_WRITER_H 1

#include <unordered_map>
#include <vector>

#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_types.h>

#include <llvm/Pass.h>

#include "generator_jit_llvm.h"
#include "generator_jit_hlsl_function.h"
#include "generator_jit_hlsl_depgraph.h"
#include "generator_jit_type_map.h"

namespace mi {
namespace mdl {

// forward
class IOutput_stream;

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

struct Type_walk_element
{
    llvm::Type  *llvm_type;
    size_t       gep_depth;
    llvm::Value *field_index_val;
    unsigned     field_index_offs;  ///!< offset to be added to field_index_val
    llvm::Type  *field_type;

    Type_walk_element(
        llvm::Type  *llvm_type,
        size_t       gep_depth,
        llvm::Value *field_index_val,
        unsigned     field_index_offs,
        llvm::Type  *field_type)
    : llvm_type(llvm_type)
    , gep_depth(gep_depth)
    , field_index_val(field_index_val)
    , field_index_offs(field_index_offs)
    , field_type(field_type)
    {}

    uint64_t get_total_size_and_update_offset(llvm::DataLayout const *dl, uint64_t offs)
    {
        uint64_t cur_align = dl->getABITypeAlignment(field_type);
        uint64_t aligned_offs = (offs + cur_align - 1) & ~(cur_align - 1);
        uint64_t cur_type_size = dl->getTypeAllocSize(field_type);
        uint64_t total_size = aligned_offs - offs + cur_type_size;
        offs = aligned_offs + cur_type_size;
        return total_size;
    }
};

typedef std::vector<Type_walk_element> Type_walk_stack;

/// This pass generates an HLSL AST from an LLVM module and dumps it into an output stream.
class HLSLWriterPass : public llvm::ModulePass
{
    friend class InsertValueObject;

public:
    static char ID;

public:
    /// Constructor.
    ///
    /// \param alloc                the allocator
    /// \param type_mapper          the type mapper
    /// \param out                  the output stream the HLSL code is written to
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param enable_debug         true, if debug info should be generated
    /// \param exp_func_list        the list of exported functions for which prototypes must
    ///                             be generated
    HLSLWriterPass(
        mi::mdl::IAllocator                                  *alloc,
        Type_mapper const                                    &type_mapper,
        mi::mdl::IOutput_stream                              &out,
        unsigned                                             num_texture_spaces,
        unsigned                                             num_texture_results,
        bool                                                 enable_debug,
        mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
        mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode);

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    llvm::StringRef getPassName() const final {
        return "HLSL writer";
    }

    bool runOnModule(llvm::Module &M) final;

private:
    /// Generate HLSL predefined entities into the definition table.
    void fillPredefinedEntities();

    /// Create the HLSL definition for a user defined LLVM function.
    hlsl::Def_function *create_definition(llvm::Function * func);

    /// Get the definition for a LLVM function, if one exists.
    hlsl::Def_function *get_definition(llvm::Function * func);

    /// Generate HLSL AST for the given function.
    void translate_function(llvm::hlsl::ASTFunction const *ast_func);

    /// Translate a region into HLSL AST.
    hlsl::Stmt *translate_region(llvm::hlsl::Region const *region);

    /// Translate a block-region into HLSL AST.
    hlsl::Stmt *translate_block(llvm::hlsl::Region const *region);

    /// Translate a natural loop-region into HLSL AST.
    hlsl::Stmt *translate_natural(llvm::hlsl::RegionNaturalLoop const *region);

    /// Translate a if-then-region into HLSL AST.
    hlsl::Stmt *translate_if_then(llvm::hlsl::RegionIfThen const *region);

    /// Translate a if-then-else-region into HLSL AST.
    hlsl::Stmt *translate_if_then_else(llvm::hlsl::RegionIfThenElse const *region);

    /// Translate a break-region into HLSL AST.
    hlsl::Stmt *translate_break(llvm::hlsl::RegionBreak const *region);

    /// Translate a continue-region into HLSL AST.
    hlsl::Stmt *translate_continue(llvm::hlsl::RegionContinue const *region);

    /// Translate a return-region into HLSL AST.
    hlsl::Stmt *translate_return(llvm::hlsl::RegionReturn const *region);

    /// Return the base pointer of the given pointer after skipping all bitcasts and GEPs.
    llvm::Value *get_base_pointer(llvm::Value *pointer);

    /// Recursive part of process_pointer_address implementation.
    llvm::Value *process_pointer_address_recurse(
        Type_walk_stack &stack,
        llvm::Value *pointer,
        uint64_t write_size);

    /// Initialize the type walk stack for the given pointer and the size to be written
    /// and return the base pointer. The stack will be filled until the largest sub-element
    /// is found which is smaller or equal to the write size.
    llvm::Value *process_pointer_address(
        Type_walk_stack &stack,
        llvm::Value *pointer,
        uint64_t write_size);

    /// Create an lvalue expression for the compound element given by the type walk stack.
    hlsl::Expr *create_compound_elem_expr(
        Type_walk_stack &stack,
        llvm::Value *base_pointer);

    /// Return true, if the index is valid for the given composite type.
    /// We define valid here as is not out of bounds according to the type.
    static bool is_valid_composite_index(llvm::Type *type, size_t index);

    /// Go to the next element in the stack.
    /// \returns false, on failure
    bool move_to_next_compound_elem(Type_walk_stack &stack);

    /// Add statements to zero initializes the given lvalue expression.
    template <unsigned N>
    void create_zero_init(llvm::SmallVector<hlsl::Stmt *, N> &stmts, hlsl::Expr *lval_expr);

    /// Translate a call into one or more statements, if it is a supported intrinsic call.
    template <unsigned N>
    bool translate_intrinsic_call(llvm::SmallVector<hlsl::Stmt *, N> &stmts, llvm::CallInst *call);

    /// Translate a struct select into an if statement writing to the given variable.
    hlsl::Stmt *translate_struct_select(llvm::SelectInst *select, hlsl::Def_variable *dst_var);

    /// Convert the given LLVM value to the given LLVM type and return an HLSL expression for it.
    /// \returns nullptr, if the conversion is not possible, the converted value otherwise
    hlsl::Expr *convert_to(llvm::Value *val, llvm::Type *dest_type);

    /// Convert the given HLSL value to the given HLSL type and return an HLSL expression for it.
    /// \returns nullptr, if the conversion is not possible, the converted value otherwise
    hlsl::Expr *convert_to(hlsl::Expr *val, hlsl::Type *dest_type);


    /// Translate an LLVM store instruction into one or more HLSL statements.
    template <unsigned N>
    void translate_store(llvm::SmallVector<hlsl::Stmt *, N> &stmts, llvm::StoreInst *inst);

    /// Translate an LLVM ConstantInt value to an HLSL Value.
    hlsl::Value *translate_constant_int(llvm::ConstantInt *ci);

    /// Translate an LLVM ConstantFP value to an HLSL Value.
    hlsl::Value *translate_constant_fp(llvm::ConstantFP *cf);

    /// Translate an LLVM ConstantDataVector value to an HLSL Value.
    hlsl::Value *translate_constant_data_vector(llvm::ConstantDataVector *cv);

    /// Translate an LLVM ConstantDataArray value to an HLSL compound initializer.
    hlsl::Expr *translate_constant_data_array(llvm::ConstantDataArray *cv);

    /// Translate an LLVM ConstantStruct value to an HLSL expression.
    /// \param cv         the constant value
    /// \param is_global  if true, compound expressions will be used,
    ///                   otherwise constructor calls will be generated where needed
    hlsl::Expr *translate_constant_struct_expr(llvm::ConstantStruct *cv, bool is_global);

    /// Translate an LLVM ConstantVector value to an HLSL Value.
    hlsl::Value *translate_constant_vector(llvm::ConstantVector *cv);

    /// Translate an LLVM ConstantArray value to an HLSL compound expression.
    /// \param cv         the constant value
    /// \param is_global  if true, compound expressions will be used,
    ///                   otherwise constructor calls will be generated where needed
    hlsl::Expr *translate_constant_array(llvm::ConstantArray *cv, bool is_global);

    /// Translate an LLVM ConstantAggregateZero value to an HLSL compound expression.
    /// \param cv         the constant value
    /// \param is_global  if true, compound expressions will be used,
    ///                   otherwise constructor calls will be generated where needed
    hlsl::Expr *translate_constant_array(llvm::ConstantAggregateZero *cv, bool is_global);

    /// Translate an LLVM ConstantArray value to an HLSL matrix value.
    hlsl::Value *translate_constant_matrix(llvm::ConstantArray *cv);

    /// Translate an LLVM ConstantAggregateZero value to an HLSL Value.
    hlsl::Value *translate_constant_aggregate_zero(llvm::ConstantAggregateZero *cv);

    /// Translate an LLVM Constant value to an HLSL Value.
    hlsl::Value *translate_constant(llvm::Constant *ci);

    /// Translate an LLVM Constant value to an HLSL expression.
    ///
    /// \param ci         the constant value
    /// \param is_global  if true, compound expressions will be used,
    ///                   otherwise constructor calls will be generated where needed
    hlsl::Expr *translate_constant_expr(llvm::Constant *ci, bool is_global);

    /// Translate an LLVM value to an HLSL expression.
    ///
    /// \param value    the LLVM value to translate
    hlsl::Expr *translate_expr(llvm::Value *value);

    /// If a given type has an unsigned variant, return it.
    ///
    /// \param type  the type
    ///
    /// \return type itself, if it si unsigned, the unsigned type, if it is signed, NULL otherwise.
    hlsl::Type *to_unsigned_type(hlsl::Type *type);

    /// Translate a binary LLVM instruction to an HLSL expression.
    hlsl::Expr *translate_expr_bin(llvm::Instruction *inst);

    /// Find a function parameter of the given type in the current function.
    hlsl::Definition *find_parameter_of_type(llvm::Type *t);

    /// Translate an LLVM select instruction to an HLSL expression.
    hlsl::Expr *translate_expr_select(llvm::SelectInst *inst);

    /// Translate an LLVM call instruction to an HLSL expression.
    ///
    /// \param inst     the call instruction to translate
    /// \param dst_var  if non-NULL, the call result should be written to this variable/parameter
    hlsl::Expr *translate_expr_call(
        llvm::CallInst   *inst,
        hlsl::Definition *dst_var);

    /// Translate an LLVM cast instruction to an HLSL expression.
    hlsl::Expr *translate_expr_cast(llvm::CastInst *inst);

    /// Creates a HLSL cast expression to the given destination type.
    ///
    /// \param dst_type  the destination type
    /// \param expr      the expression to cast
    hlsl::Expr *create_cast(hlsl::Type *dst_type, hlsl::Expr *expr);

    /// Translate an LLVM load instruction to an HLSL expression.
    hlsl::Expr *translate_expr_load(llvm::LoadInst *inst);

    /// Translate an LLVM store instruction to an HLSL expression.
    hlsl::Expr *translate_expr_store(llvm::StoreInst *inst);

    /// Translate an LLVM pointer value to an HLSL lvalue expression.
    hlsl::Expr *translate_lval_expression(llvm::Value *pointer);

    /// Translate an LLVM ShuffleVector instruction to an HLSL expression.
    hlsl::Expr *translate_expr_shufflevector(llvm::ShuffleVectorInst *inst);

    /// Translate an LLVM InsertElement instruction to an HLSL expression.
    hlsl::Expr *translate_expr_insertelement(llvm::InsertElementInst *inst);

    /// Translate an LLVM InsertValue instruction to an HLSL expression.
    hlsl::Expr *translate_expr_insertvalue(llvm::InsertValueInst *inst);

    /// Translate an LLVM ExtractElement instruction to an HLSL expression.
    hlsl::Expr *translate_expr_extractelement(llvm::ExtractElementInst *inst);

    /// Translate an LLVM ExtractValue instruction to an HLSL expression.
    hlsl::Expr *translate_expr_extractvalue(llvm::ExtractValueInst *inst);

    /// Check if a given LLVM array type is the representation of the HLSL matrix type.
    bool is_matrix_type(llvm::ArrayType *type) const;

    /// Get the type of the i-th subelement. For vector and array types, this is the
    /// element type and, for matrix types, this is the column type, if the index is not of bounds.
    /// For structs, it is the type of the i-th field.
    hlsl::Type *get_compound_sub_type(hlsl::Type_compound *comp_type, unsigned i);

    /// Load and process type debug info.
    void process_type_debug_info(llvm::Module const &module);

    /// Convert an LLVM type to an HLSL type.
    hlsl::Type *convert_type(llvm::Type *type);

    /// Add a field to a struct declaration.
    hlsl::Type_struct::Field add_struct_field(
        hlsl::Declaration_struct *decl_struct,
        hlsl::Type               *type,
        hlsl::Symbol             *sym);

    /// Add a field to a struct declaration.
    hlsl::Type_struct::Field add_struct_field(
        hlsl::Declaration_struct *decl_struct,
        hlsl::Type               *type,
        char const               *name);

    /// Convert an LLVM struct type to an HLSL type.
    hlsl::Type *convert_struct_type(llvm::StructType *type);

    /// Create an HLSL struct with the given names for the given LLVM struct type.
    ///
    /// \param type             the LLVM struct type
    /// \param type_name        the name for the HLSL struct type
    /// \param num_field_names  the number of field names, must match the number of fields
    ///                         in the LLVM struct type
    /// \param field_names      the names of the fields of the HLSL struct type
    /// \param add_to_unit      if true, add the type to the unit, so it will be printed
    hlsl::Type_struct *create_struct_from_llvm(
        llvm::StructType *type,
        char const *type_name,
        size_t num_field_names,
        char const * const *field_names,
        bool add_to_unit);

    /// Create the HLSL state core struct for the corresponding LLVM struct type.
    hlsl::Type_struct *create_state_core_struct_type(llvm::StructType *type);

    /// Create the HLSL state environment struct for the corresponding LLVM struct type.
    hlsl::Type_struct *create_state_env_struct_type(llvm::StructType *type);

    /// Create the HLSL resource data struct for the corresponding LLVM struct type.
    hlsl::Type_struct *create_res_data_struct_type(llvm::StructType *type);

    /// Create the Bsdf_sample_data struct type used by libbsdf.
    hlsl::Type_struct *create_bsdf_sample_data_struct_types(llvm::StructType *type);

    /// Create the Bsdf_evaluate_data struct type used by libbsdf.
    hlsl::Type_struct *create_bsdf_evaluate_data_struct_types(llvm::StructType *type);

    /// Create the Bsdf_pdf_data struct type used by libbsdf.
    hlsl::Type_struct *create_bsdf_pdf_data_struct_types(llvm::StructType *type);

    /// Create the Bsdf_auxiliary_data struct type used by libbsdf.
    hlsl::Type_struct *create_bsdf_auxiliary_data_struct_types(llvm::StructType *type);

    /// Create the Edf_sample_data struct type used by libbsdf.
    hlsl::Type_struct *create_edf_sample_data_struct_types(llvm::StructType *type);

    /// Create the Edf_evaluate_data struct type used by libbsdf.
    hlsl::Type_struct *create_edf_evaluate_data_struct_types(llvm::StructType *type);

    /// Create the Edf_pdf_data struct type used by libbsdf.
    hlsl::Type_struct *create_edf_pdf_data_struct_types(llvm::StructType *type);

    /// Create the Edf_auxiliary_data struct type used by libbsdf.
    hlsl::Type_struct *create_edf_auxiliary_data_struct_types(llvm::StructType *type);

    /// Get an HLSL symbol for an LLVM string.
    hlsl::Symbol *get_sym(llvm::StringRef const &str);

    /// Get an HLSL symbol for a string.
    hlsl::Symbol *get_sym(char const *str);

    /// Get a valid HLSL symbol for an LLVM string and a template.
    hlsl::Symbol *get_unique_hlsl_sym(
        llvm::StringRef const &str,
        char const            *templ);

    /// Get a valid HLSL symbol for an LLVM string and a template.
    hlsl::Symbol *get_unique_hlsl_sym(
        char const *str,
        char const *templ);

    /// Get an HLSL type name for an LLVM type.
    hlsl::Type_name *get_type_name(llvm::Type *type);

    /// Get an HLSL type name for an HLSL type.
    hlsl::Type_name *get_type_name(hlsl::Type *type);

    /// Get an HLSL type name for an HLSL symbol.
    hlsl::Type_name *get_type_name(hlsl::Symbol *sym);

    /// Get an HLSL name for the given location and LLVM string.
    hlsl::Name *get_name(Location loc, llvm::StringRef const &str);

    /// Get an HLSL name for the given location and string.
    hlsl::Name *get_name(Location loc, const char *str);

    /// Get an HLSL name for the given location and symbol.
    hlsl::Name *get_name(Location loc, hlsl::Symbol *sym);

    /// Add a parameter to the given function and the current definition table
    /// and return its symbol.
    /// Ensures, that the parameter has a unique name in the current scope.
    hlsl::Symbol *add_func_parameter(
        hlsl::Declaration_function *func,
        char const *param_name,
        hlsl::Type *param_type);

    /// Get the HLSL symbol of a generated struct constructor.
    /// This generates the struct constructor, if it does not exist, yet.
    hlsl::Def_function *get_struct_constructor(hlsl::Type_struct *type);

    /// Get a name for a given vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    char const *get_vector_index_str(uint64_t index);

    /// Get an HLSL symbol for a given vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    hlsl::Symbol *get_vector_index_sym(uint64_t index);

    /// Get an HLSL symbol for a given LLVM vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    hlsl::Symbol *get_vector_index_sym(llvm::Value *index);

    /// Create a reference to a variable of the given type.
    hlsl::Expr *create_reference(hlsl::Type_name *type_name, hlsl::Type *type);

    /// Create a reference to a variable of the given type.
    hlsl::Expr *create_reference(hlsl::Symbol *var_sym, hlsl::Type *type);

    /// Create a reference to a variable of the given type.
    hlsl::Expr *create_reference(hlsl::Symbol *var_sym, llvm::Type *type);

    /// Create a reference to an entity.
    hlsl::Expr *create_reference(hlsl::Definition *def);

    /// Create a select or array_subscript expression on a field of a compound.
    hlsl::Expr *create_compound_access(hlsl::Expr *struct_expr, unsigned comp_index);

    /// Create a select or array_subscript expression on an element of a vector.
    hlsl::Expr *create_vector_access(hlsl::Expr *vec, unsigned index);

    /// Create an array_subscript expression on an element of an array.
    hlsl::Expr *create_array_access(hlsl::Expr *array, unsigned index);

    /// Create an array_subscript expression on an matrix.
    hlsl::Expr *create_matrix_access(hlsl::Expr *matrix, unsigned index);

    /// Create a select expression on a field of a struct.
    hlsl::Expr *create_field_access(hlsl::Expr *struct_expr, unsigned field_index);

    /// Create an assign expression, assigning an expression to a variable.
    ///
    /// \param def       the symbol of the variable to be assigned
    /// \param expr      the expression to be assigned to the variable
    hlsl::Expr *create_assign_expr(hlsl::Definition *var_def, hlsl::Expr *expr);

    /// Create an assign statement, assigning an expression to an lvalue expression.
    ///
    /// \param lvalue_expr  the lvalue expression
    /// \param expr         the expression to be assigned to the lvalue expression
    hlsl::Stmt *create_assign_stmt(hlsl::Expr *lvalue, hlsl::Expr *expr);

    /// Create an assign statement, assigning an expression to a variable.
    ///
    /// \param def       the symbol of the variable to be assigned
    /// \param expr      the expression to be assigned to the variable
    hlsl::Stmt *create_assign_stmt(hlsl::Definition *var_def, hlsl::Expr *expr);

    /// Add array specifier to a declaration if necessary.
    ///
    /// \param decl  the declaration
    /// \param type  the type
    template<typename Decl_type>
    void add_array_specifiers(Decl_type *decl, hlsl::Type *type);

    /// Get the constructor for the given HLSL type.
    ///
    /// \param type  the type
    hlsl::Def_function *lookup_constructor(hlsl::Type *type);

    /// Generates a constructor call for the given HLSL type.
    ///
    /// \param type  the type
    /// \param args  the arguments to the constructor call
    /// \param loc   the location for the call
    hlsl::Expr *create_constructor_call(
        hlsl::Type                    *type,
        Array_ref<hlsl::Expr *> const &args,
        hlsl::Location                 loc);

    /// Generates a new local variable for an HLSL symbol and an LLVM type.
    ///
    /// \param var_sym  the variable symbol
    /// \param type     the LLVM type of the local to create
    hlsl::Def_variable *create_local_var(
        hlsl::Symbol *var_sym,
        llvm::Type   *type,
        bool          add_decl_statement = true);

    /// Generates a new local variable for an LLVM value and use this variable as the value's
    /// result in further generated HLSL code.
    ///
    /// \param value            the LLVM value
    /// \param do_not_register  if true, do not map this variable as the result for value
    hlsl::Def_variable *create_local_var(
        llvm::Value *value,
        bool        do_not_register = false,
        bool        add_decl_statement = true);

    /// Generates a new local const variable to hold an LLVM constant.
    ///
    /// \param cv  the LLVM constant
    hlsl::Def_variable *create_local_const(llvm::Constant *cv);

    /// Generates a new global const variable to hold an LLVM constant.
    ///
    /// \param cv  the LLVM constant
    hlsl::Def_variable *create_global_const(llvm::Constant *cv);

    /// Get or create the in- and out-variables for a PHI node.
    std::pair<hlsl::Def_variable *, hlsl::Def_variable *> get_phi_vars(llvm::PHINode *phi);

    /// Get the definition of the in-variable of a PHI node.
    hlsl::Def_variable *get_phi_in_var(llvm::PHINode *phi);

    /// Get the definition of the out-variable of a PHI node, where the in-variable
    /// will be written to at the start of a block.
    hlsl::Def_variable *get_phi_out_var(llvm::PHINode *phi);

    /// Returns true, if for the given LLVM value a variable or parameter exists.
    bool has_local_var(llvm::Value *value) {
        return m_local_var_map.find(value) != m_local_var_map.end();
    }

    /// Joins two statements into a compound statement.
    /// If one or both statements already are compound statements, they will be merged.
    /// If one statement is "empty" (nullptr), the other will be returned.
    hlsl::Stmt *join_statements(hlsl::Stmt *head, hlsl::Stmt *tail);

    /// Convert the LLVM debug location (if any is attached to the given instruction)
    /// to an HLSL location.
    hlsl::Location convert_location(llvm::Instruction *inst);

    /// Returns true, if the expression is a reference to the given definition.
    bool is_ref_to_def(hlsl::Expr *expr, hlsl::Definition *def) {
        if (hlsl::Expr_ref *ref = hlsl::as<hlsl::Expr_ref>(expr)) {
            return ref->get_definition() == def;
        }
        return false;
    }

private:
    /// MDL allocator used for generating the HLSL AST.
    IAllocator *m_alloc;

    /// the Type mapper.
    Type_mapper const &m_type_mapper;

    /// Output stream where the HLSL code will be written to.
    mi::mdl::IOutput_stream &m_out;

    /// List of exported functions for which prototypes should be generated.
    mi::mdl::LLVM_code_generator::Exported_function_list &m_exp_func_list;

    /// The HLSL compiler.
    mi::base::Handle<ICompiler> m_hlsl_compiler;

    /// The HLSL compilation unit.
    mi::base::Handle<Compilation_unit> m_unit;

    /// The dependence graph for the unit.
    Dep_graph m_dg;

    /// The current function's dependence node.
    DG_node *m_curr_node;

    /// The currently translated function.
    llvm::Function *m_curr_func;

    /// The HLSL declaration factory of the compilation unit.
    Decl_factory &m_decl_factory;

    /// The HLSL expression factory of the compilation unit.
    Expr_factory &m_expr_factory;

    /// The HLSL statement factory of the compilation unit.
    Stmt_factory &m_stmt_factory;

    /// The HLSL type factory of the compilation unit.
    Type_factory &m_type_factory;

    /// The HLSL value factory of the compilation unit.
    Value_factory &m_value_factory;

    /// The HLSL symbol table of the compilation unit.
    Symbol_table &m_symbol_table;

    /// The HLSL definition table of the compilation unit.
    Definition_table &m_def_tab;

    typedef ptr_hash_map<llvm::Type, hlsl::Type *>::Type Type2type_map;

    /// The type cache mapping from LLVM to HLSL types.
    Type2type_map m_type_cache;

    /// The start block of the current HLSL function.
    hlsl::Stmt_compound *m_cur_start_block;

    typedef ptr_hash_map<llvm::Value, hlsl::Definition *>::Type Variable_map;

    /// The map from LLVM values to local HLSL definitions for the current HLSL function.
    Variable_map m_local_var_map;

    /// The map from LLVM values to global HLSL definitions for the current HLSL function.
    Variable_map m_global_var_map;

    typedef ptr_hash_map<
        llvm::PHINode,
        std::pair<hlsl::Def_variable *, hlsl::Def_variable *>
    >::Type Phi_map;

    /// The map from PHI nodes to in- and out-variables used to implement the PHIs.
    Phi_map m_phi_var_in_out_map;

    typedef ptr_hash_map<hlsl::Type_struct, hlsl::Def_function *>::Type Struct_map;

    /// A map from HLSL type structs to the function names of generated constructor functions.
    Struct_map m_struct_constructor_map;

    typedef ptr_hash_map<llvm::Function, hlsl::Def_function *>::Type Function_map;

    /// The map from LLVM functions to HLSL function definitions.
    Function_map m_llvm_function_map;

    /// If non-null, the definition of the current out-parameter used to pass the result.
    hlsl::Def_param *m_out_def;

    /// The number of supported texture spaces.
    unsigned m_num_texture_spaces;

    /// The number of texture result entries.
    unsigned m_num_texture_results;

    /// Specified the layout of the BSDF_evaluate_data and BSDF_auxiliary_data structs.
    mi::mdl::Df_handle_slot_mode m_df_handle_slot_mode;

    /// If true, use debug info.
    bool m_use_dbg;

    /// The data layout of the current module.
    llvm::DataLayout const *m_cur_data_layout;

    typedef hash_map<string, unsigned, string_hash<string> >::Type Ref_fname_id_map;

    /// References source files.
    Ref_fname_id_map m_ref_fnames;

    typedef hash_map<string, llvm::DICompositeType *, string_hash<string> >::Type Struct_info_map;

    /// Debug info regarding struct types.
    Struct_info_map  m_struct_dbg_info;

    /// ID used to create unique names.
    unsigned m_next_unique_name_id;
};

/// Creates a HLSL writer pass.
///
/// \param[in]  alloc                the allocator
/// \param[in]  type_mapper          the type mapper
/// \param[in]  out                  the output stream the HLSL code is written to
/// \param[in]  num_texture_spaces   the number of supported texture spaces
/// \param[in]  num_texture_results  the number of texture result entries
/// \param[in]  enable_debug         true, if debug info should be generated
/// \param[in]  df_handle_slot_mode  the layout of the BSDF_{evaluate, auxiliary}_data structs
/// \param[out] exp_func_list        list of exported functions
llvm::Pass *createHLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    mi::mdl::IOutput_stream                              &out,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list);

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_HLSL_WRITER_H
