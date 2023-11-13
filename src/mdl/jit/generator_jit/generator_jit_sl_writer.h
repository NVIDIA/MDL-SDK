/******************************************************************************
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_SL_WRITER_H
#define MDL_GENERATOR_JIT_SL_WRITER_H 1

#include <unordered_map>
#include <vector>

#include <mi/base/handle.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>

#include <llvm/Pass.h>

#include "generator_jit_llvm.h"
#include "generator_jit_sl_depgraph.h"
#include "generator_jit_sl_function.h"

namespace mi {
namespace mdl {

// forward
class IOutput_stream;
class Messages_impl;
class Options_impl;
class Generated_code_source;

namespace sl {

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

/// Helper class create AST declarations in post order from visiting a dependency graph.
template<typename Ast>
class Enter_func_decl : public DG_visitor<typename Ast::Definition> {
    typedef typename Ast::Compilation_unit Compilation_unit;
    typedef typename Ast::Definition       Definition;
    typedef typename Ast::Declaration      Declaration;

public:
    /// Pre-visit a node.
    void pre_visit(DG_node<Definition> const *node) final {};

    /// Post-visit a node.
    void post_visit(DG_node<Definition> const *node) final
    {
        Definition  const *def  = node->get_definition();
        Declaration       *decl = def->get_declaration();

        if (decl != nullptr) {
            m_unit->add_decl(decl);
        }
    }

    /// Constructor.
    ///
    /// \param unit  the compilation unit where declaration will be added
    Enter_func_decl(Compilation_unit *unit) : m_unit(unit) {}

private:
    /// The compilation unit.
    Compilation_unit *m_unit;
};

/// This pass generates a structured language AST from an LLVM module and dumps
/// it into an output stream.
template<typename BasePass>
class SLWriterPass : public BasePass
{
    template<typename AST> friend class InsertValueObject;

    typedef BasePass Base;

    typedef SLWriterPass<BasePass> Self;

    // shorter name for the type traits
    typedef typename BasePass::TypeTraits AST;

    typedef typename AST::ICompiler             ICompiler;
    typedef typename AST::IPrinter              IPrinter;
    typedef typename AST::Compilation_unit      Compilation_unit;
    typedef typename AST::Definition_table      Definition_table;
    typedef typename AST::Definition            Definition;
    typedef typename AST::Def_function          Def_function;
    typedef typename AST::Def_variable          Def_variable;
    typedef typename AST::Def_param             Def_param;
    typedef typename AST::Scope                 Scope;
    typedef typename AST::Symbol_table          Symbol_table;
    typedef typename AST::Symbol                Symbol;
    typedef typename AST::Type_factory          Type_factory;
    typedef typename AST::Type                  Type;
    typedef typename AST::Type_void             Type_void;
    typedef typename AST::Type_bool             Type_bool;
    typedef typename AST::Type_int              Type_int;
    typedef typename AST::Type_uint             Type_uint;
    typedef typename AST::Type_float            Type_float;
    typedef typename AST::Type_double           Type_double;
    typedef typename AST::Type_scalar           Type_scalar;
    typedef typename AST::Type_vector           Type_vector;
    typedef typename AST::Value_matrix          Value_matrix;
    typedef typename AST::Type_array            Type_array;
    typedef typename AST::Type_struct           Type_struct;
    typedef typename AST::Type_function         Type_function;
    typedef typename AST::Type_matrix           Type_matrix;
    typedef typename AST::Type_compound         Type_compound;
    typedef typename AST::Value_factory         Value_factory;
    typedef typename AST::Value                 Value;
    typedef typename AST::Value_scalar          Value_scalar;
    typedef typename AST::Value_fp              Value_fp;
    typedef typename AST::Value_vector          Value_vector;
    typedef typename AST::Value_array           Value_array;
    typedef typename AST::Name                  Name;
    typedef typename AST::Type_name             Type_name;
    typedef typename AST::Type_qualifier        Type_qualifier;
    typedef typename AST::Declaration           Declaration;
    typedef typename AST::Declaration_struct    Declaration_struct;
    typedef typename AST::Declaration_field     Declaration_field;
    typedef typename AST::Declaration_function  Declaration_function;
    typedef typename AST::Declaration_variable  Declaration_variable;
    typedef typename AST::Declaration_param     Declaration_param;
    typedef typename AST::Field_declarator      Field_declarator;
    typedef typename AST::Init_declarator       Init_declarator;
    typedef typename AST::Stmt_factory          Stmt_factory;
    typedef typename AST::Stmt                  Stmt;
    typedef typename AST::Stmt_compound         Stmt_compound;
    typedef typename AST::Stmt_expr             Stmt_expr;
    typedef typename AST::Expr_factory          Expr_factory;
    typedef typename AST::Expr                  Expr;
    typedef typename AST::Expr_binary           Expr_binary;
    typedef typename AST::Expr_unary            Expr_unary;
    typedef typename AST::Expr_ref              Expr_ref;
    typedef typename AST::Expr_call             Expr_call;
    typedef typename AST::Expr_compound         Expr_compound;
    typedef typename AST::Expr_literal          Expr_literal;
    typedef typename AST::Parameter_qualifier   Parameter_qualifier;
    typedef typename AST::Location              Location;

public:
    static char ID;

public:
    /// Constructor.
    ///
    /// \param alloc                the allocator
    /// \param type_mapper          the type mapper
    /// \param code                 the source code object that is written to
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param enable_debug         true, if debug info should be generated
    /// \param options              backend options
    /// \param messages             backend messages
    /// \param exp_func_list        the list of exported functions for which prototypes must
    ///                             be generated
    /// \param func_remaps          function remap map
    /// \param df_handle_slot_mode  the mode of handles in API types
    /// \param enable_opt_remarks   enable OptimizationRemarks
    SLWriterPass(
        mi::mdl::IAllocator                                  *alloc,
        Type_mapper const                                    &type_mapper,
        Generated_code_source                                &code,
        unsigned                                             num_texture_spaces,
        unsigned                                             num_texture_results,
        mi::mdl::Options_impl const                          &options,
        mi::mdl::Messages_impl                               &messages,
        bool                                                 enable_debug,
        mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
        mi::mdl::Function_remap const                        &func_remaps,
        mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
        bool                                                 enable_opt_remarks);

    /// Specifies which analysis info is necessary and which is preserved.
    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnModule(llvm::Module &M) final;

private:
    /// Register all API types (structs that are used in the code, but implemented by the user).
    ///
    /// \param options              backend options
    /// \param df_handle_slot_mode  the mode of handles in API types
    void register_api_types(
        mi::mdl::Options_impl const  &options,
        mi::mdl::Df_handle_slot_mode df_handle_slot_mode);

    /// Get the definition for a LLVM function, if one exists.
    ///
    /// \param func  the LLVM function
    Def_function *get_definition(
        llvm::Function *func);

    /// Return the "inner" element type of array types.
    ///
    /// \param type  the type to process
    ///
    /// \return if type is an array type, return the inner element type, else type itself.
    static Type *inner_element_type(Type *type) { return Base::inner_element_type(type); }

    /// Generate an AST for the given structured function.
    ///
    /// \param func  the structured function
    void translate_function(
        llvm::sl::StructuredFunction const *func);

    /// Translate a region into an AST.
    ///
    /// \param region  the region to translate
    Stmt *translate_region(
        llvm::sl::Region const *region);

    /// Translate a block-region into an AST.
    ///
    /// \param region  the region to translate
    Stmt *translate_block(
        llvm::sl::Region const *region);

    /// Translate a natural loop-region into an AST.
    ///
    /// \param region  the natural loop region to translate
    Stmt *translate_natural(
        llvm::sl::RegionNaturalLoop const *region);

    /// Translate a if-then-region into an AST.
    ///
    /// \param region  the if-then region to translate
    Stmt *translate_if_then(
        llvm::sl::RegionIfThen const *region);

    /// Translate a if-then-else-region into an AST.
    ///
    /// \param region  the if-then-else region to translate
    Stmt *translate_if_then_else(
        llvm::sl::RegionIfThenElse const *region);

    /// Translate a break-region into an AST.
    ///
    /// \param region  the break region to translate
    Stmt *translate_break(
        llvm::sl::RegionBreak const *region);

    /// Translate a continue-region into an AST.
    ///
    /// \param region  the continue region to translate
    Stmt *translate_continue(
        llvm::sl::RegionContinue const *region);

    /// Translate a return-region into an AST.
    ///
    /// \param region  the return region to translate
    Stmt *translate_return(
        llvm::sl::RegionReturn const *region);

    /// Return the base pointer of the given pointer after skipping all bitcasts and GEPs.
    llvm::Value *get_base_pointer(
        llvm::Value *pointer);

    /// Recursive part of process_pointer_address implementation.
    llvm::Value *process_pointer_address_recurse(
        Type_walk_stack &stack,
        llvm::Value     *pointer,
        uint64_t        write_size);

    /// Initialize the type walk stack for the given pointer and the size to be written
    /// and return the base pointer. The stack will be filled until the largest sub-element
    /// is found which is smaller or equal to the write size.
    llvm::Value *process_pointer_address(
        Type_walk_stack &stack,
        llvm::Value     *pointer,
        uint64_t        write_size);

    /// Get the definition for a compound type field.
    ///
    /// \param type    the compound type
    /// \param sym     the symbol of the field
    ///
    /// \return the definition of the field or NULL if the field or the type was not found
    Definition *get_field_definition(
        Type   *type,
        Symbol *sym);

    /// Create an lvalue expression for the compound element given by the type walk stack.
    Expr *create_compound_elem_expr(
        Type_walk_stack &stack,
        llvm::Value     *base_pointer);

    /// Return true, if the index is valid for the given composite type.
    /// We define valid here as is not out of bounds according to the type.
    static bool is_valid_composite_index(
        llvm::Type *type,
        size_t     index);

    /// Go to the next element in the stack.
    ///
    /// \returns false, on failure
    bool move_to_next_compound_elem(
        Type_walk_stack &stack);

    /// Move into compound elements in the stack until element size is not too big anymore.
    /// \returns false, on failure
    bool move_into_compound_elem(Type_walk_stack &stack, size_t target_size);

    /// Add statements to zero initializes the given lvalue expression.
    template <unsigned N>
    void create_zero_init(
        llvm::SmallVector<Stmt *, N> &stmts,
        Expr                         *lval_expr);

    /// Translate a call into one or more statements, if it is a supported intrinsic call.
    template <unsigned N>
    bool translate_intrinsic_call(
        llvm::SmallVector<Stmt *, N> &stmts,
        llvm::CallInst               *call);

    /// Translate a struct select into an if statement writing to the given variable.
    ///
    /// \param select   an LLVM select instruction
    /// \param dst_var  the definition of an AST variable
    Stmt *translate_struct_select(
        llvm::SelectInst *select,
        Def_variable     *dst_var);

    /// Bitcast the given LLVM value to the given LLVM type and return an AST expression for it.
    ///
    /// \param val        an LLVM value
    /// \param dest_type  the LLVM destination type
    ///
    /// \returns nullptr, if the conversion is not possible, the converted value otherwise
    Expr *bitcast_to(
        llvm::Value *val,
        llvm::Type  *dest_type);

    /// Bitcast the given AST value to the given AST type and return an AST expression for it.
    /// \returns nullptr, if the conversion is not possible, the converted value otherwise
    ///
    /// \param val        an AST representation of a value
    /// \param dest_type  the AST destination type
    Expr *bitcast_to(
        Expr *val,
        Type *dest_type);

    /// Translate an LLVM store instruction into one or more AST statements.
    template <unsigned N>
    void translate_store(
        llvm::SmallVector<Stmt *, N> &stmts,
        llvm::StoreInst              *inst);

    /// Translate an LLVM ConstantInt value to an AST Value.
    Value *translate_constant_int(
        llvm::ConstantInt *ci);

    /// Translate an LLVM ConstantFP value to an AST Value.
    Value *translate_constant_fp(
        llvm::ConstantFP *cf);

    /// Translate an LLVM ConstantDataVector value to an AST Value.
    Value *translate_constant_data_vector(
        llvm::ConstantDataVector *cv);

    /// Translate an LLVM ConstantDataArray value to an AST compound initializer.
    Expr *translate_constant_data_array(
        llvm::ConstantDataArray *cv);

    /// Translate an LLVM ConstantStruct value to an AST expression.
    /// \param cv         the constant value
    /// \param is_global  if true, create a constructor for an global constant initializer
    Expr *translate_constant_struct_expr(
        llvm::ConstantStruct *cv,
        bool                 is_global);

    /// Translate an LLVM ConstantVector value to an AST Value.
    Value *translate_constant_vector(
        llvm::ConstantVector *cv);

    /// Translate an LLVM ConstantArray value to an AST compound expression.
    /// \param cv         the constant value
    /// \param is_global  if true, create a constructor for an global constant initializer
    Expr *translate_constant_array(
        llvm::ConstantArray *cv,
        bool                is_global);

    /// Translate an LLVM ConstantAggregateZero value to an AST compound expression.
    /// \param cv         the constant value
    /// \param is_global  if true, create a constructor for an global constant initializer
    Expr *translate_constant_array(
        llvm::ConstantAggregateZero *cv,
        bool                        is_global);

    /// Translate an LLVM ConstantArray value to an AST matrix value.
    Value *translate_constant_matrix(llvm::ConstantArray *cv);

    /// Translate an LLVM ConstantAggregateZero value to an AST Value.
    Value *translate_constant_aggregate_zero(llvm::ConstantAggregateZero *cv);

    /// Translate an LLVM Constant value to an AST Value.
    Value *translate_constant(llvm::Constant *ci);

    /// Translate an LLVM Constant value to an AST expression.
    ///
    /// \param ci         the constant value
    /// \param is_global  if true, create a constructor for an global constant initializer
    Expr *translate_constant_expr(
        llvm::Constant *ci,
        bool           is_global);

    /// Translate an LLVM value to an AST expression.
    ///
    /// \param value    the LLVM value to translate
    Expr *translate_expr(
        llvm::Value *value);

    /// Translate a binary LLVM instruction to an AST expression.
    ///
    /// \param inst  the LLVM instruction
    Expr *translate_expr_bin(
        llvm::Instruction *inst);

    /// Find the first function parameter of the given type in the current function.
    ///
    /// \param type  an LLVM type
    ///
    /// \return  the AST Definition of the first parameter of this type or NULL if none exists
    Definition *find_parameter_of_type(
        llvm::Type *t);

    /// Translate an LLVM select instruction to an AST expression.
    ///
    /// \param inst  the LLVM select instruction
    Expr *translate_expr_select(
        llvm::SelectInst *inst);

    /// Translate an LLVM call instruction to an AST expression.
    ///
    /// \param inst     the call instruction to translate
    /// \param dst_var  if non-NULL, the call result should be written to this variable/parameter
    Expr *translate_expr_call(
        llvm::CallInst  *inst,
        Definition      *dst_var);

    /// Translate an LLVM cast instruction to an AST expression.
    ///
    /// \param inst  the LLVM cast instruction
    Expr *translate_expr_cast(
        llvm::CastInst *inst);

    /// Translate an LLVM load instruction to an AST expression.
    ///
    /// \param inst  the LLVM load instruction
    Expr *translate_expr_load(
        llvm::LoadInst *inst);

    /// Translate an LLVM store instruction to an AST expression.
    ///
    /// \param inst  the LLVM store instruction
    Expr *translate_expr_store(
        llvm::StoreInst *inst);

    /// Translate an LLVM pointer value to an AST lvalue expression.
    ///
    /// \param pointer  the LLVM pointer value
    Expr *translate_lval_expression(
        llvm::Value *pointer);

    /// Returns the given expression as a call expression, if it is a call to a vector constructor.
    static Expr_call *as_vector_constructor_call(
        Expr *expr);

    /// Translate an LLVM ShuffleVector instruction to an AST expression.
    ///
    /// \param  the LLVM shuffle vector instruction
    Expr *translate_expr_shufflevector(
        llvm::ShuffleVectorInst *inst);

    /// Translate an LLVM InsertElement instruction to an AST expression.
    ///
    /// \param inst  the LLVM insert element instruction
    Expr *translate_expr_insertelement(
        llvm::InsertElementInst *inst);

    /// Translate an LLVM InsertValue instruction to an AST expression.
    ///
    /// \param inst  the LLVM insert value instruction
    Expr *translate_expr_insertvalue(
        llvm::InsertValueInst *inst);

    /// Translate an LLVM ExtractElement instruction to an AST expression.
    ///
    /// \param inst  the LLVM extract element instruction
    Expr *translate_expr_extractelement(
        llvm::ExtractElementInst *inst);

    /// Translate an LLVM ExtractValue instruction to an AST expression.
    ///
    /// \param inst  the LLVM extract value instruction
    Expr *translate_expr_extractvalue(
        llvm::ExtractValueInst *inst);

    /// Get the type of the i-th subelement. For vector and array types, this is the
    /// element type and, for matrix types, this is the column type, if the index is not of bounds.
    /// For structs, it is the type of the i-th field.
    Type *get_compound_sub_type(
        Type_compound *comp_type,
        unsigned      i);

    /// Add a parameter to the given function and the current definition table
    /// and return its symbol.
    /// Ensures, that the parameter has a unique name in the current scope.
    ///
    /// \param func        the AST function declaration to be modified
    /// \param param_name  the name of the new parameter
    /// \param param_type  the AST type of the parameter
    Def_param *add_func_parameter(
        Declaration_function *func,
        char const           *param_name,
        Type                 *param_type);

    /// Get the AST definition of a generated struct constructor.
    /// This generates the struct constructor, if it does not exist, yet.
    ///
    /// \param type  an AST struct type
    Def_function *get_struct_constructor(
        Type_struct *type);

    /// Get the AST symbol for a given vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    ///
    /// \param index  the vector index
    Symbol *get_vector_index_sym(
        uint64_t index);

    /// Get the AST symbol for a given LLVM vector index, if possible.
    /// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
    ///
    /// \param index  an LLVM value representing the vector index
    Symbol *get_vector_index_sym(
        llvm::Value *index);

    /// Create a select or array_subscript expression on a field of a compound.
    ///
    /// \param comp_expr   an AST expression representing an component value
    /// \param comp_index  the component index to select
    Expr *create_compound_access(
        Expr     *comp_expr,
        unsigned comp_index);

    /// Create a select or array_subscript expression on an element of a vector.
    ///
    /// \param vec    the AST vector expression
    /// \param index  the array subscription index
    ///
    /// \return the equivalent of vec[i]
    Expr *create_vector_access(
        Expr     *vec,
        unsigned index);

    /// Create an array_subscript expression on an element of an array.
    ///
    /// \param array  an AST expression representing an array typed value
    /// \param index  the index of the element to be extracted
    Expr *create_array_access(
        Expr     *array,
        unsigned index);

    /// Create an array_subscript expression on an matrix.
    ///
    /// \param matrix  an AST expression representing a matrix typed value
    /// \param index  the index of the element to be extracted
    Expr *create_matrix_access(
        Expr     *matrix,
        unsigned index);

    /// Create a select expression on a field of a struct.
    ///
    /// \param struct_expr  an AST expression representing a struct typed value
    /// \param field_index  the index of the field to be extracted
    Expr *create_field_access(
        Expr     *struct_expr,
        unsigned field_index);

    /// Create an assign expression, assigning an expression to a variable.
    ///
    /// \param var_def   the definition of the variable to be assigned
    /// \param expr      the expression to be assigned to the variable
    Expr *create_assign_expr(
        Definition *var_def,
        Expr       *expr);

    /// Create an assign statement, assigning an expression to an lvalue expression.
    ///
    /// \param lvalue   the lvalue expression
    /// \param expr     the expression to be assigned to the lvalue expression
    Stmt *create_assign_stmt(
        Expr *lvalue,
        Expr *expr);

    /// Create an assign statement, assigning an expression to a variable.
    ///
    /// \param var_def   the definition of the variable to be assigned
    /// \param expr      the expression to be assigned to the variable
    Stmt *create_assign_stmt(
        Definition *var_def,
        Expr       *expr);

    /// Generates a constructor call for the given AST type.
    ///
    /// \param type       the type
    /// \param args       the arguments to the constructor call
    /// \param loc        the location for the call
    /// \param is_global  if true, create a constructor for an global constant initializer
    Expr *create_constructor_call(
        Type                    *type,
        Array_ref<Expr *> const &args,
        Location const          &loc,
        bool                     is_global);

    /// Generates a constructor call for the given AST type.
    ///
    /// \param type  the type
    /// \param args  the arguments to the constructor call
    /// \param loc   the location for the call
    Expr *create_constructor_call(
        Type                    *type,
        Array_ref<Expr *> const &args,
        Location const          &loc);

    /// Generates a new local variable for an AST symbol and an LLVM type.
    ///
    /// \param var_sym             the variable AST symbol
    /// \param type                the LLVM type of the local to create
    /// \param add_decl_to_prolog  if true, add the variable declaration to current function prolog
    Def_variable *create_local_var(
        Symbol     *var_sym,
        llvm::Type *type,
        bool       add_decl_to_prolog = true);

    /// Generates a new local variable for an LLVM value and use this variable as the value's
    /// result in further generated AST code.
    ///
    /// \param value               the LLVM value
    /// \param do_not_register     if true, do not map this variable as the result for value
    /// \param add_decl_to_prolog  if true, add the variable declaration to current function prolog
    Def_variable *create_local_var(
        llvm::Value *value,
        bool        do_not_register = false,
        bool        add_decl_to_prolog = true);

    /// Generates a new local const variable to hold an LLVM constant.
    ///
    /// \param cv  the LLVM constant
    Def_variable *create_local_const(
        llvm::Constant *cv);

    /// Generates a new global static const entity to hold an LLVM constant.
    ///
    /// \param cv  the LLVM constant
    Definition *create_global_const(
        llvm::Constant *cv);

    /// Get or create the in- and out-variables for an LLVM PHI node.
    ///
    /// \param phi  the LLVM phi node
    std::pair<Def_variable *, Def_variable *> get_phi_vars(
        llvm::PHINode *phi,
        bool enter_in_var = false);

    /// Get the definition of the in-variable of a PHI node.
    ///
    /// \param phi  the LLVM phi node
    Def_variable *get_phi_in_var(
        llvm::PHINode *phi,
        bool enter_in_var = false);

    /// Get the definition of the out-variable of a PHI node, where the in-variable
    /// will be written to at the start of a block.
    ///
    /// \param phi  the LLVM phi node
    Def_variable *get_phi_out_var(
        llvm::PHINode *phi);

    /// Returns true, if for the given LLVM value a variable or parameter exists.
    bool has_local_var(llvm::Value *value) {
        return m_local_var_map.find(value) != m_local_var_map.end();
    }

    /// Check if the given AST statement is empty.
    ///
    /// \param stmt  the statement to check
    static bool is_empty_statment(
        Stmt *stmt);

    /// Joins two statements into a compound statement.
    /// If one or both statements already are compound statements, they will be merged.
    /// If one statement is "empty" (nullptr or ";"), the other will be returned.
    ///
    /// \param head  the head statement
    /// \param tail  the tail statement
    ///
    /// \returns the equivalent of { head; tail; }
    Stmt *join_statements(
        Stmt *head,
        Stmt *tail);

    /// Returns true, if the expression is a reference to the given definition.
    bool is_ref_to_def(Expr *expr, Definition *def) {
        if (Expr_ref *ref = as<Expr_ref>(expr)) {
            return ref->get_definition() == def;
        }
        return false;
    }

private:
    /// Output stream where the generated AST code will be written to.
    Generated_code_source &m_code;

    /// List of exported functions for which prototypes should be generated.
    mi::mdl::LLVM_code_generator::Exported_function_list &m_exp_func_list;

    /// The function remap map.
    mi::mdl::Function_remap const &m_func_remaps;

    /// The dependence graph for the unit.
    Dep_graph<Definition> m_dg;

    /// The current function's dependence node.
    DG_node<Definition> *m_curr_node;

    /// The currently translated function.
    llvm::Function *m_curr_func;

    /// The start block of the current function.
    Stmt_compound *m_cur_start_block;

    typedef typename ptr_hash_map<llvm::Value, Definition *>::Type Variable_map;

    /// The map from LLVM values to local AST definitions for the current AST function.
    Variable_map m_local_var_map;

    /// The map from LLVM constant values to global AST definitions for the current AST function.
    Variable_map m_global_const_map;

    typedef typename ptr_hash_map<
        llvm::PHINode,
        std::pair<Def_variable *, Def_variable *>
    >::Type Phi_map;

    /// The map from PHI nodes to in- and out-variables used to implement the PHIs.
    Phi_map m_phi_var_in_out_map;

    typedef typename ptr_hash_map<Type_struct, Def_function *>::Type Struct_map;

    /// A map from AST struct types to the function definitions of generated constructor functions.
    Struct_map m_struct_constructor_map;

    typedef typename ptr_hash_map<llvm::Function, Def_function *>::Type Function_map;

    /// The map from LLVM functions to AST function definitions.
    Function_map m_llvm_function_map;

    /// If non-null, the definition of the current out-parameter used to pass the result.
    Def_param *m_out_def;

    /// The number of supported texture spaces.
    unsigned m_num_texture_spaces;

    /// The number of texture result entries.
    unsigned m_num_texture_results;

    /// Specified the layout of the BSDF_evaluate_data and BSDF_auxiliary_data structs.
    mi::mdl::Df_handle_slot_mode m_df_handle_slot_mode;

    /// The data layout of the current module.
    llvm::DataLayout const *m_cur_data_layout;
};

/// Creates a HLSL writer pass.
///
/// \param[in]  alloc                    the allocator
/// \param[in]  type_mapper              the type mapper
/// \param[in]  out                      the source code object the HLSL code is written to
/// \param[in]  num_texture_spaces       the number of supported texture spaces
/// \param[in]  num_texture_results      the number of texture result entries
/// \param[in]  options                  backend options
/// \param[out] messages                 backend messages
/// \param[in]  enable_debug             true, if debug info should be generated
/// \param[in]  df_handle_slot_mode      the layout of the BSDF_{evaluate, auxiliary}_data structs
/// \param[out] exp_func_list            list of exported functions
/// \param[in]  func_remaps              function remap map
/// \param[in]  enable_opt_remarks       enable OptimizationRemarks
/// \param[in]  enable_noinline_support  enable support for noinline (otherwise noinline is ignored)
llvm::Pass *createHLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    Generated_code_source                                &code,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    mi::mdl::Options_impl const                          &options,
    mi::mdl::Messages_impl                               &messages,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Function_remap const                        &func_remaps,
    bool                                                 enable_opt_remarks,
    bool                                                 enable_noinline_support);

/// Creates a GLSL writer pass.
///
/// \param[in]  alloc                the allocator
/// \param[in]  type_mapper          the type mapper
/// \param[in]  out                  the source code object the GLSL code is written to
/// \param[in]  num_texture_spaces   the number of supported texture spaces
/// \param[in]  num_texture_results  the number of texture result entries
/// \param[in]  options              backend options
/// \param[out] messages             backend messages
/// \param[in]  enable_debug         true, if debug info should be generated
/// \param[in]  df_handle_slot_mode  the layout of the BSDF_{evaluate, auxiliary}_data structs
/// \param[out] exp_func_list        list of exported functions
/// \param[in]  func_remaps          function remap map
llvm::Pass *createGLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    Generated_code_source                                &code,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    mi::mdl::Options_impl const                          &options,
    mi::mdl::Messages_impl                               &messages,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Function_remap const                        &func_remaps);

}  // sl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_SL_WRITER_H
