/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_VISITOR_H
#define MDL_COMPILERCORE_VISITOR_H 1

#include "compilercore_cc_conf.h"

namespace mi {
namespace mdl {

class IModule;
class IAnnotation;
class IAnnotation_block;
class IArgument;
class IArgument_named;
class IArgument_positional;
class IDeclaration;
class IDeclaration_annotation;
class IDeclaration_constant;
class IDeclaration_function;
class IDeclaration_module;
class IDeclaration_namespace_alias;
class IDeclaration_import;
class IDeclaration_invalid;
class IDeclaration_type_alias;
class IDeclaration_type_enum;
class IDeclaration_type_struct;
class IDeclaration_variable;
class IExpression;
class IExpression_binary;
class IExpression_call;
class IExpression_conditional;
class IExpression_invalid;
class IExpression_let;
class IExpression_literal;
class IExpression_reference;
class IExpression_unary;
class IParameter;
class IQualified_name;
class ISimple_name;
class IStatement;
class IStatement_break;
class IStatement_case;
class IStatement_compound;
class IStatement_continue;
class IStatement_declaration;
class IStatement_do_while;
class IStatement_expression;
class IStatement_for;
class IStatement_if;
class IStatement_invalid;
class IStatement_return;
class IStatement_switch;
class IStatement_while;
class IType_name;

///
/// Stateless AST Visitor.
///
/// Supports fine grained and high level processing.
///
class Module_visitor {
public:

    /// Visit all AST nodes of a module.
    ///
    /// \param module  the module
    void visit(IModule const *module);

    /// Visit a statement and all its children.
    ///
    /// \param stmt  the statement
    void visit(IStatement const *stmt);

    /// Visit an expression and all its children.
    ///
    /// \param expr  the expression
    ///
    /// \return expr or an replacement
    IExpression const *visit(IExpression const *expr);

    /// Visit a declaration and all its children.
    ///
    /// \param decl  the declaration
    void visit(IDeclaration const *decl);

    /// Visit an Argument and all its children.
    ///
    /// \param arg  the argument
    void visit(IArgument const *arg);

    /// Visit a parameter and all its children.
    ///
    /// \param param  the parameter
    void visit(IParameter const *param);

    /// Visit a type name and all its children.
    ///
    /// \param tname  the type name
    void visit(IType_name const *tname);

    /// Visit an annotation and all its children.
    ///
    /// \param block  the annotation block
    void visit(IAnnotation_block const *block);

    /// Visit a qualified name and all its children.
    ///
    /// \param qual_name  the qualified name
    void visit(IQualified_name const *qual_name);

    /// Visit a simple name and all its children.
    ///
    /// \param sname  the simple name
    void visit(ISimple_name const *sname);

    /// Default pre visitor for statements.
    ///
    /// \param stmt  the statement
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten statement is needed.
    virtual bool pre_visit(IStatement *stmt);

    /// Default post visitor for statements.
    ///
    /// \param stmt  the statement
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten statement is needed.
    virtual void post_visit(IStatement *stmt);

    /// Default pre visitor for expressions.
    ///
    /// \param expr  the expression
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten expression is needed.
    virtual bool pre_visit(IExpression *expr);

    /// Default post visitor for expressions.
    ///
    /// \param expr  the expression
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten expression is needed.
    ///
    /// \return expr or an replacement expression
    virtual IExpression *post_visit(IExpression *expr);

    /// Default pre visitor for declarations.
    ///
    /// \param decl  the declaration
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten declaration is needed.
    virtual bool pre_visit(IDeclaration *decl);

    /// Default post visitor for declarations.
    ///
    /// \param decl  the declaration
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten declaration is needed.
    virtual void post_visit(IDeclaration *decl);

    virtual bool pre_visit(IDeclaration_invalid *decl);
    virtual void post_visit(IDeclaration_invalid *decl);

    virtual bool pre_visit(IDeclaration_import *decl);
    virtual void post_visit(IDeclaration_import *decl);

    virtual bool pre_visit(IDeclaration_annotation *anno);
    virtual void post_visit(IDeclaration_annotation *anno);

    virtual bool pre_visit(IDeclaration_constant *con);
    virtual void post_visit(IDeclaration_constant *con);

    virtual bool pre_visit(ISimple_name *name);
    virtual void post_visit(ISimple_name *name);

    virtual bool pre_visit(IQualified_name *name);
    virtual void post_visit(IQualified_name *name);

    virtual bool pre_visit(IParameter *param);
    virtual void post_visit(IParameter *param);

    virtual bool pre_visit(IAnnotation_block *block);
    virtual void post_visit(IAnnotation_block *block);

    virtual bool pre_visit(IAnnotation *anno);
    virtual void post_visit(IAnnotation *anno);

    virtual bool pre_visit(IArgument_named *arg);
    virtual void post_visit(IArgument_named *arg);

    virtual bool pre_visit(IArgument_positional *arg);
    virtual void post_visit(IArgument_positional *arg);

    virtual bool pre_visit(IType_name *tname);
    virtual void post_visit(IType_name *tname);

    virtual bool pre_visit(IDeclaration_type_alias *type_alias);
    virtual void post_visit(IDeclaration_type_alias *type_alias);

    virtual bool pre_visit(IDeclaration_type_struct *type_struct);
    virtual void post_visit(IDeclaration_type_struct *type_struct);

    virtual bool pre_visit(IDeclaration_type_enum *type_enum);
    virtual void post_visit(IDeclaration_type_enum *type_enum);

    virtual bool pre_visit(IDeclaration_variable *var_decl);
    virtual void post_visit(IDeclaration_variable *var_decl);

    virtual bool pre_visit(IDeclaration_function *fkt_decl);
    virtual void post_visit(IDeclaration_function *var_decl);

    virtual bool pre_visit(IDeclaration_module *mod_decl);
    virtual void post_visit(IDeclaration_module *mod_decl);

    virtual bool pre_visit(IDeclaration_namespace_alias *alias_decl);
    virtual void post_visit(IDeclaration_namespace_alias *alias_decl);

    virtual bool pre_visit(IExpression_invalid *expr);
    virtual IExpression *post_visit(IExpression_invalid *expr);

    virtual bool pre_visit(IExpression_literal *expr);
    virtual IExpression *post_visit(IExpression_literal *expr);

    virtual bool pre_visit(IExpression_reference *expr);
    virtual IExpression *post_visit(IExpression_reference *expr);

    virtual bool pre_visit(IExpression_unary *expr);
    virtual IExpression *post_visit(IExpression_unary *expr);

    virtual bool pre_visit(IExpression_binary *expr);
    virtual IExpression *post_visit(IExpression_binary *expr);

    virtual bool pre_visit(IExpression_conditional *expr);
    virtual IExpression *post_visit(IExpression_conditional *expr);

    virtual bool pre_visit(IExpression_call *expr);
    virtual IExpression *post_visit(IExpression_call *expr);

    virtual bool pre_visit(IExpression_let *expr);
    virtual IExpression *post_visit(IExpression_let *expr);

    virtual bool pre_visit(IStatement_invalid *stmt);
    virtual void post_visit(IStatement_invalid *stmt);

    virtual bool pre_visit(IStatement_compound *stmt);
    virtual void post_visit(IStatement_compound *stmt);

    virtual bool pre_visit(IStatement_declaration *stmt);
    virtual void post_visit(IStatement_declaration *stmt);

    virtual bool pre_visit(IStatement_expression *stmt);
    virtual void post_visit(IStatement_expression *stmt);

    virtual bool pre_visit(IStatement_if *stmt);
    virtual void post_visit(IStatement_if *stmt);

    virtual bool pre_visit(IStatement_case *stmt);
    virtual void post_visit(IStatement_case *stmt);

    virtual bool pre_visit(IStatement_switch *stmt);
    virtual void post_visit(IStatement_switch *stmt);

    virtual bool pre_visit(IStatement_while *stmt);
    virtual void post_visit(IStatement_while *stmt);

    virtual bool pre_visit(IStatement_do_while *stmt);
    virtual void post_visit(IStatement_do_while *stmt);

    virtual bool pre_visit(IStatement_for *stmt);
    virtual void post_visit(IStatement_for *stmt);

    virtual bool pre_visit(IStatement_break *stmt);
    virtual void post_visit(IStatement_break *stmt);

    virtual bool pre_visit(IStatement_continue *stmt);
    virtual void post_visit(IStatement_continue *stmt);

    virtual bool pre_visit(IStatement_return *stmt);
    virtual void post_visit(IStatement_return *stmt);

protected:
    /// Constructor.
    explicit Module_visitor();

private:
    void do_invalid_import(IDeclaration_invalid const *import);
    void do_declaration_import(IDeclaration_import const *import);
    void do_type_alias(IDeclaration_type_alias const *ta);
    void do_type_struct(IDeclaration_type_struct const *ts);
    void do_type_enum(IDeclaration_type_enum const *te);
    void do_variable_decl(IDeclaration_variable const *var_decl);
    void do_function_decl(IDeclaration_function const *fkt_decl);
    void do_module_decl(IDeclaration_module const *mod_decl);
    void do_namespace_alias(IDeclaration_namespace_alias const *alias_decl);
    void do_declaration(IDeclaration const *decl);
    void do_named_argument(IArgument_named const *arg);
    void do_positional_argument(IArgument_positional const *arg);
    void do_argument(IArgument const *arg);
    void do_annotation(IAnnotation const *anno);
    void do_annotations(IAnnotation_block const *anno);
    void do_type_name(IType_name const *tn);
    void do_parameter(IParameter const *param);
    void do_declaration_annotation(IDeclaration_annotation const *anno);
    void do_declaration_constant(IDeclaration_constant const *c);
    void do_qualified_name(IQualified_name const *name);
    void do_simple_name(ISimple_name const *name);

    MDL_CHECK_RESULT IExpression const *do_invalid_expression(
        IExpression_invalid const *expr);
    MDL_CHECK_RESULT IExpression const *do_literal_expression(
        IExpression_literal const *expr);
    MDL_CHECK_RESULT IExpression const *do_reference_expression(
        IExpression_reference const *expr);
    MDL_CHECK_RESULT IExpression const *do_unary_expression(
        IExpression_unary const *expr);
    MDL_CHECK_RESULT IExpression const *do_binary_expression(
        IExpression_binary const *expr);
    MDL_CHECK_RESULT IExpression const *do_conditional_expression(
        IExpression_conditional const *expr);
    MDL_CHECK_RESULT IExpression const *do_call_expression(
        IExpression_call const *expr);
    MDL_CHECK_RESULT IExpression const *do_let_expression(
        IExpression_let const *expr);
    MDL_CHECK_RESULT IExpression const *do_expression(
        IExpression const *expr);

    void do_invalid_statement(IStatement_invalid const *stmt);
    void do_compound_statement(IStatement_compound const *stmt);
    void do_declaration_statement(IStatement_declaration const *stmt);
    void do_expression_statement(IStatement_expression const *stmt);
    void do_if_statement(IStatement_if const *stmt);
    void do_case_statement(IStatement_case const *stmt);
    void do_switch_statement(IStatement_switch const *stmt);
    void do_while_statement(IStatement_while const *stmt);
    void do_do_while_statement(IStatement_do_while const *stmt);
    void do_for_statement(IStatement_for const *stmt);
    void do_break_statement(IStatement_break const *stmt);
    void do_continue_statement(IStatement_continue const *stmt);
    void do_return_statement(IStatement_return const *stmt);
    void do_statement(IStatement const *stmt);
};

}  // mdl
}  // mi

#endif
