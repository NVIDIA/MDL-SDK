/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_VISITOR_H
#define MDL_COMPILER_HLSL_VISITOR_H 1

namespace mi {
namespace mdl {
namespace hlsl {

class Compilation_unit;

class Declaration;
class Declaration_field;
class Declaration_function;
class Declaration_interface;
class Declaration_invalid;
class Declaration_param;
class Declaration_qualified;
class Declaration_struct;
class Declaration_variable;

class Field_declarator;
class Init_declarator;
class Layout_qualifier_id;

class Name;
class Type_name;
class Type_qualifier;
class Array_specifier;
class Instance_name;

class Expr;
class Expr_binary;
class Expr_call;
class Expr_compound;
class Expr_conditional;
class Expr_literal;
class Expr_ref;
class Expr_unary;
class Expr_invalid;

class Stmt;
class Stmt_break;
class Stmt_case;
class Stmt_compound;
class Stmt_continue;
class Stmt_decl;
class Stmt_discard;
class Stmt_do_while;
class Stmt_expr;
class Stmt_for;
class Stmt_if;
class Stmt_return;
class Stmt_switch;
class Stmt_while;
class Stmt_invalid;

///
/// AST Visitor.
///
/// Supports fine grained and high level processing.
///
class CUnit_visitor {
public:

    /// Visit all AST nodes of a compilation unit.
    ///
    /// \param unit  the compilation unit
    void visit(Compilation_unit *unit);

    /// Visit a statement and all its children.
    ///
    /// \param stmt  the statement
    void visit(Stmt *stmt);

    /// Visit an expression and all its children.
    ///
    /// \param expr  the expression
    void visit(Expr *expr);

    /// Visit a declaration and all its children.
    ///
    /// \param decl  the declaration
    void visit(Declaration *decl);

    /// Visit a type name and all its children.
    ///
    /// \param tname  the type name
    void visit(Type_name *tname);

    /// Visit a name and all its children.
    ///
    /// \param name  the name
    void visit(Name *name);

    /// Visit an array specifier and all its children.
    ///
    /// \param spec  the array specifier
    void visit(Array_specifier *spec);

    /// Default pre visitor for statements.
    ///
    /// \param stmt  the statement
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten statement is needed.
    virtual bool pre_visit(Stmt *stmt);

    /// Default post visitor for statements.
    ///
    /// \param stmt  the statement
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten statement is needed.
    virtual void post_visit(Stmt *stmt);

    /// Default pre visitor for expressions.
    ///
    /// \param expr  the expression
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten expression is needed.
    virtual bool pre_visit(Expr *expr);

    /// Default post visitor for expressions.
    ///
    /// \param expr  the expression
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten expression is needed.
    virtual void post_visit(Expr *expr);

    /// Default pre visitor for declarations.
    ///
    /// \param decl  the declaration
    ///
    /// \return true  if the children should be visited
    ///         false if the children should NOT be visited
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten declaration is needed.
    virtual bool pre_visit(Declaration *decl);

    /// Default post visitor for declarations.
    ///
    /// \param decl  the declaration
    ///
    /// Overwrite this method if some general processing for every
    /// not explicitly overwritten expression is needed.
    virtual void post_visit(Declaration *decl);

    // ----------------------- declarations -----------------------

    virtual bool pre_visit(Name *name);
    virtual void post_visit(Name *name);

    virtual bool pre_visit(Layout_qualifier_id *id);
    virtual void post_visit(Layout_qualifier_id *id);

    virtual bool pre_visit(Type_qualifier *tq);
    virtual void post_visit(Type_qualifier *tq);

    virtual bool pre_visit(Array_specifier *as);
    virtual void post_visit(Array_specifier *as);

    virtual bool pre_visit(Type_name *tname);
    virtual void post_visit(Type_name *tname);

    virtual bool pre_visit(Declaration_invalid *decl);
    virtual void post_visit(Declaration_invalid *decl);

    virtual bool pre_visit(Init_declarator *init);
    virtual void post_visit(Init_declarator *init);

    virtual bool pre_visit(Declaration_variable *decl);
    virtual void post_visit(Declaration_variable *decl);

    virtual bool pre_visit(Declaration_param *decl);
    virtual void post_visit(Declaration_param *decl);

    virtual bool pre_visit(Declaration_function *decl);
    virtual void post_visit(Declaration_function *decl);

    virtual bool pre_visit(Field_declarator *field);
    virtual void post_visit(Field_declarator *field);

    virtual bool pre_visit(Declaration_field *decl);
    virtual void post_visit(Declaration_field *decl);

    virtual bool pre_visit(Declaration_struct *decl);
    virtual void post_visit(Declaration_struct *decl);

    virtual bool pre_visit(Declaration_interface *decl);
    virtual void post_visit(Declaration_interface *decl);

    virtual bool pre_visit(Instance_name *name);
    virtual void post_visit(Instance_name *name);

    virtual bool pre_visit(Declaration_qualified *decl);
    virtual void post_visit(Declaration_qualified *decl);

    // ----------------------- expressions -----------------------

    virtual bool pre_visit(Expr_invalid *expr);
    virtual void post_visit(Expr_invalid *expr);

    virtual bool pre_visit(Expr_literal *expr);
    virtual void post_visit(Expr_literal *expr);

    virtual bool pre_visit(Expr_ref *expr);
    virtual void post_visit(Expr_ref *expr);

    virtual bool pre_visit(Expr_unary *expr);
    virtual void post_visit(Expr_unary *expr);

    virtual bool pre_visit(Expr_binary *expr);
    virtual void post_visit(Expr_binary *expr);

    virtual bool pre_visit(Expr_conditional *expr);
    virtual void post_visit(Expr_conditional *expr);

    virtual bool pre_visit(Expr_call *expr);
    virtual void post_visit(Expr_call *expr);

    virtual bool pre_visit(Expr_compound *expr);
    virtual void post_visit(Expr_compound *expr);

    // ----------------------- statements -----------------------

    virtual bool pre_visit(Stmt_invalid *stmt);
    virtual void post_visit(Stmt_invalid *stmt);

    virtual bool pre_visit(Stmt_decl *stmt);
    virtual void post_visit(Stmt_decl *stmt);

    virtual bool pre_visit(Stmt_compound *stmt);
    virtual void post_visit(Stmt_compound *stmt);

    virtual bool pre_visit(Stmt_expr *stmt);
    virtual void post_visit(Stmt_expr *stmt);

    virtual bool pre_visit(Stmt_if *stmt);
    virtual void post_visit(Stmt_if *stmt);

    virtual bool pre_visit(Stmt_case *stmt);
    virtual void post_visit(Stmt_case *stmt);

    virtual bool pre_visit(Stmt_switch *stmt);
    virtual void post_visit(Stmt_switch *stmt);

    virtual bool pre_visit(Stmt_while *stmt);
    virtual void post_visit(Stmt_while *stmt);

    virtual bool pre_visit(Stmt_do_while *stmt);
    virtual void post_visit(Stmt_do_while *stmt);

    virtual bool pre_visit(Stmt_for *stmt);
    virtual void post_visit(Stmt_for *stmt);

    virtual bool pre_visit(Stmt_break *stmt);
    virtual void post_visit(Stmt_break *stmt);

    virtual bool pre_visit(Stmt_continue *stmt);
    virtual void post_visit(Stmt_continue *stmt);

    virtual bool pre_visit(Stmt_discard *stmt);
    virtual void post_visit(Stmt_discard *stmt);

    virtual bool pre_visit(Stmt_return *stmt);
    virtual void post_visit(Stmt_return *stmt);

protected:
    /// Constructor.
    explicit CUnit_visitor();

private:
    // ----------------------- declarations -----------------------

    void do_declaration(Declaration *decl);
    void do_invalid(Declaration_invalid *decl);
    void do_variable_decl(Declaration_variable *decl);
    void do_init(Init_declarator *init);
    void do_parameter(Declaration_param *decl);
    void do_function_decl(Declaration_function *decl);
    void do_field_decl(Declaration_field *decl);
    void do_declarator(Field_declarator *field);
    void do_type_struct(Declaration_struct *decl);
    void do_interface(Declaration_interface *decl);
    void do_qualified(Declaration_qualified *decl);
    void do_name(Name *name);
    void do_type_name(Type_name *tn);
    void do_type_qualifier(Type_qualifier *tq);
    void do_array_specifier(Array_specifier *spec);
    void do_instance_name(Instance_name *name);

    // ----------------------- expressions -----------------------

    void do_expression(Expr *expr);
    void do_invalid_expression(Expr_invalid *expr);
    void do_literal_expression(Expr_literal *expr);
    void do_reference_expression(Expr_ref *expr);
    void do_unary_expression(Expr_unary *expr);
    void do_binary_expression(Expr_binary *expr);
    void do_conditional_expression(Expr_conditional *expr);
    void do_call_expression(Expr_call *expr);
    void do_compound_expression(Expr_compound *expr);

    // ----------------------- statements -----------------------

    void do_statement(Stmt *stmt);
    void do_invalid_statement(Stmt_invalid *stmt);
    void do_compound_statement(Stmt_compound *stmt);
    void do_declaration_statement(Stmt_decl *stmt);
    void do_expression_statement(Stmt_expr *stmt);
    void do_if_statement(Stmt_if *stmt);
    void do_case_statement(Stmt_case *stmt);
    void do_switch_statement(Stmt_switch *stmt);
    void do_while_statement(Stmt_while *stmt);
    void do_do_while_statement(Stmt_do_while *stmt);
    void do_for_statement(Stmt_for *stmt);
    void do_break_statement(Stmt_break *stmt);
    void do_continue_statement(Stmt_continue *stmt);
    void do_discard_statement(Stmt_discard *stmt);
    void do_return_statement(Stmt_return *stmt);
};

}  // hlsl
}  // mdl
}  // mi

#endif
