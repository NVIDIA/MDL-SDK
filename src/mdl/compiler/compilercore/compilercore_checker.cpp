/******************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_checker.h"
#include "compilercore_factories.h"
#include "compilercore_mdl.h"
#include "compilercore_modules.h"
#include "compilercore_tools.h"

#include <cstdio>

namespace mi {
namespace mdl {

// Constructor.
Code_checker::Code_checker(
    bool     verbose,
    IPrinter *printer)
: m_verbose(verbose)
, m_error_count(0)
, m_printer(printer)
{
}

// Check a value factory for soundness.
void Code_checker::check_value_factory(
    Value_factory const *factory)
{
    if (m_verbose) {
        if (m_printer.is_valid_interface()) {
            m_printer->print("Checking value factory:\n");
        }
    }

    Value_factory::const_value_iterator it(factory->values_begin());
    Value_factory::const_value_iterator end(factory->values_end());

    for (; it != end; ++it) {
        IValue const *v = *it;

        check_value(factory, v);
    }
}

// Check a given value for soundness.
void Code_checker::check_value(
    Value_factory const *owner,
    IValue const        *v)
{
    if (v == NULL) {
        report("value is NULL");
        return;
    }

    if (m_verbose) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "0x%p", v);

        if (m_printer.is_valid_interface()) {
            m_printer->print("Checking value ");
            m_printer->print(buffer);
            m_printer->print(" ");
            m_printer->print(v);
            m_printer->print("\n");
        }
    }

    if (!owner->is_owner(v)) {
        report("value is not owned by given factory");
    }

    IType const *v_type = v->get_type();

    check_type(const_cast<Value_factory *>(owner)->get_type_factory(), v_type);

    switch (v->get_kind()) {
        case IValue::VK_BAD:
            if (!is<IType_error>(v_type)) {
                report("Type of an IValue_bad is not error type");
            }
            break;
        case IValue::VK_BOOL:
            if (!is<IType_bool>(v_type)) {
                report("Type of an IValue_bool is not the bool type");
            }
            break;
        case IValue::VK_INT:
            if (!is<IType_int>(v_type)) {
                report("Type of an IValue_int is not the int type");
            }
            break;
        case IValue::VK_ENUM:
            if (!is<IType_enum>(v_type)) {
                report("Type of an IValue_enum is not a enum type");
            }
            break;
        case IValue::VK_FLOAT:
            if (!is<IType_float>(v_type)) {
                report("Type of an IValue_float is not the float type");
            }
            break;
        case IValue::VK_DOUBLE:
            if (!is<IType_double>(v_type)) {
                report("Type of an IValue_double is not the double type");
            }
            break;
        case IValue::VK_STRING:
            if (!is<IType_string>(v_type)) {
                report("Type of an IValue_string is not the string type");
            }
            break;
        case IValue::VK_VECTOR:
            if (!is<IType_vector>(v_type)) {
                report("Type of an IValue_vector is not a vector type");
            }
            break;
        case IValue::VK_MATRIX:
            if (!is<IType_matrix>(v_type)) {
                report("Type of an IValue_matrix is not a matrix type");
            }
            break;
        case IValue::VK_ARRAY:
            if (!is<IType_array>(v_type)) {
                report("Type of an IValue_array is not an array type");
            }
            break;
        case IValue::VK_RGB_COLOR:
            if (!is<IType_color>(v_type)) {
                report("Type of an IValue_rgb_color is not the color type");
            }
            break;
        case IValue::VK_STRUCT:
            if (!is<IType_struct>(v_type)) {
                report("Type of an IValue_struct is not a struct type");
            }
            break;
        case IValue::VK_INVALID_REF:
        case IValue::VK_TEXTURE:
        case IValue::VK_LIGHT_PROFILE:
        case IValue::VK_BSDF_MEASUREMENT:
            break;
        default:
            report("IValue has a wrong kind");
    }
}

// Check a given type for soundness.
void Code_checker::check_type(
    Type_factory const *owner,
    IType const        *type)
{
    if (type == NULL) {
        report("type is NULL");
        return;
    }

    if (m_verbose) {
        if (m_printer.is_valid_interface()) {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "0x%p", type);

            m_printer->print("Checking type ");
            m_printer->print(buffer);
            m_printer->print(" ");
            m_printer->print(type);
            m_printer->print("\n");
        }
    }

    if (!owner->is_owner(type)) {
        report("type is not owned by given factory");
    }

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
        break;
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const *e_type = a_type->get_element_type();

            check_type(owner, e_type);

            if (!a_type->is_immediate_sized()) {
                IType_array_size const *s = a_type->get_deferred_size();
                if (s == NULL) {
                    report("deferred array size of an deferred sized array is NULL");
                }
            }
        }
        break;
    case IType::TK_COLOR:
    case IType::TK_FUNCTION:
    case IType::TK_STRUCT:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_ERROR:
        break;
    case IType::TK_PTR:
        MDL_ASSERT(!"pointer type occured unexpected");
        break;
    case IType::TK_REF:
        MDL_ASSERT(!"reference type occured unexpected");
        break;
    case IType::TK_VOID:
        MDL_ASSERT(!"void type occured unexpected");
        break;
    case IType::TK_AUTO:
        MDL_ASSERT(!"auto type occured unexpected");
        break;
    default:
        report("Type has a wrong kind");
        break;
    }
}

// Check a given AST expression for soundness.
void Code_checker::check_expr(
    IModule const     *owner,
    IExpression const *expr)
{
    // Visitor for expressions.
    class Expr_checker : public Module_visitor {
    public:
        /// Constructor.
        Expr_checker(Code_checker &cc, Module const *m)
        : m_cc(cc)
        , m_module(m)
        , m_tf(m->get_type_factory())
        , m_vf(m->get_value_factory())
        {
        }

        /// Default post visitor for expressions.
        ///
        /// \param expr  the expression
        IExpression *post_visit(IExpression *expr) MDL_FINAL
        {
            IType const *type = expr->get_type();

            // check only the type in generic cases
            m_cc.check_type(m_tf, type);

            return expr;
        }

        // Post visitor for literal expressions.
        ///
        /// \param expr  the expression
        IExpression *post_visit(IExpression_literal *expr) MDL_FINAL
        {
            IValue const *value = expr->get_value();

            m_cc.check_value(m_vf, value);

            return expr;
        }

        // Post visitor for reference expressions.
        ///
        /// \param expr  the expression
        IExpression *post_visit(IExpression_reference *expr) MDL_FINAL
        {
            IType const *type = expr->get_type();

            m_cc.check_type(m_tf, type);

            if (expr->is_array_constructor()) {
                // no definition
                return expr;
            }

            IDefinition const *def = expr->get_definition();
            m_cc.check_definition(m_module, def);

            return expr;
        }

    private:
        Code_checker &m_cc;
        Module const *m_module;
        Type_factory *m_tf;
        Value_factory *m_vf;
    };

    Expr_checker ec(*this, impl_cast<Module>(owner));

    ec.visit(expr);
}

// Check the parameter initializer of a definition for soundness.
void Code_checker::check_parameter_initializers(
    IModule const      *owner,
    IDefinition const  *def)
{
    if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
        // access of parameter initializers on imported entities is forbidden
        return;
    }

    IType_function const *f_tp = as<IType_function>(def->get_type());

    if (f_tp == NULL) {
        return;
    }

    if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
        is<IType_texture>(f_tp->get_return_type()))
    {
        // FIXME: this is an ugly work-around for texture constructors. The 2D
        // constructor references tex::gamma_mode but are built-in (which in theory creates a
        // dependency circle). One could call this a "design flaw".
        // Anyway the constructor references the definition from the tex module
        // instead. But, because they are built-in, the auto import fixer does not "reach"
        // them for rewrite. So far, we just ignore the error here, nothing bad can happen, as
        // the gamma_mode definitions are always in RAM being stdlib.
        return;
    }

    for (int i = 0, n = f_tp->get_parameter_count(); i < n; ++i) {
        if (IExpression const *init = def->get_default_param_initializer(i)) {
            check_expr(owner, init);
        }
    }
}

// Check a given definition for soundness.
void Code_checker::check_definition(
    IModule const     *owner,
    IDefinition const *def)
{
    Module const *module = impl_cast<Module>(owner);

    if (def == NULL) {
        report("def is NULL");
        return;
    }

    if (m_verbose) {
        if (m_printer.is_valid_interface()) {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "0x%p", def);

            m_printer->print("Checking definition ");
            m_printer->print(buffer);
            m_printer->print(" ");
            m_printer->print(def);
            m_printer->print("\n");
        }
    }

    if (!module->is_owner(def)) {
        report("Def is not owned by given module");
    }

    check_type(module->get_type_factory(), def->get_type());

    switch (def->get_kind()) {
    case IDefinition::DK_ERROR:
        break;
    case IDefinition::DK_CONSTANT:
    case IDefinition::DK_ENUM_VALUE:
        check_value(module->get_value_factory(), def->get_constant_value());
        break;
    case IDefinition::DK_ANNOTATION:
    case IDefinition::DK_FUNCTION:
    case IDefinition::DK_CONSTRUCTOR:
        check_parameter_initializers(owner, def);
        break;
    case IDefinition::DK_STRUCT_CATEGORY:
        break;
    case IDefinition::DK_TYPE:
        break;
    case IDefinition::DK_VARIABLE:
        break;
    case IDefinition::DK_MEMBER:
    case IDefinition::DK_PARAMETER:
    case IDefinition::DK_ARRAY_SIZE:
    case IDefinition::DK_OPERATOR:
    case IDefinition::DK_NAMESPACE:
        break;
    }
}

// Report an error.
void Code_checker::report(char const *msg)
{
    ++m_error_count;
    if (m_printer.is_valid_interface()) {
        m_printer->print("Error check: ");
        m_printer->print(msg);
        m_printer->print("\n");
    }
}

// ------------------------ module checker ------------------------

// Constructor.
Module_checker::Module_checker(
    Module const *module,
    bool         verbose,
    IPrinter     *printer)
: Base(verbose, printer)
, m_module(module)
{
}

// Check a module for soundness.
void Module_checker::check_module(
    Module const *module)
{
    visit(module);
}

// Default post visitor for expressions.
IExpression *Module_checker::post_visit(
    IExpression *expr)
{
    check_expr(m_module, expr);
    return expr;
}

// Check a module.
bool Module_checker::check(
    IMDL const    *imdl,
    IModule const *imodule,
    bool          verbose)
{
    MDL const    *compiler = impl_cast<MDL>(imdl);
    Module const *module   = impl_cast<Module>(imodule);

    if (verbose && compiler != NULL) {
        mi::base::Handle<IOutput_stream> os_stderr(compiler->create_std_stream(IMDL::OS_STDERR));
        IPrinter *printer = compiler->create_printer(os_stderr.get());
        printer->enable_color(true);

        Module_checker checker(module, verbose, printer);

        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print("\n");

        checker.check_value_factory(module->get_value_factory());

        checker.check_module(module);

        if (checker.get_error_count() != 0) {
            printer->print("Checking ");
            printer->print(module->get_name());
            printer->print(" FAILED!\n\n");
            return false;
        }

        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print(" OK!\n\n");
        return true;
    } else {
        Module_checker checker(module, /*verbose=*/false, /*printer=*/NULL);
        checker.check_value_factory(module->get_value_factory());
        checker.check_module(module);
        return checker.get_error_count() == 0;
    }
}

// Checker.
bool Tree_checker::check(
    IMDL const    *imdl,
    IModule const *imodule,
    bool          verbose)
{
    MDL const    *compiler = impl_cast<MDL>(imdl);
    Module const *module   = impl_cast<Module>(imodule);

    if (verbose && compiler != NULL) {
        mi::base::Handle<IOutput_stream> os_stderr(compiler->create_std_stream(IMDL::OS_STDERR));
        IPrinter *printer = compiler->create_printer(os_stderr.get());
        printer->enable_color(true);

        Tree_checker checker(compiler->get_allocator(), printer, verbose);

        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print(" for TREE property\n");

        checker.visit(module);

        if (checker.get_error_count() != 0) {
            printer->print("Checking ");
            printer->print(module->get_name());
            printer->print(" FAILED!\n\n");
            return false;
        }

        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print(" OK!\n\n");
        return true;
    } else {
        Tree_checker checker(module->get_allocator(), NULL, false);
        checker.visit(module);

        return checker.get_error_count() == 0;
    }
}

// Constructor.
Tree_checker::Tree_checker(
    IAllocator *alloc,
    IPrinter   *printer,
    bool       verbose)
: Code_checker(verbose, printer)
, m_ast_set(0, Ptr_set::hasher(), Ptr_set::key_equal(), alloc)
{
}

void Tree_checker::post_visit(ISimple_name *sname)
{
    if (!m_ast_set.insert(sname).second) {
        // found a reuse, this IS BAD
        report("AST contains a simple name reuse");
    }
}

void Tree_checker::post_visit(IQualified_name *qname)
{
    if (!m_ast_set.insert(qname).second) {
        // found a reuse, this IS BAD
        report("AST contains a qualified name reuse");
    }
}

void Tree_checker::post_visit(IType_name *tname)
{
    if (!m_ast_set.insert(tname).second) {
        // found a reuse, this IS BAD
        report("AST contains a type name reuse");
    }
}

IExpression *Tree_checker::post_visit(IExpression *expr)
{
    if (!m_ast_set.insert(expr).second) {
        // found a reuse, this IS BAD
        report("AST contains an expression reuse");
    }
    return expr;
}

void Tree_checker::post_visit(IStatement *stmt)
{
    if (!m_ast_set.insert(stmt).second) {
        // found a reuse, this IS BAD
        report("AST contains a statement reuse");
    }
}

void Tree_checker::post_visit(IDeclaration *decl)
{
    if (!m_ast_set.insert(decl).second) {
        // found a reuse, this IS BAD
        report("AST contains a declaration reuse");
    }
}

}  // mdl
}  // mi
 
