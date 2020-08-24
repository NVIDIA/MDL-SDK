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
void Code_checker::check_factory(Value_factory const *factory)
{
    if (m_verbose) {
        m_printer->print("Checking value factory:\n");
    }

    Value_factory::const_value_iterator it(factory->values_begin());
    Value_factory::const_value_iterator end(factory->values_end());

    for (; it != end; ++it) {
        IValue const *v = *it;

        check_value(v);
    }
}

// Check a given value for soundness.
void Code_checker::check_value(IValue const *v)
{
    if (v == NULL) {
        report("value is NULL");
        return;
    }

    if (m_verbose) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "0x%p", v);

        m_printer->print("Checking value ");
        m_printer->print(buffer);
        m_printer->print(" ");
        m_printer->print(v);
        m_printer->print("\n");
    }

    IType const *v_type = v->get_type();

    check_type(v_type);

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
void Code_checker::check_type(IType const *type)
{
    if (type == NULL) {
        report("type is NULL");
        return;
    }

    if (m_verbose) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "0x%p", type);

        m_printer->print("Checking type ");
        m_printer->print(buffer);
        m_printer->print(" ");
        m_printer->print(type);
        m_printer->print("\n");
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

            check_type(e_type);

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
    case IType::TK_INCOMPLETE:
        MDL_ASSERT(!"incomplete type occured unexpected");
        break;
    default:
        report("Type has a wrong kind");
        break;
    }
}

// Report an error.
void Code_checker::report(char const *msg)
{
    ++m_error_count;
    m_printer->print("Error check: ");
    m_printer->print(msg);
    m_printer->print("\n");
}

// ------------------------ module checker ------------------------

// Constructor.
Module_checker::Module_checker(
    bool     verbose,
    IPrinter *printer)
: Base(verbose, printer)
{
}

// Check a module.
bool Module_checker::check(MDL const *compiler, Module const *module, bool verbose)
{
    mi::base::Handle<IOutput_stream> os_stderr(compiler->create_std_stream(IMDL::OS_STDERR));
    IPrinter *printer = compiler->create_printer(os_stderr.get());
    printer->enable_color(true);

    Module_checker checker(verbose, printer);

    if (verbose) {
        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print("\n");
    }

    checker.check_factory(module->get_value_factory());

    if (checker.get_error_count() != 0) {
        if (verbose) {
            printer->print("Checking ");
            printer->print(module->get_name());
            printer->print(" FAILED!\n\n");
        }
        return false;
    }

    if (verbose) {
        printer->print("Checking ");
        printer->print(module->get_name());
        printer->print(" OK!\n\n");
    }
    return true;
}

}  // mdl
}  // mi
 