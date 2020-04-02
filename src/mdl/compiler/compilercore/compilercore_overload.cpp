/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_overload.h"
#include "compilercore_def_table.h"
#include "compilercore_factories.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

// Constructor.
Overload_solver::Overload_solver(Module &module)
: m_module(module)
, m_tf(*module.get_type_factory())
, m_def_tab(module.get_definition_table())
, m_type_bindings(0, Bind_type_map::hasher(), Bind_type_map::key_equal(), get_allocator())
, m_size_bindings(0, Bind_size_map::hasher(), Bind_size_map::key_equal(), get_allocator())
, m_sym_bindings(0, Bind_symbol_map::hasher(), Bind_symbol_map::key_equal(), get_allocator())
{

}

// Retrieve the used allocator.
IAllocator *Overload_solver::get_allocator() const { return m_module.get_allocator(); }

// Checks if types are equal.
bool Overload_solver::equal_types(IType const *a, IType const *b)
{
    if (a == b)
        return true;
    if (a->skip_type_alias() != b->skip_type_alias()) {
        return false;
    }
    return a->get_type_modifiers() == b->get_type_modifiers();
}

// Compare two signature entries representing functions for "specific-ness".
bool Overload_solver::is_more_specific(Signature_entry const &a, Signature_entry const &b)
{
    size_t n_params = a.sig_length;

    MDL_ASSERT(n_params == b.sig_length);

    for (size_t i = 0; i < n_params; ++i) {
        IType const *param_a = a.signature[i];
        IType const *param_b = b.signature[i];

        param_a = param_a->skip_type_alias();
        param_b = param_b->skip_type_alias();

        if (equal_types(param_a, param_b)) {
            if (is<IType_array>(param_a) && is<IType_array>(param_b)) {
                if (a.bounds != NULL && b.bounds != NULL) {
                    if (a.bounds[i] && !b.bounds[i])
                        return false;
                }
            }
            continue;
        }

        if (is<IType_enum>(param_b)) {
            switch (param_a->get_kind()) {
            case IType::TK_ENUM:
                // no enum is more specific
            case IType::TK_INT:
                // enum -> int
            case IType::TK_FLOAT:
                // enum -> float
            case IType::TK_DOUBLE:
                // enum -> double
                return false;
            default:
                break;
            }
        }

        if (find_implicit_conversion(param_b, param_a) != NULL)
            return false;
    }
    return true;
}

// Given a list and a (call) signature, kill any less specific definition from the list.
bool Overload_solver::kill_less_specific(Signature_list &list, Signature_entry const &new_sig)
{
    for (Signature_list::iterator it(list.begin()), end(list.end()); it != end; ++it) {
        Signature_entry const &curr_sig = *it;

        bool curr_is_more = is_more_specific(curr_sig, new_sig);
        bool new_is_more  = is_more_specific(new_sig, curr_sig);

        if (curr_is_more && !new_is_more) {
            // current def is more specific than new def, no need for new def
            return false;
        }
        if (!curr_is_more && new_is_more) {
            // current def is less specific the new def, kill it
            list.erase(it);
            return true;
        }

        // FIXME: is the following still true?
        // Note: due to the policy that import "redefinitions" are only reported
        // at use, it CAN happen that we find two definitions that are really the same,
        // in which case (curr_is_more && new_is_more) == true
        // Return true in THAT case, so both are added and an overload error is reported.
        if (curr_is_more && new_is_more)
            return true;
    }
    return true;
}

// Check if a parameter type is already bound.
bool Overload_solver::is_bound_type(IType_array const *abs_type) const
{
    Bind_type_map::const_iterator it = m_type_bindings.find(abs_type);
    return it != m_type_bindings.end();
}

// Bind the given deferred type to a type.
void Overload_solver::bind_array_type(IType_array const *abs_type, IType_array const *type)
{
    MDL_ASSERT(!abs_type->is_immediate_sized() && "Wrong type binding");

    IType_array_size const *abs_size = abs_type->get_deferred_size();
    if (type->is_immediate_sized()) {
        m_size_bindings[abs_size] = type->get_size();
    } else {
        m_sym_bindings[abs_size] = type->get_deferred_size();
    }

    // bind the size NOT the element type
    IType const *e_type = abs_type->get_element_type();
    IType const *n_type;
    if (type->is_immediate_sized()) {
        n_type = m_tf.create_array(e_type, type->get_size());
    } else {
        n_type = m_tf.create_array(e_type, type->get_deferred_size());
    }
    m_type_bindings[abs_type] = n_type;
}

// Return the bound type for a deferred type.
IType const *Overload_solver::get_bound_type(IType const *type)
{
    Bind_type_map::const_iterator it = m_type_bindings.find(type);
    if (it != m_type_bindings.end())
        return it->second;

    if (IType_array const *a_type = as<IType_array>(type)) {
        // check if the size is bound
        if (!a_type->is_immediate_sized()) {
            IType_array_size const *abs_size = a_type->get_deferred_size();

            Bind_size_map::const_iterator sit = m_size_bindings.find(abs_size);
            if (sit != m_size_bindings.end()) {
                int size = sit->second;

                IType const *e_type = a_type->get_element_type();

                IType const *r_type = m_tf.create_array(e_type, size);

                m_type_bindings[type] = r_type;
                return r_type;
            }

            Bind_symbol_map::const_iterator ait = m_sym_bindings.find(abs_size);
            if (ait != m_sym_bindings.end()) {
                IType_array_size const *size = ait->second;

                IType const *e_type = a_type->get_element_type();

                IType const *r_type =  m_tf.create_array(e_type, size);

                m_type_bindings[type] = r_type;
                return r_type;
            }
        }
    }
    return type;
}

// Clear all bindings of deferred sized array types.
void Overload_solver::clear_type_bindings()
{
    m_type_bindings.clear();
    m_size_bindings.clear();
    m_sym_bindings.clear();
}

// Check if it is possible to assign an argument type to the parameter
// type of a call.
bool Overload_solver::can_assign_param(
    IType const *param_type,
    IType const *arg_type,
    bool        &new_bound)
{
    new_bound = false;
    if (param_type == arg_type)
        return true;

    if (is<IType_error>(param_type)) {
        // this should only happen in materials, where overload is forbidden, so
        // we can create definitions with error parameters to improve error checking
        return true;
    }

    if (as<IType_enum>(arg_type) != NULL) {
        // special case for enums: an enum can be assigned to int, float, double
        IType const *base = param_type->skip_type_alias();
        IType::Kind kind = base->get_kind();

        bool res = (kind == IType::TK_INT || kind == IType::TK_FLOAT || kind == IType::TK_DOUBLE);
        if (res)
            return true;
        return false;
    }

    if (IType_array const *a_param_type = as<IType_array>(param_type)) {
        if (IType_array const *a_arg_type = as<IType_array>(arg_type)) {
            if (!a_param_type->is_immediate_sized()) {
                // the parameter type is abstract, check for bindings
                a_param_type = cast<IType_array>(get_bound_type(a_param_type));
            }

            if (a_param_type->is_immediate_sized()) {
                // concrete parameter type, size must match
                if (!a_arg_type->is_immediate_sized())
                    return false;
                if (a_param_type->get_size() != a_arg_type->get_size())
                    return false;
                return equal_types(
                    a_param_type->get_element_type()->skip_type_alias(),
                    a_arg_type->get_element_type()->skip_type_alias());
            } else {
                // param type is an deferred size array
                if (a_arg_type->is_immediate_sized()) {
                    // can pass a immediate size array to a deferred size parameter, but this will
                    // bind the parameter type
                    bool res = equal_types(
                        a_param_type->get_element_type()->skip_type_alias(),
                        a_arg_type->get_element_type()->skip_type_alias());
                    if (res) {
                        new_bound = true;
                        bind_array_type(a_param_type, a_arg_type);
                    }
                    return res;
                } else {
                    if (is_bound_type(a_param_type)) {
                        // must use the same deferred size
                        if (a_param_type->get_deferred_size() != a_arg_type->get_deferred_size())
                            return false;
                        return equal_types(
                            a_param_type->get_element_type()->skip_type_alias(),
                            a_arg_type->get_element_type()->skip_type_alias());
                    } else {
                        // can pass a deferred size array to a deferred size parameter, but
                        // this will bind the parameter type
                        bool res = equal_types(
                            a_param_type->get_element_type()->skip_type_alias(),
                            a_arg_type->get_element_type()->skip_type_alias());
                        if (res) {
                            new_bound = true;
                            bind_array_type(a_param_type, a_arg_type);
                        }
                        return res;
                    }
                }
            }
        }
    }

    return find_implicit_conversion(arg_type, param_type) != NULL;
}

// Returns a short signature from a (function) definition.
string Overload_solver::get_short_signature(Definition const *def) const
{
    char const *s = "";
    switch (def->get_kind()) {
    case Definition::DK_CONSTRUCTOR:
        s = "constructor ";
        break;
    case Definition::DK_ANNOTATION:
        s = "annotation ";
        break;
    default:
        break;
    }
    string func_name(s, m_module.get_allocator());
    func_name += def->get_sym()->get_name();
    return func_name;
}

// Find the Definition of an implicit conversion constructor or an conversion
// operator.
Definition *Overload_solver::find_implicit_conversion(
    IType const    *from_tp,
    IType const    *to_tp)
{
    MDL_ASSERT(!is<IType_error>(from_tp) && !is<IType_error>(to_tp));

    IType const *dt = to_tp->skip_type_alias();
    if (is<IType_array>(dt) || is<IType_function>(dt)) {
        // cannot convert to array or function
        return NULL;
    }

    IType const *st = from_tp->skip_type_alias();

    // in the MDL compiler all implicit conversions are implemented "the OO-way", i.e.
    // to convert from st to dt, there must either exists an implicit dt(st) constructor
    // or an operator st(dt).
    Scope const *src_scope = m_def_tab.get_type_scope(st);
    if (src_scope == NULL)
        return NULL;

    Scope const *dst_scope = m_def_tab.get_type_scope(dt);
    if (dst_scope == NULL) {
        // If this happens, something really bad goes from, for instance the type of an
        // argument is a function type
        MDL_ASSERT(!"find_implicit_conversion(): destination type has no scope");
        return NULL;
    }

    ISymbol const *construct_sym = dst_scope->get_scope_name();

    Definition *def = NULL;

    // search first for operator TYPE()
    Definition *set = src_scope->find_definition_in_scope(construct_sym);
    for (def = set; def != NULL; def = def->get_prev_def()) {
        if (def->get_semantics() != Definition::DS_CONV_OPERATOR) {
            // search conversion operator
            continue;
        }
        IType_function const *f_type = cast<IType_function>(def->get_type());

        MDL_ASSERT(f_type->get_parameter_count() == 1);

        IType const *ret_type = f_type->get_return_type();

        if (ret_type->skip_type_alias() == dt) {
            // found it
            return def;
        }
    }

    // not found, search for an implicit constructor TYPE()
    set = dst_scope->find_definition_in_scope(construct_sym);
    for (def = set; def != NULL; def = def->get_prev_def()) {
        if (def->has_flag(Definition::DEF_IS_EXPLICIT)) {
            // we are searching for implicit constructors only
            continue;
        }
        if (def->get_kind() != Definition::DK_CONSTRUCTOR) {
            // ignore operators/members ...
            continue;
        }
        IType_function const *f_type = cast<IType_function>(def->get_type());

        if (f_type->get_parameter_count() != 1) {
            // not exactly one argument: not a conversion
            continue;
        }
        IType const   *p_type;
        ISymbol const *p_sym;
        f_type->get_parameter(0, p_type, p_sym);

        if (p_type->skip_type_alias() == st) {
            // found it
            return def;
        }
    }
    // not found
    return NULL;
}

#ifdef ENABLE_ASSERT
/// Returns true for error definitions.
///
/// \param def  the definition to check
static bool is_error(IDefinition const *def)
{
    IType const *type = def->get_type();
    if (is<IType_error>(type))
        return true;
    return def->get_symbol()->get_id() == ISymbol::SYM_ERROR;
}
#endif

// Get the default expression of a parameter of a function, constructor or annotation.
IExpression const *Overload_solver::get_default_param_initializer(
    Definition const *def,
    int              param_idx) const
{
    Definition const *orig = m_module.get_original_definition(def);
    return orig->get_default_param_initializer(param_idx);
}

// Find function overloads.
Overload_solver::Definition_list Overload_solver::find_positional_overload(
    Definition const *def,
    IType const *arg_types[],
    size_t num_args)
{
    MDL_ASSERT(!is_error(def));
    IAllocator *alloc = get_allocator();

    // collect the possible set
    typedef list<Definition const *>::Type Definition_list;

    Definition_list def_list(alloc);

    size_t cnt = 0;
    for (Definition const *curr_def = def;
        curr_def != NULL;
        curr_def = curr_def->get_prev_def())
    {
        if (!curr_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)) {
            def_list.push_back(curr_def);
            ++cnt;
        }
    }

    bool is_overloaded = cnt > 1;

    Memory_arena arena(alloc);

    Signature_list best_matches(&arena);

    int best_value = 0;

    VLA<IType const *> signature(m_module.get_allocator(), num_args);
    VLA<bool>          bounds(m_module.get_allocator(), num_args);

    // so far we do not support named parameters
    size_t num_pos_args = num_args;
    size_t min_params   = num_pos_args;

    for (Definition_list::iterator it(def_list.begin()), end(def_list.end());
        it != end;
        ++it)
    {
        Definition const     *candidate = *it;
        IType const          *type = candidate->get_type();

        // TODO: there should be no errors at this place
        if (is<IType_error>(type))
            continue;

        IType_function const *func_type = cast<IType_function>(type);
        size_t num_params = func_type->get_parameter_count();

        if (is_overloaded && num_params < min_params)
            continue;

        int this_value = 0;
        bool type_must_match = candidate->has_flag(Definition::DEF_OP_LVALUE);
        for (size_t k = 0; k < num_pos_args && k < num_params; ++k, type_must_match = false) {
            IType const   *param_type;
            ISymbol const *name;

            IType const *arg_type = arg_types[k];
            func_type->get_parameter(k, param_type, name);

            param_type = param_type->skip_type_alias();

            bool new_bound = false;
            if (equal_types(param_type, arg_type)) {
                // types match
                this_value += 2;
            } else if (!type_must_match && can_assign_param(param_type, arg_type, new_bound)) {
                // can be implicitly converted
                this_value += 1;
            } else {
                this_value = -1;
                break;
            }
            signature[k] = get_bound_type(param_type);
            bounds[k]    = new_bound;
        }
        if (!is_overloaded && num_params < num_pos_args) {
            this_value = -1;
        }

        if (this_value >= 0) {
            // check default parameters
            for (size_t k = num_pos_args; k < num_params; ++k) {
                IType const   *param_type;
                ISymbol const *name;

                func_type->get_parameter(k, param_type, name);

                IType const    *arg_type      = NULL;
                size_t         idx            = k;
                bool           is_default_arg = false;

                // must have a default value
                IExpression const *def_expr = get_default_param_initializer(candidate, k);
                if (def_expr == NULL) {
                    this_value = -1;
                    break;
                } else {
                    // remember this default parameter
                    is_default_arg = true;

                    // beware: the default parameter might be owned by another module,
                    // import its type
                    arg_type = def_expr->get_type();
                    arg_type = m_module.import_type(arg_type->skip_type_alias());
                }

                // check types and calculate type binding
                param_type = param_type->skip_type_alias();

                bool new_bound = false;
                if (equal_types(param_type, arg_type)) {
                    // types match
                    if (!is_default_arg)
                        this_value += 2;
                } else if (can_assign_param(param_type, arg_type, new_bound)) {
                    // direct assignment without conversion
                    if (!is_default_arg)
                        this_value += 1;
                } else {
                    this_value = -1;
                    break;
                }
                if (!is_default_arg) {
                    signature[idx] = get_bound_type(param_type);
                    bounds[idx]    = new_bound;
                }
            }
        }

        clear_type_bindings();

        if (this_value > best_value) {
            // higher values mean "more specific"
            best_matches.clear();
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            best_matches.push_back(entry);
            best_value = this_value;
        } else if (this_value == best_value) {
            Signature_entry entry(
                (IType const *const *)Arena_memdup(
                arena, signature.data(), signature.size() * sizeof(signature[0])),
                (bool const *const)Arena_memdup(
                arena, bounds.data(), bounds.size() * sizeof(bounds[0])),
                signature.size(),
                candidate);
            if (kill_less_specific(best_matches, entry)) {
                best_matches.push_back(entry);
            }
        }
    }

    Definition_list res(get_allocator());

    for (Signature_list::iterator it(best_matches.begin()), end(best_matches.end());
        it != end;
        ++it)
    {
        Signature_entry const &entry = *it;
        res.push_back(entry.def);
    }

    return res;
}


}  // mdl
}  // mi
