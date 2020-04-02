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

#include "pch.h"

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/base/iallocator.h>

#include <vector>

#include "compilercore_cc_conf.h"
#include "compilercore_factories.h"
#include "compilercore_symbols.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_memory_arena.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

/// A mixin for all base type methods.
template <typename Interface>
class Type_base : public Interface
{
    typedef Interface Base;
public:

    /// Get the kind of type.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Get the type modifiers of a type
    IType::Modifiers get_type_modifiers() const MDL_OVERRIDE {
        IType::Modifiers mod = IType::MK_NONE;
        IType const *tp = this;
        while (const IType_alias *alias = as<IType_alias>(tp)) {
            mod |= alias->get_type_modifiers();
            tp = alias;
        }
        return mod;
    }

    /// Access to the base type.
    IType *skip_type_alias() MDL_FINAL {
        IType const *tp = this;
        return const_cast<IType *>(tp->skip_type_alias());
    }

    /// Access to the base type.
    IType const *skip_type_alias() const MDL_FINAL {
        IType const *tp = this;
        while (is<IType_alias>(tp)) {
            tp = static_cast<IType_alias const *>(tp)->get_aliased_type();
        }
        return tp;
    }

protected:
    /// Constructor.
    explicit Type_base()
    : Base()
    {
    }

private:
    // non copyable
    Type_base(Type_base const &) MDL_DELETED_FUNCTION;
    Type_base &operator=(Type_base const &) MDL_DELETED_FUNCTION;
};

/// An expression base mixin for types with variadic number of arguments.
template <typename Interface, typename ArgIf>
class Type_base_variadic : public Type_base<Interface>
{
    typedef Type_base<Interface> Base;
public:

protected:
    /// Constructor.
    ///
    /// \param arena  the memory arena variadic parts are allocated on
    explicit Type_base_variadic(Memory_arena *arena)
    : Base()
    , m_args(arena)
    {
    }

    /// Return the number of variadic arguments.
    size_t argument_count() const { return m_args.size(); }

    /// Add a new argument.
    size_t add_argument(ArgIf arg) {
        size_t res = m_args.size();
        m_args.push_back(arg);
        return res;
    }

    /// Get the argument at given position.
    ArgIf argument_at(size_t pos) const { return m_args.at(pos); }

protected:
    typename Arena_vector<ArgIf>::Type m_args;
};

/// Implementation of the type of kind alias.
/// Note that types with modifiers are represented solely using alias types,
/// so a const T is an alias of the type T.
class Type_alias : public Type_base<IType_alias>
{
    typedef Type_base<IType_alias> Base;
    friend class Arena_builder;
public:

    /// Get the name of the type.
    ISymbol const *get_symbol() const MDL_FINAL { return m_name; }

    /// Get the type aliased by this type.
    IType const *get_aliased_type() const MDL_FINAL { return m_aliased_type; }

    /// Get the modifier set of this type.
    IType::Modifiers get_type_modifiers() const MDL_FINAL { return m_modifiers; }

    // ---------------------- non-interface ----------------------

    /// Get the owner id.
    size_t get_owner_id() const { return m_owner_id; }

private:
    /// Constructor
    ///
    /// \param owner_id      the id of the type factory owning this type
    /// \param aliased_type  the type that is aliased
    /// \param name          the fully qualified name of this type or NULL
    /// \param modifiers     modifiers of this alias type
    explicit Type_alias(
        size_t           owner_id,
        IType const      *aliased_type,
        ISymbol const    *name,
        IType::Modifiers modifiers)
    : Base()
    , m_owner_id(owner_id)
    , m_aliased_type(aliased_type)
    , m_name(name)
    , m_modifiers(modifiers)
    {
        MDL_ASSERT(Type_factory::is_owned(m_owner_id, aliased_type));
    }

private:
    /// An id representing the owner of this type (for debugging).
    size_t const m_owner_id;

    /// The aliased type.
    IType const *m_aliased_type;

    /// The alias name of the type if any.
    ISymbol const *m_name;

    /// The modifiers of this type.
    IType::Modifiers m_modifiers;
};

/// Implementation of the error type.
/// The error type represents a type error in the syntax tree.
class Type_error : public Type_base<IType_error>
{
    typedef Type_base<IType_error> Base;
public:
    /// Constructor.
    explicit Type_error()
    : Base()
    {
    }
};

/// Implementation of the incomplete type.
class Type_incomplete : public Type_base<IType_incomplete>
{
    typedef Type_base<IType_incomplete> Base;
public:
    /// Constructor.
    explicit Type_incomplete()
    : Base()
    {
    }
};

/// Implementation of the bool type.
class Type_bool : public Type_base<IType_bool>
{
    typedef Type_base<IType_bool> Base;
public:
    /// Constructor.
    explicit Type_bool()
    : Base()
    {
    }
};

/// Implementation of the int type.
class Type_int : public Type_base<IType_int>
{
    typedef Type_base<IType_int> Base;
public:
    /// Constructor.
    explicit Type_int()
    : Base()
    {
    }
};

/// A member of a enum type.
class Enum_member
{
public:
    /// Return the name of this member.
    ISymbol const *get_symbol() const { return m_name; }

    /// Return the code of this member.
    int get_code() const { return m_code; }

    /// Constructor.
    ///
    /// \param name  the name of this enum member
    /// \param code  the code if this enum member
    explicit Enum_member(ISymbol const *name, int code)
    : m_name(name)
    , m_code(code)
    {
    }

private:
    /// The name of this member.
    ISymbol const *m_name;

    /// The code of this member.
    int m_code;
};

/// Implementation of the enum type.
class Type_enum : public Type_base_variadic<IType_enum, Enum_member>
{
    typedef Type_base_variadic<IType_enum, Enum_member> Base;
    friend class Arena_builder;
public:
    /// Get the name of the enum type.
    ISymbol const *get_symbol() const MDL_FINAL { return m_name; }

    /// Add a value.
    ///
    /// \param name The name of the value.
    /// \param code The code of the value.
    ///
    /// \return The index of this new value.
    int add_value(ISymbol const *name, int code) MDL_FINAL {
        return Base::add_argument(Enum_member(name, code));
    }

    /// Get the number of values.
    int get_value_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get a value.
    ///
    /// \param index    The index of the value.
    /// \param name     The name of the value.
    /// \param code     The code of the value.
    ///
    /// \return  true of success, false on index error.
    bool get_value(int index, ISymbol const *&symbol, int &code) const MDL_FINAL {
        if (0 <= index && size_t(index) < Base::argument_count()) {
            Enum_member const &member = Base::argument_at(index);
            symbol = member.get_symbol();
            code = member.get_code();
            return true;
        }
        return false;
    }

    /// Lookup a value in O(N).
    ///
    /// \param name     The name of the value.
    /// \param code     The code of the value.
    ///
    /// \return true if the name was found, false otherwise.
    bool lookup(ISymbol const *symbol, int &code) const MDL_FINAL {
        for (size_t idx = 0, end = Base::argument_count(); idx < end; ++idx) {
            Enum_member const &member = Base::argument_at(idx);
            if (symbol == member.get_symbol()) {
                code = member.get_code();
                return true;
            }
        }
        return false;
    }

    /// If this enum is a predefined one, return its ID, else EID_USER.
    Predefined_id get_predefined_id() const MDL_FINAL {
        return m_predefined_id;
    }

    // ---------------------- non-interface ----------------------

    /// Get the owner id.
    size_t get_owner_id() const { return m_owner_id; }

private:
    /// Constructor.
    ///
    /// \param owner_id   the id of the type factory owning this type
    /// \param arena      the memory arena the enum values are allocated on
    /// \param name       the fully qualified name of this type
    /// \param id         the predefined id of this enum
    explicit Type_enum(
        size_t        owner_id,
        Memory_arena  *arena,
        ISymbol const *name,
        Predefined_id id = EID_USER)
    : Base(arena)
    , m_owner_id(owner_id)
    , m_name(name)
    , m_predefined_id(id)
    {
    }

private:
    /// An id representing the owner of this type (for debugging).
    size_t const m_owner_id;

    /// The name of this enum type.
    ISymbol const * const m_name;

    /// The predefined ID of this enum type.
    Predefined_id const m_predefined_id;
};

/// Implementation of the float type.
class Type_float : public Type_base<IType_float>
{
    typedef Type_base<IType_float> Base;
public:
    /// Constructor.
    explicit Type_float()
    : Base()
    {
    }
};

// The singleton for the float type.
namespace {
Type_float const the_float_type;
};

/// Implementation of the double type.
class Type_double : public Type_base<IType_double>
{
    typedef Type_base<IType_double> Base;
public:
    /// Constructor.
    explicit Type_double()
    : Base()
    {
    }
};

/// Implementation of the string type.
class Type_string : public Type_base<IType_string>
{
    typedef Type_base<IType_string> Base;
public:
    /// Constructor.
    explicit Type_string()
    : Base()
    {
    }
};

/// Implementation of the light_profile type.
class Type_light_profile : public Type_base<IType_light_profile>
{
    typedef Type_base<IType_light_profile> Base;
public:
    /// Constructor.
    explicit Type_light_profile()
    : Base()
    {
    }
};

/// Implementation of the bsdf type.
class Type_bsdf : public Type_base<IType_bsdf>
{
    typedef Type_base<IType_bsdf> Base;
public:
    /// Constructor.
    explicit Type_bsdf()
    : Base()
    {
    }
};

/// Implementation of the hair_bsdf type.
class Type_hair_bsdf : public Type_base<IType_hair_bsdf>
{
    typedef Type_base<IType_hair_bsdf> Base;
public:
    /// Constructor.
    explicit Type_hair_bsdf()
    : Base()
    {
    }
};

/// Implementation of the edf type.
class Type_edf : public Type_base<IType_edf>
{
    typedef Type_base<IType_edf> Base;
public:
    /// Constructor.
    explicit Type_edf()
    : Base()
    {
    }
};

/// Implementation of the vdf type.
class Type_vdf : public Type_base<IType_vdf>
{
    typedef Type_base<IType_vdf> Base;
public:
    /// Constructor.
    explicit Type_vdf()
    : Base()
    {
    }
};

/// Implementation of the vector type.
class Type_vector : public Type_base<IType_vector>
{
    typedef Type_base<IType_vector> Base;
public:
    /// Get the type of the vector elements.
    IType_atomic const *get_element_type() const MDL_FINAL { return m_element_type; }

    /// Get the number of vector elements.
    int get_size() const  MDL_FINAL{ return m_size; }

    /// Get the compound type at index i.
    IType const *get_compound_type(int index) const MDL_FINAL {
        if (0 <= index && index < m_size)
            return m_element_type;
        return NULL;
    }

    /// Get the number of compound elements.
    int get_compound_size() const MDL_FINAL { return m_size; }

    /// Constructor.
    ///
    /// \param element_type  the element type of this vector type
    /// \param size          the size of this vector type
    explicit Type_vector(IType_atomic const *element_type, int size)
    : Base()
    , m_element_type(element_type)
    , m_size(size)
    {
    }

private:
    /// The element type of this vector.
    IType_atomic const *m_element_type;

    /// The size of this vector.
    int const m_size;
};

/// Implementation of the matrix type.
class Type_matrix : public Type_base<IType_matrix>
{
    typedef Type_base<IType_matrix> Base;
public:
    /// Get the type of the matrix elements.
    IType_vector const *get_element_type() const MDL_FINAL { return m_element_type; }

    /// Get the number of matrix columns.
    int get_columns() const MDL_FINAL { return m_columns; }

    /// Get the compound type at index i.
    IType const *get_compound_type(int index) const MDL_FINAL {
        if (0 <= index && index < m_columns)
            return m_element_type;
        return NULL;
    }

    /// Get the number of compound elements.
    int get_compound_size() const MDL_FINAL { return m_columns; }

    /// Constructor.
    ///
    /// \param element_type  the element type of this matrix type
    /// \param columns       the number of columns of this matrix type
    explicit Type_matrix(IType_vector const *element_type, int columns)
    : Base()
    , m_element_type(element_type)
    , m_columns(columns)
    {
    }

private:
    /// The element type of this matrix.
    IType_vector const *m_element_type;

    /// The number of columns of this matrix.
    int const m_columns;
};

/// Implementation of the abstract array size
class Type_array_size : public IType_array_size
{
    typedef IType_array_size Base;
    friend class Arena_builder;
public:
    /// Get the name of the abstract array length symbol.
    ISymbol const *get_size_symbol() const MDL_FINAL { return m_sym; }

    /// Get the absolute name of the this array length.
    ISymbol const *get_name() const MDL_FINAL { return m_name; }

    /// Constructor.
    ///
    /// \param sym   the name of the abstract array length symbol
    /// \param name  the absolute name of this array length symbol
    explicit Type_array_size(ISymbol const *sym, ISymbol const *name)
    : m_sym(sym)
    , m_name(name)
    {
    }

private:
    /// The name of the abstract array length symbol.
    ISymbol const *m_sym;

    /// The absolute name of this array length symbol.
    ISymbol const *m_name;
};

template<>
inline Type_array_size const *impl_cast(IType_array_size const *t) {
    return static_cast<Type_array_size const *>(t);
}

/// Implementation of the array type.
class Type_array : public Type_base<IType_array>
{
    typedef Type_base<IType_array> Base;
    friend class Arena_builder;
public:

    /// Get the type of the array elements.
    IType const *get_element_type() const MDL_FINAL { return m_element_type; }

    /// Test if the array is immediate sized.
    bool is_immediate_sized() const MDL_FINAL { return m_deferred_size == NULL; }

    /// Get the size of the (immediate sized) array.
    int get_size() const MDL_FINAL { return m_size; }

    /// Get the deferred array size.
    IType_array_size const *get_deferred_size() const MDL_FINAL { return m_deferred_size; }

    /// Get the compound type at index i.
    IType const *get_compound_type(int index) const MDL_FINAL {
        if (0 <= index && index < m_size)
            return m_element_type;
        return NULL;
    }

    /// Get the number of compound elements.
    int get_compound_size() const MDL_FINAL { return m_size; }

    // ---------------------- non-interface ----------------------

    /// Get the owner id.
    size_t get_owner_id() const { return m_owner_id; }

private:
    /// Constructor for an abstract array.
    ///
    /// \param owner_id      the id of the type factory owning this type
    /// \param element_type  the element type of this array type
    /// \param abs_size      the abstract array size
    explicit Type_array(
        size_t                owner_id,
        IType const           *element_type,
        Type_array_size const *abs_size)
    : Base()
    , m_owner_id(owner_id)
    , m_element_type(element_type)
    , m_deferred_size(abs_size)
    , m_size(-1)
    {
        MDL_ASSERT(Type_factory::is_owned(m_owner_id, element_type));
    }

    /// Constructor for an abstract array.
    ///
    /// \param owner_id      the id of the type factory owning this type
    /// \param element_type  the element type of this array type
    /// \param length        the array size
    explicit Type_array(
        size_t      owner_id,
        IType const *element_type,
        int         length)
    : Base()
    , m_owner_id(owner_id)
    , m_element_type(element_type)
    , m_deferred_size(NULL)
    , m_size(length)
    {
    }

private:
    /// An id representing the owner of this type (for debugging).
    size_t const m_owner_id;

    /// The element type of this array.
    IType const *m_element_type;

    /// The definition of the deferred length of this array if any.
    Type_array_size const * const m_deferred_size;

    /// The length of the array if not abstract.
    int const m_size;
};

/// Implementation of the color type.
class Type_color : public Type_base<IType_color>
{
    typedef Type_base<IType_color> Base;
public:

    /// Get the compound type at index i.
    IType const *get_compound_type(int index) const MDL_FINAL {
        if (0 <= index && index < 3) {
            return &the_float_type;
        }
        return NULL;
    }

    /// Get the number of compound elements.
    int get_compound_size() const MDL_FINAL { return 3; }

    /// Constructor.
    explicit Type_color()
    : Base()
    {
    }
};

/// Implementation of the function type.
class Type_function : public Type_base<IType_function>
{
    typedef Type_base<IType_function> Base;
    friend class Arena_builder;
public:
    typedef IType_factory::Function_parameter Function_parameter;

    /// Get the return type of the function.
    IType const *get_return_type() const MDL_FINAL { return m_ret_type; }

    /// Get the number of parameters of the function.
    int get_parameter_count() const MDL_FINAL { return int(m_n_parameters); }

    /// Get a parameter of the function.
    ///
    /// \param index    The index of the parameter in the parameter list.
    /// \param type     The type of the parameter.
    /// \param symbol   The symbol of the parameter.
    void get_parameter(
        int           index,
        IType const   *&type,
        ISymbol const *&symbol) const MDL_FINAL
    {
        if (0 <= index && size_t(index) < m_n_parameters) {
            Function_parameter const &param = m_parameters[index];
            type   = param.p_type;
            symbol = param.p_sym;
        } else {
            type   = NULL;
            symbol = NULL;
        }
    }

    // ---------------------- non-interface ----------------------

    /// Get the owner id.
    size_t get_owner_id() const { return m_owner_id; }

private:
    /// Constructor.
    ///
    /// \param owner_id      the id of the type factory owning this type
    /// \param arena         the arena used to allocate the function parameters on
    /// \param parameters    the function parameters of this type
    /// \param n_parameters  number of function parameters
    explicit Type_function(
        size_t                           owner_id,
        Memory_arena                     *arena,
        IType const                      *ret_type,
        Function_parameter const * const parameters,
        size_t                           n_parameters)
    : Base()
    , m_owner_id(owner_id)
    , m_ret_type(ret_type)
    , m_parameters(NULL)
    , m_n_parameters(n_parameters)
    {
        // return type might be NULL for annotation types
        MDL_ASSERT(m_ret_type == NULL || Type_factory::is_owned(m_owner_id, m_ret_type));
        if (n_parameters > 0) {
            m_parameters = reinterpret_cast<Function_parameter *>(
                Arena_memdup(*arena, parameters, n_parameters * sizeof(Function_parameter)));

            for (size_t i = 0; i < n_parameters; ++i) {
                MDL_ASSERT(Type_factory::is_owned(m_owner_id, m_parameters[i].p_type));
            }
        }
    }

private:
    /// An id representing the owner of this type (for debugging).
    size_t const m_owner_id;

    /// The return type of this function type.
    IType const *m_ret_type;

    /// The parameters of this function type.
    Function_parameter const *m_parameters;

    /// Number of function parameters.
    size_t m_n_parameters;
};

/// A structure member.
class Struct_member {
public:
    /// Return the type of this member.
    IType const *get_type() const { return m_type; }

    /// Return the name of this member.
    ISymbol const *get_symbol() const { return m_name; }

    /// Constructor.
    ///
    /// \param type  the type of this structure member
    /// \param name  the name of this structure member
    explicit Struct_member(IType const *type, ISymbol const *name)
    : m_type(type)
    , m_name(name)
    {
    }

private:
    /// The type of this field.
    IType const *m_type;

    /// The name of this field.
    ISymbol const *m_name;
};

/// Implementation of the structure type.
class Type_struct : public Type_base_variadic<IType_struct, Struct_member>
{
    typedef Type_base_variadic<IType_struct, Struct_member> Base;
    friend class Arena_builder;
public:

    /// Get the name of the struct type.
    ISymbol const *get_symbol() const MDL_FINAL { return m_name; }

    /// Add a field to the struct type.
    /// \param type The type of the field.
    /// \param name The name of the field.
    void add_field(IType const *type, ISymbol const *name) MDL_FINAL {
        MDL_ASSERT(m_predefined_id != SID_USER || Type_factory::is_owned(m_owner_id, type));
        Base::add_argument(Struct_member(type, name));
    }

    /// Get the number of fields.
    int get_field_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get a field.
    /// \param index    The index of the field.
    /// \param type     The type of the field.
    /// \param symbol   The symbol of the field.
    void get_field(
        int index,
        IType const *&type,
        ISymbol const *&symbol) const MDL_FINAL
    {
        Struct_member const &field = Base::argument_at(index);
        type = field.get_type();
        symbol = field.get_symbol();
    }

    /// Return the index of a field in O(N) if it is present and -1 otherwise.
    /// \param symbol   The name of the field.
    int find_field(ISymbol const *symbol) const MDL_FINAL {
        for (size_t idx = 0, end = Base::argument_count(); idx < end; ++idx) {
            Struct_member const &field = Base::argument_at(idx);
            if (symbol == field.get_symbol())
                return int(idx);
        }
        return -1;
    }

    /// Return the index of a field in O(N) if it is present and -1 otherwise.
    /// \param symbol   The name of the field.
    int find_field(char const *name) const MDL_FINAL {
        for (size_t idx = 0, end = Base::argument_count(); idx < end; ++idx) {
            Struct_member const &field = Base::argument_at(idx);
            if (strcmp(field.get_symbol()->get_name(), name) == 0)
                return int(idx);
        }
        return -1;
    }

    /// Return the index of a method in O(N) if it is present and -1 otherwise.
    /// \param symbol   The name of the method.
    int find_method(ISymbol const *symbol) const MDL_FINAL
    {
        for (size_t idx = 0, end = m_methods.size(); idx < end; ++idx) {
            Struct_member const &field = m_methods.at(idx);
            if (symbol == field.get_symbol())
                return int(idx);
        }
        return -1;

    }

    /// Add a method.
    /// \param type The type of the method.
    /// \param name The name of the method.
    void add_method(IType_function const *type, ISymbol const *name) MDL_FINAL {
        MDL_ASSERT(Type_factory::is_owned(m_owner_id, type));
        m_methods.push_back(Struct_member(type, name));
    }

    /// Get the number of methods.
    int get_method_count() const MDL_FINAL{ return m_methods.size(); }

    /// Get a method.
    /// \param index    The index of the method.
    /// \param type     The type of the field.
    /// \param symbol   The symbol of the field.
    void get_method(int index,
        IType_function const *&type,
        ISymbol const *&symbol) const MDL_FINAL
    {
        Struct_member const &field = m_methods.at(index);
        type = static_cast<const IType_function *>(field.get_type());
        symbol = field.get_symbol();
    }

    /// If this struct is a predefined one, return its ID, else SID_USER.
    Predefined_id get_predefined_id() const MDL_FINAL {
        return m_predefined_id;
    }

    /// Get the compound type at index i.
    IType const *get_compound_type(int index) const MDL_FINAL {
        if (0 <= index && index < Type_struct::get_field_count()) {
            Struct_member const &field = Base::argument_at(index);
            return field.get_type();
        }
        return NULL;
    }

    /// Get the number of compound elements.
    int get_compound_size() const  MDL_FINAL{ return Type_struct::get_field_count(); }

    // ---------------------- non-interface ----------------------

    /// Get the owner id.
    size_t get_owner_id() const { return m_owner_id; }

private:
    /// Constructor.
    ///
    /// \param owner_id  the id of the type factory owning this type
    /// \param arena     the arena used to allocate the structure members on
    /// \param name      the absolute name of this structure
    /// \param id        the predefined id of this structure
    explicit Type_struct(
        size_t        owner_id,
        Memory_arena  *arena,
        ISymbol const *name,
        Predefined_id id = SID_USER)
    : Base(arena)
    , m_owner_id(owner_id)
    , m_name(name)
    , m_predefined_id(id)
    , m_methods(arena)
    {
    }

private:
    /// An id representing the owner of this type (for debugging).
    size_t const m_owner_id;

    /// The name of this structure type.
    ISymbol const * const m_name;

    /// The predefined ID of this structure type.
    Predefined_id const m_predefined_id;

    /// All methods of this struct.
    Arena_vector<Struct_member>::Type m_methods;
};

/// Implementation of the texture type.
class Type_texture : public Type_base<IType_texture>
{
    typedef Type_base<IType_texture> Base;
    friend class Arena_builder;
public:

    /// Get the texture shape.
    Shape get_shape() const MDL_FINAL { return m_shape; }

    /// Get the coordinate type.
    IType const *get_coord_type() const MDL_FINAL { return m_coord_type; }

    explicit Type_texture(
        Shape shape,
        IType const *coord_type)
    : Base()
    , m_shape(shape)
    , m_coord_type(coord_type)
    {
    }

private:
    /// The texture shape.
    Shape const m_shape;

    /// The coordinate type.
    IType const * const m_coord_type;
};

/// Implementation of the bsdf_measurement type.
class Type_bsdf_measurement : public Type_base<IType_bsdf_measurement>
{
    typedef Type_base<IType_bsdf_measurement> Base;
public:
    /// Constructor.
    explicit Type_bsdf_measurement()
    : Base()
    {
    }
};

// -------------------------------------- type factory --------------------------------------

namespace {
// create the builtin types except of the float type which was already created.
#define NO_FLOAT_TYPE
#define BUILTIN_TYPE(type, name, args) type const name args;

#include "compilercore_builtin_types.h"
}  // anonymous

static size_t g_id = 0;

Type_factory::Type_factory(
    Memory_arena  &arena,
    IMDL          *compiler,
    Symbol_table  *symtab)
: Base()
, m_builder(arena)
, m_id(++g_id)
, m_compiler_factory(compiler != NULL ? compiler->get_type_factory() : NULL)
, m_symtab(symtab)
, m_type_cache(0, Type_cache::hasher(), Type_cache::key_equal(), &arena)
, m_array_size_cache(0, Array_size_cache::hasher(), Array_size_cache::key_equal(), &arena)
, m_imported_types_cache(0, Type_import_map::hasher(), Type_import_map::key_equal(), &arena)
{
    if (compiler == NULL) {
        // must be create on heap, so do it here

        // Note: create only the types here, its members will be added by the builder
        // from compilercore_known_defs.h
        m_predefined_structs[IType_struct::SID_MATERIAL_EMISSION] = m_builder.create<Type_struct>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_MATERIAL_EMISSION),
            IType_struct::SID_MATERIAL_EMISSION);
        m_predefined_structs[IType_struct::SID_MATERIAL_SURFACE]  = m_builder.create<Type_struct>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_MATERIAL_SURFACE),
            IType_struct::SID_MATERIAL_SURFACE);
        m_predefined_structs[IType_struct::SID_MATERIAL_VOLUME]   = m_builder.create<Type_struct>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_MATERIAL_VOLUME),
            IType_struct::SID_MATERIAL_VOLUME);
        m_predefined_structs[IType_struct::SID_MATERIAL_GEOMETRY] = m_builder.create<Type_struct>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_MATERIAL_GEOMETRY),
            IType_struct::SID_MATERIAL_GEOMETRY);
        m_predefined_structs[IType_struct::SID_MATERIAL] = m_builder.create<Type_struct>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_MATERIAL),
            IType_struct::SID_MATERIAL);

        // This IS ugly: the texture type constructors depend on the ::tex::gamma_mode type.
        // However, these types are builtin, so we must create them in every module and than
        // means ::tex::gamma_mode must be available in advance ...
        // Build the type here!
        Type_enum *gamma_mode =
            m_builder.create<Type_enum>(
                m_id,
                m_builder.get_arena(),
                Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_TEX_GAMMA_MODE),
                IType_enum::EID_TEX_GAMMA_MODE);

        // Note: values for the builtin gamma_mode enum are added when the ::tex module
        // is processed inside NT_analysis::pre_visit(IDeclaration_type_enum *enum_decl).

        m_predefined_enums[IType_enum::EID_TEX_GAMMA_MODE] = gamma_mode;

        // MDL 1.1 intensity mode enum. Its values will be added by the builder
        // from compilercore_known_defs.h
        Type_enum *intensity_mode =
            m_builder.create<Type_enum>(
            m_id,
            m_builder.get_arena(),
            Symbol_table::get_predefined_symbol(ISymbol::SYM_TYPE_INTENSITY_MODE),
            IType_enum::EID_INTENSITY_MODE);
        m_predefined_enums[IType_enum::EID_INTENSITY_MODE] = intensity_mode;
    } else {
        memset(m_predefined_structs, 0, sizeof(m_predefined_structs));
        memset(m_predefined_enums,   0, sizeof(m_predefined_enums));
    }
}

// Create a new type alias instance.
IType const *Type_factory::create_alias(
    IType const      *type,
    ISymbol const    *name,
    IType::Modifiers modifiers)
{
    // only allowed on the module factories
    if (! m_compiler_factory)
        return NULL;

    // an alias of the error type is still the error type
    if (is<IType_error>(type))
        return type;

    if (name != NULL) {
        // import the name into our symbol table
        name = m_symtab->get_symbol(name->get_name());
    } else {
        if (modifiers == IType::MK_NONE)
            return type;
    }

    Type_cache_key key(type, name, modifiers);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IType const *alias_type = m_builder.create<Type_alias>(m_id, type, name, modifiers);

        it = m_type_cache.insert(Type_cache::value_type(key, alias_type)).first;
    }
    return it->second;
}

// Create a new type error instance.
IType_error const *Type_factory::create_error()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_error();
    return &the_error_type;
}

// Create a new type incomplete instance.
IType_incomplete const *Type_factory::create_incomplete()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_incomplete();
    return &the_incomplete_type;
}

// Create a new type bool instance.
IType_bool const *Type_factory::create_bool()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_bool();
    return &the_bool_type;
}

/// Create a new type int instance.
const IType_int *Type_factory::create_int()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_int();
    return &the_int_type;
}

// Create a new type enum instance.
IType_enum *Type_factory::create_enum(ISymbol const *name)
{
    // only allowed on the module factories
    if (!m_compiler_factory)
        return NULL;

    // import the name
    name = m_symtab->get_user_type_symbol(name->get_name());
    IType_enum *e_type = m_builder.create<Type_enum>(m_id, m_builder.get_arena(), name);

    // register this enum type, so we can "import" it
    m_imported_types_cache[name->get_name()] = e_type;

    return e_type;
}

// Lookup an enum type.
IType_enum const *Type_factory::lookup_enum(char const *name) const
{
    // only allowed on the module factories
    if (!m_compiler_factory)
        return NULL;

    Type_import_map::const_iterator it = m_imported_types_cache.find(name);
    if (it == m_imported_types_cache.end()) return NULL;

    // cast to enum type or NULL, if it's not an enum type
    IType_enum const *enum_type = as<IType_enum>(it->second);
    return enum_type;
}

// Create a new type float instance.
IType_float const *Type_factory::create_float()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_float();
    return &the_float_type;
}

// Create a new type double instance.
IType_double const *Type_factory::create_double()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_double();
    return &the_double_type;
}

// Create a new type string instance.
IType_string const *Type_factory::create_string()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_string();
    return &the_string_type;
}

// Create the type bsdf instance.
IType_bsdf const *Type_factory::create_bsdf()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_bsdf();
    return &the_bsdf_type;
}

// Create a new type hair_bsdf instance.
IType_hair_bsdf const *Type_factory::create_hair_bsdf()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_hair_bsdf();
    return &the_hair_bsdf_type;
}

// Create a new type edf instance.
IType_edf const *Type_factory::create_edf()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_edf();
    return &the_edf_type;
}

// Create a new type vdf instance.
IType_vdf const *Type_factory::create_vdf()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_vdf();
    return &the_vdf_type;
}

// Create a new type light profile instance.
IType_light_profile const *Type_factory::create_light_profile()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_light_profile();
    return &the_light_profile_type;
}

// Create a new type vector instance.
IType_vector const *Type_factory::create_vector(
    IType_atomic const *element_type,
    int size)
{
    if (m_compiler_factory)
        return m_compiler_factory->create_vector(element_type, size);
    switch (size) {
    case 2:
        switch (element_type->get_kind()) {
        case IType::TK_BOOL:
            return &the_bool2_type;
        case IType::TK_INT:
            return &the_int2_type;
        case IType::TK_FLOAT:
            return &the_float2_type;
        case IType::TK_DOUBLE:
            return &the_double2_type;
        default:
            break;
        }
        break;
    case 3:
        switch (element_type->get_kind()) {
        case IType::TK_BOOL:
            return &the_bool3_type;
        case IType::TK_INT:
            return &the_int3_type;
        case IType::TK_FLOAT:
            return &the_float3_type;
        case IType::TK_DOUBLE:
            return &the_double3_type;
        default:
            break;
        }
        break;
    case 4:
        switch (element_type->get_kind()) {
        case IType::TK_BOOL:
            return &the_bool4_type;
        case IType::TK_INT:
            return &the_int4_type;
        case IType::TK_FLOAT:
            return &the_float4_type;
        case IType::TK_DOUBLE:
            return &the_double4_type;
        default:
            break;
        }
        break;
    }
    // unsupported
    return NULL;
}

// Create a new type matrix instance.
IType_matrix const *Type_factory::create_matrix(
    IType_vector const *element_type,
    int columns)
{
    if (m_compiler_factory)
        return m_compiler_factory->create_matrix(element_type, columns);

    IType::Kind kind = element_type->get_element_type()->get_kind();

    switch (columns) {
    case 2:
        switch (element_type->get_size()) {
        case 2:
            if (kind == IType::TK_FLOAT)
                return &the_float2x2_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double2x2_type;
            break;
        case 3:
            if (kind == IType::TK_FLOAT)
                return &the_float3x2_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double3x2_type;
            break;
        case 4:
            if (kind == IType::TK_FLOAT)
                return &the_float4x2_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double4x2_type;
            break;
        }
        break;
    case 3:
        switch (element_type->get_size()) {
        case 2:
            if (kind == IType::TK_FLOAT)
                return &the_float2x3_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double2x3_type;
            break;
        case 3:
            if (kind == IType::TK_FLOAT)
                return &the_float3x3_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double3x3_type;
            break;
        case 4:
            if (kind == IType::TK_FLOAT)
                return &the_float4x3_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double4x3_type;
            break;
        }
        break;
    case 4:
        switch (element_type->get_size()) {
        case 2:
            if (kind == IType::TK_FLOAT)
                return &the_float2x4_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double2x4_type;
            break;
        case 3:
            if (kind == IType::TK_FLOAT)
                return &the_float3x4_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double3x4_type;
            break;
        case 4:
            if (kind == IType::TK_FLOAT)
                return &the_float4x4_type;
            else if (kind == IType::TK_DOUBLE)
                return &the_double4x4_type;
            break;
        }
        break;
    }
    // unsupported
    return NULL;
}

// Find an sized array type instance.
IType const *Type_factory::find_array(IType const *element_type, int size) const
{
    // only allowed on the module factories
    if (! m_compiler_factory)
        return NULL;

    Type_cache_key key(size, element_type);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it != m_type_cache.end()) {
        return it->second;
    }
    return NULL;
}

// Find any deferred sized array type with the same base type.
IType const *Type_factory::find_any_deferred_array(IType const *element_type) const
{
    // only allowed on the module factories
    if (m_compiler_factory == NULL)
        return NULL;

    // FIXME: slow linear search here, because I cannot guess the hash value ...
    // for now it is only used from get_overload_by_signature() neuray helper, so it is fine
    // for me
    for (Type_cache::const_iterator it(m_type_cache.begin()), end(m_type_cache.end());
        it != end;
        ++it)
    {
        Type_cache_key const &key = it->first;
        if (key.kind == Type_cache_key::KEY_ABSTRACT_ARRAY) {
            if (key.type == element_type) {
                // found it
                return it->second;
            }
        }
    }
    return NULL;
}

// Create a new type sized array instance.
IType const *Type_factory::create_array(
    IType const *element_type,
    size_t      size)
{
    // only allowed on the module factories
    if (! m_compiler_factory)
        return NULL;

    if (is<IType_error>(element_type)) {
        // cannot create an array of error type
        return element_type;
    }

    Type_cache_key key(size, element_type);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IType const *array_type = m_builder.create<Type_array>(m_id, element_type, int(size));

        it = m_type_cache.insert(Type_cache::value_type(key, array_type)).first;
    }
    return it->second;
}

// Create a new type color instance.
IType_color const *Type_factory::create_color()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_color();
    return &the_color_type;
}

// Create a new type function instance.
IType_function const *Type_factory::create_function(
    IType const                      *return_type,
    Function_parameter const * const parameters,
    size_t                           n_parameters)
{
    // only allowed on the module factories
    if (! m_compiler_factory)
        return NULL;

    Type_cache_key key(return_type, parameters, n_parameters);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IType_function const *fun_type = m_builder.create<Type_function>(
            m_id, m_builder.get_arena(), return_type, parameters, n_parameters);

        it = m_type_cache.insert(Type_cache::value_type(fun_type, fun_type)).first;
    }
    return cast<IType_function>(it->second);
}

// Create a new type struct instance.
IType_struct *Type_factory::create_struct(ISymbol const *name)
{
    // only allowed on the module factories
    if (!m_compiler_factory)
        return NULL;

    // import the name
    name = m_symtab->get_user_type_symbol(name->get_name());
    IType_struct *s_type = m_builder.create<Type_struct>(m_id, m_builder.get_arena(), name);

    // register this struct type, so we can "import" it
    m_imported_types_cache[name->get_name()] = s_type;
    return s_type;
}

// Lookup a struct type.
IType_struct const *Type_factory::lookup_struct(char const *name) const
{
    // only allowed on the module factories
    if (!m_compiler_factory)
        return NULL;

    Type_import_map::const_iterator it = m_imported_types_cache.find(name);
    if (it == m_imported_types_cache.end()) return NULL;

    // cast to struct type or NULL, if it's not a struct type
    IType_struct const *struct_type = as<IType_struct>(it->second);
    return struct_type;
}

// Create a new type texture instance.
IType_texture const *Type_factory::create_texture(
    IType_texture::Shape const shape)
{
    if (m_compiler_factory)
        return m_compiler_factory->create_texture(shape);

    switch (shape) {
    case IType_texture::TS_2D:        return &the_texture_2d_type;
    case IType_texture::TS_3D:        return &the_texture_3d_type;
    case IType_texture::TS_CUBE:      return &the_texture_cube_type;
    case IType_texture::TS_PTEX:      return &the_texture_ptex_type;
    case IType_texture::TS_BSDF_DATA: return &the_texture_bsdf_data_type;
    }
    MDL_ASSERT(!"Unsupported texture shape");
    return NULL;
}

// Create a new type bsdf_measurement instance.
IType_bsdf_measurement const *Type_factory::create_bsdf_measurement()
{
    if (m_compiler_factory)
        return m_compiler_factory->create_bsdf_measurement();
    return &the_bsdf_measurement_type;
}

// Import a type from another type factory.
IType const *Type_factory::import(IType const *type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            // Note: we do not cache alias type imports here. This should be safe,
            // because alias types are never coupled to definition table scopes. so it is
            // safe to just recreate them.
            IType_alias const *a_type = cast<IType_alias>(type);
            ISymbol const     *sym    = a_type->get_symbol();
            IType const       *i_type = a_type->get_aliased_type();
            i_type = import(i_type);

            IType::Modifiers mod = a_type->get_type_modifiers();
            IType const *n_type = create_alias(i_type, sym, mod);
            return n_type;
        }
    case IType::TK_BOOL:
        return create_bool();
    case IType::TK_INT:
        return create_int();
    case IType::TK_ENUM:
        {
            IType_enum const *e_type     = cast<IType_enum>(type);
            IType_enum::Predefined_id id = e_type->get_predefined_id();

            if (id != IType_enum::EID_USER) {
                // a builtin enum
                return get_predefined_enum(id);
            }

            ISymbol const *e_type_sym = e_type->get_symbol();
            Type_import_map::iterator it = m_imported_types_cache.find(e_type_sym->get_name());
            if (it != m_imported_types_cache.end()) {
                // We have this name;
                IType const *n_type = it->second;

                if (is<IType_enum>(n_type))
                    return n_type;
                // The names of user defined types must be unique, so this should not happen ...
                // Except something really bad happens.
            }

            IType_enum *n_type = create_enum(e_type_sym);

            for (int i = 0, n = e_type->get_value_count(); i < n; ++i) {
                ISymbol const *v_sym;
                int           v_code;
                e_type->get_value(i, v_sym, v_code);

                // FIXME: add_value is dangerous
                v_sym = m_symtab->get_symbol(v_sym->get_name());
                n_type->add_value(v_sym, v_code);
            }
            return n_type;
        }
    case IType::TK_FLOAT:
        return create_float();
    case IType::TK_DOUBLE:
        return create_double();
    case IType::TK_STRING:
        return create_string();
    case IType::TK_LIGHT_PROFILE:
        return create_light_profile();
    case IType::TK_BSDF:
        return create_bsdf();
    case IType::TK_HAIR_BSDF:
        return create_hair_bsdf();
    case IType::TK_EDF:
        return create_edf();
    case IType::TK_VDF:
        return create_vdf();
    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            IType_atomic const *a_type = v_type->get_element_type();

            a_type = static_cast<IType_atomic const *>(import(a_type));

            return create_vector(a_type, v_type->get_size());
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            IType_vector const *v_type = m_type->get_element_type();

            v_type = cast<IType_vector>(import(v_type));

            return create_matrix(v_type, m_type->get_columns());
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const       *e_type = a_type->get_element_type();

            e_type = import(e_type);

            if (a_type->is_immediate_sized())
                return create_array(e_type, a_type->get_size());

            Type_array_size const *a_size =
                impl_cast<Type_array_size>(a_type->get_deferred_size());

            return create_array(
                e_type,
                a_size->get_name(),
                a_size->get_size_symbol());
        }
    case IType::TK_COLOR:
        return create_color();
    case IType::TK_FUNCTION:
        {
            IType_function const *f_type = cast<IType_function>(type);

            int n_params = f_type->get_parameter_count();
            VLA<Function_parameter> params(m_builder.get_arena()->get_allocator(), n_params);

            for (int i = 0; i < n_params; ++i) {
                IType const   *p_type;
                ISymbol const *p_sym;

                f_type->get_parameter(i, p_type, p_sym);

                params[i].p_type = import(p_type);
                params[i].p_sym  = m_symtab->get_symbol(p_sym->get_name());
            }

            IType const *ret_tp = f_type->get_return_type();

            // annotations have NO return type
            if (ret_tp != NULL)
                ret_tp = import(f_type->get_return_type());

            return create_function(ret_tp, params.data(), n_params);
        }
    case IType::TK_STRUCT:
        {
            IType_struct const          *s_type = cast<IType_struct>(type);
            IType_struct::Predefined_id id      = s_type->get_predefined_id();

            if (id != IType_struct::SID_USER) {
                // a builtin-material struct
                return get_predefined_struct(id);
            }

            ISymbol const *s_sym = s_type->get_symbol();

            Type_import_map::iterator it = m_imported_types_cache.find(s_sym->get_name());
            if (it != m_imported_types_cache.end()) {
                // We have this name;
                IType const *n_type = it->second;
                if (is<IType_struct>(n_type))
                    return n_type;
                // The names of user defined types must be unique, so this should not happen ...
                // Except something really bad happens.
            }

            IType_struct *n_type = create_struct(s_sym);

            for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                IType const   *f_type;
                ISymbol const *f_sym;

                s_type->get_field(i, f_type, f_sym);

                f_type = import(f_type);
                f_sym  = m_symtab->get_symbol(f_sym->get_name());

                // FIXME: add_field is dangerous
                n_type->add_field(f_type, f_sym);
            }
            return n_type;
        }
    case IType::TK_TEXTURE:
        {
            IType_texture const *t_type = cast<IType_texture>(type);
            return create_texture(t_type->get_shape());
        }
    case IType::TK_BSDF_MEASUREMENT:
        return create_bsdf_measurement();
    case IType::TK_INCOMPLETE:
        return create_incomplete();
    case IType::TK_ERROR:
        return create_error();
    }
    MDL_ASSERT(!"Unsupported type kind");
    return NULL;
}

// Create a new type abstract array instance.
IType const *Type_factory::create_array(
    IType const   *element_type,
    ISymbol const *abs_name,
    ISymbol const *sym)
{
    // only allowed on the module factories
    if (! m_compiler_factory)
        return NULL;

    if (is<IType_error>(element_type)) {
        // cannot create an array of error type
        return element_type;
    }

    Type_array_size const *array_size = impl_cast<Type_array_size>(get_array_size(abs_name, sym));

    Type_cache_key key(element_type, array_size);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IType const *array_type = m_builder.create<Type_array>(m_id, element_type, array_size);

        it = m_type_cache.insert(Type_cache::value_type(key, array_type)).first;
    }
    return it->second;
}

// Return a predefined struct.
IType_struct *Type_factory::get_predefined_struct(IType_struct::Predefined_id part)
{
    if (m_compiler_factory) {
        // this cast IS ugly, but we know that the top level type factory
        // is of type Type_factory (and not a proxy), so it's ok
        return static_cast<Type_factory *>(m_compiler_factory)->get_predefined_struct(part);
    }
    if (0 <= part && part <= IType_struct::SID_LAST)
        return m_predefined_structs[part];
    return NULL;
}

// Return a predefined enum.
IType_enum *Type_factory::get_predefined_enum(IType_enum::Predefined_id part)
{
    if (m_compiler_factory) {
        // this cast IS ugly, but we know that the top level type factory
        // is of type Type_factory (and not a proxy), so it's ok
        return static_cast<Type_factory *>(m_compiler_factory)->get_predefined_enum(part);
    }
    if (0 <= part && part <= IType_enum::EID_LAST)
        return m_predefined_enums[part];
    return NULL;
}

// Return the symbol table of this type factory.
Symbol_table *Type_factory::get_symbol_table()
{
    return m_symtab;
}

// Get the equivalent type for a given type in our type factory or return NULL if
IType const *Type_factory::get_equal(IType const *type) const
{
    Type_factory *safe = const_cast<Type_factory *>(this);

    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        // not supported
        return NULL;
    case IType::TK_BOOL:
        return safe->create_bool();
    case IType::TK_INT:
        return safe->create_int();
    case IType::TK_ENUM:
        {
            IType_enum const *e_type     = cast<IType_enum>(type);
            IType_enum::Predefined_id id = e_type->get_predefined_id();

            if (id != IType_enum::EID_USER) {
                // a builtin-material struct
                return safe->get_predefined_enum(id);
            }

            ISymbol const *e_type_sym = e_type->get_symbol();

            Type_import_map::const_iterator it =
                m_imported_types_cache.find(e_type_sym->get_name());
            if (it != m_imported_types_cache.end()) {
                // We have this name;
                IType const *n_type = it->second;

                if (is<IType_enum>(n_type))
                    return n_type;
                // The names of user defined types must be unique, so this should not happen ...
                // Except something really bad happens.
            }
            return NULL;
        }
    case IType::TK_FLOAT:
        return safe->create_float();
    case IType::TK_DOUBLE:
        return safe->create_double();
    case IType::TK_STRING:
        return safe->create_string();
    case IType::TK_LIGHT_PROFILE:
        return safe->create_light_profile();
    case IType::TK_BSDF:
        return safe->create_bsdf();
    case IType::TK_HAIR_BSDF:
        return safe->create_hair_bsdf();
    case IType::TK_EDF:
        return safe->create_edf();
    case IType::TK_VDF:
        return safe->create_vdf();
    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            IType_atomic const *a_type = v_type->get_element_type();

            a_type = static_cast<IType_atomic const *>(get_equal(a_type));
            return safe->create_vector(a_type, v_type->get_size());
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            IType_vector const *v_type = m_type->get_element_type();

            v_type = cast<IType_vector>(get_equal(v_type));
            return safe->create_matrix(v_type, m_type->get_columns());
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const       *e_type = a_type->get_element_type();

            e_type = get_equal(e_type);
            if (e_type == NULL)
                return NULL;

            if (a_type->is_immediate_sized()) {
                Type_cache_key key(a_type->get_size(), e_type);

                Type_cache::const_iterator it = m_type_cache.find(key);
                if (it != m_type_cache.end())
                    return it->second;
                return NULL;
            }
            // abstract array not supported
            return NULL;
        }
    case IType::TK_COLOR:
        return safe->create_color();
    case IType::TK_FUNCTION:
        // not supported
        return NULL;
    case IType::TK_STRUCT:
        {
            IType_struct const          *s_type = cast<IType_struct>(type);
            IType_struct::Predefined_id id      = s_type->get_predefined_id();

            if (id != IType_struct::SID_USER) {
                // a builtin-material struct
                return safe->get_predefined_struct(id);
            }

            ISymbol const *s_sym = s_type->get_symbol();

            Type_import_map::const_iterator it = m_imported_types_cache.find(s_sym->get_name());
            if (it != m_imported_types_cache.end()) {
                // We have this name;
                IType const *n_type = it->second;
                if (is<IType_struct>(n_type))
                    return n_type;
                // The names of user defined types must be unique, so this should not happen ...
                // Except something really bad happens.
            }
            return NULL;
        }
    case IType::TK_TEXTURE:
        {
            IType_texture const *t_type = cast<IType_texture>(type);
            return safe->create_texture(t_type->get_shape());
        }
    case IType::TK_BSDF_MEASUREMENT:
        return safe->create_bsdf_measurement();
    case IType::TK_INCOMPLETE:
        return safe->create_incomplete();
    case IType::TK_ERROR:
        return safe->create_error();
    }
    MDL_ASSERT(!"Unsupported type kind");
    return NULL;
}

// Get the array size for a given absolute abstract array length name.
IType_array_size const *Type_factory::get_array_size(
    ISymbol const *abs_name,
    ISymbol const *sym)
{
    Type_array_size const *array_size;

    // import the symbols into our symbol table
    sym      = m_symtab->get_symbol(sym->get_name());
    abs_name = m_symtab->get_user_type_symbol(abs_name->get_name());

    Array_size_cache::const_iterator a_it = m_array_size_cache.find(abs_name);
    if (a_it == m_array_size_cache.end()) {
        array_size = m_builder.create<Type_array_size>(sym, abs_name);
        m_array_size_cache.insert(Array_size_cache::value_type(abs_name, array_size));
    } else
        array_size = a_it->second;
    return array_size;
}

// Create a new type abstract array instance.
IType const *Type_factory::create_array(
    IType const            *element_type,
    IType_array_size const *iarray_size)
{
    // only allowed on the module factories
    if (!m_compiler_factory)
        return NULL;

    if (is<IType_error>(element_type)) {
        // cannot create an array of error type
        return element_type;
    }

    Type_array_size const *array_size = impl_cast<Type_array_size>(iarray_size);
    Type_cache_key key(element_type, array_size);

    Type_cache::const_iterator it = m_type_cache.find(key);
    if (it == m_type_cache.end()) {
        IType const *array_type = m_builder.create<Type_array>(m_id, element_type, array_size);

        it = m_type_cache.insert(Type_cache::value_type(key, array_type)).first;
    }
    return it->second;
}

namespace {

/// Helper class to compare Array_sizes.
struct Array_size_less {
    bool operator() (Type_array_size const *s, Type_array_size const *t)
    {
        ISymbol const *s_sym = s->get_name();
        ISymbol const *t_sym = t->get_name();

        return strcmp(s_sym->get_name(), t_sym->get_name()) < 0;
    }
};

}  // anonymous

// Serialize the type table.
void Type_factory::serialize(Factory_serializer &serializer) const
{
    register_builtins(serializer);

    serializer.write_section_tag(Serializer::ST_TYPE_TABLE);

    // register this type factory
    Tag_t factory_tag = serializer.register_type_factory(this);
    serializer.write_encoded_tag(factory_tag);
    DOUT(("type factory %u {\n", unsigned(factory_tag)));
    INC_SCOPE();

    // do not serialize m_id
    // do not serialize the link to the compiler factory

    // the symbol table must be already serialized at this moment
    Tag_t symtab_tag = serializer.get_symbol_table_tag(m_symtab);
    serializer.write_encoded_tag(symtab_tag);

    DOUT(("using symtab %u\n", unsigned(symtab_tag)));

    // do not serialize m_predefined_structs, already done in register_builtins()

    // now serialize all types; first step array sizes.
    size_t array_sizes = m_array_size_cache.size();
    typedef vector<Type_array_size const *>::Type Type_array_sizes;

    Type_array_sizes sizes(serializer.get_allocator());
    sizes.reserve(array_sizes);

    for (Array_size_cache::const_iterator
        it(m_array_size_cache.begin()), end(m_array_size_cache.end());
        it != end;
        ++it)
    {
        Type_array_size const *size = it->second;
        sizes.push_back(size);
    }

    std::sort(sizes.begin(), sizes.end(), Array_size_less());

    serializer.write_encoded_tag(array_sizes);

    DOUT(("#array_sizes %u\n", unsigned(array_sizes)));
    INC_SCOPE();

    for (Type_array_sizes::iterator it(sizes.begin()), end(sizes.end()); it != end; ++it) {
        Type_array_size const *size = *it;

        Tag_t array_size_tag = serializer.register_array_size(size);
        serializer.write_encoded_tag(array_size_tag);

        ISymbol const *size_sym = size->get_size_symbol();
        Tag_t          size_tag = serializer.get_symbol_tag(size_sym);
        serializer.write_encoded_tag(size_tag);

        ISymbol const *name_sym = size->get_name();
        Tag_t          name_tag = serializer.get_symbol_tag(name_sym);
        serializer.write_encoded_tag(name_tag);

        DOUT(("array_size %u %u %u (%s %s)\n",
            unsigned(array_size_tag),
            unsigned(size_tag),
            unsigned(name_tag),
            size_sym->get_name(),
            name_sym->get_name()));
    }
    DEC_SCOPE();

    // second step: iterate over both caches and enqueue types
    for (Type_cache::const_iterator it(m_type_cache.begin()), end(m_type_cache.end());
        it != end;
        ++it)
    {
        IType const *type = it->second;

        MDL_ASSERT(is_owned(m_id, type));
        serializer.enqueue_type(type);
    }

    for (Type_import_map::const_iterator
        it(m_imported_types_cache.begin()), end(m_imported_types_cache.end());
        it != end;
        ++it)
    {
        IType const *type = it->second;

        MDL_ASSERT(is_owned(m_id, type));
        serializer.enqueue_type(type);
    }

    serializer.write_enqueued_types();
    DEC_SCOPE();
    DOUT(("Type factory }\n"));
}

// Deserialize the type table.
void Type_factory::deserialize(Factory_deserializer &deserializer)
{
    register_builtins(deserializer);

    Tag_t t;

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_TYPE_TABLE);

    // register this type factory
    Tag_t factory_tag = deserializer.read_encoded_tag();
    deserializer.register_type_factory(factory_tag, this);
    DOUT(("type factory %u (%p)\n", unsigned(factory_tag), this));
    INC_SCOPE();

    // do not deserialize m_id
    // do not deserialize the link to the compiler factory

    // the symbol table must be already serialized at this moment
    Tag_t symtab_tag = deserializer.read_encoded_tag();
    Symbol_table const *symtab;

    symtab = deserializer.get_symbol_table(symtab_tag);
    MDL_ASSERT(symtab == m_symtab);

    DOUT(("using symtab %u\n", unsigned(symtab_tag)));

    // do not deserialize m_predefined_structs, already done in register_builtins()

    // now deserialize all types; first step array sizes.
    size_t array_sizes = deserializer.read_encoded_tag();

    DOUT(("#array_sizes %u\n", unsigned(array_sizes)));
    INC_SCOPE();

    for (size_t i = 0; i < array_sizes; ++i) {
        Tag_t array_size_tag = deserializer.read_encoded_tag();

        Tag_t          size_tag = deserializer.read_encoded_tag();
        ISymbol const *size_sym = deserializer.get_symbol(size_tag);

        Tag_t          name_tag = deserializer.read_encoded_tag();
        ISymbol const *name_sym = deserializer.get_symbol(name_tag);

        IType_array_size const *size = get_array_size(name_sym, size_sym);
        deserializer.register_array_size(array_size_tag, size);

        DOUT(("array_size %u %u %u (%s %s)\n",
            unsigned(array_size_tag),
            unsigned(size_tag),
            unsigned(name_tag),
            size_sym->get_name(),
            name_sym->get_name()));
    }
    DEC_SCOPE();

    // second step: read types (and rebuilt caches)
    size_t n_types = deserializer.read_encoded_tag();
    DOUT(("#types %u\n", unsigned(n_types)));
    INC_SCOPE();

    for (size_t i = 0; i < n_types; ++i) {
        deserializer.read_type(*this);
    }
    DEC_SCOPE();

    DEC_SCOPE();
    DOUT(("Type factory }\n"));
}

// Register builtin-types.
void Type_factory::register_builtins(Factory_serializer &serializer) const
{
    // register all builtin types. When this happens, the type
    // tag set must be empty, check that.
    Tag_t t, check = Tag_t(0);

#define BUILTIN_TYPE(type, name, args) \
    t = serializer.register_type(&name); ++check; MDL_ASSERT(t == check);
#include "compilercore_builtin_types.h"

    // register predefined structs
    for (int i = 0, n = IType_struct::SID_LAST; i <= n; ++i) {
        IType_struct const *predef =
            const_cast<Type_factory *>(this)->get_predefined_struct(IType_struct::Predefined_id(i));
        t = serializer.register_type(predef); ++check; MDL_ASSERT(t == check);
    }
    // register predefined enums
    for (int i = 0, n = IType_enum::EID_LAST; i <= n; ++i) {
        IType_enum const *predef =
            const_cast<Type_factory *>(this)->get_predefined_enum(IType_enum::Predefined_id(i));
        t = serializer.register_type(predef); ++check; MDL_ASSERT(t == check);
    }
}

// Register builtin-types.
void Type_factory::register_builtins(Factory_deserializer &deserializer)
{
    // register all builtin types. When this happens, the type
    // tag set must be empty, check that.
    Tag_t t = Tag_t(0);

#define BUILTIN_TYPE(type, name, args) \
    ++t; deserializer.register_type(t, &name);
#include "compilercore_builtin_types.h"

    // register predefined structs
    for (int i = 0, n = IType_struct::SID_LAST; i <= n; ++i) {
        IType_struct const *predef = get_predefined_struct(IType_struct::Predefined_id(i));
        ++t; deserializer.register_type(t, predef);
    }
    // register predefined enums
    for (int i = 0, n = IType_enum::EID_LAST; i <= n; ++i) {
        IType_enum const *predef = get_predefined_enum(IType_enum::Predefined_id(i));
        ++t; deserializer.register_type(t, predef);
    }
}

// Check if the given type is owned by this type factory.
bool Type_factory::is_owned(
    size_t      owner,
    IType const *type)
{
    // can be only checked for user types
    size_t owner_id = get_owner_id(type);

    if (owner_id != 0)
        return owner == owner_id;
    return true;
}

// Get the owner id of a type.
size_t Type_factory::get_owner_id(IType const *type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        {
            Type_alias const *s_tp = static_cast<Type_alias const *>(type);
            return s_tp->get_owner_id();
        }
        break;
    case IType::TK_ENUM:
        {
            Type_enum const *e_tp = static_cast<Type_enum const *>(type);
            if (e_tp->get_predefined_id() != IType_enum::EID_USER)
                return 0;
            return e_tp->get_owner_id();
        }
        break;
    case IType::TK_ARRAY:
        {
            Type_array const *a_tp = static_cast<Type_array const *>(type);
            return a_tp->get_owner_id();
        }
        break;
    case IType::TK_FUNCTION:
        {
            Type_function const *f_tp = static_cast<Type_function const *>(type);
            return f_tp->get_owner_id();
        }
        break;
    case IType::TK_STRUCT:
        {
            Type_struct const *s_tp = static_cast<Type_struct const *>(type);
            if (s_tp->get_predefined_id() != IType_struct::SID_USER)
                return 0;
            return s_tp->get_owner_id();
        }
        break;
    default:
        // strictly speaking other types are owned by the compiler factory
        return 0;
    }
}

// Checks if this type factory owns the given type
bool Type_factory::is_owner(IType const *type) const
{
    size_t id = get_owner_id(type);
    if (id == m_id)
        return true;

    if (m_compiler_factory != NULL) {
        if (impl_cast<Type_factory>(m_compiler_factory)->is_owner(type)) {
            return true;
        }
    } else {
        // we are inside the compiler factory, predefined types have zero ID
        return id == 0;
    }

    return false;
}

}  // mdl
}  // mi
