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

#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_types.h>

#include "compilercore_allocator.h"
#include "compilercore_mdl.h"
#include "compilercore_assert.h"
#include "compilercore_tools.h"

#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

typedef unsigned uint32_t;
typedef unsigned char byte;

// Constructor.
Base_pointer_serializer::Base_pointer_serializer(IAllocator *alloc)
: m_pointer_map(0, Pointer_map::hasher(), Pointer_map::key_equal(), alloc)
, m_last_tag(0)
{
}

// Returns true if the given pointer is already known.
bool Base_pointer_serializer::is_known_pointer(void *obj) const
{
    return m_pointer_map.find(obj) != m_pointer_map.end();
}

// Create a new tag for the given pointer.
Tag_t Base_pointer_serializer::create_tag_for_pointer(void *obj)
{
    Tag_t tag = ++m_last_tag;
    m_pointer_map[obj] = tag;
    return tag;
}

// Get the tag for the given pointer.
Tag_t Base_pointer_serializer::get_tag_for_pointer(void *obj) const
{
    Pointer_map::const_iterator it = m_pointer_map.find(obj);
    if (it != m_pointer_map.end())
        return it->second;
    MDL_ASSERT(!"pointer tag not found");
    return Tag_t(0);
}

// Get or create the tag for the given pointer.
Tag_t Base_pointer_serializer::get_or_create_tag_for_pointer(void *obj)
{
    Pointer_map::const_iterator it = m_pointer_map.find(obj);
    if (it != m_pointer_map.end())
        return it->second;
    return create_tag_for_pointer(obj);
}

// Get the tag for the given pointer and delete the pointer from the pointer set.
Tag_t Base_pointer_serializer::get_tag_for_pointer_and_drop(void *obj)
{
    Pointer_map::iterator it = m_pointer_map.find(obj);
    if (it != m_pointer_map.end()) {
        Tag_t res = it->second;
        m_pointer_map.erase(it);
        return res;
    }
    MDL_ASSERT(!"pointer tag not found");
    return Tag_t(0);
}

// Constructor.
Base_pointer_deserializer::Base_pointer_deserializer(IAllocator *alloc)
: m_tag_map(0, Tag_map::hasher(), Tag_map::key_equal(), alloc)
{
}

// Returns true if the given tag is already known.
bool Base_pointer_deserializer::is_known_tag(Tag_t tag) const
{
    return m_tag_map.find(tag) != m_tag_map.end();
}

// Register a pointer for a given tag.
void Base_pointer_deserializer::register_pointer(Tag_t tag, void *obj)
{
    MDL_ASSERT(!is_known_tag(tag));
    m_tag_map[tag] = obj;
}

// Get the pointer for a given tag.
void *Base_pointer_deserializer::get_pointer(Tag_t tag) const
{
    return m_tag_map.find(tag)->second;
}

// --------------------- Base serializer ---------------------

// Write an int.
void Base_serializer::write_int(int v)
{
    size_t code = v < 0 ? ~(unsigned(v) << 1) : unsigned(v) << 1;

    write_encoded_tag(code);
}

// Write a float.
void Base_serializer::write_float(float v)
{
    // assume 32bit single float
    union {
        float f;
        byte  bytes[4];
    } u;

    u.f = v;

    // FIXME: not portable, byte order may vary
    for (size_t i = 0; i < 4; ++i)
        write(u.bytes[i]);
}

// Write a double.
void Base_serializer::write_double(double v)
{
    // assume 64bit double float
    union {
        double f;
        byte  bytes[8];
    } u;

    u.f = v;

    // FIXME: not portable, byte order may vary
    for (size_t i = 0; i < 8; ++i)
        write(u.bytes[i]);
}

// Write an MDL section tag.
void Base_serializer::write_section_tag(Serializer::Serializer_tags tag)
{
    // Tags are written as 32bit LE
    uint32_t v = tag;

    write(byte(v)); v >>= 8;
    write(byte(v)); v >>= 8;
    write(byte(v)); v >>= 8;
    write(byte(v)); v >>= 8;
}

// Write a (general) tag, assuming small values.
void Base_serializer::write_encoded_tag(size_t tag)
{
    if (tag < 0x80) {
        // one byte
        write(byte(tag));
        return;
    }
    if (tag < 0x4000) {
        write(byte(0x80 | (tag >> 8)));
        write(byte(tag));
        return;
    }
    if (tag < 0x20000000) {
        write(byte(0xC0 | (tag >> 24)));
        write(byte(tag >> 16));
        write(byte(tag >> 8));
        write(byte(tag));
        return;
    }
    // full range
    write(byte(0xE0));
#ifdef BIT64
    write(byte(tag >> 56));
    write(byte(tag >> 48));
    write(byte(tag >> 40));
    write(byte(tag >> 32));
#else
    // on 32bit, size_t is 32bit only
    write(byte(0));
    write(byte(0));
    write(byte(0));
    write(byte(0));
#endif
    write(byte(tag >> 24));
    write(byte(tag >> 16));
    write(byte(tag >> 8));
    write(byte(tag));
}

// Write a c-string, supports NULL pointer.
void Base_serializer::write_cstring(char const *s)
{
    if (s == NULL) {
        write_encoded_tag(0);
        return;
    }
    size_t len = strlen(s);

    write_encoded_tag(len + 1);
    for (size_t i = 0; i < len; ++i)
        write(s[i]);
}

// Write a DB::Tag.
void Base_serializer::write_db_tag(unsigned tag)
{
    write_int(tag);
}

// Read an int.
int Base_deserializer::read_int()
{
    size_t code = read_encoded_tag();

    return int(code & 1 ? ~(code >> 1) : code >> 1);
}

// Read a float.
float Base_deserializer::read_float()
{
    // assume 32bit single float
    union {
        float f;
        byte  bytes[4];
    } u;

    // FIXME: not portable, byte order may vary
    for (size_t i = 0; i < 4; ++i)
        u.bytes[i] = read();

    return u.f;
}

// Read a double.
double Base_deserializer::read_double()
{
    // assume 64bit double float
    union {
        double f;
        byte  bytes[8];
    } u;

    // FIXME: not portable, byte order may vary
    for (size_t i = 0; i < 8; ++i)
        u.bytes[i] = read();

    return u.f;
}

// Read an MDL section tag.
Serializer::Serializer_tags Base_deserializer::read_section_tag()
{
    uint32_t v = read();
    v |= read() << 8;
    v |= read() << 16;
    v |= read() << 24;

    return Serializer::Serializer_tags(v);
}

// Read a (general) tag, assuming small values.
size_t Base_deserializer::read_encoded_tag()
{
    size_t tag = 0;

    byte b = read();

    if (b < 0x80) {
        return b;
    }
    if (b < 0xC0) {
        tag  = (b & ~0x80) << 8;
        tag |= read();

        return tag;
    }
    if (b < 0xE0) {
        tag  = (b & ~0xC0) << 24;
        tag |= read() << 16;
        tag |= read() << 8;
        tag |= read();

        return tag;
    }

    // full range
    MDL_ASSERT(b == 0xE0);

    tag  = size_t(read()) << 56;
    tag |= size_t(read()) << 48;
    tag |= size_t(read()) << 40;
    tag |= size_t(read()) << 32;
    tag |= size_t(read()) << 24;
    tag |= size_t(read()) << 16;
    tag |= size_t(read()) << 8;
    tag |= size_t(read());

    return tag;
}

// Destructor.
Base_serializer::~Base_serializer()
{
}

// Read a c-string, supports NULL pointer.
char const *Base_deserializer::read_cstring()
{
    size_t len = read_encoded_tag();

    if (len == 0) {
        // the NULL pointer
        return NULL;
    }

    if (m_len < len) {
        if (m_string_buf != NULL)
            m_alloc->free(m_string_buf);
        m_string_buf = reinterpret_cast<char *>(m_alloc->malloc(len));
        m_len = len;
    }

    --len;
    m_string_buf[len] = '\0';
    for (size_t i = 0; i < len; ++i) {
        m_string_buf[i] = char(read());
    }
    return m_string_buf;
}

// Reads a DB::Tag 32bit encoding.
unsigned Base_deserializer::read_db_tag()
{
    return read_int();
}

// Constructor.
Base_deserializer::Base_deserializer(IAllocator *alloc)
: m_alloc(alloc)
, m_string_buf(NULL)
, m_len(0)
{
}

// Destructor.
Base_deserializer::~Base_deserializer()
{
    if (m_string_buf != NULL)
        m_alloc->free(m_string_buf);
}

// --------------------- Buffer serializer ---------------------

// Write a byte.
void Buffer_serializer::write(Byte b)
{
    if (m_next == m_end) {
        // reached end of current buffer

        // allocate a new header
        Header *h = reinterpret_cast<Header *>(m_alloc->malloc(sizeof(*h) - 1 + LOAD_SIZE));

        h->next = NULL;
        h->size = LOAD_SIZE;

        if (m_first == NULL)
            m_first = h;
        if (m_curr != NULL)
            m_curr->next = h;
        m_curr = h;

        m_next = h->load;
        m_end  = &h->load[h->size];
    }
    *m_next = b;
    ++m_next;
    ++m_size;
}

// Constructor.
Buffer_serializer::Buffer_serializer(IAllocator *alloc)
: Base()
, m_alloc(alloc)
, m_first(NULL)
, m_curr(NULL)
, m_next(NULL)
, m_end(NULL)
, m_size(0)
{
}

// Destructor.
Buffer_serializer::~Buffer_serializer()
{
    for (Header *next, *h = m_first; h != NULL; h = next) {
        next = h->next;

        m_alloc->free(h);
    }
}

// Get the data stream.
Buffer_serializer::Byte const *Buffer_serializer::get_data() const
{
    if (m_size == 0) {
        return NULL;
    }
    if (m_size > m_first->size) {
        // data is in several headers, combine into one
        Header *dst = reinterpret_cast<Header *>(m_alloc->malloc(sizeof(*dst) - 1 + m_size));

        dst->next = NULL;
        dst->size = m_size;

        Header *h = m_first;
        Byte   *p = dst->load;
        for (size_t s = m_size; s > 0;) {
            size_t c = s < h->size ? s : h->size;

            memcpy(p, h->load, c);
            p += c;
            s -= c;
            Header *n = h->next;
            m_alloc->free(h);
            h = n;
        }
        m_first = m_curr = dst;
        m_next  = m_end  = &dst->load[dst->size];
    }
    // all data is now in the first header
    return m_first->load;
}

// Read a byte.
byte Buffer_deserializer::read()
{
    if (m_data < m_end) {
        return *m_data++;
    }
    return 0;
}

// Constructor.
Buffer_deserializer::Buffer_deserializer(
    IAllocator *alloc,
    Byte const *data,
    size_t     size)
: Base(alloc)
, m_data(data)
, m_end(data + size)
{
}

// Destructor.
Buffer_deserializer::~Buffer_deserializer()
{
    MDL_ASSERT(m_data == m_end && "not all data has been consumed");
}

// --------------------- Stream serializer ---------------------

// Write a byte.
void Stream_serializer::write(Byte b)
{
    m_os->write_char(char(b));
}

// Constructor.
Stream_serializer::Stream_serializer(IOutput_stream *os)
: Base()
, m_os(mi::base::make_handle_dup(os))
{
}

// Read a byte.
byte Stream_deserializer::read()
{
    return byte(m_is->read_char());
}

// Constructor.
Stream_deserializer::Stream_deserializer(IAllocator *alloc, IInput_stream *is)
: Base(alloc)
, m_is(mi::base::make_handle_dup(is))
{
}

// --------------------- Entity serializer ---------------------

// Constructor.
Entity_serializer::Entity_serializer(
    IAllocator  *alloc,
    ISerializer *serializer)
: m_alloc(alloc)
, m_serializer(serializer)
{
}

// --------------------- Binary serializer ---------------------

// Constructor.
MDL_binary_serializer::MDL_binary_serializer(
    IAllocator  *alloc,
    MDL const   *compiler,
    ISerializer *serializer)
: Entity_serializer(alloc, serializer)
, m_modules(alloc)
, m_id_map(0, Id_map::hasher(), Id_map::key_equal(), alloc)
{
    // register all builtin modules. When this happens, the module
    // tag set must be empty, check that.
    Tag_t t, check;

    check = Tag_t(0);

    for (size_t i = 0, n = compiler->get_builtin_module_count(); i < n; ++i) {
        Module const *builtin_mod = compiler->get_builtin_module(i);

        t = register_module(builtin_mod);
        MDL_ASSERT(t == ++check);
    }
}

// Check if the given module is known by this binary serializer.
bool MDL_binary_serializer::is_known_module(Module const *mod) const
{
    return m_modules.is_known(mod);
}

// Register a module.
Tag_t MDL_binary_serializer::register_module(Module const *mod)
{
    m_id_map[mod->get_unique_id()] = mod;
    return m_modules.create_tag(mod);
}

// Get the tag for a known module.
Tag_t MDL_binary_serializer::get_module_tag(Module const *mod) const
{
    Tag_t tag = m_modules.get_tag(mod);
    MDL_ASSERT(tag != 0);
    return tag;
}

// Get the tag for a known module ID.
Tag_t MDL_binary_serializer::get_module_tag(size_t id) const
{
    Id_map::const_iterator it = m_id_map.find(id);
    MDL_ASSERT(it != m_id_map.end() && "Could not find module ID in binary serializer");
    Module const *mod = it->second;
    return get_module_tag(mod);
}

// --------------------- Factory serializer ---------------------

// Put the given type into the type wait queue.
void Factory_serializer::push_type(IType const *type)
{
    if (m_building_root_set) {
        // build the estimated root set
        if (m_child_types.find(type) == m_child_types.end())
            m_type_root_queue.push_back(type);
    } else {
        MDL_ASSERT(
            std::find(m_type_queue.begin(), m_type_queue.end(), type) == m_type_queue.end());
        m_type_queue.push_back(type);
    }
}

// Mark a type as used by an compound type.
void Factory_serializer::mark_child_type(IType const *type)
{
    if (m_building_root_set) {
        m_child_types.insert(type);
    }
}

// Process a type and its sub-types in DFS order.
void Factory_serializer::dfs_type(IType const *type, bool is_child_type)
{
    if (is_child_type)
        mark_child_type(type);

    if (!m_types_visited.insert(type).second) {
        // type was already in the set
        return;
    }

    switch (type->get_kind()) {
    case IType::TK_BOOL:
    case IType::TK_INT:
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
    case IType::TK_COLOR:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        // these types are builtin, no need to serialize them
        break;

    case IType::TK_ALIAS:
        {
            IType_alias const *a_type = cast<IType_alias>(type);

            dfs_type(a_type->get_aliased_type());
            push_type(a_type);
        }
        break;

    case IType::TK_ENUM:
        push_type(type);
        break;

    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);

            dfs_type(a_type->get_element_type());
            push_type(a_type);
        }
        break;
    case IType::TK_FUNCTION:
        {
            IType_function const *f_type = cast<IType_function>(type);

            if (IType const *ret_type = f_type->get_return_type())
                dfs_type(ret_type);

            for (int i = 0, n = f_type->get_parameter_count(); i < n; ++i) {
                IType const   *p_type;
                ISymbol const *p_sym;

                f_type->get_parameter(i, p_type, p_sym);

                dfs_type(p_type);
            }
            push_type(f_type);
        }
        break;
    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);

            for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                IType const   *f_type;
                ISymbol const *f_sym;

                s_type->get_field(i, f_type, f_sym);

                dfs_type(f_type);
            }
            for (int i = 0, n = s_type->get_method_count(); i < n; ++i) {
                IType_function const *m_type;
                ISymbol const        *m_sym;

                s_type->get_method(i, m_type, m_sym);

                dfs_type(m_type);
            }
            push_type(s_type);
        }
        break;
    }
}

// Process a type and its sub-types in DFS order.
void Factory_serializer::enqueue_type(IType const *type)
{
    dfs_type(type, /*is_child_type=*/false);
}

// Put the given value into the value wait queue.
void Factory_serializer::push_value(IValue const *value)
{
    if (m_building_root_set) {
        // build the estimated root set
        if (m_child_values.find(value) == m_child_values.end())
            m_value_root_queue.push_back(value);
    } else {
        MDL_ASSERT(m_value_queue_marker.insert(value).second == true);
        m_value_queue.push_back(value);
    }
}

// Mark a value as used by an compound value.
void Factory_serializer::mark_child_value(IValue const *value)
{
    if (m_building_root_set) {
        m_child_values.insert(value);
    }
}

// Process a value and its sub-values in DFS order.
void Factory_serializer::dfs_value(IValue const *v, bool is_child_value)
{
    if (is_child_value)
        mark_child_value(v);

    if (!m_values_visited.insert(v).second) {
        // value was already in the set
        return;
    }

    switch (v->get_kind()) {
    case IValue::VK_BAD:
    case IValue::VK_BOOL:
        // these values are builtin, no need to serialize them
        break;

    case IValue::VK_INT:
    case IValue::VK_ENUM:
    case IValue::VK_FLOAT:
    case IValue::VK_DOUBLE:
    case IValue::VK_STRING:
    case IValue::VK_INVALID_REF:
    case IValue::VK_TEXTURE:
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        // these values are atomic
        push_value(v);
        break;

    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            // these values are compound
            IValue_compound const *compound = cast<IValue_compound>(v);

            for (int i = 0, n = compound->get_component_count(); i < n; ++i) {
                IValue const *e = compound->get_value(i);

                dfs_value(e);
            }
            push_value(compound);
        }
        break;
    }
}

// Enqueue the given value for processing.
void Factory_serializer::enqueue_value(IValue const *v)
{
    dfs_value(v, /*is_child_value=*/false);
}

namespace {

/// Helper class to compare ITypes.
struct IType_less {
    bool operator() (IType const *s, IType const *t)
    {
        if (s == t)
            return false;

        IType::Kind s_kind = s->get_kind();
        IType::Kind t_kind = t->get_kind();

        if (s_kind == t_kind) {
            switch (s_kind) {
            case IType::TK_BOOL:
            case IType::TK_INT:
            case IType::TK_FLOAT:
            case IType::TK_DOUBLE:
            case IType::TK_STRING:
            case IType::TK_LIGHT_PROFILE:
            case IType::TK_BSDF:
            case IType::TK_HAIR_BSDF:
            case IType::TK_EDF:
            case IType::TK_VDF:
            case IType::TK_COLOR:
            case IType::TK_BSDF_MEASUREMENT:
            case IType::TK_INCOMPLETE:
            case IType::TK_ERROR:
                // these types are singletons, so it should never happen
                // that two instances are compared
                MDL_ASSERT(!"singleton type occured more than once in list");
                return false;

            case IType::TK_VECTOR:
                {
                    IType_vector const *v_s = cast<IType_vector>(s);
                    IType_vector const *v_t = cast<IType_vector>(t);

                    int s_s = v_s->get_size();
                    int s_t = v_t->get_size();

                    if (s_s == s_t) {
                        IType const *e_s = v_s->get_element_type();
                        IType const *e_t = v_t->get_element_type();

                        return operator()(e_s, e_t);
                    }
                    return s_s < s_t;
                }
            case IType::TK_MATRIX:
                {
                    IType_matrix const *m_s = cast<IType_matrix>(s);
                    IType_matrix const *m_t = cast<IType_matrix>(t);

                    int c_s = m_s->get_columns();
                    int c_t = m_t->get_columns();

                    if (c_s == c_t) {
                        IType const *e_s = m_s->get_element_type();
                        IType const *e_t = m_t->get_element_type();

                        return operator()(e_s, e_t);
                    }
                    return c_s < c_t;
                }

            case IType::TK_ALIAS:
                {
                    IType_alias const *a_s = cast<IType_alias>(s);
                    IType_alias const *a_t = cast<IType_alias>(t);

                    ISymbol const *s_s = a_s->get_symbol();
                    ISymbol const *s_t = a_t->get_symbol();
                    if (s_s == NULL) {
                        if (s_t == NULL) {
                            // both aliases have no name, check modifiers

                            IType::Modifiers m_s = a_s->get_type_modifiers();
                            IType::Modifiers m_t = a_t->get_type_modifiers();

                            if (m_s == m_t)
                                return operator()(
                                    a_s->get_aliased_type(), a_t->get_aliased_type());
                            return m_s < m_t;
                        } else {
                            // t has a name, s < t ==> true
                            return true;
                        }
                    } else {
                        // s has a name
                        if (s_t == NULL) {
                            // t has NO name, s < t ==> false
                            return false;
                        } else {
                            // both have a name
                            return strcmp(s_s->get_name(), s_t->get_name()) < 0;
                        }
                    }
                }

            case IType::TK_ENUM:
                {
                    IType_enum const *e_s = cast<IType_enum>(s);
                    IType_enum const *e_t = cast<IType_enum>(t);

                    // compare the names
                    return strcmp(
                        e_s->get_symbol()->get_name(), e_t->get_symbol()->get_name()) < 0;
                }

            case IType::TK_ARRAY:
                {
                    IType_array const *a_s = cast<IType_array>(s);
                    IType_array const *a_t = cast<IType_array>(t);

                    bool is_imm_sized = a_s->is_immediate_sized();

                    if (is_imm_sized == a_t->is_immediate_sized()) {
                        if (is_imm_sized) {
                            // both are immediate sized
                            int s_size = a_s->get_size();
                            int t_size = a_t->get_size();

                            if (s_size == t_size) {
                                // same size, compare the element types
                                return operator()(
                                    a_s->get_element_type(), a_t->get_element_type());
                            }
                            return s_size < t_size;
                        } else {
                            // both are deferred sized
                            IType_array_size const *as_s = a_s->get_deferred_size();
                            IType_array_size const *as_t = a_t->get_deferred_size();

                            if (as_s == as_t) {
                                // same size, compare the element types
                                return operator()(
                                    a_s->get_element_type(), a_t->get_element_type());
                            }
                            // compare the absolute size symbol name
                            return strcmp(
                                as_s->get_name()->get_name(), as_t->get_name()->get_name()) < 0;
                        }
                    }
                    // the immediate sized is smaller then the deferred sized
                    return is_imm_sized;
                }

            case IType::TK_FUNCTION:
                {
                    IType_function const *f_s = cast<IType_function>(s);
                    IType_function const *f_t = cast<IType_function>(t);

                    int n_param_s = f_s->get_parameter_count();
                    int n_param_t = f_t->get_parameter_count();

                    if (n_param_s == n_param_t) {
                        // same number of parameters, check return types
                        IType const *rt_s = f_s->get_return_type();
                        IType const *rt_t = f_t->get_return_type();

                        if (rt_s == NULL) {
                            if (rt_t != NULL) {
                                // s < t ==> true
                                return true;
                            }
                            // else both are NULL
                        } else {
                            if (rt_t == NULL) {
                                // s < t ==> false
                                return false;
                            }
                            if (rt_s != rt_t) {
                                return operator()(rt_s, rt_t);
                            }
                            // both types are equal
                        }

                        for (int i = 0; i < n_param_s; ++i) {
                            IType   const *pt_s;
                            ISymbol const *ps_s;

                            f_s->get_parameter(i, pt_s, ps_s);

                            IType   const *pt_t;
                            ISymbol const *ps_t;

                            f_t->get_parameter(i, pt_t, ps_t);

                            if (pt_s != pt_t) {
                                return operator()(pt_s, pt_t);
                            }
                            if (ps_s != ps_t) {
                                return strcmp(ps_s->get_name(), ps_t->get_name()) < 0;
                            }
                        }
                        // should not happen
                        MDL_ASSERT(!"equal function types detected");
                        return false;
                    }
                    return n_param_s < n_param_t;
                }

            case IType::TK_STRUCT:
                {
                    IType_struct const *s_s = cast<IType_struct>(s);
                    IType_struct const *s_t = cast<IType_struct>(t);

                    int n_s = s_s->get_field_count();
                    int n_t = s_t->get_field_count();

                    if (n_s == n_t) {
                        // compare the names, these are absolute
                        return strcmp(
                            s_s->get_symbol()->get_name(), s_t->get_symbol()->get_name()) < 0;
                    }
                    return n_s < n_t;
                }

            case IType::TK_TEXTURE:
                {
                    IType_texture const *t_s = cast<IType_texture>(s);
                    IType_texture const *t_t = cast<IType_texture>(t);

                    IType_texture::Shape shape_s = t_s->get_shape();
                    IType_texture::Shape shape_t = t_t->get_shape();

                    if (shape_s == shape_t) {
                        IType const *c_s = t_s->get_coord_type();
                        IType const *c_t = t_t->get_coord_type();

                        MDL_ASSERT(c_s != c_t && "equal texture types detected");
                        return operator()(c_s, c_t);
                    }
                    return shape_s < shape_t;
                }
            }
        }
        return s_kind < t_kind;
    }
};

/// Helper class to compare IValues.
struct IValue_less {
    bool operator() (IValue const *v, IValue const *w)
    {
        if (v == w)
            return false;

        IValue::Kind v_kind = v->get_kind();
        IValue::Kind w_kind = w->get_kind();

        if (v_kind == w_kind) {
            switch (v_kind) {
            case IValue::VK_BAD:
                // singleton value, should not happen
                MDL_ASSERT(!"more than one value bad detected");
                return false;

            case IValue::VK_BOOL:
                {
                    IValue_bool const *b_v = cast<IValue_bool>(v);
                    MDL_ASSERT(
                        b_v->get_value() != cast<IValue_bool>(w)->get_value() &&
                        "two equal bool values detected");

                    // false < true
                    return !b_v->get_value();
                }

            case IValue::VK_INT:
                {
                    IValue_int const *i_v = cast<IValue_int>(v);
                    IValue_int const *i_w = cast<IValue_int>(w);

                    return i_v->get_value() < i_w->get_value();
                }

            case IValue::VK_ENUM:
                {
                    IValue_enum const *e_v = cast<IValue_enum>(v);
                    IValue_enum const *e_w = cast<IValue_enum>(w);

                    IType_enum const *t_v = e_v->get_type();
                    IType_enum const *t_w = e_w->get_type();

                    if (t_v == t_w)
                        return e_v->get_value() < e_w->get_value();

                    IType_less type_less;
                    return type_less(t_v, t_w);
                }

            case IValue::VK_FLOAT:
                {
                    IValue_float const *f_v = cast<IValue_float>(v);
                    IValue_float const *f_w = cast<IValue_float>(w);

                    return f_v->get_value() < f_w->get_value();
                }

            case IValue::VK_DOUBLE:
                {
                    IValue_double const *d_v = cast<IValue_double>(v);
                    IValue_double const *d_w = cast<IValue_double>(w);

                    return d_v->get_value() < d_w->get_value();
                }

            case IValue::VK_STRING:
                {
                    IValue_string const *d_v = cast<IValue_string>(v);
                    IValue_string const *d_w = cast<IValue_string>(w);

                    return strcmp(d_v->get_value(), d_w->get_value()) < 0;
                }

            case IValue::VK_VECTOR:
            case IValue::VK_MATRIX:
            case IValue::VK_ARRAY:
            case IValue::VK_RGB_COLOR:
            case IValue::VK_STRUCT:
                {
                    IValue_compound const *c_v = cast<IValue_compound>(v);
                    IValue_compound const *c_w = cast<IValue_compound>(w);

                    int v_cnt = c_v->get_component_count();
                    int w_cnt = c_w->get_component_count();

                    if (v_cnt == w_cnt) {
                        IType const *t_v = c_v->get_type();
                        IType const *t_w = c_w->get_type();

                        if (t_v == t_w) {
                            for (int i = 0; i < v_cnt; ++i) {
                                IValue const *v_child = c_v->get_value(i);
                                IValue const *w_child = c_w->get_value(i);

                                if (v_child != w_child) {
                                    return operator()(v_child, w_child);
                                }
                            }
                            MDL_ASSERT(!"different compound values with equal childs detected");
                            return false;
                        }
                        IType_less type_less;
                        return type_less(t_v, t_w);
                    }
                    return v_cnt < w_cnt;
                }

            case IValue::VK_INVALID_REF:
                {
                    IType const *t_v = v->get_type();
                    IType const *t_w = w->get_type();

                    MDL_ASSERT(
                        t_v != t_w && "different invalid ref values with same type detected");

                    IType_less type_less;
                    return type_less(t_v, t_w);
                }

            case IValue::VK_TEXTURE:
                {
                    IValue_texture const *r_v = cast<IValue_texture>(v);
                    IValue_texture const *r_w = cast<IValue_texture>(w);

                    IType const *t_v = r_v->get_type();
                    IType const *t_w = r_w->get_type();

                    if (t_v != t_w) {
                        IType_less type_less;
                        return type_less(t_v, t_w);
                    }

                    int res = strcmp(r_v->get_string_value(), r_w->get_string_value());
                    if (res != 0)
                        return res < 0;

                    IValue_texture::gamma_mode g_v = r_v->get_gamma_mode();
                    IValue_texture::gamma_mode g_w = r_w->get_gamma_mode();

                    if (g_v != g_w)
                        return g_v < g_w;

                    int tag_v = r_v->get_tag_value();
                    int tag_w = r_w->get_tag_value();
                    if (tag_v != tag_w)
                        return tag_v < tag_w;
                    return r_v->get_bsdf_data_kind() < r_w->get_bsdf_data_kind();
                }

            case IValue::VK_LIGHT_PROFILE:
            case IValue::VK_BSDF_MEASUREMENT:
                {
                    IValue_resource const *r_v = cast<IValue_resource>(v);
                    IValue_resource const *r_w = cast<IValue_resource>(w);

                    IType const *t_v = r_v->get_type();
                    IType const *t_w = r_w->get_type();

                    if (t_v != t_w) {
                        IType_less type_less;
                        return type_less(t_v, t_w);
                    }

                    int res = strcmp(r_v->get_string_value(), r_w->get_string_value());
                    if (res != 0)
                        return res < 0;

                    return r_v->get_tag_value() < r_w->get_tag_value();
                }
            }
        }
        return v_kind < w_kind;
    }
};

}  // anonymous

// Write all enqueued types.
void Factory_serializer::write_enqueued_types()
{
    // build the root set from the estimated root set
    Type_queue type_root_queue(get_allocator());
    for (Type_queue::const_iterator it(m_type_root_queue.begin()), end(m_type_root_queue.end());
        it != end;
        ++it)
    {
        IType const *type = *it;

        if (m_child_types.find(type) == m_child_types.end())
            type_root_queue.push_back(type);
    }
    m_type_root_queue.swap(type_root_queue);

#ifndef NO_MDL_SERIALIZATION_SORT
    std::sort(m_type_root_queue.begin(), m_type_root_queue.end(), IType_less());
#endif

    // create the queue from the root set
    m_building_root_set = false;
    m_types_visited.clear();

    for (Type_queue::const_iterator it(m_type_root_queue.begin()), end(m_type_root_queue.end());
        it != end;
        ++it)
    {
        IType const *root = *it;

        dfs_type(root, /*is_child_type=*/false);
    }

    // ready
    m_building_root_set = true;
    m_type_root_queue.clear();
    m_types_visited.clear();

    size_t n_types = m_type_queue.size();
    write_encoded_tag(n_types);
    DOUT(("#types %u\n", unsigned(n_types)));
    INC_SCOPE();

    size_t i = 0;
    for (Type_queue::const_iterator it(m_type_queue.begin()), end(m_type_queue.end());
        it != end;
        ++it)
    {
        IType const *type = *it;

        write_type(type);
        ++i;
    }

    // clear the queue and visit set
    m_types_visited.clear();
    m_type_queue.clear();

    DEC_SCOPE();
}

// Write all enqueued values.
void Factory_serializer::write_enqueued_values()
{
    // build the root set from the estimated root set
    Value_queue value_root_queue(get_allocator());
    for (Value_queue::const_iterator it(m_value_root_queue.begin()), end(m_value_root_queue.end());
        it != end;
        ++it)
    {
        IValue const *value = *it;

        if (m_child_values.find(value) == m_child_values.end())
            value_root_queue.push_back(value);
    }
    m_value_root_queue.swap(value_root_queue);

#ifndef NO_MDL_SERIALIZATION_SORT
    std::sort(m_value_root_queue.begin(), m_value_root_queue.end(), IValue_less());
#endif

    // create the queue from the root set
    m_building_root_set = false;
    m_values_visited.clear();

    for (Value_queue::const_iterator it(m_value_root_queue.begin()), end(m_value_root_queue.end());
        it != end;
        ++it)
    {
        IValue const *root = *it;

        dfs_value(root, /*is_child_value=*/false);
    }

    // ready
    m_building_root_set = true;
    m_value_root_queue.clear();
    m_values_visited.clear();

    // write the number of values
    size_t n_values = m_value_queue.size();
    write_encoded_tag(n_values);

    DOUT(("#values %u\n", unsigned(n_values)));
    INC_SCOPE();

    for (Value_queue::const_iterator it(m_value_queue.begin()), end(m_value_queue.end());
        it != end;
        ++it)
    {
        IValue const *value = *it;

        write_value(value);
    }
    DEC_SCOPE();

    // clear the queue and visit set
    m_values_visited.clear();
    m_value_queue.clear();
}

/// Helper, write the IValue header.
static void write_value_header(Factory_serializer *s, IValue const *v)
{
    Tag_t value_tag = s->register_value(v);
    s->write_encoded_tag(value_tag);

    DOUT(("value tag %u\n", unsigned(value_tag)));

    IType const *t = v->get_type();
    Tag_t type_tag = s->get_type_tag(t);
    s->write_encoded_tag(type_tag);

    DOUT(("type tag %u\n", unsigned(type_tag)));
}

// Write the given value.
void Factory_serializer::write_value(IValue const *v)
{
    IValue::Kind kind = v->get_kind();
    write_encoded_tag(kind);

    switch (kind) {
    case IValue::VK_BAD:
        DOUT(("value BAD\n"));
        INC_SCOPE();
        {
            // these values are builtin, no need to serialize them, just reference its tag
            Tag_t value_tag = get_value_tag(v);
            write_encoded_tag(value_tag);

            DOUT(("value tag %u\n", unsigned(value_tag)));
            DEC_SCOPE();
        }
        break;
    case IValue::VK_BOOL:
        DOUT(("value BOOL\n"));
        INC_SCOPE();
        {
            // these values are builtin, no need to serialize them, just reference its tag
            Tag_t value_tag = get_value_tag(v);
            write_encoded_tag(value_tag);

            DOUT(("value tag %u\n", unsigned(value_tag)));
        }
        DEC_SCOPE();
        break;

    case IValue::VK_INT:
        DOUT(("value INT\n"));
        INC_SCOPE();
        {
            IValue_int const *iv = cast<IValue_int>(v);
            write_value_header(this, iv);

            int vv = iv->get_value();
            write_int(vv);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_ENUM:
        DOUT(("value ENUM\n"));
        INC_SCOPE();
        {
            IValue_enum const *ev = cast<IValue_enum>(v);
            write_value_header(this, ev);

            int idx = ev->get_index();
            write_int(idx);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_FLOAT:
        DOUT(("value FLOAT\n"));
        INC_SCOPE();
        {
            IValue_float const *fv = cast<IValue_float>(v);
            write_value_header(this, fv);

            float vv = fv->get_value();
            write_float(vv);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_DOUBLE:
        DOUT(("value DOUBLE\n"));
        INC_SCOPE();
        {
            IValue_double const *dv = cast<IValue_double>(v);
            write_value_header(this, dv);

            double vv = dv->get_value();
            write_double(vv);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_STRING:
        DOUT(("value STRING\n"));
        INC_SCOPE();
        {
            IValue_string const *sv = cast<IValue_string>(v);
            write_value_header(this, sv);

            char const *vv = sv->get_value();
            write_cstring(vv);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_INVALID_REF:
        DOUT(("value INVALID_REF\n"));
        INC_SCOPE();
        {
            IValue_invalid_ref const *iv = cast<IValue_invalid_ref>(v);
            write_value_header(this, iv);
        }
        DEC_SCOPE();
        break;

    case IValue::VK_TEXTURE:
        DOUT(("value TEXTURE\n"));
        INC_SCOPE();
        {
            IValue_texture const *rv = cast<IValue_texture>(v);
            write_value_header(this, rv);

            char const *vv = rv->get_string_value();
            write_cstring(vv);
            write_db_tag(rv->get_tag_value());
            write_unsigned(rv->get_tag_version());
            write_int(rv->get_gamma_mode());
            write_int(rv->get_bsdf_data_kind());
        }
        DEC_SCOPE();
        break;

    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        DOUT(("value %s\n",
            kind == IValue::VK_LIGHT_PROFILE ? "LIGHT_PROFILE" : "BSDF_MEASUREMENT"));
        INC_SCOPE();
        {
            IValue_resource const *rv = cast<IValue_resource>(v);
            write_value_header(this, rv);

            char const *vv = rv->get_string_value();
            write_cstring(vv);
            write_db_tag(rv->get_tag_value());
            write_unsigned(rv->get_tag_version());
        }
        DEC_SCOPE();
        break;

    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        DOUT(("value COMPOUND\n"));
        INC_SCOPE();
        {
            // these values are compound
            IValue_compound const *compound = cast<IValue_compound>(v);

            write_value_header(this, compound);

            int n = compound->get_component_count();
            write_unsigned(n);

            DOUT(("#sub-values %d\n", n));

            for (int i = 0; i < n; ++i) {
                IValue const *e = compound->get_value(i);

                Tag_t e_tag = get_value_tag(e);
                write_encoded_tag(e_tag);

                DOUT(("#subvalue %u\n", unsigned(e_tag)));
            }
        }
        DEC_SCOPE();
        break;
    }
}

// Write the given type.
void Factory_serializer::write_type(IType const *type)
{
    IType::Kind kind = type->get_kind();
    write_encoded_tag(kind);

    DOUT(("type kind %u\n", kind));

    switch (kind) {
    case IType::TK_BOOL:
    case IType::TK_INT:
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
    case IType::TK_COLOR:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        {
            // these types are builtin, no need to serialize them, just reference its tag
            Tag_t type_tag = get_type_tag(type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));
        }
        break;

    case IType::TK_ALIAS:
        {
            IType_alias const *a_type = cast<IType_alias>(type);
            IType const       *e_type = a_type->get_aliased_type();

            Tag_t type_tag = register_type(a_type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));

            // write the name if any
            if (ISymbol const *sym = a_type->get_symbol()) {
                write_bool(true);
                Tag_t sym_tag = get_symbol_tag(sym);
                write_encoded_tag(sym_tag);

                DOUT(("type name %u (%s)\n", unsigned(sym_tag), sym->get_name()));
            } else {
                write_bool(false);
            }

            // write the aliased type
            Tag_t e_type_tag = get_type_tag(e_type);
            write_encoded_tag(e_type_tag);

            DOUT(("type alias %u\n", unsigned(e_type_tag)));

            // write the modifiers
            IType::Modifiers mod = a_type->get_type_modifiers();
            write_encoded_tag(mod);

            DOUT(("type modifiers %u\n", unsigned(mod)));
        }
        break;

    case IType::TK_ENUM:
        {
            IType_enum const *e_type = cast<IType_enum>(type);

            Tag_t type_tag = register_type(e_type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_enum::Predefined_id pid = e_type->get_predefined_id();
            if (pid == IType_enum::EID_USER) {
                // a user defined enum
                write_bool(false);

                // write the type name
                ISymbol const *sym = e_type->get_symbol();
                Tag_t sym_tag = get_symbol_tag(sym);
                write_encoded_tag(sym_tag);

                DOUT(("type name %u (%s)\n", unsigned(sym_tag), sym->get_name()));

                int n = e_type->get_value_count();
                write_unsigned(n);

                DOUT(("#values %d\n", n));
                INC_SCOPE();

                for (int i = 0; i < n; ++i) {
                    ISymbol const *sym;
                    int           code;

                    e_type->get_value(i, sym, code);

                    Tag_t sym_tag = get_symbol_tag(sym);
                    write_encoded_tag(sym_tag);
                    write_int(code);

                    DOUT(("value sym_tag %u code %d (%s)\n",
                        unsigned(sym_tag), code, sym->get_name()));
                }
                DEC_SCOPE();
            } else {
                // a predefined type
                write_bool(true);
                write_encoded_tag(pid);
            }
        }
        break;

    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const       *e_type = a_type->get_element_type();

            Tag_t type_tag = register_type(a_type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));

            // write the element type
            Tag_t e_type_tag = get_type_tag(e_type);
            write_encoded_tag(e_type_tag);

            DOUT(("e_type %u\n", unsigned(e_type_tag)));

            if (a_type->is_immediate_sized()) {
                write_bool(true);
                write_unsigned(a_type->get_size());

                DOUT(("imm size %d\n", a_type->get_size()));
            } else {
                write_bool(false);
                Tag_t size_tag = get_array_size_tag(a_type->get_deferred_size());
                write_encoded_tag(size_tag);

                DOUT(("def size %u\n", unsigned(size_tag)));
            }
        }
        break;

    case IType::TK_FUNCTION:
        {
            IType_function const *f_type = cast<IType_function>(type);

            Tag_t type_tag = register_type(f_type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));

            // write the return type
            if (IType const *ret_type = f_type->get_return_type()) {
                write_bool(true);
                Tag_t ret_type_tag = get_type_tag(ret_type);
                write_encoded_tag(ret_type_tag);

                DOUT(("ret_type %u\n", unsigned(ret_type_tag)));
            } else {
                write_bool(false);
            }

            // write parameter count
            int n = f_type->get_parameter_count();
            write_unsigned(n);

            DOUT(("#params %d\n", n));
            INC_SCOPE();

            // write parameters
            for (int i = 0; i < n; ++i) {
                IType const   *p_type;
                ISymbol const *p_sym;

                f_type->get_parameter(i, p_type, p_sym);

                Tag_t p_type_tag = get_type_tag(p_type);
                write_encoded_tag(p_type_tag);
                Tag_t p_sym_tag = get_symbol_tag(p_sym);
                write_encoded_tag(p_sym_tag);

                DOUT(("param %u %u (%s)\n",
                    unsigned(p_type_tag), unsigned(p_sym_tag), p_sym->get_name()));
            }
            DEC_SCOPE();
        }
        break;

    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);

            Tag_t type_tag = register_type(s_type);
            write_encoded_tag(type_tag);

            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_struct::Predefined_id pid = s_type->get_predefined_id();
            if (pid == IType_struct::SID_USER) {
                // a user defined struct
                write_bool(false);

                // write the struct name
                ISymbol const *sym = s_type->get_symbol();
                Tag_t sym_tag = get_symbol_tag(sym);
                write_encoded_tag(sym_tag);

                DOUT(("name %u (%s)\n", unsigned(sym_tag), sym->get_name()));

                // write the field count
                int n = s_type->get_field_count();
                write_unsigned(n);

                DOUT(("#fields %d\n", n));
                INC_SCOPE();

                // write the fields
                for (int i = 0; i < n; ++i) {
                    IType const   *f_type;
                    ISymbol const *f_sym;

                    s_type->get_field(i, f_type, f_sym);

                    Tag_t f_type_tag = get_type_tag(f_type);
                    write_encoded_tag(f_type_tag);
                    Tag_t f_sym_tag = get_symbol_tag(f_sym);
                    write_encoded_tag(f_sym_tag);

                    DOUT(("field %u %u (%s)\n",
                        unsigned(f_type_tag), unsigned(f_sym_tag), f_sym->get_name()));
                }
                DEC_SCOPE();

                // write the method count
                n = s_type->get_method_count();
                write_unsigned(n);

                DOUT(("#methods %d\n", n));
                INC_SCOPE();

                // write the methods
                for (int i = 0; i < n; ++i) {
                    IType_function const *m_type;
                    ISymbol const        *m_sym;

                    s_type->get_method(i, m_type, m_sym);

                    Tag_t m_type_tag = get_type_tag(m_type);
                    write_encoded_tag(m_type_tag);
                    Tag_t m_sym_tag = get_symbol_tag(m_sym);
                    write_encoded_tag(m_sym_tag);

                    DOUT(("method %u %u (%s)\n",
                        unsigned(m_type_tag), unsigned(m_sym_tag), m_sym->get_name()));
                }
                DEC_SCOPE();
            } else {
                // a predefined type
                write_bool(true);
                write_encoded_tag(pid);
            }
        }
        break;
    }
}

// Constructor.
Factory_serializer::Factory_serializer(
    IAllocator            *alloc,
    ISerializer           *serializer,
    MDL_binary_serializer *bin_serializer)
: Entity_serializer(alloc, serializer)
, m_symbols(alloc)
, m_types(alloc)
, m_array_sizes(alloc)
, m_values(alloc)
, m_types_visited(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_child_types(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_type_root_queue(alloc)
, m_type_queue(alloc)
, m_values_visited(0, Value_set::hasher(), Value_set::key_equal(), alloc)
, m_child_values(0, Value_set::hasher(), Value_set::key_equal(), alloc)
, m_value_root_queue(alloc)
, m_value_queue(alloc)
, m_building_root_set(true)
, m_bin_serializer(bin_serializer)
#ifdef ENABLE_ASSERT
, m_value_queue_marker(0, Value_set::hasher(), Value_set::key_equal(), alloc)
#endif
{
}

// --------------------- Module serializer ---------------------

// Write an AST declaration
void Module_serializer::write_decl(
    IDeclaration const *decl)
{
    Tag_t decl_tag = register_declaration(decl);
    write_encoded_tag(decl_tag);
    DOUT(("decl %u\n", unsigned(decl_tag)));
    INC_SCOPE();

    IDeclaration::Kind kind = decl->get_kind();

    // write the kind
    write_unsigned(kind);

    bool exported = decl->is_exported();
    write_bool(exported);
    DOUT(("exported %u\n", unsigned(exported)));

    switch (kind) {
    case IDeclaration::DK_INVALID:
        // ready
        DOUT(("INVALID\n"));
        break;

    case IDeclaration::DK_IMPORT:
        DOUT(("Import_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_import const *i_decl = cast<IDeclaration_import>(decl);

            if (IQualified_name const *qname = i_decl->get_module_name()) {
                write_bool(true);
                write_name(qname);
            } else {
                write_bool(false);
            }

            int name_count = i_decl->get_name_count();
            write_unsigned(name_count);
            DOUT(("#names %d\n", name_count));
            INC_SCOPE();

            for (int i = 0; i < name_count; ++i) {
                IQualified_name const *name = i_decl->get_name(i);
                write_name(name);
            }
            DEC_SCOPE();
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_ANNOTATION:
        DOUT(("Anno_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_annotation const *a_decl = cast<IDeclaration_annotation>(decl);

            ISimple_name const *anno_name = a_decl->get_name();
            write_name(anno_name);

            IDefinition const *def = a_decl->get_definition();
            Tag_t def_tag = get_definition_tag(def);
            write_encoded_tag(def_tag);
            DOUT(("def %u\n", unsigned(def_tag)));

            int param_count = a_decl->get_parameter_count();
            write_unsigned(param_count);
            DOUT(("#params %d\n", param_count));
            INC_SCOPE();

            for (int i = 0; i < param_count; ++i) {
                IParameter const *param = a_decl->get_parameter(i);
                write_parameter(param);
            }
            DEC_SCOPE();

            IAnnotation_block const *annos = a_decl->get_annotations();
            write_annos(annos);
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_CONSTANT:
        DOUT(("Const_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_constant const *c_decl = cast<IDeclaration_constant>(decl);

            IType_name const *tname = c_decl->get_type_name();
            write_name(tname);

            int cst_count = c_decl->get_constant_count();
            write_unsigned(cst_count);
            DOUT(("#consts %d\n", cst_count));
            INC_SCOPE();

            for (int i = 0; i < cst_count; ++i) {
                ISimple_name const *sname = c_decl->get_constant_name(i);
                write_name(sname);

                IExpression const *expr = c_decl->get_constant_exp(i);
                write_expr(expr);

                IAnnotation_block const *annos = c_decl->get_annotations(i);
                write_annos(annos);
            }
            DEC_SCOPE();
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_ALIAS:
        DOUT(("Alias_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_type_alias const *a_decl = cast<IDeclaration_type_alias>(decl);

            IType_name const *tname = a_decl->get_type_name();
            write_name(tname);

            ISimple_name const *sname = a_decl->get_alias_name();
            write_name(sname);
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_STRUCT:
        DOUT(("Struct_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_type_struct const *s_decl = cast<IDeclaration_type_struct>(decl);

            ISimple_name const *sname = s_decl->get_name();
            write_name(sname);

            IAnnotation_block const *annos = s_decl->get_annotations();
            write_annos(annos);

            IDefinition const *def = s_decl->get_definition();
            Tag_t def_tag = get_definition_tag(def);
            write_encoded_tag(def_tag);
            DOUT(("def %u\n", unsigned(def_tag)));

            int f_count = s_decl->get_field_count();
            write_unsigned(f_count);
            DOUT(("#fields %d\n", f_count));
            INC_SCOPE();

            for (int i = 0; i < f_count; ++i) {
                IType_name const *tname = s_decl->get_field_type_name(i);
                write_name(tname);

                ISimple_name const *sname = s_decl->get_field_name(i);
                write_name(sname);

                if (IExpression const *expr = s_decl->get_field_init(i)) {
                    write_bool(true);
                    write_expr(expr);
                } else {
                    write_bool(false);
                }

                IAnnotation_block const *annos = s_decl->get_annotations(i);
                write_annos(annos);
            }
            DEC_SCOPE();
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_ENUM:
        DOUT(("Enum_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_type_enum const *e_decl = cast<IDeclaration_type_enum>(decl);

            ISimple_name const *sname = e_decl->get_name();
            write_name(sname);

            IAnnotation_block const *annos = e_decl->get_annotations();
            write_annos(annos);

            IDefinition const *def = e_decl->get_definition();
            Tag_t def_tag = get_definition_tag(def);
            write_encoded_tag(def_tag);
            DOUT(("def %u\n", unsigned(def_tag)));

            int v_count = e_decl->get_value_count();
            write_unsigned(v_count);
            DOUT(("#values %d\n", v_count));
            INC_SCOPE();

            for (int i = 0; i < v_count; ++i) {
                ISimple_name const *sname = e_decl->get_value_name(i);
                write_name(sname);

                if (IExpression const *expr = e_decl->get_value_init(i)) {
                    write_bool(true);
                    write_expr(expr);
                } else {
                    write_bool(false);
                }

                IAnnotation_block const *annos = e_decl->get_annotations(i);
                write_annos(annos);
            }
            DEC_SCOPE();
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_VARIABLE:
        DOUT(("Var_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_variable const *v_decl = cast<IDeclaration_variable>(decl);

            IType_name const *tname = v_decl->get_type_name();
            write_name(tname);

            int v_count = v_decl->get_variable_count();
            write_unsigned(v_count);
            DOUT(("#vars %d\n", v_count));
            INC_SCOPE();

            for (int i = 0; i < v_count; ++i) {
                ISimple_name const *sname = v_decl->get_variable_name(i);
                write_name(sname);

                if (IExpression const *expr = v_decl->get_variable_init(i)) {
                    write_bool(true);
                    write_expr(expr);
                } else {
                    write_bool(false);
                }

                IAnnotation_block const *annos = v_decl->get_annotations(i);
                write_annos(annos);
            }
            DEC_SCOPE();
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_FUNCTION:
        DOUT(("Func_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_function const *f_decl = cast<IDeclaration_function>(decl);

            IType_name const *tname = f_decl->get_return_type_name();
            write_name(tname);

            IAnnotation_block const *ret_annos = f_decl->get_return_annotations();
            write_annos(ret_annos);

            ISimple_name const *sname = f_decl->get_name();
            write_name(sname);

            bool b = f_decl->is_preset();
            write_bool(b);
            DOUT(("preset %u\n", unsigned(b)));

            if (IStatement const *stmt = f_decl->get_body()) {
                write_bool(true);
                DOUT(("body\n"));
                INC_SCOPE();
                write_stmt(stmt);
                DEC_SCOPE();
            } else {
                write_bool(false);
            }

            IAnnotation_block const *annos = f_decl->get_annotations();
            write_annos(annos);

            IDefinition const *def = f_decl->get_definition();
            Tag_t def_tag = get_definition_tag(def);
            write_encoded_tag(def_tag);
            DOUT(("def %u\n", unsigned(def_tag)));

            int p_count = f_decl->get_parameter_count();
            write_unsigned(p_count);
            DOUT(("#params %d\n", p_count));
            INC_SCOPE();

            for (int i = 0; i < p_count; ++i) {
                IParameter const *param = f_decl->get_parameter(i);
                write_parameter(param);
            }
            DEC_SCOPE();

            Qualifier q = f_decl->get_qualifier();
            write_unsigned(q);
            DOUT(("qualifier %u\n", unsigned(q)));
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_MODULE:
        DOUT(("Module_decl\n"));
        INC_SCOPE();
        {
            IDeclaration_module const *f_decl = cast<IDeclaration_module>(decl);

            IAnnotation_block const *annos = f_decl->get_annotations();
            write_annos(annos);
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_NAMESPACE_ALIAS:
        DOUT(("Namespace_alias\n"));
        INC_SCOPE();
        {
            IDeclaration_namespace_alias const *n_decl = cast<IDeclaration_namespace_alias>(decl);

            ISimple_name const *alias = n_decl->get_alias();
            write_name(alias);

            IQualified_name const *ns = n_decl->get_namespace();
            write_name(ns);
        }
        DEC_SCOPE();
        break;
    }

    Position const *pos = &decl->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST simple name.
void Module_serializer::write_name(ISimple_name const *sname)
{
    DOUT(("sname\n"));
    INC_SCOPE();

    ISymbol const *sym = sname->get_symbol();
    Tag_t sym_tag = get_symbol_tag(sym);
    write_encoded_tag(sym_tag);
    DOUT(("sym %u (%s)\n", unsigned(sym_tag), sym->get_name()));

    if (IDefinition const *def = sname->get_definition()) {
        write_bool(true);
        Tag_t def_tag = get_definition_tag(def);
        write_encoded_tag(def_tag);
        DOUT(("def %u\n", unsigned(def_tag)));
    } else {
        write_bool(false);
    }

    Position const *pos = &sname->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST qualified name.
void Module_serializer::write_name(IQualified_name const *qname)
{
    DOUT(("qname\n"));
    INC_SCOPE();

    bool abs = qname->is_absolute();
    write_bool(abs);
    DOUT(("absolute %u\n", unsigned(abs)));

    int c_count = qname->get_component_count();
    write_unsigned(c_count);
    DOUT(("#components %d\n", c_count));
    INC_SCOPE();

    for (int i = 0; i < c_count; ++i) {
        ISimple_name const *sname = qname->get_component(i);
        write_name(sname);
    }
    DEC_SCOPE();

    if (IDefinition const *def = qname->get_definition()) {
        write_bool(true);
        Tag_t def_tag = get_definition_tag(def);
        write_encoded_tag(def_tag);
        DOUT(("def %u\n", unsigned(def_tag)));
    } else {
        write_bool(false);
    }

    Position const *pos = &qname->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST type name.
void Module_serializer::write_name(IType_name const *tname)
{
    DOUT(("tname\n"));
    INC_SCOPE();

    IQualified_name const *qname = tname->get_qualified_name();
    write_name(qname);

    bool abs = tname->is_absolute();
    write_bool(abs);
    DOUT(("absolute %u\n", unsigned(abs)));

    Qualifier q = tname->get_qualifier();
    write_unsigned(q);
    DOUT(("qualifier %u\n", unsigned(q)));

    bool arr   = tname->is_array();
    bool c_arr = tname->is_concrete_array();

    unsigned code = 0;

    // encode 4 cases into 2 bools to maintain compatibility with previous protocol version
    if (arr) {
        if (c_arr) {
            if (tname->is_incomplete_array()) {
                code = 0x01;
            } else {
                code = 0x03;
            }
        } else {
            code = 0x02;
        }
    } else {
        code = 0x00;
    }

    write_bool((code & 0x02) != 0);
    DOUT(("is_array %u\n", unsigned(arr)));

    write_bool((code & 0x01) != 0);
    DOUT(("is_concreate_array %u\n", unsigned(c_arr)));

    switch (code) {
    case 0x00:
        // no array
        break;
    case 0x01:
        // incomplete array
        break;
    case 0x02:
        // array size symbol
        {
            ISimple_name const *size = tname->get_size_name();
            write_name(size);
        }
        break;
    case 0x03:
        // array with size expression
        {
            IExpression const *size = tname->get_array_size();
            write_expr(size);
        }
        break;
    }

    if (IType const *type = tname->get_type()) {
        write_bool(true);

        Tag_t type_tag = get_type_tag(type);
        write_encoded_tag(type_tag);
        DOUT(("type %u\n", unsigned(type_tag)));
    } else {
        write_bool(false);
    }

    Position const *pos = &tname->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST parameter.
void Module_serializer::write_parameter(IParameter const *param)
{
    DOUT(("param\n"));
    INC_SCOPE();

    IType_name const *tname = param->get_type_name();
    write_name(tname);

    ISimple_name const *sname = param->get_name();
    write_name(sname);

    if (IExpression const *expr = param->get_init_expr()) {
        write_bool(true);
        write_expr(expr);
    } else {
        write_bool(false);
    }

    IAnnotation_block const *annos = param->get_annotations();
    write_annos(annos);

    Position const *pos = &param->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST expression.
void Module_serializer::write_expr(IExpression const *expr)
{
    DOUT(("expr\n"));
    INC_SCOPE();

    Tag_t t = get_expression_tag_and_drop(expr);
    write_encoded_tag(t);
    DOUT(("expr tag %u\n", unsigned(t)));

    IExpression::Kind kind = expr->get_kind();
    write_unsigned(kind);

    int s_count = expr->get_sub_expression_count();
    if (kind == IExpression::EK_CALL) {
        // for calls, dump only the callee reference
        s_count = 1;
    }
    write_unsigned(s_count);
    DOUT(("#sub-expr %d\n", s_count));
    INC_SCOPE();

    for (int i = 0; i < s_count; ++i) {
        IExpression const *child = expr->get_sub_expression(i);
        write_expr(child);
    }
    DEC_SCOPE();

    switch (kind) {
    case IExpression::EK_INVALID:
        // no extra attributes
        DOUT(("INVALID_EXPR\n"));
        break;

    case IExpression::EK_CONDITIONAL:
        // no extra attributes
        DOUT(("Cond_expr\n"));
        break;

    case IExpression::EK_CALL:
        DOUT(("Call_expr\n"));
        INC_SCOPE();
        {
            IExpression_call const *call = cast<IExpression_call>(expr);

            int a_count = call->get_argument_count();
            write_unsigned(a_count);
            DOUT(("#args %d\n", a_count));

            for (int i = 0; i < a_count; ++i) {
                IArgument const *arg = call->get_argument(i);
                write_argument(arg);
            }
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_LITERAL:
        DOUT(("Literal_expr\n"));
        INC_SCOPE();
        {
            IExpression_literal const *l_expr = cast<IExpression_literal>(expr);

            IValue const *v = l_expr->get_value();
            Tag_t value_tag = get_value_tag(v);
            write_encoded_tag(value_tag);
            DOUT(("value %u\n", unsigned(value_tag)));
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_REFERENCE:
        DOUT(("REF_expr\n"));
        INC_SCOPE();
        {
            IExpression_reference const *r_expr = cast<IExpression_reference>(expr);

            IType_name const *tname = r_expr->get_name();
            write_name(tname);

            bool is_arr = r_expr->is_array_constructor();
            write_bool(is_arr);
            DOUT(("is_arr_const %u\n", unsigned(is_arr)));

            if (!is_arr) {
                IDefinition const *def = r_expr->get_definition();
                Tag_t def_tag = get_definition_tag(def);
                write_encoded_tag(def_tag);
                DOUT(("def tag %u\n", unsigned(def_tag)));
            }
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_UNARY:
        DOUT(("Unary_expr\n"));
        INC_SCOPE();
        {
            IExpression_unary const *u_expr = cast<IExpression_unary>(expr);

            IExpression_unary::Operator op = u_expr->get_operator();
            write_unsigned(op);
            DOUT(("op %u\n", unsigned(op)));

            if (op == IExpression_unary::OK_CAST) {
                write_name(u_expr->get_type_name());
            }
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_BINARY:
        DOUT(("Binary_expr\n"));
        INC_SCOPE();
        {
            IExpression_binary const *b_expr = cast<IExpression_binary>(expr);

            IExpression_binary::Operator op = b_expr->get_operator();
            write_unsigned(op);
            DOUT(("op %u\n", unsigned(op)));
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_LET:
        DOUT(("Let_expr\n"));
        INC_SCOPE();
        {
            IExpression_let const *l_expr = cast<IExpression_let>(expr);

            int d_count = l_expr->get_declaration_count();
            write_unsigned(d_count);
            DOUT(("#decls %d\n", d_count));

            for (int i = 0; i < d_count; ++i) {
                IDeclaration const *decl = l_expr->get_declaration(i);
                write_decl(decl);
            }
        }
        DEC_SCOPE();
        break;
    }

    bool in_para = expr->in_parenthesis();
    write_bool(in_para);
    DOUT(("in_para %u\n", unsigned(in_para)));

    IType const *type = expr->get_type();
    Tag_t type_tag = get_type_tag(type);
    write_encoded_tag(type_tag);
    DOUT(("type %u\n", unsigned(type_tag)));

    Position const *pos = &expr->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST annotation block.
void Module_serializer::write_annos(IAnnotation_block const *block)
{
    // annotations are always optional, so check it here
    if (block == NULL) {
        write_bool(false);
        return;
    }
    write_bool(true);

    DOUT(("anno_block\n"));
    INC_SCOPE();

    int a_count = block->get_annotation_count();
    write_unsigned(a_count);
    DOUT(("#annos %d\n", a_count));
    INC_SCOPE();

    for (int i = 0; i < a_count; ++i) {
        IAnnotation const *anno = block->get_annotation(i);
        write_anno(anno);
    }
    DEC_SCOPE();

    Position const *pos = &block->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST annotation.
void Module_serializer::write_anno(IAnnotation const *anno)
{
    DOUT(("anno\n"));
    INC_SCOPE();

    IQualified_name const *qname = anno->get_name();
    write_name(qname);

    int a_count = anno->get_argument_count();
    write_unsigned(a_count);
    DOUT(("#args %d\n", a_count));
    INC_SCOPE();

    for (int i = 0; i < a_count; ++i) {
        IArgument const *arg = anno->get_argument(i);
        write_argument(arg);
    }
    DEC_SCOPE();

    Position const *pos = &anno->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST statement.
void Module_serializer::write_stmt(IStatement const *stmt)
{
    DOUT(("stmt\n"));
    INC_SCOPE();

    IStatement::Kind kind = stmt->get_kind();
    write_unsigned(kind);

    switch (kind) {
    case IStatement::SK_INVALID:
        // ready
        DOUT(("INVALID\n"));
        break;

    case IStatement::SK_COMPOUND:
        {
            DOUT(("Block_stmt\n"));
            IStatement_compound const *c_stmt = cast<IStatement_compound>(stmt);

            int s_count = c_stmt->get_statement_count();
            write_unsigned(s_count);
            DOUT(("#stmts %d\n", s_count));
            INC_SCOPE();

            for (int i = 0; i < s_count; ++i) {
                IStatement const *child = c_stmt->get_statement(i);
                write_stmt(child);
            }
            DEC_SCOPE();
        }
        break;

    case IStatement::SK_DECLARATION:
        {
            DOUT(("Decl_stmt\n"));
            IStatement_declaration const *d_stmt = cast<IStatement_declaration>(stmt);

            IDeclaration const *decl = d_stmt->get_declaration();
            write_decl(decl);
        }
        break;

    case IStatement::SK_EXPRESSION:
        {
            DOUT(("Expr_stmt\n"));
            IStatement_expression const *e_stmt = cast<IStatement_expression>(stmt);

            if (IExpression const *expr = e_stmt->get_expression()) {
                write_bool(true);
                write_expr(expr);
            } else {
                write_bool(false);
            }
        }
        break;

    case IStatement::SK_IF:
        {
            DOUT(("If_stmt\n"));
            IStatement_if const *i_stmt = cast<IStatement_if>(stmt);

            IExpression const *cond = i_stmt->get_condition();
            write_expr(cond);

            IStatement const *then_stmt = i_stmt->get_then_statement();
            write_stmt(then_stmt);

            if (IStatement const *else_stmt = i_stmt->get_else_statement()) {
                write_bool(true);
                write_stmt(else_stmt);
            } else {
                write_bool(false);
            }
        }
        break;

    case IStatement::SK_CASE:
        {
            DOUT(("Case_stmt\n"));
            IStatement_case const *c_stmt = cast<IStatement_case>(stmt);

            if (IExpression const *label = c_stmt->get_label()) {
                write_bool(true);
                write_expr(label);
            } else {
                write_bool(false);
            }

            int s_count = c_stmt->get_statement_count();
            write_unsigned(s_count);
            DOUT(("#stmts %d\n", s_count));
            INC_SCOPE();

            for (int i = 0; i < s_count; ++i) {
                IStatement const *child = c_stmt->get_statement(i);
                write_stmt(child);
            }
            DEC_SCOPE();
        }
        break;

    case IStatement::SK_SWITCH:
        {
            DOUT(("Switch_stmt\n"));
            IStatement_switch const *s_stmt = cast<IStatement_switch>(stmt);

            IExpression const *cond = s_stmt->get_condition();
            write_expr(cond);

            int c_count = s_stmt->get_case_count();
            write_unsigned(c_count);
            DOUT(("#cases %d\n", c_count));
            INC_SCOPE();

            for (int i = 0; i < c_count; ++i) {
                IStatement const *child = s_stmt->get_case(i);
                write_stmt(child);
            }
            DEC_SCOPE();
        }
        break;

    case IStatement::SK_WHILE:
    case IStatement::SK_DO_WHILE:
        {
            DOUT(("%s_stmt\n", kind == IStatement::SK_WHILE ? "While" : "Do_while"));
            IStatement_loop const *l_stmt = cast<IStatement_loop>(stmt);

            IExpression const *cond = l_stmt->get_condition();
            write_expr(cond);

            IStatement const *body = l_stmt->get_body();
            write_stmt(body);
        }
        break;

    case IStatement::SK_FOR:
        {
            DOUT(("For_stmt\n"));
            IStatement_for const *f_stmt = cast<IStatement_for>(stmt);

            if (IStatement const *init = f_stmt->get_init()) {
                write_bool(true);
                write_stmt(init);
            } else {
                write_bool(false);
            }

            if (IExpression const *update = f_stmt->get_update()) {
                write_bool(true);
                write_expr(update);
            } else {
                write_bool(false);
            }

            if (IExpression const *cond = f_stmt->get_condition()) {
                write_bool(true);
                write_expr(cond);
            } else {
                write_bool(false);
            }

            IStatement const *body = f_stmt->get_body();
            write_stmt(body);
        }
        break;

    case IStatement::SK_BREAK:
        // ready
        DOUT(("Break_stmt\n"));
        break;

    case IStatement::SK_CONTINUE:
        // ready
        DOUT(("Continue_stmt\n"));
        break;

    case IStatement::SK_RETURN:
        {
            DOUT(("Return_stmt\n"));
            IStatement_return const *r_stmt = cast<IStatement_return>(stmt);

            if (IExpression const *expr = r_stmt->get_expression()) {
                write_bool(true);
                write_expr(expr);
            } else {
                write_bool(false);
            }
        }
        break;
    }

    Position const *pos = &stmt->access_position();
    write_pos(pos);
    DEC_SCOPE();
}

// Write an AST argument.
void Module_serializer::write_argument(IArgument const *arg)
{
    DOUT(("arg\n"));

    IArgument::Kind kind = arg->get_kind();
    write_unsigned(kind);
    DOUT(("kind %u\n", unsigned(kind)));

    IExpression const *expr = arg->get_argument_expr();
    write_expr(expr);

    if (kind == IArgument::AK_NAMED) {
        IArgument_named const *narg = cast<IArgument_named>(arg);

        ISimple_name const *sname = narg->get_parameter_name();
        write_name(sname);
    }

    Position const *pos = &arg->access_position();
    write_pos(pos);
}

// Write an AST position.
void Module_serializer::write_pos(Position const *pos)
{
    int sl = pos->get_start_line();
    write_unsigned(sl);

    int el = pos->get_end_line();
    write_unsigned(el);

    int sc = pos->get_start_column();
    write_unsigned(sc);

    int ec = pos->get_end_column();
    write_unsigned(ec);

    size_t id = pos->get_filename_id();
    write_encoded_tag(id);

    DOUT(("pos %d,%d %d,%d %u\n", sl, sc, el, ec, unsigned(id)));
}

// Write all initializer expressions unreferenced so far.
void Module_serializer::write_unreferenced_init_expressions()
{
    typedef Pointer_serializer<IExpression> PS;

    struct Entry {
        Entry(PS::value_type v) : expr((IExpression const *)v.first), tag(v.second) {}

        IExpression const *expr;
        Tag_t             tag;
    };

    // sort value types by tags.
    struct Tag_comparator {
        bool operator()(Entry const &a, Entry const &b) {
            return a.tag < b.tag;
        }
    };

    // copy them into vector because we need to sort them and write_expr() will
    // delete them from the pointer map
    vector<Entry>::Type inits(m_init_exprs.begin(), m_init_exprs.end(), m_alloc);

    std::sort(inits.begin(), inits.end(), Tag_comparator());

    size_t count = inits.size();
    write_encoded_tag(count);
    DOUT(("#unref expr %u\n", unsigned(count)));

    for (size_t i = 0; i < count; ++i) {
        write_expr(inits[i].expr);
    }

    MDL_ASSERT(m_init_exprs.empty() && "init expressions still not empty");
}

Module_serializer::Module_serializer(
    IAllocator            *alloc,
    ISerializer           *serializer,
    MDL_binary_serializer *bin_serializer)
: Factory_serializer(alloc, serializer, bin_serializer)
, m_definitions(alloc)
, m_declaratios(alloc)
, m_init_exprs(alloc)
{
}

// --------------------- Entity wait queue ---------------------

// Constructor.
Base_wait_queue::Base_wait_queue(IAllocator *alloc)
: m_alloc(alloc)
, m_builder(alloc)
, m_free_list(NULL)
, m_wait_list(Wait_lists::key_compare(), alloc)
{
}

// Destructor.
Base_wait_queue::~Base_wait_queue()
{
    MDL_ASSERT(m_wait_list.empty() && "Still waiting objects");
    for (Entry *q, *p = m_free_list; p != NULL; p = q) {
        q = p->m_next;

        m_builder.destroy(p);
    }
}

// Wait for the given tag.
void Base_wait_queue::wait(Tag_t tag, void **adr)
{
    Entry *e = get_entry(adr);

    std::pair<Wait_lists::iterator, bool> res =
        m_wait_list.insert(Wait_lists::value_type(tag, e));
    if (!res.second) {
        // there was already an entry
        Entry *old = res.first->second;
        e->m_next  = old->m_next;
        old->m_next = e;
    }
}

// Object gets ready.
void Base_wait_queue::ready(Tag_t tag, void *obj)
{
    Wait_lists::iterator it = m_wait_list.find(tag);

    if (it != m_wait_list.end()) {
        // trigger
        for (Entry *q, *p = it->second; p != NULL; p = q) {
            q           = p->m_next;
            *p->m_dst   = obj;
            p->m_next   = m_free_list;
            m_free_list = p;
        }
        m_wait_list.erase(it);
    }
}

// Get a new free entry.
Base_wait_queue::Entry *Base_wait_queue::get_entry(void **adr)
{
    if (m_free_list == NULL)
        return m_builder.create<Entry>(adr);
    Entry *p = m_free_list;
    m_free_list = p->m_next;

    return new (p) Entry(adr);
}

// --------------------- Entity deserializer ---------------------

// Constructor.
Entity_deserializer::Entity_deserializer(
    IAllocator    *alloc,
    IDeserializer *deserializer)
: m_alloc(alloc)
, m_deserializer(deserializer)
{
}

// --------------------- Binary deserializer ---------------------

// Constructor.
MDL_binary_deserializer::MDL_binary_deserializer(
    IAllocator    *alloc,
    IDeserializer *deserializer,
    MDL           *compiler)
: Entity_deserializer(alloc, deserializer)
, m_modules(alloc)
{
    // register all builtin modules
    Tag_t t = Tag_t(0);

    for (size_t i = 0, n = compiler->get_builtin_module_count(); i < n; ++i) {
        Module const *builtin_mod = compiler->get_builtin_module(i);

        register_module(++t, builtin_mod);
    }
}

// --------------------- Factory deserializer ---------------------

// Constructor.
Factory_deserializer::Factory_deserializer(
    IAllocator              *alloc,
    IDeserializer           *deserializer,
    MDL_binary_deserializer *bin_deserializer)
    : Entity_deserializer(alloc, deserializer)
    , m_symbols(alloc)
    , m_types(alloc)
    , m_array_sizes(alloc)
    , m_values(alloc)
    , m_bin_deserializer(bin_deserializer)
{
}

// Read a type.
IType const *Factory_deserializer::read_type(Type_factory &tf)
{
    IType::Kind kind = IType::Kind(read_encoded_tag());

    DOUT(("type kind %u\n", kind));

    switch (kind) {
    case IType::TK_BOOL:
    case IType::TK_INT:
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
    case IType::TK_COLOR:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        {
            // these types are builtin, no need to deserialize them, just reference its tag
            Tag_t type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(type_tag)));

            return get_type(type_tag);
        }

    case IType::TK_ALIAS:
        {
            Tag_t a_type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(a_type_tag)));

            // read the name if any
            ISymbol const *sym = NULL;
            if (read_bool()) {
                Tag_t sym_tag = read_encoded_tag();
                sym = get_symbol(sym_tag);

                DOUT(("type name %u (%s)\n", unsigned(sym_tag), sym->get_name()));
            }

            // read the aliased type
            Tag_t e_type_tag = read_encoded_tag();
            IType const *e_type = get_type(e_type_tag);

            DOUT(("type alias %u\n", unsigned(e_type_tag)));

            // read the modifiers
            IType::Modifiers mod(read_encoded_tag());

            DOUT(("type modifiers %u\n", unsigned(mod)));

            IType const *a_type = tf.create_alias(e_type, sym, mod);

            register_type(a_type_tag, a_type);
            return a_type;
        }

    case IType::TK_ENUM:
        {
            IType_enum *e_type = NULL;

            Tag_t e_type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(e_type_tag)));

            if (read_bool()) {
                // a predefined type
                IType_enum::Predefined_id pid = IType_enum::Predefined_id(read_encoded_tag());
                e_type = tf.get_predefined_enum(pid);
            } else {
                // a user defined type

                // read the type name
                Tag_t sym_tag = read_encoded_tag();
                ISymbol const *sym = get_symbol(sym_tag);

                DOUT(("type name %u (%s)\n", unsigned(sym_tag), sym->get_name()));

                e_type = tf.create_enum(sym);

                int n = read_unsigned();

                DOUT(("#values %d\n", n));
                INC_SCOPE();

                for (int i = 0; i < n; ++i) {
                    Tag_t         sym_tag = read_encoded_tag();
                    ISymbol const *sym    = get_symbol(sym_tag);
                    int           code    = read_int();

                    e_type->add_value(sym, code);

                    DOUT(("value sym_tag %u code %d (%s)\n",
                        unsigned(sym_tag), code, sym->get_name()));
                }
                DEC_SCOPE();
            }

            register_type(e_type_tag, e_type);
            return e_type;
        }

    case IType::TK_ARRAY:
        {
            Tag_t a_type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(a_type_tag)));

            // read the element type
            Tag_t e_type_tag = read_encoded_tag();
            IType const *e_type = get_type(e_type_tag);

            DOUT(("e_type %u\n", unsigned(e_type_tag)));

            IType const *a_type;
            if (read_bool()) {
                int size = read_unsigned();
                a_type = tf.create_array(e_type, size);

                DOUT(("imm size %d\n", size));
            } else {
                Tag_t                  size_tag = read_encoded_tag();
                IType_array_size const *size    = get_array_size(size_tag);
                a_type = tf.create_array(e_type, size);

                DOUT(("def size %u\n", unsigned(size_tag)));
            }
            register_type(a_type_tag, a_type);
            return a_type;
        }

    case IType::TK_FUNCTION:
        {
            Tag_t f_type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(f_type_tag)));

            // read the return type
            IType const *ret_type = NULL;
            if (read_bool()) {
                Tag_t ret_type_tag = read_encoded_tag();
                ret_type = get_type(ret_type_tag);

                DOUT(("ret_type %u\n", unsigned(ret_type_tag)));
            }

            // read parameter count
            int n = read_unsigned();

            DOUT(("#params %d\n", n));
            INC_SCOPE();

            // read parameters
            VLA<IType_factory::Function_parameter> params(m_alloc, n);
            for (int i = 0; i < n; ++i) {
                Tag_t p_type_tag = read_encoded_tag();
                params[i].p_type = get_type(p_type_tag);
                Tag_t p_sym_tag  = read_encoded_tag();
                params[i].p_sym  = get_symbol(p_sym_tag);

                DOUT(("param %u %u (%s)\n",
                    unsigned(p_type_tag), unsigned(p_sym_tag), params[i].p_sym->get_name()));
            }
            DEC_SCOPE();

            IType_function const *f_type = tf.create_function(ret_type, params.data(), n);
            register_type(f_type_tag, f_type);
            return f_type;
        }

    case IType::TK_STRUCT:
        {
            IType_struct *s_type = NULL;

            Tag_t s_type_tag = read_encoded_tag();

            DOUT(("type tag %u\n", unsigned(s_type_tag)));

            if (read_bool()) {
                // a predefined type
                IType_struct::Predefined_id pid = IType_struct::Predefined_id(read_encoded_tag());
                s_type = tf.get_predefined_struct(pid);
            } else {
                // a user defined type

                // read the struct name
                Tag_t sym_tag = read_encoded_tag();
                ISymbol const *sym = get_symbol(sym_tag);

                DOUT(("name %u (%s)\n", unsigned(sym_tag), sym->get_name()));

                s_type = tf.create_struct(sym);

                // read the field count
                int n = read_unsigned();

                DOUT(("#fields %d\n", n));

                // read the fields
                for (int i = 0; i < n; ++i) {
                    Tag_t f_type_tag = read_encoded_tag();
                    IType const   *f_type = get_type(f_type_tag);
                    Tag_t f_sym_tag = read_encoded_tag();
                    ISymbol const *f_sym  = get_symbol(f_sym_tag);

                    s_type->add_field(f_type, f_sym);

                    DOUT(("field %u %u (%s)\n",
                        unsigned(f_type_tag), unsigned(f_sym_tag), f_sym->get_name()));
                }

                // read the method count
                n = read_unsigned();

                DOUT(("#methods %d\n", n));

                // read the methods
                for (int i = 0; i < n; ++i) {
                    Tag_t m_type_tag = read_encoded_tag();
                    IType_function const *m_type = cast<IType_function>(get_type(m_type_tag));
                    Tag_t m_sym_tag = read_encoded_tag();
                    ISymbol const        *m_sym  = get_symbol(m_sym_tag);

                    s_type->add_method(m_type, m_sym);

                    DOUT(("method %u %u (%s)\n",
                        unsigned(m_type_tag), unsigned(m_sym_tag), m_sym->get_name()));
                }
            }

            register_type(s_type_tag, s_type);
            return s_type;
        }
    }
    MDL_ASSERT(!"Unknown type kind");
    return NULL;
}

// Read a value.
IValue const *Factory_deserializer::read_value(Value_factory &vf)
{
    IValue::Kind kind = IValue::Kind(read_encoded_tag());

    switch (kind) {
    case IValue::VK_BAD:
        // these values are builtin, no need to serialize them, just reference its tag
        {
            DOUT(("value BAD\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();

            DOUT(("value tag %u\n", unsigned(value_tag)));

            IValue const *v = get_value(value_tag);
            DEC_SCOPE();
            return v;
        }
    case IValue::VK_BOOL:
        // these values are builtin, no need to serialize them, just reference its tag
        {
            DOUT(("value BOOL\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();

            DOUT(("value tag %u\n", unsigned(value_tag)));

            IValue const *v = get_value(value_tag);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_INT:
        {
            DOUT(("value INT\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            // ensure it is the int type
            cast<IType_int>(get_type(type_tag));

            int vv = read_int();

            IValue const *v = vf.create_int(vv);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_ENUM:
        {
            DOUT(("value ENUM\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_enum const *e_type = cast<IType_enum>(get_type(type_tag));

            int idx = read_int();

            IValue const *v = vf.create_enum(e_type, idx);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_FLOAT:
        {
            DOUT(("value FLOAT\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            // ensure it is the float type
            cast<IType_float>(get_type(type_tag));

            float vv = read_float();

            IValue const *v = vf.create_float(vv);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_DOUBLE:
        {
            DOUT(("value DOUBLE\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            // ensure it is the double type
            cast<IType_double>(get_type(type_tag));

            double vv = read_double();

            IValue const *v = vf.create_double(vv);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_STRING:
        {
            DOUT(("value STRING\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            // ensure it is the string type
            cast<IType_string>(get_type(type_tag));

            char const *vv = read_cstring();

            IValue const *v = vf.create_string(vv);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_INVALID_REF:
        {
            DOUT(("value INVALID_REF\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_reference const *rt = cast<IType_reference>(get_type(type_tag));

            IValue const *v = vf.create_invalid_ref(rt);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_TEXTURE:
        {
            DOUT(("value TEXTURE\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_texture const *rt = cast<IType_texture>(get_type(type_tag));

            char const *vv     = read_cstring();
            unsigned    vv_tag = read_db_tag();
            unsigned    vv_ver = read_unsigned();
            IValue_texture::gamma_mode gamma = IValue_texture::gamma_mode(read_int());
            IValue_texture::Bsdf_data_kind bsdf_data_kind =
                IValue_texture::Bsdf_data_kind(read_int());
            IValue const *v = rt->get_shape() == IType_texture::TS_BSDF_DATA ?
                vf.create_bsdf_data_texture(bsdf_data_kind, vv_tag, vv_ver) :
                vf.create_texture(rt, vv, gamma, vv_tag, vv_ver);

            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_LIGHT_PROFILE:
        {
            DOUT(("value LIGHT_PROFILE\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_light_profile const *rt = cast<IType_light_profile>(get_type(type_tag));

            char const *vv    = read_cstring();
            unsigned   vv_tag = read_db_tag();
            unsigned   vv_ver = read_unsigned();
            IValue const *v = vf.create_light_profile(rt, vv, vv_tag, vv_ver);

            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_BSDF_MEASUREMENT:
        {
            DOUT(("value BSDF_MEASUREMENT\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_bsdf_measurement const *rt = cast<IType_bsdf_measurement>(get_type(type_tag));

            char const *vv    = read_cstring();
            unsigned   vv_tag = read_db_tag();
            unsigned   vv_ver = read_unsigned();
            IValue const *v = vf.create_bsdf_measurement(rt, vv, vv_tag, vv_ver);

            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }

    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            DOUT(("value COMPOUND\n"));
            INC_SCOPE();
            Tag_t value_tag = read_encoded_tag();
            DOUT(("value tag %u\n", unsigned(value_tag)));

            Tag_t type_tag = read_encoded_tag();
            DOUT(("type tag %u\n", unsigned(type_tag)));

            IType_compound const *rt = cast<IType_compound>(get_type(type_tag));

            int n = read_unsigned();

            DOUT(("#sub-values %d\n", n));

            VLA<IValue const *> values(m_alloc, n);
            for (int i = 0; i < n; ++i) {
                Tag_t e_tag = read_encoded_tag();

                DOUT(("#subvalue %u\n", unsigned(e_tag)));

                values[i] = get_value(e_tag);
            }

            IValue const *v = vf.create_compound(rt, values.data(), n);
            register_value(value_tag, v);
            DEC_SCOPE();
            return v;
        }
    }
    MDL_ASSERT(!"Unknown value kind");
    return NULL;
}

// --------------------- Module deserializer ---------------------

// Constructor.
Module_deserializer::Module_deserializer(
    IAllocator              *alloc,
    IDeserializer           *deserializer,
    MDL_binary_deserializer *bin_deserializer,
    MDL                     *compiler)
: Factory_deserializer(alloc, deserializer, bin_deserializer)
, m_compiler(compiler)
, m_definitions(alloc)
, m_declarations(alloc)
, m_init_exprs(alloc)
, m_decl_wait_queue(alloc)
, m_def_wait_queue(alloc)
, m_expr_wait_queue(alloc)
{
}

// The given location waits for a declaration to become ready.
void Module_deserializer::wait_for_declaration(Tag_t tag, IDeclaration const **loc)
{
    if (m_declarations.is_known_tag(tag)) {
        // already ready
        *loc = get_declaration(tag);
        return;
    }

    m_decl_wait_queue.wait(tag, loc);
}

// The given location waits for a definition to become ready.
void Module_deserializer::wait_for_definition(Tag_t tag, Definition **loc)
{
    if (m_definitions.is_known_tag(tag)) {
        // already ready
        *loc = get_definition(tag);
    }

    m_def_wait_queue.wait(tag, loc);
}

// The given location waits for an expression to become ready.
void Module_deserializer::wait_for_expression(Tag_t tag, IExpression const **loc)
{
    if (m_init_exprs.is_known_tag(tag)) {
        // already ready
        *loc = get_expression(tag);
    }

    m_expr_wait_queue.wait(tag, loc);
}

// Read an AST declaration.
IDeclaration *Module_deserializer::read_decl(Module &mod)
{
    Tag_t decl_tag = read_encoded_tag();
    DOUT(("decl %u\n", unsigned(decl_tag)));
    INC_SCOPE();

    // read the kind
    IDeclaration::Kind kind = IDeclaration::Kind(read_unsigned());

    bool exported = read_bool();
    DOUT(("exported %u\n", unsigned(exported)));

    IDeclaration         *decl = NULL;
    IDeclaration_factory *df   = mod.get_declaration_factory();

    switch (kind) {
    case IDeclaration::DK_INVALID:
        DOUT(("INVALID\n"));
        decl = df->create_invalid(exported);
        break;

    case IDeclaration::DK_IMPORT:
        DOUT(("Import_decl\n"));
        INC_SCOPE();
        {
            IQualified_name const *qname = NULL;

            if (read_bool()) {
                qname = read_qname(mod);
            }
            IDeclaration_import *i_decl = df->create_import(qname, exported);

            int name_count = read_unsigned();
            DOUT(("#names %d\n", name_count));
            INC_SCOPE();

            for (int i = 0; i < name_count; ++i) {
                IQualified_name const *qname = read_qname(mod);

                i_decl->add_name(qname);
            }
            DEC_SCOPE();
            decl = i_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_ANNOTATION:
        DOUT(("Anno_decl\n"));
        INC_SCOPE();
        {
            ISimple_name const      *anno_name = read_sname(mod);
            IDeclaration_annotation *a_decl = df->create_annotation(anno_name, NULL, exported);

            Tag_t def_tag = read_encoded_tag();
            IDefinition const *def = get_definition(def_tag);
            a_decl->set_definition(def);
            DOUT(("def %u\n", unsigned(def_tag)));

            int param_count = read_unsigned();
            DOUT(("#params %d\n", param_count));
            INC_SCOPE();

            for (int i = 0; i < param_count; ++i) {
                IParameter const *param = read_parameter(mod);

                a_decl->add_parameter(param);
            }
            DEC_SCOPE();

            IAnnotation_block const *annos = read_annos(mod);
            a_decl->set_annotations(annos);

            decl = a_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_CONSTANT:
        DOUT(("Const_decl\n"));
        INC_SCOPE();
        {
            IType_name const *tname = read_tname(mod);

            IDeclaration_constant *c_decl = df->create_constant(tname, exported);

            int cst_count = read_unsigned();
            DOUT(("#consts %d\n", cst_count));
            INC_SCOPE();

            for (int i = 0; i < cst_count; ++i) {
                ISimple_name const      *sname = read_sname(mod);
                IExpression const       *expr  = read_expr(mod);
                IAnnotation_block const *annos = read_annos(mod);

                c_decl->add_constant(sname, expr, annos);
            }
            DEC_SCOPE();
            decl = c_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_ALIAS:
        DOUT(("Alias_decl\n"));
        INC_SCOPE();
        {
            IType_name const   *tname = read_tname(mod);
            ISimple_name const *sname = read_sname(mod);

            decl = df->create_alias(tname, sname, exported);
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_STRUCT:
        DOUT(("Struct_decl\n"));
        INC_SCOPE();
        {
            ISimple_name const      *sname = read_sname(mod);
            IAnnotation_block const *annos = read_annos(mod);

            IDeclaration_type_struct *s_decl = df->create_struct(sname, annos, exported);

            Tag_t def_tag = read_encoded_tag();
            IDefinition const *def = get_definition(def_tag);
            s_decl->set_definition(def);
            DOUT(("def %u\n", unsigned(def_tag)));

            int f_count = read_unsigned();
            DOUT(("#fields %d\n", f_count));
            INC_SCOPE();

            for (int i = 0; i < f_count; ++i) {
                IType_name const   *tname = read_tname(mod);
                ISimple_name const *sname = read_sname(mod);
                IExpression const  *expr  = NULL;

                if (read_bool()) {
                    expr = read_expr(mod);
                }

                IAnnotation_block const *annos = read_annos(mod);

                s_decl->add_field(tname, sname, expr, annos);
            }
            DEC_SCOPE();
            decl = s_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_TYPE_ENUM:
        DOUT(("Enum_decl\n"));
        INC_SCOPE();
        {
            ISimple_name const      *sname  = read_sname(mod);
            IAnnotation_block const *annos  = read_annos(mod);
            IDeclaration_type_enum  *e_decl = df->create_enum(sname, annos, exported);

            Tag_t def_tag = read_encoded_tag();
            IDefinition const *def = get_definition(def_tag);
            e_decl->set_definition(def);
            DOUT(("def %u\n", unsigned(def_tag)));

            int v_count = read_unsigned();
            DOUT(("#values %d\n", v_count));
            INC_SCOPE();

            for (int i = 0; i < v_count; ++i) {
                ISimple_name const *sname = read_sname(mod);
                IExpression const  *expr  = NULL;

                if (read_bool()) {
                    expr = read_expr(mod);
                }

                IAnnotation_block const *annos = read_annos(mod);

                e_decl->add_value(sname, expr, annos);
            }
            DEC_SCOPE();
            decl = e_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_VARIABLE:
        DOUT(("Var_decl\n"));
        INC_SCOPE();
        {
            IType_name const      *tname  = read_tname(mod);
            IDeclaration_variable *v_decl = df->create_variable(tname, exported);

            int v_count = read_unsigned();
            DOUT(("#vars %d\n", v_count));
            INC_SCOPE();

            for (int i = 0; i < v_count; ++i) {
                ISimple_name const *sname = read_sname(mod);
                IExpression const  *expr  = NULL;

                if (read_bool()) {
                    expr = read_expr(mod);
                }

                IAnnotation_block const *annos = read_annos(mod);

                v_decl->add_variable(sname, expr, annos);
            }
            DEC_SCOPE();
            decl = v_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_FUNCTION:
        DOUT(("Func_decl\n"));
        INC_SCOPE();
        {
            IType_name const        *tname     = read_tname(mod);
            IAnnotation_block const *ret_annos = read_annos(mod);
            ISimple_name const      *sname     = read_sname(mod);

            bool is_pr = read_bool();
            DOUT(("preset %u\n", unsigned(is_pr)));

            IStatement const *stmt = NULL;
            if (read_bool()) {
                DOUT(("body\n"));
                INC_SCOPE();
                stmt = read_stmt(mod);
                DEC_SCOPE();
            }

            IAnnotation_block const *annos = read_annos(mod);

            IDeclaration_function *f_decl = df->create_function(
                tname, ret_annos, sname, is_pr, stmt, annos, exported);

            Tag_t def_tag = read_encoded_tag();
            IDefinition const *def = get_definition(def_tag);
            f_decl->set_definition(def);
            DOUT(("def %u\n", unsigned(def_tag)));

            int p_count = read_unsigned();
            DOUT(("#params %d\n", p_count));
            INC_SCOPE();

            for (int i = 0; i < p_count; ++i) {
                IParameter const *param = read_parameter(mod);

                f_decl->add_parameter(param);
            }
            DEC_SCOPE();

            Qualifier q = Qualifier(read_unsigned());
            DOUT(("qualifier %u\n", unsigned(q)));
            f_decl->set_qualifier(q);

            decl = f_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_MODULE:
        DOUT(("Module_decl\n"));
        INC_SCOPE();
        {
            IAnnotation_block const *annos = read_annos(mod);
            IDeclaration_module *m_decl = df->create_module(annos);
            decl = m_decl;
        }
        DEC_SCOPE();
        break;

    case IDeclaration::DK_NAMESPACE_ALIAS:
        DOUT(("Namespace_alias\n"));
        INC_SCOPE();
        {
            ISimple_name const    *alias = read_sname(mod);
            IQualified_name const *ns    = read_qname(mod);
            IDeclaration_namespace_alias *a_decl = df->create_namespace_alias(alias, ns);
            decl = a_decl;
        }
        DEC_SCOPE();
        break;
    }

    MDL_ASSERT(decl != NULL);
    read_pos(decl->access_position());

    register_declaration(decl_tag, decl);

    DEC_SCOPE();
    return decl;
}

// Read all initializer expressions unreferenced so far.
void Module_deserializer::read_unreferenced_init_expressions(Module &mod)
{
    size_t count = read_encoded_tag();
    DOUT(("#unref expr %u\n", unsigned(count)));

    for (size_t i = 0; i < count; ++i) {
        (void)read_expr(mod);
    }
}

// Creates a new (empty) module.
Module *Module_deserializer::create_module(
    IMDL::MDL_version mdl_version,
    bool              analyzed)
{
    // create an new empty module
    Module *mod = m_compiler->create_module(/*context=*/NULL, /*module_name=*/NULL, mdl_version);

    if (analyzed) {
        // analyze it, this will create all the predefined entities the deserializer needs
        mod->analyze(/*cache=*/NULL, /*ctx=*/NULL);
    }
    return mod;
}

// Read an AST simple name.
ISimple_name const *Module_deserializer::read_sname(Module &mod)
{
    DOUT(("sname\n"));
    INC_SCOPE();

    Tag_t sym_tag = read_encoded_tag();
    ISymbol const *sym = get_symbol(sym_tag);
    DOUT(("sym %u (%s)\n", unsigned(sym_tag), sym->get_name()));

    ISimple_name *sname =
        const_cast<ISimple_name *>(mod.get_name_factory()->create_simple_name(sym));

    IDefinition const *def = NULL;
    if (read_bool()) {
        Tag_t def_tag = read_encoded_tag();
        def = get_definition(def_tag);
        DOUT(("def %u\n", unsigned(def_tag)));
    }
    sname->set_definition(def);

    read_pos(sname->access_position());
    DEC_SCOPE();

    return sname;
}

// Read an AST qualified name.
IQualified_name *Module_deserializer::read_qname(Module &mod)
{
    IQualified_name *qname = mod.get_name_factory()->create_qualified_name();

    DOUT(("qname\n"));
    INC_SCOPE();

    bool abs = read_bool();
    if (abs)
        qname->set_absolute();
    DOUT(("absolute %u\n", unsigned(abs)));

    int c_count = read_unsigned();
    DOUT(("#components %d\n", c_count));
    INC_SCOPE();

    for (int i = 0; i < c_count; ++i) {
        ISimple_name const *sname = read_sname(mod);

        qname->add_component(sname);
    }
    DEC_SCOPE();

    IDefinition const *def = NULL;
    if (read_bool()) {
        Tag_t def_tag = read_encoded_tag();
        def = get_definition(def_tag);
        DOUT(("def %u\n", unsigned(def_tag)));
    }
    qname->set_definition(def);

    read_pos(qname->access_position());

    DEC_SCOPE();
    return qname;
}

// Read an AST type name.
IType_name const *Module_deserializer::read_tname(Module &mod)
{
    DOUT(("tname\n"));
    INC_SCOPE();

    IQualified_name *qname = read_qname(mod);

    IType_name *tname = mod.get_name_factory()-> create_type_name(qname);

    bool abs = read_bool();
    if (abs)
        tname->set_absolute();
    DOUT(("absolute %u\n", unsigned(abs)));

    Qualifier q = Qualifier(read_unsigned());
    tname->set_qualifier(q);
    DOUT(("qualifier %u\n", unsigned(q)));

    bool high_bit = read_bool();
    bool low_bit  = read_bool();

    unsigned code = (high_bit ? 0x02 : 0) | (low_bit ? 0x01 : 0);

    DOUT(("is_array %u\n", code != 0));
    DOUT(("is_concreate_array %u\n", code & 0x01));

    switch (code) {
    case 0x00:
        // no array
        break;
    case 0x01:
        // incomplete array
        tname->set_array_size(NULL);
        break;
    case 0x02:
        // array size symbol
        {
            ISimple_name const *size = read_sname(mod);
            tname->set_size_name(size);
        }
        break;
    case 0x03:
        // array with size expression
        {
            IExpression const *size = read_expr(mod);
            tname->set_array_size(size);
        }
        break;
    }

    if (read_bool()) {
        Tag_t type_tag = read_encoded_tag();
        IType const *type = get_type(type_tag);
        DOUT(("type %u\n", unsigned(type_tag)));
        tname->set_type(type);
    }

    read_pos(tname->access_position());
    DEC_SCOPE();

    return tname;
}

// Read an AST parameter.
IParameter const *Module_deserializer::read_parameter(Module &mod)
{
    DOUT(("param\n"));
    INC_SCOPE();

    IType_name const   *tname = read_tname(mod);
    ISimple_name const *sname = read_sname(mod);
    IExpression const  *expr  = NULL;

    if (read_bool()) {
        expr = read_expr(mod);
    }

    IAnnotation_block const *annos = read_annos(mod);

    IDeclaration_factory *df = mod.get_declaration_factory();
    IParameter *param = const_cast<IParameter *>(df->create_parameter(tname, sname, expr, annos));

    read_pos(param->access_position());
    DEC_SCOPE();

    return param;
}

// Read an AST expression.
IExpression *Module_deserializer::read_expr(Module &mod)
{
    IExpression *expr = NULL;

    DOUT(("expr\n"));
    INC_SCOPE();

    Tag_t expr_tag = read_encoded_tag();
    DOUT(("expr tag %u\n", unsigned(expr_tag)));

    IExpression::Kind kind = IExpression::Kind(read_unsigned());

    int s_count = read_unsigned();
    DOUT(("#sub-expr %d\n", s_count));
    INC_SCOPE();

    VLA<IExpression const *> children(m_alloc, s_count);
    for (int i = 0; i < s_count; ++i) {
        children[i] = read_expr(mod);
    }
    DEC_SCOPE();

    IExpression_factory *ef = mod.get_expression_factory();

    switch (kind) {
    case IExpression::EK_INVALID:
        DOUT(("INVALID_EXPR\n"));
        expr = ef->create_invalid();
        break;

    case IExpression::EK_CONDITIONAL:
        DOUT(("Cond_expr\n"));
        expr = ef->create_conditional(children[0], children[1], children[2]);
        break;

    case IExpression::EK_CALL:
        DOUT(("Call_expr\n"));
        INC_SCOPE();
        {
            IExpression_call *call = ef->create_call(children[0]);

            int a_count = read_unsigned();
            DOUT(("#args %d\n", a_count));

            for (int i = 0; i < a_count; ++i) {
                IArgument const *arg = read_argument(mod);

                call->add_argument(arg);
            }
            expr = call;
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_LITERAL:
        DOUT(("Literal_expr\n"));
        INC_SCOPE();
        {
            Tag_t value_tag = read_encoded_tag();
            IValue const *v = get_value(value_tag);
            DOUT(("value %u\n", unsigned(value_tag)));

            expr = ef->create_literal(v);
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_REFERENCE:
        DOUT(("REF_expr\n"));
        INC_SCOPE();
        {
            IType_name const      *tname  = read_tname(mod);
            IExpression_reference *r_expr = ef->create_reference(tname);


            bool is_arr = read_bool();
            if (is_arr)
                r_expr->set_array_constructor();
            DOUT(("is_arr_const %u\n", unsigned(is_arr)));

            if (!is_arr) {
                Tag_t def_tag = read_encoded_tag();
                IDefinition const *def = get_definition(def_tag);
                r_expr->set_definition(def);
                DOUT(("def tag %u\n", unsigned(def_tag)));
            }
            expr = r_expr;
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_UNARY:
        DOUT(("Unary_expr\n"));
        INC_SCOPE();
        {
            IExpression_unary::Operator op = IExpression_unary::Operator(read_unsigned());
            DOUT(("op %u\n", unsigned(op)));

            IExpression_unary *u_expr = ef->create_unary(op, children[0]);
            if (op == IExpression_unary::OK_CAST) {
                u_expr->set_type_name(read_tname(mod));
            }
            expr = u_expr;
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_BINARY:
        DOUT(("Binary_expr\n"));
        INC_SCOPE();
        {
            IExpression_binary::Operator op = IExpression_binary::Operator(read_unsigned());
            DOUT(("op %u\n", unsigned(op)));

            expr = ef->create_binary(op, children[0], children[1]);
        }
        DEC_SCOPE();
        break;

    case IExpression::EK_LET:
        DOUT(("Let_expr\n"));
        INC_SCOPE();
        {
            IExpression_let *l_expr = ef->create_let(children[0]);

            int d_count = read_unsigned();
            DOUT(("#decls %d\n", d_count));

            for (int i = 0; i < d_count; ++i) {
                IDeclaration const *decl = read_decl(mod);

                l_expr->add_declaration(decl);
            }
            expr = l_expr;
        }
        DEC_SCOPE();
        break;
    }

    bool in_para = read_bool();
    if (in_para)
        expr->mark_parenthesis();
    DOUT(("in_para %u\n", unsigned(in_para)));

    Tag_t type_tag = read_encoded_tag();
    IType const *type = get_type(type_tag);
    expr->set_type(type);
    DOUT(("type %u\n", unsigned(type_tag)));

    read_pos(expr->access_position());

    if (expr_tag != 0) {
        // this expression is tagged, someone waits for it
        m_expr_wait_queue.ready(expr_tag, expr);
    }
    DEC_SCOPE();
    return expr;
}

// Read an AST annotation block.
IAnnotation_block const *Module_deserializer::read_annos(Module &mod)
{
    // annotations are always optional, so check it here
    if (!read_bool()) {
        return NULL;
    }
    IAnnotation_block *block = mod.get_annotation_factory()->create_annotation_block();

    DOUT(("anno_block\n"));
    INC_SCOPE();

    int a_count = read_unsigned();
    DOUT(("#annos %d\n", a_count));
    INC_SCOPE();

    for (int i = 0; i < a_count; ++i) {
        IAnnotation const *anno = read_anno(mod);

        block->add_annotation(anno);
    }
    DEC_SCOPE();

    read_pos(block->access_position());
    DEC_SCOPE();

    return block;
}

// Read an AST annotation.
IAnnotation const *Module_deserializer::read_anno(Module &mod)
{
    DOUT(("anno\n"));
    INC_SCOPE();

    IQualified_name const *qname = read_qname(mod);

    IAnnotation *anno = mod.get_annotation_factory()->create_annotation(qname);

    int a_count = read_unsigned();
    DOUT(("#args %d\n", a_count));
    INC_SCOPE();

    for (int i = 0; i < a_count; ++i) {
        IArgument const *arg = read_argument(mod);

        anno->add_argument(arg);
    }
    DEC_SCOPE();

    read_pos(anno->access_position());
    DEC_SCOPE();

    return anno;
}

// Read an AST statement.
IStatement const *Module_deserializer::read_stmt(Module &mod)
{
    DOUT(("stmt\n"));
    INC_SCOPE();

    IStatement::Kind kind = IStatement::Kind(read_unsigned());

    IStatement         *stmt = NULL;
    IStatement_factory *sf   = mod.get_statement_factory();

    switch (kind) {
    case IStatement::SK_INVALID:
        // ready
        DOUT(("INVALID\n"));
        stmt = sf->create_invalid();
        break;

    case IStatement::SK_COMPOUND:
        {
            DOUT(("Block_stmt\n"));
            IStatement_compound *c_stmt = sf->create_compound();

            int s_count = read_unsigned();
            DOUT(("#stmts %d\n", s_count));
            INC_SCOPE();

            for (int i = 0; i < s_count; ++i) {
                IStatement const *child = read_stmt(mod);

                c_stmt->add_statement(child);
            }
            DEC_SCOPE();
            stmt = c_stmt;
        }
        break;

    case IStatement::SK_DECLARATION:
        {
            DOUT(("Decl_stmt\n"));
            IDeclaration const *decl = read_decl(mod);
            stmt = sf->create_declaration(decl);
        }
        break;

    case IStatement::SK_EXPRESSION:
        {
            DOUT(("Expr_stmt\n"));
            IExpression const *expr = NULL;
            if (read_bool()) {
                expr = read_expr(mod);
            }
            stmt = sf->create_expression(expr);
        }
        break;

    case IStatement::SK_IF:
        {
            DOUT(("If_stmt\n"));
            IExpression const *cond      = read_expr(mod);
            IStatement const  *then_stmt = read_stmt(mod);
            IStatement const  *else_stmt = NULL;

            if (read_bool()) {
                else_stmt = read_stmt(mod);
            }
            stmt = sf->create_if(cond, then_stmt, else_stmt);
        }
        break;

    case IStatement::SK_CASE:
        {
            DOUT(("Case_stmt\n"));
            IExpression const *label = NULL;
            if (read_bool()) {
                label = read_expr(mod);
            }

            IStatement_case *c_stmt = sf->create_switch_case(label);

            int s_count = read_unsigned();
            DOUT(("#stmts %d\n", s_count));
            INC_SCOPE();

            for (int i = 0; i < s_count; ++i) {
                IStatement const *child = read_stmt(mod);

                c_stmt->add_statement(child);
            }
            DEC_SCOPE();
            stmt = c_stmt;
        }
        break;

    case IStatement::SK_SWITCH:
        {
            DOUT(("Switch_stmt\n"));
            IExpression const *cond   = read_expr(mod);
            IStatement_switch *s_stmt = sf->create_switch(cond);

            int c_count = read_unsigned();
            DOUT(("#cases %d\n", c_count));
            INC_SCOPE();

            for (int i = 0; i < c_count; ++i) {
                IStatement const *child = read_stmt(mod);

                s_stmt->add_case(child);
            }
            DEC_SCOPE();
            stmt = s_stmt;
        }
        break;

    case IStatement::SK_WHILE:
        {
            DOUT(("While_stmt\n"));
            IExpression const *cond = read_expr(mod);
            IStatement const  *body = read_stmt(mod);

            stmt = sf->create_while(cond, body);
        }
        break;

    case IStatement::SK_DO_WHILE:
        {
            DOUT(("Do_while_stmt\n"));
            IExpression const *cond = read_expr(mod);
            IStatement const  *body = read_stmt(mod);

            stmt = sf->create_do_while(cond, body);
        }
        break;

    case IStatement::SK_FOR:
        {
            DOUT(("For_stmt\n"));
            IStatement const *init = NULL;
            if (read_bool()) {
                init = read_stmt(mod);
            }

            IExpression const *update = NULL;
            if (read_bool()) {
                update = read_expr(mod);
            }

            IExpression const *cond = NULL;
            if (read_bool()) {
                cond = read_expr(mod);
            }

            IStatement const *body = read_stmt(mod);

            stmt = sf->create_for(init, cond, update, body);
        }
        break;

    case IStatement::SK_BREAK:
        DOUT(("Break_stmt\n"));
        stmt = sf->create_break();
        break;

    case IStatement::SK_CONTINUE:
        DOUT(("Continue_stmt\n"));
        stmt = sf->create_continue();
        break;

    case IStatement::SK_RETURN:
        {
            DOUT(("Return_stmt\n"));
            IExpression const *expr = NULL;
            if (read_bool()) {
                expr = read_expr(mod);
            }
            stmt = sf->create_return(expr);
        }
        break;
    }

    MDL_ASSERT(stmt != NULL);

    read_pos(stmt->access_position());
    DEC_SCOPE();

    return stmt;
}

// Read an AST argument.
IArgument const *Module_deserializer::read_argument(Module &mod)
{
    DOUT(("arg\n"));

    IArgument::Kind kind = IArgument::Kind(read_unsigned());
    DOUT(("kind %u\n", unsigned(kind)));

    IExpression const *expr = read_expr(mod);

    IArgument *arg;
    IExpression_factory *ef = mod.get_expression_factory();

    if (kind == IArgument::AK_NAMED) {
        ISimple_name const *sname = read_sname(mod);

        arg = const_cast<IArgument_named *>(ef->create_named_argument(sname, expr));
    } else {
        arg = const_cast<IArgument_positional *>(ef->create_positional_argument(expr));
    }

    read_pos(arg->access_position());

    return arg;
}

// Read an AST position.
void Module_deserializer::read_pos(Position &pos)
{
    int sl = read_unsigned();
    pos.set_start_line(sl);

    int el = read_unsigned();
    pos.set_end_line(el);

    int sc = read_unsigned();
    pos.set_start_column(sc);

    int ec = read_unsigned();
    pos.set_end_column(ec);

    size_t id = read_encoded_tag();
    pos.set_filename_id(id);

    DOUT(("pos %d,%d %d,%d %u\n", sl, sc, el, ec, unsigned(id)));
}

}  // mdl
}  // mi

#if 0

#include <cstdarg>
#include <cstdio>

namespace mi {
namespace mdl {

static unsigned indent = 0;

void dprintf(char const *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    for (unsigned i = 0, n = indent; i < n; ++i)
        printf(" ");

    vprintf(fmt, ap);
    fflush(stdout);

    va_end(ap);
}

void dprintf_incscope()
{
    ++indent;
}

void dprintf_decscope()
{
    --indent;
}

}  // mdl
}  // mi

#endif
