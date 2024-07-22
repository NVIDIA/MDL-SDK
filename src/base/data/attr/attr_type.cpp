/***************************************************************************************************
 * Copyright (c) 2005-2024, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/// \file
/// \brief The implementation of the ATTR::Type non-inline members

#include "pch.h"
#include "i_attr_type.h"
#include "attr.h"
#ifdef DEBUG_NON_INLINE
 #define MI_INLINE
  #include "attr_inline_type.h"
#endif

#include <base/data/serial/i_serializer.h>
#include <base/lib/cont/i_cont_array.h>
#include <base/lib/cont/i_cont_rle_array.h>

#undef EXPERIMENTAL_ARRAYS_OF_ARRAYS_MODE

namespace
{
using namespace MI;

/// Dump attribute to info messages. This is for debugging only.
/// \param types all types to dump
void rec_print_type(CONT::Array<const ATTR::Type *> *types);

// This is for convenience in the following table only.
typedef CONT::Rle_array<Uint>* RLE_PTR;
}

namespace MI {
namespace ATTR {

using namespace LOG;
using namespace std;

//==================================================================================================

/// information about each basic type. This is not exposed outside the class.
/// Note that \c TYPE_RGBA_FP is an alias for \c TYPE_COLOR and doesn't show here.
/// Also, \c TYPE_MATRIX4X4 is an alias for \c TYPE_MATRIX.
/// A method for looking up types given their name is available in
/// \c Mod_attr::get_type_code(name).
/// The string representations of the attributes should all be lower case,
/// since the lookup in \c Mod_attr::get_type_code(name) is case-insensitive.
const Type::Typeinfo Type::m_typeinfo[] = {
        //  name               comp    base (Type_code)      size
        { "<illegal>",          0,      TYPE_UNDEF,        0                    },
        { "boolean",            1,      TYPE_BOOLEAN,      sizeof(bool)         },
        { "int8",               1,      TYPE_INT8,         sizeof(Sint8)        },
        { "int16",              1,      TYPE_INT16,        sizeof(Sint16)       },
        { "int32",              1,      TYPE_INT32,        sizeof(Sint32)       },
        { "int64",              1,      TYPE_INT64,        sizeof(Sint64)       },
        { "scalar",             1,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "vector2",            2,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "vector3",            3,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "vector4",            4,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "dscalar",            1,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dvector2",           2,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dvector3",           3,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dvector4",           4,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "matrix",             16,     TYPE_SCALAR,       sizeof(Scalar)       },
        { "dmatrix",            16,     TYPE_DSCALAR,      sizeof(Dscalar)      },
        // TYPE_GMATRIX = TYPE_DMATRIX
        { "quaternion",         4,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "string",             1,      TYPE_STRING,       sizeof(char *)       },
        { "tag",                1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "color",              4,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "rgb",                3,      TYPE_INT8,         sizeof(Uint8)        },
        { "rgba",               4,      TYPE_INT8,         sizeof(Uint8)        },
        { "rgbe",               4,      TYPE_INT8,         sizeof(Uint8)        },
        { "rgbea",              5,      TYPE_INT8,         sizeof(Uint8)        },
        { "rgb_16",             3,      TYPE_INT16,        sizeof(Uint16)       },
        { "rgba_16",            4,      TYPE_INT16,        sizeof(Uint16)       },
        { "rgb_fp",             3,      TYPE_SCALAR,       sizeof(Scalar)       },
        // TYPE_RGBA_FP = TYPE_COLOR
        { "struct",             0,      TYPE_UNDEF,        0                    },
        { "array",              0,      TYPE_UNDEF,        0                    },
        { "rle_uint_ptr",       1,      TYPE_RLE_UINT_PTR, sizeof(RLE_PTR)      },
        { "vector2i",           2,      TYPE_INT32,        sizeof(Sint32)       },
        { "vector3i",           3,      TYPE_INT32,        sizeof(Sint32)       },
        { "vector4i",           4,      TYPE_INT32,        sizeof(Sint32)       },
        { "vector2b",           2,      TYPE_BOOLEAN,      sizeof(bool)         },
        { "vector3b",           3,      TYPE_BOOLEAN,      sizeof(bool)         },
        { "vector4b",           4,      TYPE_BOOLEAN,      sizeof(bool)         },
        { "matrix2x2",          4,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix2x3",          6,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix3x2",          6,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix3x3",          9,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix4x3",          12,     TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix3x4",          12,     TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix4x2",          8,      TYPE_SCALAR,       sizeof(Scalar)       },
        { "matrix2x4",          8,      TYPE_SCALAR,       sizeof(Scalar)       },
        // MetaSL-specific attribute types
        { "texture1d",          1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "texture2d",          1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "texture3d",          1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "texture_cube",       1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "brdf",               1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "light",              1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "light_profile",      1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "shader",             1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "particle_map",       1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "spectrum",           3,      TYPE_SCALAR,       sizeof(Scalar)       },
        // MDL-specific attribute types
        { "id",                 1,      TYPE_INT32,        sizeof(Uint32)       },
        { "parameter",          1,      TYPE_INT32,        sizeof(Uint32)       },
        { "temporary",          1,      TYPE_INT32,        sizeof(Uint32)       },
        { "attachable",         0,      TYPE_ATTACHABLE,   0                    },
        // it is actually one TYPE_TAG and one TYPE_STRING!!!
        { "call",               2,      TYPE_STRING,       sizeof(char *)       },
        { "texture",            1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "bsdf_measurement",   1,      TYPE_TAG,          sizeof(DB::Tag)      },
        { "enum",               1,      TYPE_INT32,        sizeof(Sint32)       },
        // general attribute types
        { "dmatrix2x2",         4,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix2x3",         6,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix3x2",         6,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix3x3",         9,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix4x3",         12,     TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix3x4",         12,     TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix4x2",         8,      TYPE_DSCALAR,      sizeof(Dscalar)      },
        { "dmatrix2x4",         8,      TYPE_DSCALAR,      sizeof(Dscalar)      },

};

//
// constructor.
//

Type::Type(
    Type_code           type,           // primitive type: bool, int, ...
    const char          *name,          // name of new type, 0 means none
    Uint                arraysize)      // number of elements, 0=dynamic
{
    mi_static_assert(TYPE_NUM == (sizeof(m_typeinfo) / sizeof(Typeinfo)));
    mi_static_assert(sizeof(Type::m_typeinfo) == TYPE_NUM * sizeof(Type::Typeinfo));
    ASSERT(M_ATTR, type != TYPE_ID);

    type = to_valid_range(type);
    if (name)
        m_name = name;
    m_typecode  = type;
    m_const     = false;
    m_spare     = false;
    m_arraysize = m_typecode != TYPE_ARRAY? arraysize : 0;
    // enable this check to find out where arrays are created the old way w/o using TYPE_ARRAY
    //ASSERT(M_ATTR, arraysize <= 1 || m_typecode == TYPE_ARRAY);
    m_next      = 0;
    m_child     = 0;
}


Type::Type(
    const Type          &other)         // primitive type: bool, int, ...
  : m_typecode(TYPE_UNDEF), m_next(0), m_child(0)
{
    do_the_deep_copy(other);
}


Type::Type(
    const Type          &other,         // primitive type: bool, int, ...
    Uint                array_size)     // the array size
  : m_typecode(TYPE_UNDEF), m_next(0), m_child(0)
{
    do_the_deep_copy(other);
    m_arraysize = array_size;
}


Type::~Type()
{
    ASSERT(M_ATTR, m_typecode != TYPE_ID);
    delete m_next;
    if (m_typecode != TYPE_ENUM)
        delete m_child;
    else
        delete m_enum;
}


// The normal handling for structs is unchanged. Only adding a child Type to an array Type requires
// special handling.
void Type::set_child(
    const Type& child)
{
    ASSERT(M_ATTR,
        get_typecode() == TYPE_STRUCT ||
        get_typecode() == TYPE_ARRAY ||
        get_typecode() == TYPE_ATTACHABLE ||
        get_typecode() == TYPE_CALL);
#ifndef EXPERIMENTAL_ARRAYS_OF_ARRAYS_MODE
    if (get_typecode() == TYPE_ARRAY && child.get_typecode() == TYPE_ARRAY) {
        mod_log->info(M_ATTR, LOG::Mod_log::C_DATABASE,
            "Nested arrays are currently not supported");
        return;
    }
#endif
    delete m_child;
    m_child = new Type(child);

    // update this' array count iff it is an array
    if (get_typecode() == TYPE_ARRAY) {
        if (child.get_typecode() == TYPE_ARRAY) {
            //delete m_child;
            //m_child = 0;
            //return;
        }
        else {
            // using direct setting here is a bit more performant than the member set_arraysize(),
            // since the latter sets the array_size of the child (again)
            //set_arraysize(child.get_arraysize());
            m_arraysize = child.get_arraysize();

            // since the array should behave as if it is the child it should take over its name, too
            set_name(child.get_name());

            // ...and the next(): this allows handling the next through the TYPE_ARRAY "meta-type"
            if (m_child->get_next()) {
                // the TYPE_ARRAY's current m_next value would be lost!
                ASSERT(M_ATTR, m_next == 0);
                std::swap(m_next, m_child->m_next);
            }
        }
    }
}


//--------------------------------------------------------------------------------------------------

// Given a complete name, return the Type of the subtree, and an offset into a value structure
// where that Type stores its data. The name must be a complete path, such as a[2].b if a is a
// struct array containing b.
// If the name that is found is an array but no [n] is given, return the address of the array
// (which is identical to [0]) and the array type. If an index [n] is given, return the array's
// element type.
// If the name that is found is a struct but no .sub is given, return the address of the struct.
// This allows partial lookups.
// For example, if the type tree is {a,b[3]{c,d}}, b is the same as b[0].c.
// Dynamic arrays are not handled here because the value is a pointer.
const Type *Type::lookup(
    const char  *name,                                  // name to look up
    Uint        *ret_offs,                              // return offset in value struct
    Uint        offs) const                             // add this to returned offset
{
    ASSERT(M_ATTR, name);

    Uint        align;                                  // for ensuring alignment
    Uint        member = null_index;                    // if array, array member; else 0
    const char  *end   = 0;                             // end of word in name, at one of [.\0
    const char  *dot   = 0;                             // if there is a '.', dot points to it

    // cast from size_t to Uint is safe here
    align = Uint(align_all()) - 1;
    offs = (offs + align) & ~align;
    dot = strchr(name, '.');                            // split into name[member].sub
                                                        //           end--^  dot--^
    // contains name a bracket, ie an index? Then set member accordingly.
    if (const char* bracket = strchr(name, '[')) {
        // is bracket *before* any dot? Ie, is first part of the name an indexed array access?
        if (!dot || bracket < dot) {
            end = bracket;
            member = atoi(bracket+1);
        }
    }
    if (!end)
        end = dot ? dot : name + strlen(name);


    // where are we now?
    // - end points to end of current member or at current member's bracket '['
    // - dot points to first '.' or 0
    // - member contains current member's array index or null_index


    // if this type matches, ie if name == m_name, then resolve lookup by
    // either setting offset directly or continue descending into substruct
    if (get_name() && !strncmp(name, get_name(), end-name) && !(get_name()[end-name])) {
        // handle TYPE_ARRAY by forwarding to its child except when no member was given
        if (get_typecode() == TYPE_ARRAY) {
            if (member == null_index) {
                if (ret_offs)
                    *ret_offs = offs;
                return this;
            }
            else {
                return get_child()->lookup(name, ret_offs, offs);
            }
        }
        if (member != null_index) {             // name has "[member]"
            if (!m_arraysize)                   // dynamic array: can't offset
                return 0;
            if (member >= get_arraysize())      // static array & out of bounds
                return 0;
                                                // static array: step to member
            // cast from size_t to Uint is safe here
            offs += member * Uint(sizeof_elem());
        }


        if (dot) {                              // look up .substruct
            if (!get_child())                   // have no substruct: fail.
                return 0;
            else                                // else recurse into substruct
                return get_child()->lookup(dot+1, ret_offs, offs);
        } else {                                // full match: return this type
            if (ret_offs)                       // (even if we have substructs)
                *ret_offs = offs;
            return this;
        }
    }
    // else continue with the next type
    if (get_next()) {
        if (get_typecode() != TYPE_STRUCT &&
            get_typecode() != TYPE_ARRAY &&
            get_typecode() != TYPE_ATTACHABLE &&
            get_typecode() != TYPE_CALL)
            // cast from size_t to Uint is safe here
            offs += Uint(
                m_arraysize ? sizeof_one() : sizeof(Uint) + sizeof(void *));
        else
            // cast from size_t to Uint is safe here
            offs += Uint(sizeof_all());

        offs = (offs + align) & ~align;
        return get_next()->lookup(name, ret_offs, offs);
    }
    return 0;                                   // end of type chain: fail.
}

// Given a complete name, return the Type of the subtree, and an offset into a value structure
// where that Type stores its data. The name must be a complete path, such as a[2].b if a is a
// struct array containing b.
// If the name that is found is an array but no [n] is given, return the address of the array
// (which is identical to [0]) and the array type. If an index [n] is given, return the array's
// element type.
// If the name that is found is a struct but no .sub is given, return the address of the struct.
// This allows partial lookups.
// For example, if the type tree is {a,b[3]{c,d}}, b is the same as b[0].c.
//
// In contrast to the method above, dynamic arrays are handled correctly.
const Type* Type::lookup(
    const char* name,
    const char* base_address,
    const char** ret_address,
    Uint offs) const
{
    ASSERT(M_ATTR, name);

    Uint        align;                                  // for ensuring alignment
    Uint        member = null_index;                    // if array, array member; else 0
    const char  *end   = 0;                             // end of word in name, at one of [.\0
    const char  *dot   = 0;                             // if there is a '.', dot points to it

    // cast from size_t to Uint is safe here
    align = Uint(align_all()) - 1;
    offs = (offs + align) & ~align;
    dot = strchr(name, '.');                            // split into name[member].sub
                                                        //           end--^  dot--^
    // contains name a bracket, ie an index? Then set member accordingly.
    if (const char* bracket = strchr(name, '[')) {
        // is bracket *before* any dot? Ie, is first part of the name an indexed array access?
        if (!dot || bracket < dot) {
            end = bracket;
            member = atoi(bracket+1);
        }
    }
    if (!end)
        end = dot ? dot : name + strlen(name);


    // where are we now?
    // - end points to end of current member or at current member's bracket '['
    // - dot points to first '.' or 0
    // - member contains current member's array index or null_index


    // if this type matches, ie if name == m_name, then resolve lookup by
    // either setting offset directly or continue descending into substruct
    if (get_name() && !strncmp(name, get_name(), end-name) && !(get_name()[end-name])) {
        // handle TYPE_ARRAY by forwarding to its child except when no member was given
        if (get_typecode() == TYPE_ARRAY) {
            if (member == null_index) {
                if (ret_address)
                    *ret_address = base_address + offs;
                return this;
            }
            else {
                return get_child()->lookup(name, base_address, ret_address, offs);
            }
        }
        if (member != null_index) {             // name has "[member]"
            if (m_arraysize == 0) {             // dynamic array
                Uint dynamic_size = ((Dynamic_array*)(void*)(base_address+offs))->m_count;
                if (member >= dynamic_size)    // out of bounds
                    return 0;
                base_address = ((Dynamic_array*)(void*)(base_address+offs))->m_value;
                offs = 0;
            } else {                            // static array
                if (member >= get_arraysize())  // out of bounds
                    return 0;
            }
            // cast from size_t to Uint is safe here
            offs += member * Uint(sizeof_elem()); // step to member
        }


        if (dot) {                              // look up .substruct
            if (!get_child())                   // have no substruct: fail.
                return 0;
            else {                              // else recurse into substruct
                if (get_typecode() == TYPE_CALL)
                    offs += 2*Type::sizeof_one(TYPE_STRING);
                return get_child()->lookup(dot+1, base_address, ret_address, offs);
            }
        } else {                                // full match: return this type
            if (ret_address)                    // (even if we have substructs)
                *ret_address = base_address + offs;
            return this;
        }
    }
    // else continue with the next type
    if (get_next()) {
        if (get_typecode() != TYPE_STRUCT
            && get_typecode() != TYPE_ARRAY
            && get_typecode() != TYPE_ATTACHABLE
            && get_typecode() != TYPE_CALL)
            // cast from size_t to Uint is safe here
            offs += Uint(
                m_arraysize ? sizeof_one() : sizeof(Uint) + sizeof(void *));
        else
            // cast from size_t to Uint is safe here
            offs += Uint(sizeof_all());

        offs = (offs + align) & ~align;
        return get_next()->lookup(name, base_address, ret_address, offs);
    }
    return 0;                                   // end of type chain: fail.
}
                        

//
// serialize the object to the given serializer including all sub elements.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
// Also recurse into children and successors.
// The first version is for standard Type serialization. The second is used
// as a prefix when serializing a data value, like a shader parameter, where
// the receiving end cannot be assumed to have the shader declaration (and
// can't get it because it has no transaction pointer). In this case, only
// send the absolute minimum needed for deserializing the data because shader
// parameter blocks can be large and there may be a large number of them.
//

const SERIAL::Serializable *Type::serialize(
    SERIAL::Serializer  *serializer) const // useful functions for byte streams
{
    return do_serialize(serializer);
}


const SERIAL::Serializable *Type::do_serialize(
    SERIAL::Serializer  *serializer) const      // useful functions for byte streams
{
    mi_static_assert(TYPE_NUM < 256);                   // type and flags
    SERIAL::write(serializer,m_name);
    serializer->write(m_typecode);
    serializer->write(m_const);
    serializer->write(m_spare);
    serializer->write(m_arraysize);
    serializer->write(!!m_next);            // write a flag indicating whether there is a m_next
    if (m_next)
        m_next->do_serialize(serializer);
    if (m_typecode != TYPE_ENUM) {
        serializer->write(!!m_child);       // write a flag indicating whether there is a m_child
        if (m_child)
            m_child->do_serialize(serializer);
    }
    else {
        serializer->write(!!m_enum);
        if (m_enum) {
            serializer->write_size_t(m_enum->size());
            for (size_t i=0; i<m_enum->size(); ++i) {
                SERIAL::write(serializer,(*m_enum)[i].first);
                SERIAL::write(serializer,(*m_enum)[i].second);
            }
        }
    }
    SERIAL::write(serializer,m_type_name);

    return this+1;
}


//
// deserialize the object and all sub-objects from the given deserializer.
// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
// Also recurse into children and successors. See the serializer above for
// the meaning of the "parts" bitmap, which precedes the actual data as a
// table of contents or manifest.
//

SERIAL::Serializable *Type::deserialize(
    SERIAL::Deserializer *deser)        // useful functions for byte streams
{
    SERIAL::read(deser,&m_name);
    deser->read(&m_typecode);
    deser->read(&m_const);
    deser->read(&m_spare);
    deser->read(&m_arraysize);
    bool has_next=false;
    deser->read(&has_next);
    if (has_next) {
        m_next = new Type;
        m_next->deserialize(deser);
    }
    else
        m_next = 0;
    if (m_typecode != TYPE_ENUM) {
        bool has_child=false;
        deser->read(&has_child);
        if (has_child) {
            m_child = new Type;
            m_child->deserialize(deser);
        }
        else
            m_child = 0;
    }
    else {
        bool has_enum = false;
        deser->read(&has_enum);
        if (has_enum) {
            size_t s;
            deser->read_size_t(&s);
            m_enum = new vector<pair<int, string> >;
            m_enum->reserve(s);
            for (size_t i=0; i<s; ++i) {
                int i_val;
                SERIAL::read(deser,&i_val);
                string s_val;
                SERIAL::read(deser,&s_val);
                m_enum->push_back(std::make_pair(i_val, s_val));
            }
        }
        else
            m_enum = 0;
    }
    SERIAL::read(deser,&m_type_name);

    return this+1;
}


//
// copy constructor, makes a deep copy of the type
//

Type &Type::operator=(
    const Type          &other)         // deep-copy this type
{
    // check against self-assignment
    if (this == &other)
        return *this;

    // copying can start now...
    do_the_deep_copy(other);

    return *this;
}


void Type::dump() const
{
    CONT::Array<const Type *> types(1, this);
    rec_print_type(&types);
    ASSERT(M_ATTR, types.empty());
}


//
// return a string that describes the type including subtypes.
//
std::string Type::print() const
{
    std::string result;
    if (m_const)
        result += "const ";
    result += type_name(Type_code(m_typecode));
    if (m_arraysize != 1) {
        result += '[';
        if (m_arraysize)
            result += std::to_string(m_arraysize);
        result += ']';
    }
    if (!m_type_name.empty()) {
        result += ' ';
        result += m_type_name;
    }
    if (m_typecode == TYPE_STRUCT) {
        result += " {";
        for(Type *c = m_child; c; c = c->m_next) {
            result += ' ';
            result += c->print();
            result += ' ';
            if(const char *field_name = c->get_name())
                result += field_name;
            else
                result += "<none>";
            result += ';';
        }
        result += " }";
    }
    else if (m_typecode == TYPE_ATTACHABLE) {
        result += '<';
        result += m_child->m_next->print();
        result += '>';
    }
    else if (m_typecode == TYPE_CALL) {
        result += " {";
        for(Type *c = m_child; c; c = c->m_next) {
            result += ' ';
            result += c->print();
            result += ' ';
            if(const char *field_name = c->get_name())
                result += field_name;
            else
                result += "<none>";
            result += ';';
        }
        result += " }";
    }
    return result;
}


//
// factory function used for deserialization
//

SERIAL::Serializable *Type::factory()
{
    return new Type;
}


// private implementation of the deep copy
// Note that this code assumes a proper initialization of all members prior
// to this call (since it simply calls delete on m_child, m_next, m_name).
// As soon as there was a change to std::shared_ptr the clean-up code can go away,
// since assignment will properly free all allocated resources.
void Type::do_the_deep_copy(
    const Type  &other)                 // deep-copy this type
{
    Type *old_next  = m_next;
    Type *old_child = m_typecode != TYPE_ENUM? m_child : 0;

    // deep copy
    m_name      = other.m_name;
    m_typecode  = other.m_typecode;
    m_const     = other.m_const;
    m_spare     = other.m_spare;
    m_arraysize = other.m_arraysize;
    m_next      = other.m_next  ? new Type(*other.m_next)  : 0;
    if (other.m_typecode != TYPE_ENUM)
        m_child = other.m_child ? new Type(*other.m_child) : 0;
    else
        create_enum(other.m_enum);
    m_type_name = other.get_type_name();

    // clean-up: do this AFTER the copy is done: this allows to copy a child/next to this ...
    delete old_next;
    delete old_child;
}


// Helper to copy the enum values.
void Type::create_enum(
    vector<pair<int, string> >* enum_values)
{
    ASSERT(M_ATTR, m_typecode == TYPE_ENUM);
    ASSERT(M_ATTR, m_enum == 0); // else we need to free it

    if (enum_values) {
        m_enum = new vector<pair<int, string> >();
        *m_enum = *enum_values;
    }
}


//==================================================================================================

bool contains_expected_type(
    const Type& type,
    Type_code expected)
{
    const Type_code type_code = type.get_typecode();

    if (type_code == expected)
        return true;

    // some types are represented by a tag,
    // allow a match for those as well.
    if (expected == TYPE_TAG &&
        (type_code == TYPE_TEXTURE ||
         type_code == TYPE_TEXTURE1D ||
         type_code == TYPE_TEXTURE2D ||
         type_code == TYPE_TEXTURE3D ||
         type_code == TYPE_TEXTURE_CUBE ||
         type_code == TYPE_BRDF ||
         type_code == TYPE_LIGHT ||
         type_code == TYPE_LIGHTPROFILE ||
         type_code == TYPE_SHADER ||
         type_code == TYPE_PARTICLE_MAP ||
         type_code == TYPE_BSDF_MEASUREMENT))
    {
        return true;
    }
    // some are represented by int
    if (expected == TYPE_INT32 &&
        (type_code == TYPE_ENUM))
    {
        return true;
    }

    // inspect array's child
    if (type_code == TYPE_ARRAY) {
        if (type.get_child())
            return contains_expected_type(*type.get_child(), expected);
    }
    return false;
}


// Utility helper. Comparing two enum collections.
bool compare_enum_collections(
    const vector<pair<int, string> >& one,
    const vector<pair<int, string> >& other)
{
    if (one.size() != other.size())
        return false;
    if (one.empty())
        return true;
    for (size_t i=0; i<one.size(); ++i) {
        const pair<int, string> & p = one[i];
        const pair<int, string> & q = other[i];
        if (p.first != q.first || p.second != q.second)
            return false;
    }
    return true;
}

}
}


namespace
{
using namespace MI::ATTR;
using namespace MI::LOG;

//
// dump attribute to info messages, for debugging only. The actual printing
// code is a separate function to allow recursion.
//

void rec_print_type(
    CONT::Array<const Type *>   *types)
{
    const Type *current = (*types)[types->size()-1];

    while (current) {
        // current Type
        mod_log->debug(M_ATTR, Mod_log::C_IO, "%*s%s %s (%s)",
            static_cast<int>(3 * (types->size()-1)), " ",
            Type::type_name(current->get_typecode()), current->get_name(),
            current->get_arraysize()==0 && current->get_typecode()!=TYPE_ARRAY ? "dynamic array" :
            current->get_arraysize()==1 && current->get_typecode()!=TYPE_ARRAY ? "single value"
                                          : "static array");

        // dive recursively into children
        const Type *child = current->get_child();
        if (child) {
            types->append(child);
            rec_print_type(types);
        }
        current = current->get_next();
    }
    // remove last element
    types->remove(types->size()-1);
}

}       // namespace
