/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief Collection of useful tools.

#include "pch.h"

#include "i_attr_utilities.h"
#include "i_attr_type.h"
#include "attr.h"

#include <base/lib/cont/i_cont_rle_array.h>
#include <base/lib/log/log.h>

#include <mi/math.h>

#include <sstream>

namespace MI {
namespace ATTR {

using namespace std;
using namespace CONT;
using namespace LOG;

using namespace mi::math;


/// Generate n tabs.
string tabs(size_t n)
{
    string result;
    for (size_t i=0; i < n; ++i)
        result += "\t";
    return result;
}


string get_type_value_rep(
    Type_iterator& iter,
    size_t depth=0)
{
    stringstream result;

    for (; !iter.at_end(); iter.to_next()) {
        char* value = iter.get_value();
        int arraysize = iter->get_arraysize();

        const char* name = iter->get_name();
        result << '\n' << tabs(depth) << '\"' << (name?name:"(no name)") << "\" ("
            << Type::type_name(iter->get_typecode()) << ") ";

        // a dynamic array - hence reset the value pointer accordingly
        if (!arraysize && iter->get_typecode() != TYPE_ARRAY) {
            Dynamic_array* array = (Dynamic_array*)value;
            // no alignment required here
            value = array->m_value;
            arraysize = array->m_count;
        }

        // arrays can be handled recursively without any changes to the value pointers
        if (iter->get_typecode() == TYPE_ARRAY) {
            Type_iterator sub_iter(&iter, value);
            size_t size = iter->sizeof_elem() * iter->get_arraysize();
            result << get_type_value_rep(sub_iter, depth+1);
            // increase offset by the array element type's size since an array has no size itself
            value += size;
        }
        // rle arrays reset the value pointers accordingly
        else if (iter->get_typecode() == TYPE_RLE_UINT_PTR) {
            Rle_array<Uint>* in_array = *(Rle_array<Uint>**)value;
            Rle_chunk_iterator<Uint> iterator = in_array->begin_chunk();
            size_t size = in_array->get_index_size();
            for (size_t i=0; i < size; ++i) {
                //size_t n = iterator.count();
                //Uint v = iterator.data();
                // TODO add output: value v n times
                ++iterator;
            }
            result << "Support for RLE arrays still missing." << endl;
            value += sizeof(Rle_array<Uint>*);
        }
        // structs can be handled recursively
        else if (iter->get_typecode() == TYPE_STRUCT || iter->get_typecode() == TYPE_ATTACHABLE
            || iter->get_typecode() == TYPE_CALL)
        {
            size_t size = iter->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                if (arraysize > 1)
                    result << "\n" << tabs(depth) << '[' << a << "] ";
                if (iter->get_typecode() == TYPE_CALL) {
                    // name - which is now actually a Tag but occupies space of a string pointer
                    result << " Tag " << (*(DB::Tag *)value).get_uint() << ",";
                    value += Type::sizeof_one(TYPE_STRING);
                    // parameters
                    result << " return_type \"" << (*value?(*(char**)value):"(null)") << "\"";
                    value += Type::sizeof_one(TYPE_STRING);
                }
                Type_iterator sub_iter(&iter, value);
                result << endl << tabs(depth) << "{" << get_type_value_rep(sub_iter, depth + 1)
                    << '\n' << tabs(depth) << "}";
                value += size;
                if (iter->get_typecode() == TYPE_CALL) {
                   value -= 2*Type::sizeof_one(TYPE_STRING);
                }
            }
        }
        else {
            int type, count, size;
            Type_code type_code = iter->get_typecode();
            eval_typecode(type_code, &type, &count, &size);
            if (arraysize > 1)
                result << '[';
            for (int a=0; a < arraysize; ++a) {
                if (a)
                    result << ", ";
                for (int i=0; i < count; ++i) {
                    switch(type) {
                      case '*': {
                          result << '\"' << (*value?(*(char **)value):"(null)") << '\"';
                          break;
                      }
                      case 'c': {
                          if (type_code == TYPE_BOOLEAN)
                              result << (*(Sint8*)value == 0? "false" : "true");
                          else
                              result << *(Sint8*)value;
                          break;
                      }
                      case 's': result << *(Sint16  *)value; break;
                      case 'i': result << *(Sint32  *)value; break;
                      case 'q': result << *(Sint64  *)value; break;
                      case 'f': result << *(Scalar  *)value; break;
                      case 'd': result << *(Dscalar *)value; break;
                      case 't': result << "Tag " << (*(DB::Tag *)value).get_uint(); break;
                      default:  ASSERT(M_ATTR, 0);
                    }
                    value += size;
                }
            }
            if (arraysize > 1)
                result << ']';
        }
    }

    return result.str();
}

//--------------------------------------------------------------------------------------------------

// Return the value of the given type as a const char* string.
string get_type_value_desc(
    const Type& type,
    const char* value)
{
    Type_iterator iter(&type, const_cast<char*>(value));
    return get_type_value_rep(iter);
}

#if 0
{
    stringstream result;
    result << (m_type->get_name()? m_type->get_name() : "<unknown>");
    const char* val_ptr = m_value;

    Type_code code = m_type->get_typecode();
    result << " [" << Type::type_name(code) << "]";
    size_t array_size = m_type->get_arraysize();
//    bool is_dyn_array = false;
//    if (!array_size && code != TYPE_ARRAY)
//	is_dyn_array = true;
    result << " with arraysize " << array_size << " (";

    switch (code) {
      case TYPE_RGB: {
        Uint8* ptr = (Uint8*)val_ptr;
        result << (int)ptr[0] << ", " << (int)ptr[1] << ", " << (int)ptr[2];
        break;
      }
      case TYPE_RGBA: {
        Uint8* ptr = (Uint8*)val_ptr;
        result <<
            (int)ptr[0] << ", " << (int)ptr[1] << ", " << (int)ptr[2] << ", " << (int)ptr[3];
        break;
      }
      case TYPE_COLOR: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2];
        break;
      }
      case TYPE_BOOLEAN: {
        bool* ptr = (bool*)val_ptr;
        result << (*ptr? "true" : "false");
        break;
      }
      case TYPE_VECTOR2B: {
        bool* ptr = (bool*)val_ptr;
        result << (ptr[0]? "true" : "false") << ", " << (ptr[1]? "true" : "false");
        break;
      }
      case TYPE_VECTOR3B: {
          bool* ptr = (bool*)val_ptr;
          result << (ptr[0]? "true" : "false") << ", " << (ptr[1]? "true" : "false")
              << ", " << (ptr[2]? "true" : "false");
          break;
      }
      case TYPE_VECTOR4B: {
          bool* ptr = (bool*)val_ptr;
          result << (ptr[0]? "true" : "false") << ", " << (ptr[1]? "true" : "false")
              << ", " << (ptr[2]? "true" : "false") << ", " << (ptr[3]? "true" : "false");
          break;
      }
      case TYPE_SCALAR: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0];
        if (array_size > 1)
            result << ", ...";
            //print_array<Scalar>(array_size, val_ptr);
        break;
      }
      case TYPE_VECTOR2: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1];
        break;
      }
      case TYPE_VECTOR2I: {
          int* ptr = (int*)val_ptr;
          result << ptr[0] << ", " << ptr[1];
          break;
      }
      case TYPE_VECTOR3: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2];
        break;
      }
      case TYPE_VECTOR3I: {
        int* ptr = (int*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2];
        break;
      }
      case TYPE_VECTOR4: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3];
        break;
      }
      case TYPE_VECTOR4I: {
        int* ptr = (int*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3];
        break;
      }
      case TYPE_DSCALAR: {
        Dscalar* ptr = (Dscalar*)val_ptr;
        result << ptr[0];
        if (array_size > 1)
            result << ", ...";
            //print_array<Dscalar>(array_size, val_ptr);
        break;
      }
      case TYPE_DVECTOR2: {
        Dscalar* ptr = (Dscalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1];
        break;
      }
      case TYPE_DVECTOR3: {
        Dscalar* ptr = (Dscalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2];
        break;
      }
      case TYPE_DVECTOR4: {
        Dscalar* ptr = (Dscalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3];
        break;
      }
      case TYPE_MATRIX: {
        Scalar* ptr = (Scalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << "...";
        break;
      }
      case TYPE_DMATRIX: {
        Dscalar* ptr = (Dscalar*)val_ptr;
        result << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << "...";
        break;
      }
      case TYPE_ID:
      case TYPE_PARAMETER:
      case TYPE_TEMPORARY:
      case TYPE_TAG:
      case TYPE_LIGHT:
      case TYPE_INT32: {
        Uint32* ptr = (Uint32*)val_ptr;
        result << *ptr;
        if (array_size > 1)
            result << ", ...";
        break;
      }
      case TYPE_STRUCT: {
          result << "struct";
          break;
      }
      case TYPE_ATTACHABLE: {
          result << "ref proxy";
          break;
      }
      case TYPE_CALL: {
          result << "call";
          break;
      }
      case TYPE_RLE_UINT_PTR: {
        Rle_array<Uint>* ptr = (Rle_array<Uint>*)val_ptr;
        result << "rle array with " << ptr->size() << " elements";
        break;
      }
      case TYPE_ARRAY: {
        result << "array";
        break;
      }
      case TYPE_STRING: {
        char* ptr = (char*)val_ptr;
        if (ptr)
            result << ptr;
        if (array_size > 1)
            result << ", ...";
        break;
      }
      case TYPE_TEXTURE:
      case TYPE_TEXTURE1D:
      case TYPE_TEXTURE2D:
      case TYPE_TEXTURE3D:
      case TYPE_TEXTURE_CUBE: {
          result << "texture";
          break;
      }
      case TYPE_BSDF_MEASUREMENT:
          result << "bsdf measurement";
          break;
      }
      default:
        result << "UNKNOWN";
    }

    result << ")";
    return result.str();
}
#endif

// evaluate the given typecode and find out how many fields of which type are
// available and what is the overall size of the data described by the type.
// It's important to identify tags (type 't') separately so the serializer
// stores them as type Tag in .mib files, because impmib remaps tag values.
void eval_typecode(
    Type_code typecode,					// the typecode to be evaluated
    int* type,						// store the type of the primitives here
    int* count,						// store the count of the primitives here
    int* size)						// store the overall size here
{
    *type = *count = *size = 0;
    switch(typecode) {
      case TYPE_BOOLEAN:		*type = 'c'; *size = 1; *count = 1;	break;
      case TYPE_INT8:			*type = 'c'; *size = 1; *count = 1;	break;
      case TYPE_INT16:			*type = 's'; *size = 2; *count = 1;	break;
      case TYPE_INT32:			*type = 'i'; *size = 4; *count = 1;	break;
      case TYPE_INT64:			*type = 'q'; *size = 8; *count = 1;	break;
      case TYPE_SCALAR:			*type = 'f'; *size = 4; *count = 1;	break;
      case TYPE_VECTOR2:		*type = 'f'; *size = 4; *count = 2;	break;
      case TYPE_VECTOR3:		*type = 'f'; *size = 4; *count = 3;	break;
      case TYPE_VECTOR4:		*type = 'f'; *size = 4; *count = 4;	break;
      case TYPE_DSCALAR:		*type = 'd'; *size = 8; *count = 1;	break;
      case TYPE_DVECTOR2:		*type = 'd'; *size = 8; *count = 2;	break;
      case TYPE_DVECTOR3:		*type = 'd'; *size = 8; *count = 3;	break;
      case TYPE_DVECTOR4:		*type = 'd'; *size = 8; *count = 4;	break;
          // case TYPE_MATRIX4X4:	// identical to type matrix
      case TYPE_MATRIX:			*type = 'f'; *size = 4; *count = 16;	break;
      case TYPE_DMATRIX:		*type = 'd'; *size = 8; *count = 16;	break;
      case TYPE_QUATERNION:		*type = 'f'; *size = 4; *count = 4;	break;
      case TYPE_STRING:			*type = '*'; *size = sizeof(void*); *count = 1;	break;
          // case TYPE_RGBA_FP:	// identical to type color
      case TYPE_COLOR:			*type = 'f'; *size = 4; *count = 4;	break;
      case TYPE_RGB:			*type = 'c'; *size = 1; *count = 3;	break;
      case TYPE_RGBA:			*type = 'c'; *size = 1; *count = 4;	break;
      case TYPE_RGBE:			*type = 'c'; *size = 1; *count = 4;	break;
      case TYPE_RGBEA:			*type = 'c'; *size = 1; *count = 5;	break;
      case TYPE_RGB_16:			*type = 's'; *size = 2; *count = 3;	break;
      case TYPE_RGBA_16:		*type = 's'; *size = 2; *count = 4;	break;
      case TYPE_RGB_FP:			*type = 'f'; *size = 4; *count = 3;	break;
      case TYPE_TAG:			*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_TEXTURE:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_TEXTURE1D:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_TEXTURE2D:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_TEXTURE3D:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_TEXTURE_CUBE:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_LIGHTPROFILE:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_BRDF:			*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_LIGHT:			*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_BSDF_MEASUREMENT:	*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_ENUM:                   *type = 'i'; *size = 4; *count = 1;     break;
      case TYPE_VECTOR2I:       	*type = 'i'; *size = 4; *count = 2;     break;
      case TYPE_VECTOR3I:       	*type = 'i'; *size = 4; *count = 3;     break;
      case TYPE_VECTOR4I:       	*type = 'i'; *size = 4; *count = 4;     break;
      case TYPE_VECTOR2B:       	*type = 'c'; *size = 1; *count = 2;     break;
      case TYPE_VECTOR3B:       	*type = 'c'; *size = 1; *count = 3;     break;
      case TYPE_VECTOR4B:       	*type = 'c'; *size = 1; *count = 4;     break;
      case TYPE_MATRIX2X2:		*type = 'f'; *size = 4; *count = 4;	break;
      case TYPE_MATRIX2X3:		*type = 'f'; *size = 4; *count = 6;	break;
      case TYPE_MATRIX3X2:		*type = 'f'; *size = 4; *count = 6;	break;
      case TYPE_MATRIX3X3:		*type = 'f'; *size = 4; *count = 9;	break;
      case TYPE_MATRIX4X3:		*type = 'f'; *size = 4; *count = 12;	break;
      case TYPE_MATRIX3X4:		*type = 'f'; *size = 4; *count = 12;	break;
      case TYPE_MATRIX4X2:		*type = 'f'; *size = 4; *count = 8;	break;
      case TYPE_MATRIX2X4:		*type = 'f'; *size = 4; *count = 8;	break;
      case TYPE_SHADER:			*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_PARTICLE_MAP:		*type = 't'; *size = 4; *count = 1;	break;
      case TYPE_SPECTRUM:		*type = 'f'; *size = 4; *count = 3;     break;
      case TYPE_ID:			*type = 'i'; *size = 4; *count = 1;	break;
      case TYPE_PARAMETER:		*type = 'i'; *size = 4; *count = 1;	break;
      case TYPE_TEMPORARY:		*type = 'i'; *size = 4; *count = 1;	break;
      case TYPE_DMATRIX2X2:		*type = 'd'; *size = 8; *count = 4;	break;
      case TYPE_DMATRIX2X3:		*type = 'd'; *size = 8; *count = 6;	break;
      case TYPE_DMATRIX3X2:		*type = 'd'; *size = 8; *count = 6;	break;
      case TYPE_DMATRIX3X3:		*type = 'd'; *size = 8; *count = 9;	break;
      case TYPE_DMATRIX4X3:		*type = 'd'; *size = 8; *count = 12;	break;
      case TYPE_DMATRIX3X4:		*type = 'd'; *size = 8; *count = 12;	break;
      case TYPE_DMATRIX4X2:		*type = 'd'; *size = 8; *count = 8;	break;
      case TYPE_DMATRIX2X4:		*type = 'd'; *size = 8; *count = 8;	break;
      case TYPE_CALL:           	*type = '*'; *size = sizeof(void*); *count = 2;	break;

      case TYPE_UNDEF:
      case TYPE_STRUCT:
      case TYPE_ARRAY:
      case TYPE_RLE_UINT_PTR:
      case TYPE_ATTACHABLE:
      case TYPE_NUM:
          ASSERT(M_ATTR, 0);
    }
}


// Generate n tabs.
static std::string tabs(int n)
{
    std::string result;
    for(int i = 0; i < n; i++)
        result += "\t";
    return result;
}

template<typename T>
static std::string to_string(T value)
{
    return std::to_string(value);
}

template<>
std::string to_string(bool value)
{
    return std::string(value ? "true" : "false");
}

template<Size n>
std::string to_string(Vector<bool,n> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += to_string(value[i]);
    }
    result += ")";
    return result;
}

template<Size n>
std::string to_string(Vector<Sint32,n> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += std::to_string(value[i]);
    }
    result += ")";
    return result;
}

template<Size n>
std::string to_string(Vector<Float32,n> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += std::to_string(value[i]);
    }
    result += ")";
    return result;
}

template<Size n>
std::string to_string(Vector<Float64,n> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += std::to_string(value[i]);
    }
    result += ")";
    return result;
}

template<Size n,Size m>
std::string to_string(Matrix<Float32,n,m> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += to_string(value[i]);
    }
    result += ")";
    return result;
}

template<Size n,Size m>
std::string to_string(Matrix<Float64,n,m> value)
{
    std::string result;
    result += "(";
    for(Size i = 0; i < n; i++) {
        if(i) result += ",";
        result += to_string(value[i]);
    }
    result += ")";
    return result;
}

template<>
std::string to_string(Color value)
{
    std::string result;
    result += "(";
    result += std::to_string(value.r);
    result += ",";
    result += std::to_string(value.g);
    result += ",";
    result += std::to_string(value.b);
    result += ",";
    result += std::to_string(value.a);
    result += ")";
    return result;
}

std::string to_string(Uint8 *value,int n)
{
    std::string result;
    result += "(";
    for(int i = 0; i < n; i++) {
        if(i) result += ",";
        result += std::to_string(value[i]);
    }
    result += ")";
    return result;
}

std::string to_string(Uint16 *value,int n)
{
    std::string result;
    result += "(";
    for(int i = 0; i < n; i++) {
        if(i) result += ",";
        result += std::to_string(value[i]);
    }
    result += ")";
    return result;
}


// Dump attribute value to debug messages, for debugging only.
void dump_attr_values(
    const Type& type,
    const char* name,
    const char* data,
    int depth)
{
    const char* output_name = name ? name : "(no name)";
    Type_code type_code = type.get_typecode();
    const char *code_name = Type::type_name(type_code);
    int arraysize = type.get_arraysize();
    if (!arraysize && (type_code != TYPE_ARRAY)) {
        const Dynamic_array* da = reinterpret_cast<const Dynamic_array*>(data);
        int count = da->m_count;
        const char* value = da->m_value;
        Type element_type(type,1);
        size_t element_size = type.sizeof_elem();
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s[%d] %s",
            tabs(depth).c_str(),code_name,count,output_name);
        const char* element_value = value;
        for (int i=0; i < count; ++i) {
            dump_attr_values(element_type, name, element_value, depth+1);
            element_value += element_size;
        }
        return;
    }
    switch (type_code) {
    case TYPE_UNDEF:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s",
            tabs(depth).c_str(),code_name,output_name);
        break;
    case TYPE_BOOLEAN:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<bool>(type,data,name)).c_str());
        break;
    case TYPE_INT8:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Sint8>(type,data,name)).c_str());
        break;
    case TYPE_INT16:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Sint16>(type,data,name)).c_str());
        break;
    case TYPE_INT32:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Sint32>(type,data,name)).c_str());
        break;
    case TYPE_INT64:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Sint64>(type,data,name)).c_str());
        break;
    case TYPE_SCALAR:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Float32>(type,data,name)).c_str());
        break;
    case TYPE_VECTOR2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,2> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,4> >(type,data,name)).c_str());
        break;
    case TYPE_DSCALAR:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Float64>(type,data,name)).c_str());
        break;
    case TYPE_DVECTOR2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float64,2> >(type,data,name)).c_str());
        break;
    case TYPE_DVECTOR3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float64,3> >(type,data,name)).c_str());
        break;
    case TYPE_DVECTOR4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float64,4> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,4,4> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,4,4> >(type,data,name)).c_str());
        break;
    case TYPE_QUATERNION:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,4> >(type,data,name)).c_str());
        break;
    case TYPE_STRING:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = \"%s\"",
            tabs(depth).c_str(),code_name,output_name,
            get_value<char *>(type,data,name));
        break;
    case TYPE_TAG:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint32>(type,data,name)).c_str());
        break;
    case TYPE_COLOR:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Color>(type,data,name)).c_str());
        break;
    case TYPE_RGB:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint8*>(type,data,name),3).c_str()); //-V525 PVS
        break;
    case TYPE_RGBA:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint8*>(type,data,name),4).c_str());
        break;
    case TYPE_RGBE:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint8*>(type,data,name),4).c_str());
        break;
    case TYPE_RGBEA:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint8*>(type,data,name),5).c_str());
        break;
    case TYPE_RGB_16:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint16*>(type,data,name),3).c_str());
        break;
    case TYPE_RGBA_16:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint16*>(type,data,name),4).c_str());
        break;
    case TYPE_RGB_FP:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str());
        break;
    case TYPE_ATTACHABLE:
        {
            mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s %s",
                tabs(depth).c_str(),code_name,type.get_type_name().c_str(),name);
            int field_index = 0;
            for (const Type* c=type.get_child(); c; c=c->get_next()) {
                std::string element_address = name;
                const char* field_name = c->get_name();
                if (field_name) {
                    element_address += ".";
                    element_address += field_name;
                }
                char* ret_address;
                type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                dump_attr_values(*c,field_name,ret_address,depth+1);
                ++field_index;
            }
        }
        break;
    case TYPE_STRUCT:
        {
            mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s %s",
                tabs(depth).c_str(),code_name,type.get_type_name().c_str(),name);
            int field_index = 0;
            for (const Type* c=type.get_child(); c; c=c->get_next()) {
                std::string element_address = name;
                element_address += ".";
                if (const char* field_name = c->get_name()) {
                    element_address += field_name;
                    char* ret_address;
                    type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                    dump_attr_values(*c,field_name,ret_address,depth+1);
                }
                else {
                    mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%sfield %d has no name",
                        tabs(depth+1).c_str(),field_index);
                }
                ++field_index;
            }
        }
        break;
    case TYPE_CALL: {
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s %s",
            tabs(depth).c_str(),code_name,type.get_type_name().c_str(),name);
        // dump the first string - it is actually a Tag
        const Uint32 tag = DB::Tag(*reinterpret_cast<const DB::Tag*>(data)).get_uint();
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%stag %u", tabs(depth+1).c_str(), tag);

        int field_index = 0;
        for (const Type *c = type.get_child(); c; c = c->get_next()) {
            std::string element_address = name;
            element_address += ".";
            if (const char* field_name = c->get_name()) {
                element_address += field_name;
                char* ret_address;
                type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                dump_attr_values(*c,field_name,ret_address,depth+1);
            }
            else {
                mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%sfield %d has no name",
                    tabs(depth+1).c_str(),field_index);
            }
            ++field_index;
        }
    }
    break;
    case TYPE_ARRAY: {
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s",
            tabs(depth).c_str(),code_name,name);
        int size = type.get_arraysize();
        for (int i=0; i < size; ++i) {
            std::string element_address = name;
            element_address += "[";
            element_address += std::to_string(i);
            element_address += "]";
            char* ret_address;
            type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
            dump_attr_values(*type.get_child(),type.get_child()->get_name(),ret_address,depth+1);
        }
    }
    break;
    case TYPE_RLE_UINT_PTR:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint32>(type,data,name)).c_str());
        break;
    case TYPE_VECTOR2I:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Sint32,2> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR3I:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Sint32,3> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR4I:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Sint32,4> >(type,data,name)).c_str());
        break;

    case TYPE_VECTOR2B:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<bool,2> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR3B:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<bool,3> >(type,data,name)).c_str());
        break;
    case TYPE_VECTOR4B:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<bool,4> >(type,data,name)).c_str());
        break;

    case TYPE_MATRIX2X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,2,2> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX2X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,2,3> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX3X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,3,2> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX3X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,3,3> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX4X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,4,3> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX3X4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,3,4> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX4X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,4,2> >(type,data,name)).c_str());
        break;
    case TYPE_MATRIX2X4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float32,2,4> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX2X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,2,2> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX2X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,2,3> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX3X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,3,2> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX3X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,3,3> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX4X3:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,4,3> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX3X4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,3,4> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX4X2:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,4,2> >(type,data,name)).c_str());
        break;
    case TYPE_DMATRIX2X4:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Matrix<Float64,2,4> >(type,data,name)).c_str());
        break;

    case TYPE_TEXTURE:
    case TYPE_TEXTURE1D:
    case TYPE_TEXTURE2D:
    case TYPE_TEXTURE3D:
    case TYPE_TEXTURE_CUBE:
    case TYPE_BRDF:
    case TYPE_LIGHT:
    case TYPE_LIGHTPROFILE:
    case TYPE_SHADER:
    case TYPE_PARTICLE_MAP:
    case TYPE_BSDF_MEASUREMENT:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint32>(type,data,name)).c_str());
        break;
    case TYPE_SPECTRUM:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str());
        break;
    case TYPE_ID:
    case TYPE_PARAMETER:
    case TYPE_TEMPORARY:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = %s",
            tabs(depth).c_str(),code_name,output_name,
            to_string(get_value<Uint32>(type,data,name)).c_str());
        break;
    case TYPE_ENUM: {
        // the value of a TYPE_ENUM attribute is the index
        // TODO better reporting, incl the referenced values
        const int* index = reinterpret_cast<const int*>(data);
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s has index %d",
            tabs(depth).c_str(), code_name, output_name,
            *index);
        break;
    }
    default:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s<unknown> %s",
            tabs(depth).c_str(),name);
    }
}


// Retrieve the dump of the attribute value as a string.
void get_dump_attr_values_string(
    const Type& type,
    const char* name,
    const char* data,
    int depth,
    stringstream& os)
{
    const char* output_name = name ? name : "(no name)";
    Type_code type_code = type.get_typecode();
    const char *code_name = Type::type_name(type_code);
    int arraysize = type.get_arraysize();
    if (!arraysize && (type_code != TYPE_ARRAY)) {
        const Dynamic_array* da = reinterpret_cast<const Dynamic_array*>(data);
        int count = da->m_count;
        const char* value = da->m_value;
        Type element_type(type,1);
        size_t element_size = type.sizeof_elem();
        os << tabs(depth).c_str() << code_name << '[' << count << "] " << name << endl;
        const char* element_value = value;
        for (int i=0; i < count; ++i) {
            get_dump_attr_values_string(element_type, name, element_value, depth+1, os);
            element_value += element_size;
        }
        return;
    }
    switch (type_code) {
    case TYPE_UNDEF:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << endl;
        break;
    case TYPE_BOOLEAN:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<bool>(type,data,name)).c_str() << endl;
        break;
    case TYPE_INT8:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Sint8>(type,data,name)).c_str() << endl;
        break;
    case TYPE_INT16:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Sint16>(type,data,name)).c_str() << endl;
        break;
    case TYPE_INT32:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Sint32>(type,data,name)).c_str() << endl;
        break;
    case TYPE_INT64:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Sint64>(type,data,name)).c_str() << endl;
        break;
    case TYPE_SCALAR:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Float32>(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DSCALAR:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Float64>(type,data,name)).c_str() << endl;
        break;
    case TYPE_DVECTOR2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float64,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DVECTOR3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float64,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DVECTOR4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float64,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,4,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,4,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_QUATERNION:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_STRING:
        mod_log->debug(M_ATTR, LOG::Mod_log::C_DATABASE,"%s%s %s = \"%s\"",
            tabs(depth).c_str(),code_name,output_name,
            get_value<char *>(type,data,name));
        break;
    case TYPE_TAG:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint32>(type,data,name)).c_str() << endl;
        break;
    case TYPE_COLOR:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Color>(type,data,name)).c_str() << endl;
        break;
    case TYPE_RGB:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint8*>(type,data,name),3).c_str() << endl; //-V525 PVS
        break;
    case TYPE_RGBA:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint8*>(type,data,name),4).c_str() << endl;
        break;
    case TYPE_RGBE:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint8*>(type,data,name),4).c_str() << endl;
        break;
    case TYPE_RGBEA:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint8*>(type,data,name),5).c_str() << endl;
        break;
    case TYPE_RGB_16:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint16*>(type,data,name),3).c_str() << endl;
        break;
    case TYPE_RGBA_16:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint16*>(type,data,name),4).c_str() << endl;
        break;
    case TYPE_RGB_FP:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_ATTACHABLE:
        {
            os << tabs(depth).c_str() << code_name << ' ' << type.get_type_name().c_str()
                << ' ' << output_name << endl;
            int field_index = 0;
            for (const Type* c=type.get_child(); c; c=c->get_next()) {
                std::string element_address = name;
                element_address += ".";
                const char* field_name = c->get_name();
                if (field_name) {
                    element_address += ".";
                    element_address += field_name;
                }
                char* ret_address;
                type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                get_dump_attr_values_string(*c,field_name,ret_address,depth+1, os);
                ++field_index;
            }
        }
        break;
    case TYPE_STRUCT:
        {
            os << tabs(depth).c_str() << code_name << ' ' << type.get_type_name().c_str()
                << ' ' << output_name << endl;
            int field_index = 0;
            for (const Type* c=type.get_child(); c; c=c->get_next()) {
                std::string element_address = name;
                element_address += ".";
                if (const char* field_name = c->get_name()) {
                    element_address += field_name;
                    char* ret_address;
                    type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                    get_dump_attr_values_string(*c,field_name,ret_address,depth+1, os);
                }
                else {
                    os << tabs(depth+1).c_str() << "field " << field_index << " has no name"
                       << endl;
                }
                ++field_index;
            }
        }
        break;
    case TYPE_CALL: {
        os << tabs(depth).c_str() << code_name << ' ' << type.get_type_name().c_str()
            << ' ' << output_name << endl;
        // dump the initial string - it is actually a Tag
        Uint32 tag = DB::Tag(*reinterpret_cast<const DB::Tag*>(data)).get_uint();
        os << tabs(depth+1).c_str() << "tag " << tag << endl;

        int field_index = 0;
        for (const Type *c = type.get_child(); c; c = c->get_next()) {
            std::string element_address = name;
            element_address += ".";
            if (const char* field_name = c->get_name()) {
                element_address += field_name;
                char* ret_address;
                type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
                get_dump_attr_values_string(*c,field_name,ret_address,depth+1, os);
            }
            else {
                os << tabs(depth+1).c_str() << "field " << field_index << " has no name" << endl;
            }
            ++field_index;
        }
    }
    break;
    case TYPE_ARRAY: {
        os << tabs(depth).c_str() << code_name << ' ' << output_name << endl;
        int size = type.get_arraysize();
        for (int i=0; i < size; ++i) {
            std::string element_address = name;
            element_address += "[";
            element_address += std::to_string(i);
            element_address += "]";
            char* ret_address;
            type.lookup(element_address.c_str(),const_cast<char*>(data),&ret_address);
            get_dump_attr_values_string(
                *type.get_child(),
                type.get_child()->get_name(),
                ret_address,
                depth+1,
                os);
        }
    }
    break;
    case TYPE_RLE_UINT_PTR:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint32>(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR2I:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Sint32,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR3I:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Sint32,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR4I:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Sint32,4> >(type,data,name)).c_str() << endl;
        break;

    case TYPE_VECTOR2B:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<bool,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR3B:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<bool,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_VECTOR4B:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<bool,4> >(type,data,name)).c_str() << endl;
        break;

    case TYPE_MATRIX2X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,2,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX2X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,2,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX3X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,3,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX3X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,3,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX4X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,4,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX3X4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,3,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX4X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,4,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_MATRIX2X4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float32,2,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX2X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,2,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX2X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,2,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX3X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,3,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX3X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,3,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX4X3:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,4,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX3X4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,3,4> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX4X2:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,4,2> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_DMATRIX2X4:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Matrix<Float64,2,4> >(type,data,name)).c_str() << endl;
        break;

    case TYPE_TEXTURE:
    case TYPE_TEXTURE1D:
    case TYPE_TEXTURE2D:
    case TYPE_TEXTURE3D:
    case TYPE_TEXTURE_CUBE:
    case TYPE_BRDF:
    case TYPE_LIGHT:
    case TYPE_LIGHTPROFILE:
    case TYPE_SHADER:
    case TYPE_PARTICLE_MAP:
    case TYPE_BSDF_MEASUREMENT:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint32>(type,data,name)).c_str() << endl;
    case TYPE_SPECTRUM:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Vector<Float32,3> >(type,data,name)).c_str() << endl;
        break;
    case TYPE_ID:
    case TYPE_PARAMETER:
    case TYPE_TEMPORARY:
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " = "
            << to_string(get_value<Uint32>(type,data,name)).c_str() << endl;
        break;
    case TYPE_ENUM: {
        // the value of a TYPE_ENUM attribute is the index
        // TODO better reporting, incl the referenced values
        const int* index = reinterpret_cast<const int*>(data);
        os << tabs(depth).c_str() << code_name << ' ' << output_name << " has index "
            << index << endl;
        break;
    }
    default:
        os << tabs(depth).c_str() << "<unknown> " << output_name << endl;
    }
}


// Retrieve the dump of the attribute value as a string.
string get_dump_attr_values_string(
    const Type& type,
    const char* name,
    const char* data,
    int depth)
{
    stringstream os;
    get_dump_attr_values_string(type, name, data, depth, os);
    return os.str();
}

}
}
