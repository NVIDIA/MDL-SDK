/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the INumber implementation.
 **/

#include "pch.h"

#include "neuray_number_impl.h"

#include <mi/base/config.h>

// disable C4800: 'T' : forcing value to bool 'true' or 'false' (performance warning)
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4800 )
#endif

namespace MI {

namespace NEURAY {

template <typename I, typename T>
mi::base::IInterface* Number_impl<I, T>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Number_impl<I, T>();
}

template <typename I, typename T>
Number_impl<I, T>::Number_impl()
  : m_storage( T( 0))
{
}

template <typename I, typename T>
const char* Number_impl<I, T>::get_type_name() const
{
    return Type_traits<T>::get_type_name();
}

template <>
const char* Number_impl<mi::ISize, mi::Size>::get_type_name() const
{
    return "Size";
}

template <>
const char* Number_impl<mi::IDifference, mi::Difference>::get_type_name() const
{
    return "Difference";
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( bool& value) const
{
    value = static_cast<bool>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( bool value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint8& value) const
{
    value = static_cast<mi::Uint8>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint8 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint16& value) const
{
    value = static_cast<mi::Uint16>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint16 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint32& value) const
{
    value = static_cast<mi::Uint32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint64& value) const
{
    value = static_cast<mi::Uint64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint8& value) const
{
    value = static_cast<mi::Sint8>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint8 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint16& value) const
{
    value = static_cast<mi::Sint16>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint16 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint32& value) const
{
    value = static_cast<mi::Sint32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint64& value) const
{
    value = static_cast<mi::Sint64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Float32& value) const
{
    value = static_cast<mi::Float32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Float32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Float64& value) const
{
    value = static_cast<mi::Float64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Float64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
mi::base::IInterface* Number_impl_proxy<I, T>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Number_impl_proxy<I, T>())->cast_to_major();
}

template <typename I, typename T>
Number_impl_proxy<I, T>::Number_impl_proxy()
{
    m_pointer = nullptr;
}

template <typename I, typename T>
const char* Number_impl_proxy<I, T>::get_type_name() const
{
    return Type_traits<T>::get_type_name();
}

template <>
const char* Number_impl_proxy<mi::ISize, mi::Size>::get_type_name() const
{
    return "Size";
}

template <>
const char* Number_impl_proxy<mi::IDifference, mi::Difference>::get_type_name() const
{
    return "Difference";
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( bool& value) const
{
    value = static_cast<bool>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( bool value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint8& value) const
{
    value = static_cast<mi::Uint8>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint8 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint16& value) const
{
    value = static_cast<mi::Uint16>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint16 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint32& value) const
{
    value = static_cast<mi::Uint32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint64& value) const
{
    value = static_cast<mi::Uint64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint8& value) const
{
    value = static_cast<mi::Sint8>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint8 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint16& value) const
{
    value = static_cast<mi::Sint16>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint16 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint32& value) const
{
    value = static_cast<mi::Sint32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint64& value) const
{
    value = static_cast<mi::Sint64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Float32& value) const
{
    value = static_cast<mi::Float32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Float32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Float64& value) const
{
    value = static_cast<mi::Float64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Float64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<T*>( pointer);
    m_owner = make_handle_dup( owner);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::release_referenced_memory()
{
    // nothing to do
}

template<> const char* Type_traits<bool       >::get_type_name() { return "Boolean"; }
template<> const char* Type_traits<mi::Sint8  >::get_type_name() { return "Sint8";   }
template<> const char* Type_traits<mi::Sint16 >::get_type_name() { return "Sint16";  }
template<> const char* Type_traits<mi::Sint32 >::get_type_name() { return "Sint32";  }
template<> const char* Type_traits<mi::Sint64 >::get_type_name() { return "Sint64";  }
template<> const char* Type_traits<mi::Uint8  >::get_type_name() { return "Uint8";   }
template<> const char* Type_traits<mi::Uint16 >::get_type_name() { return "Uint16";  }
template<> const char* Type_traits<mi::Uint32 >::get_type_name() { return "Uint32";  }
template<> const char* Type_traits<mi::Uint64 >::get_type_name() { return "Uint64";  }
template<> const char* Type_traits<mi::Float32>::get_type_name() { return "Float32"; }
template<> const char* Type_traits<mi::Float64>::get_type_name() { return "Float64"; }

// explicit template instantiation for Number_impl<I, T>
template class Number_impl<mi::IBoolean,    bool>;
template class Number_impl<mi::ISint8,      mi::Sint8>;
template class Number_impl<mi::ISint16,     mi::Sint16>;
template class Number_impl<mi::ISint32,     mi::Sint32>;
template class Number_impl<mi::ISint64,     mi::Sint64>;
template class Number_impl<mi::IUint8,      mi::Uint8>;
template class Number_impl<mi::IUint16,     mi::Uint16>;
template class Number_impl<mi::IUint32,     mi::Uint32>;
template class Number_impl<mi::IUint64,     mi::Uint64>;
template class Number_impl<mi::IFloat32,    mi::Float32>;
template class Number_impl<mi::IFloat64,    mi::Float64>;
template class Number_impl<mi::ISize,       mi::Size>;
template class Number_impl<mi::IDifference, mi::Difference>;

// explicit template instantiation for Number_impl_proxy<I, T>
template class Number_impl_proxy<mi::IBoolean,    bool>;
template class Number_impl_proxy<mi::ISint8,      mi::Sint8>;
template class Number_impl_proxy<mi::ISint16,     mi::Sint16>;
template class Number_impl_proxy<mi::ISint32,     mi::Sint32>;
template class Number_impl_proxy<mi::ISint64,     mi::Sint64>;
template class Number_impl_proxy<mi::IUint8,      mi::Uint8>;
template class Number_impl_proxy<mi::IUint16,     mi::Uint16>;
template class Number_impl_proxy<mi::IUint32,     mi::Uint32>;
template class Number_impl_proxy<mi::IUint64,     mi::Uint64>;
template class Number_impl_proxy<mi::IFloat32,    mi::Float32>;
template class Number_impl_proxy<mi::IFloat64,    mi::Float64>;

// expliciti template instantiation for Type_trais<T>
template class Type_traits<mi::Sint8>;
template class Type_traits<mi::Sint16>;
template class Type_traits<mi::Sint32>;
template class Type_traits<mi::Sint64>;
template class Type_traits<mi::Uint8>;
template class Type_traits<mi::Uint16>;
template class Type_traits<mi::Uint32>;
template class Type_traits<mi::Uint64>;
template class Type_traits<mi::Float32>;
template class Type_traits<mi::Float64>;

} // namespace NEURAY

} // namespace MI
