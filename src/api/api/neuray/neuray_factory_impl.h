/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IFactory
 **
 ** Implements the IFactory interface
 **/

#ifndef API_API_NEURAY_NEURAY_FACTORY_IMPL_H
#define API_API_NEURAY_NEURAY_FACTORY_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ifactory.h>

#include <iosfwd>
#include <cstring>
#include <boost/core/noncopyable.hpp>

namespace mi {

class IArray;
class ICompound;
class IConst_pointer;
class IData_simple;
class IData_collection;
class IDynamic_array;
class IEnum;
class IMap;
class IPointer;
class IRef;
class IString;
class IStructure;
class INumber;
class IUuid;

namespace neuraylib { class ITransaction; }

}

namespace MI {

namespace NEURAY {

class Class_factory;

extern mi::neuraylib::IFactory* s_factory;

class Factory_impl
  : public mi::base::Interface_implement<mi::neuraylib::IFactory>,
    public boost::noncopyable
{
public:
    /// Constructor of Factory_impl
    Factory_impl( Class_factory* class_factory);

    /// Destructor of Factory_impl
    ~Factory_impl();

    // public API methods

    mi::base::IInterface* create(
        const char* type_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr);

    using mi::neuraylib::IFactory::create;

    mi::Uint32 assign_from_to( const mi::IData* source, mi::IData* target, mi::Uint32 options);

    mi::IData* clone( const mi::IData* source, mi::Uint32 options);

    using mi::neuraylib::IFactory::clone;

    mi::Sint32 compare( const mi::IData* lhs, const mi::IData* rhs);

    const mi::IString* dump( const mi::IData* data, const char* name, mi::Size depth);

    const mi::IString* dump(
        mi::neuraylib::ITransaction* transaction,
        const mi::IData* data,
        const char* name,
        mi::Size depth);

    const mi::IStructure_decl* get_structure_decl( const char* structure_name) const;

    const mi::IEnum_decl* get_enum_decl( const char* enum_name) const;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    /// Dumps \p data.
    ///
    /// See #dump(IData,const char*,Size) and #dump(ITransaction,IData,const char*,Size) for
    /// details.
    ///
    /// \param transaction   Might be \c nullptr. Only to be used for nested non-IData interfaces.
    /// \param[out] s        Dumped representation of \p data.
    void dump(
        mi::neuraylib::ITransaction* transaction,
        const char* name,
        const mi::IData* data,
        mi::Size depth,
        std::ostringstream& s);

    /// Dumps \p data (helper function for non-IData interfaces).
    ///
    /// Supports MDL types, values, and expressions, and lists thereof.
    ///
    /// \param transaction   Might be \c nullptr. A \c nullptr transaction might affect or prevent
    ///                      successful dumping for some interfaces.
    /// \param[out] s        Dumped representation of \p data.
    void dump_non_idata(
        mi::neuraylib::ITransaction* transaction,
        const char* name,
        const mi::base::IInterface* data,
        mi::Size depth,
        std::ostringstream& s);

    /// Convenience wrapper around create() for interfaces that need a transaction.
    ///
    /// First calls #create(). If that fails, extracts the transaction from \p prototype, and calls
    /// #mi::neuraylib::ITransaction::create() (if it succeeded to extract the transaction).
    mi::base::IInterface* create_with_transaction(
        const char* type_name,
        const mi::IData* prototype,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr);

    /// Convenience wrapper around create() for interfaces that need a transaction.
    ///
    /// First calls #create(). If that fails, extracts the transaction from \p prototype, and calls
    /// #mi::neuraylib::ITransaction::create() (if it succeeded to extract the transaction).
    template<class T>
    T* create_with_transaction(
        const char* type_name,
        const mi::IData* prototype,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr)
    {
        mi::base::IInterface* ptr_iinterface
            = create_with_transaction( type_name, prototype, argc, argv);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Pointer to the class factory
    Class_factory* m_class_factory;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_FACTORY_IMPL_H
