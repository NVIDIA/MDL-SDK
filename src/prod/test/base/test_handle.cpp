/******************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Test driver for public/mi/base/handle.h.
 **/

#include "pch.h"

#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>
#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>


// Test interface extension, provides access to reference count for testing
class ITest_interface : public
    mi::base::Interface_declare<0xe06f13a3,0x11b7,0x4e40,0xb9,0x69,0xc1,0x4b,0x7a,0x10,0xc8,0x01>
{
public:
    virtual mi::Uint32 get_reference_count() const = 0;
};


// Test interface implementation
// The refcount reporting works only single threaded, which is the case for this test.
class Test_interface : public mi::base::Interface_implement<ITest_interface>
{
public:
    using Base = mi::base::Interface_implement<ITest_interface>;

    mi::Uint32 get_reference_count() const final
    {
        Base::retain();
        return Base::release();
    }
};

// A second test interface
class ITest_interface_2 : public
    mi::base::Interface_declare<0x3f237ee6,0x8907,0x4200,0x84,0xe2,0xdf,0x38,0x80,0xf5,0x08,0x97,
                          ITest_interface>
{
};

// Implementation of second test interface
class Test_interface_2 : public mi::base::Interface_implement<ITest_interface_2>
{
public:
    using Base = mi::base::Interface_implement<ITest_interface_2>;

    mi::Uint32 get_reference_count() const final
    {
        Base::retain();
        return Base::release();
    }
};

// Test interface pointer passing without additional use of Handle
// Ref-count need to be 2 in h before calling this.
void foo_no_handle( ITest_interface* h)
{
    MI_CHECK( 2 == h->get_reference_count());
}

// Test interface pointer passing with additional use of Handle
// Ref-count need to be 2 in h before calling this.
void foo_with_handle( ITest_interface* h)
{
    MI_CHECK( 2 == h->get_reference_count());
    mi::base::Handle<ITest_interface> handle( h, mi::base::DUP_INTERFACE);
    MI_CHECK( 3 == h->get_reference_count());
}

// Test interface pointer passing without additional use of const Handle
// Ref-count need to be 2 in h before calling this.
void foo_no_const_handle( const ITest_interface* h)
{
    MI_CHECK( 2 == h->get_reference_count());
}

// Test interface pointer passing with additional use of const Handle
// Ref-count need to be 2 in h before calling this.
void foo_with_const_handle( const ITest_interface* h)
{
    MI_CHECK( 2 == h->get_reference_count());
    mi::base::Handle<const ITest_interface> handle(h, mi::base::DUP_INTERFACE);
    MI_CHECK( 3 == h->get_reference_count());
}

// Example of interface-returning functions
ITest_interface* interface_factory()
{
    return new Test_interface;
}

const ITest_interface* const_interface_factory()
{
    return new Test_interface;
}

ITest_interface_2* interface_2_factory()
{
    return new Test_interface_2;
}

const ITest_interface_2* const_interface_2_factory()
{
    return new Test_interface_2;
}


MI_TEST_AUTO_FUNCTION( test_handle )
{
    mi::base::Handle<ITest_interface> h0;
    MI_CHECK( ! h0.is_valid_interface());
    MI_CHECK( ! h0);

    MI_CHECK( h0      == h0);
    MI_CHECK( h0      == nullptr);
    MI_CHECK( nullptr == h0);

    mi::base::Handle<ITest_interface> handle( interface_factory());
    MI_CHECK( handle.is_valid_interface());
    MI_CHECK( handle);
    MI_CHECK( handle  == handle);
    MI_CHECK( handle  != h0);
    MI_CHECK( h0      != handle);
    MI_CHECK( handle  != nullptr);
    MI_CHECK( nullptr != handle);

    MI_CHECK( 1 == handle->get_reference_count());
    MI_CHECK( 1 == (*handle).get_reference_count());

    mi::base::Handle<mi::base::IInterface> h1 =
        handle.get_interface<mi::base::IInterface>();
    MI_CHECK( h1.is_valid_interface());
    MI_CHECK( h1);
    MI_CHECK( 2 == handle->get_reference_count());
    MI_CHECK( 2 == (*handle).get_reference_count());

    {
        mi::base::Handle<ITest_interface> h2 = h1.get_interface<ITest_interface>();
        MI_CHECK( h2.is_valid_interface());
        MI_CHECK( 3 == h2->get_reference_count());
        MI_CHECK( 3 == handle->get_reference_count());
    }
    MI_CHECK( 2 == handle->get_reference_count());

    mi::base::Handle<ITest_interface> h3 = handle;
    MI_CHECK( 3 == handle->get_reference_count());
    MI_CHECK( 3 == h3->get_reference_count());

    mi::base::Handle<ITest_interface> h4( interface_factory());
    MI_CHECK( 3 == h3->get_reference_count());
    MI_CHECK( 1 == h4->get_reference_count());

    mi::base::Handle<ITest_interface> h5 = h4;
    MI_CHECK( 2 == h4->get_reference_count());

    h4 = h3;
    MI_CHECK( 4 == h3->get_reference_count());
    MI_CHECK( 1 == h5->get_reference_count());

    h3.swap( h5);
    MI_CHECK( 1 == h3->get_reference_count());
    MI_CHECK( 4 == h5->get_reference_count());

    h4 = nullptr;
    MI_CHECK( ! h4.is_valid_interface());
    MI_CHECK( 3 == handle->get_reference_count());

    h5 = nullptr;
    MI_CHECK( ! h5.is_valid_interface());
    MI_CHECK( 2 == handle->get_reference_count());

    foo_no_handle( handle.get());
    MI_CHECK( 2 == handle->get_reference_count());

    foo_with_handle( handle.get());
    MI_CHECK( 2 == handle->get_reference_count());

    MI_CHECK( handle == handle.get());
    MI_CHECK( handle.get() == handle);

    MI_CHECK( ! ( handle != handle.get()));
    MI_CHECK( ! ( handle.get() != handle));

    mi::base::Handle<ITest_interface  > h6 ( interface_factory());
    mi::base::Handle<ITest_interface_2> h7 ( interface_2_factory());

    MI_CHECK( h6 != h7);
    MI_CHECK( ! ( h6 == h7));
}


MI_TEST_AUTO_FUNCTION( test_const_handle )
{
    mi::base::Handle<const ITest_interface> h0;
    MI_CHECK( ! h0.is_valid_interface());
    MI_CHECK( ! h0);

    MI_CHECK( h0      == h0);
    MI_CHECK( h0      == nullptr);
    MI_CHECK( nullptr == h0);

    mi::base::Handle<const ITest_interface> handle( const_interface_factory());
    MI_CHECK( handle.is_valid_interface());
    MI_CHECK( handle);
    MI_CHECK( handle  == handle);
    MI_CHECK( handle  != h0);
    MI_CHECK( h0      != handle);
    MI_CHECK( handle  != nullptr);
    MI_CHECK( nullptr != handle);

    MI_CHECK( handle == handle.get());
    MI_CHECK( handle.get() == handle);

    MI_CHECK( ! ( handle != handle.get()));
    MI_CHECK( ! ( handle.get() != handle));

    MI_CHECK( 1 == handle->get_reference_count());
    MI_CHECK( 1 == (*handle).get_reference_count());

    mi::base::Handle<const mi::base::IInterface> h1 =
        handle.get_interface<const mi::base::IInterface>();
    MI_CHECK( h1.is_valid_interface());
    MI_CHECK( 2 == handle->get_reference_count());
    MI_CHECK( 2 == (*handle).get_reference_count());

    {
        mi::base::Handle<const ITest_interface> h2 =
            h1.get_interface<const ITest_interface>();
        MI_CHECK( h2.is_valid_interface());
        MI_CHECK( 3 == h2->get_reference_count());
        MI_CHECK( 3 == handle->get_reference_count());
    }
    MI_CHECK( 2 == handle->get_reference_count());

    mi::base::Handle<const ITest_interface> h3 = handle;
    MI_CHECK( 3 == handle->get_reference_count());
    MI_CHECK( 3 == h3->get_reference_count());

    mi::base::Handle<const ITest_interface> h4( const_interface_factory());
    MI_CHECK( 3 == h3->get_reference_count());
    MI_CHECK( 1 == h4->get_reference_count());

    mi::base::Handle<const ITest_interface> h5 = h4;
    MI_CHECK( 2 == h4->get_reference_count());

    h4 = h3;
    MI_CHECK( 4 == h3->get_reference_count());
    MI_CHECK( 1 == h5->get_reference_count());

    h3.swap( h5);
    MI_CHECK( 1 == h3->get_reference_count());
    MI_CHECK( 4 == h5->get_reference_count());

    h4 = nullptr;
    MI_CHECK( ! h4.is_valid_interface());
    MI_CHECK( 3 == handle->get_reference_count());

    h5 = nullptr;
    MI_CHECK( ! h5.is_valid_interface());
    MI_CHECK( 2 == handle->get_reference_count());

    foo_no_const_handle( handle.get());
    MI_CHECK( 2 == handle->get_reference_count());

    foo_with_const_handle( handle.get());
    MI_CHECK( 2 == handle->get_reference_count());

    MI_CHECK( handle == handle.get());
    MI_CHECK( handle.get() == handle);

    MI_CHECK( ! ( handle != handle.get()));
    MI_CHECK( ! ( handle.get() != handle));

    mi::base::Handle<const ITest_interface  > h6 ( const_interface_factory());
    mi::base::Handle<const ITest_interface_2> h7 ( const_interface_2_factory());

    MI_CHECK( h6 != h7);
    MI_CHECK( ! ( h6 == h7));
}

// Test legal conversions from Handle/mutable pointer to const Handle
MI_TEST_AUTO_FUNCTION( test_handle_const_handle )
{
    // Get some non-trivial interface
    mi::base::Handle<ITest_interface> handle( interface_factory());
    mi::base::Handle<ITest_interface> h1 = handle = interface_factory();
    MI_CHECK( handle.is_valid_interface());
    MI_CHECK( 2 == handle->get_reference_count());

    // Constructor from Handle
    mi::base::Handle<const ITest_interface> h2 = handle;
    MI_CHECK( h2.is_valid_interface());
    MI_CHECK( 3 == handle->get_reference_count());
    MI_CHECK( 3 == h2->get_reference_count());

    // Constructor from mutable interface pointer
    mi::base::Handle<const ITest_interface> h3( interface_factory());
    mi::base::Handle<const ITest_interface> h4 = h3;
    MI_CHECK( h3.is_valid_interface());
    MI_CHECK( 2 == h3->get_reference_count());

    // Assignment from mutable Handle
    h3 = h2;
    MI_CHECK( h3.is_valid_interface());
    MI_CHECK( 1 == h4->get_reference_count());
    MI_CHECK( 4 == h3->get_reference_count());
    MI_CHECK( 4 == h2->get_reference_count());

    // Assignment from mutable pointer
    h2 = interface_factory();
    MI_CHECK( h2.is_valid_interface());
    MI_CHECK( 1 == h2->get_reference_count());
    MI_CHECK( 3 == h3->get_reference_count());

    // Comparison of const and mutable Handles
    h2 = h1;
    MI_CHECK( h2 == h1);
    MI_CHECK( h2.get() == h1);
    MI_CHECK( h2 == h1.get());
    MI_CHECK( ! ( h2 != h1));
    MI_CHECK( ! ( h2.get() != h1));
    MI_CHECK( ! ( h2 != h1.get()));
}

MI_TEST_AUTO_FUNCTION( test_iinterface_implement )
{
    auto* i = new Test_interface;
    MI_CHECK_EQUAL( 1, i->get_reference_count());
    i->retain();
    i->retain();
    MI_CHECK_EQUAL( 3, i->get_reference_count());

    auto* j = new Test_interface;
    j->retain();
    MI_CHECK_EQUAL( 2, j->get_reference_count());

    // Test ref-count on implementation value assignment
    *j = *i;
    MI_CHECK_EQUAL( 2, j->get_reference_count());
    MI_CHECK_EQUAL( 3, i->get_reference_count());

    // Test ref-count on implementation copy construction
    auto* k = new Test_interface( *i);
    MI_CHECK_EQUAL( 1, k->get_reference_count());
    MI_CHECK_EQUAL( 3, i->get_reference_count());

    // Free all implementations
    MI_CHECK_EQUAL( 2, i->release());
    MI_CHECK_EQUAL( 1, i->release());
    MI_CHECK_EQUAL( 0, i->release());
    MI_CHECK_EQUAL( 1, j->release());
    MI_CHECK_EQUAL( 0, j->release());
    MI_CHECK_EQUAL( 0, k->release());
}

MI_TEST_AUTO_FUNCTION( test_compare_iid )
{
    mi::base::Uuid null_id;
    null_id.m_id1 = 0;
    null_id.m_id2 = 0;
    null_id.m_id3 = 0;
    null_id.m_id4 = 0;
    mi::base::Uuid one_id;
    one_id.m_id1 = 1;
    one_id.m_id2 = 0;
    one_id.m_id3 = 0;
    one_id.m_id4 = 0;
    MI_CHECK(   ITest_interface::compare_iid( null_id));
    MI_CHECK( ! ITest_interface::compare_iid( one_id));
    MI_CHECK(   ITest_interface::compare_iid( ITest_interface::IID()));
    MI_CHECK( ! ITest_interface::compare_iid( ITest_interface_2::IID()));
    MI_CHECK(   ITest_interface_2::compare_iid( ITest_interface::IID()));
    MI_CHECK(   ITest_interface_2::compare_iid( ITest_interface_2::IID()));
}


MI_TEST_AUTO_FUNCTION( test_make_handle )
{
    mi::base::Handle< ITest_interface> iptr =
        mi::base::make_handle( interface_factory());
    MI_CHECK( iptr.is_valid_interface());
    MI_CHECK( 1 == iptr->get_reference_count());
    MI_CHECK( mi::base::make_handle( interface_factory()).get());

    mi::base::Handle< const ITest_interface> const_iptr =
        mi::base::make_handle( const_interface_factory());
    MI_CHECK( const_iptr.is_valid_interface());
    MI_CHECK( 1 == const_iptr->get_reference_count());
    MI_CHECK( mi::base::make_handle( const_interface_factory()).get());

    mi::base::Handle<  ITest_interface> iptr2 =
        mi::base::make_handle_dup( iptr.get());
    MI_CHECK( iptr2.is_valid_interface());
    MI_CHECK( 2 == iptr->get_reference_count());
    MI_CHECK( 2 == iptr2->get_reference_count());
    MI_CHECK( mi::base::make_handle_dup( iptr.get()));
    MI_CHECK( 2 == iptr->get_reference_count());

    mi::base::Handle< const ITest_interface> const_iptr2 =
        mi::base::make_handle_dup( const_iptr.get());
    MI_CHECK( const_iptr2.is_valid_interface());
    MI_CHECK( 2 == const_iptr->get_reference_count());
    MI_CHECK( 2 == const_iptr2->get_reference_count());
    MI_CHECK( mi::base::make_handle_dup( const_iptr.get()));
    MI_CHECK( 2 == const_iptr->get_reference_count());
}
