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
/// \file mi/mdl/mdl_iowned.h
/// \brief Defines the Interface_owned interface.
#ifndef MDL_IOWNED_H
#define MDL_IOWNED_H 1

namespace mi {
namespace mdl {

/// This is the base class of all "owned" interfaces.
///
/// The lifetime of owned objects is coupled with the lifetime of its owners.
/// These objects can only be constructed through factories which become its owners
/// (and might be owned themselves).
class Interface_owned
{
protected:
    /// Interface_owned objects cannot be destructed, so hide its destructor.
    virtual ~Interface_owned() = 0;
};

/*!
 * \page mdl_ownership Ownership of objects in the MDL Core library
 *
 * Most object inside the MDL Core library are created by the MDL core compiler
 * or one of its backends. For those, the lifetime can be easily defined:
 * Once a higher level object is gone, there is no reason why they should survive.
 * For instance, if an IModule (the representation of a compiled MDL module) is
 * freed, there is no use case for AST nodes or types of this module.
 *
 * Hence, most objects are "owned" in the MDL Core. An owned interface is
 * always derived from the empty Interface_owned interface. This interface does not
 * add any functionality, it just serves as a marker.
 *
 * \section mdl_factories_and_objects Factories and objects
 *
 * Objects in the MDL Core world are always created by factories, in fact there
 * is no single \c new operator in the code itself. Because most interfaces returned
 * by factories are also immutable, sometimes factories even return singletons
 * without extra notice.
 *
 * Factories also are the owner of all object they create. However, all factories
 * are also owned by higher level objects like IModule or IGenerated_code_dag
 * which then are reference counted.
 *
 * Note that the ownership cannot be transfered. Especially it is illegal
 * to use a type owned by one IModule (through its type factory) inside another module.
 * This can be the source of bad crashes.
 *
 * However, it is possible to copy objects from one factory to another.
 * Factories that support such a copy operatiorn have a method \c import() which creates a
 * copy owned by the destination factory.
 *
 * \section mdl_no_new Allocation and the new operator in MDL Core
 *
 * The MDL Core library does not use the \c new operator. Instead, all objects
 * are allocated through an allocator interface mi::base::IAllocator.
 * Because all objects are created by factories, there is no need for operator \c new.
 */

} // mdl
} // mi

#endif // MDL_IOWNED_H
