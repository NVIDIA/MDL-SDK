/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/base.h
/// \brief Base API.
///
/// See \ref mi_base.

#ifndef MI_BASE_H
#define MI_BASE_H

#include <mi/base/assert.h>
#include <mi/base/atom.h>
#include <mi/base/condition.h>
#include <mi/base/config.h>
#include <mi/base/default_allocator.h>
#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/iallocator.h>
#include <mi/base/ilogger.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>
#include <mi/base/lock.h>
#include <mi/base/plugin.h>
#include <mi/base/std_allocator.h>
#include <mi/base/types.h>
#include <mi/base/uuid.h>
#include <mi/base/version.h>

/// Common namespace for APIs of NVIDIA Advanced Rendering Center GmbH.
/// \ingroup mi_base
namespace mi {

/// Namespace for the Base API.
/// \ingroup mi_base
namespace base {

/// \defgroup mi_base Base API
/// \brief Basic types, configuration, and assertion support.
///
/// \par Include File:
/// <tt> \#include <mi/base.h></tt>

} // namespace base

} // namespace mi

#endif // MI_BASE_H
