/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief qt image provided that allows to load images thar are embedded in an archive


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_ARCHIVE_IMAGE_PROVIDER_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_MDL_ARCHIVE_IMAGE_PROVIDER_H


#include <QQuickImageProvider>
#include <QtGui/QImage>

#include "example_shared.h"

namespace mi
{
    namespace neuraylib
    {
        class INeuray;
        class IMdl_archive_api;
    }
}


class Mdl_archive_image_provider : public QQuickImageProvider
{
public:
    explicit Mdl_archive_image_provider(mi::neuraylib::INeuray* neuray);
    virtual ~Mdl_archive_image_provider() = default;

    QPixmap requestPixmap(const QString &id, QSize *size, const QSize &requestedSize) override;

private:
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> m_mdl_archive_api;
};

#endif
