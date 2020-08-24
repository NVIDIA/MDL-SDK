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


#include "mdl_archive_image_provider.h"
#include "utilities/string_helper.h"

Mdl_archive_image_provider::Mdl_archive_image_provider(mi::neuraylib::INeuray* neuray)
    : QQuickImageProvider(QQuickImageProvider::Pixmap)
{
    m_mdl_archive_api = neuray->get_api_component<mi::neuraylib::IMdl_archive_api>();
}

QPixmap Mdl_archive_image_provider::requestPixmap(const QString& id, QSize* size,
                                                  const QSize& requestedSize)
{
    // split into archive and file name
    std::string resource_path = id.toUtf8().constData();
    resource_path = String_helper::replace(resource_path, "%5C", "\\");

    // fetch file
    mi::base::Handle<mi::neuraylib::IReader> reader(
        m_mdl_archive_api->get_file(resource_path.c_str()));

    const mi::Sint64 file_size = reader->get_file_size();
    QByteArray buffer(file_size, 0);

    // load the image
    QPixmap image;
    QPixmap result;
    if(reader->read(buffer.data(), file_size) > 0 && image.loadFromData(buffer))
    {
        // resize
        if (requestedSize.isValid())
            result = image.scaled(requestedSize, Qt::KeepAspectRatio);
        else
            result = image;
    }

    // update size information
    *size = result.size();
    return result;
}
