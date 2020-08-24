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
/// \brief Plugin registration

#ifndef MDL_SDK_EXAMPLES_MDL_PLUGIN_H 
#define MDL_SDK_EXAMPLES_MDL_PLUGIN_H

#include <QtQml/QQmlExtensionPlugin>
#include "include/mdl_qt_plugin.h"

class Mdl_archive_image_provider;
class View_model;

class QPluginLoader;



class Mdl_qt_plugin : public QQmlExtensionPlugin, public Mdl_qt_plugin_interface
{
    Q_OBJECT
    Q_INTERFACES(Mdl_qt_plugin_interface)
    Q_PLUGIN_METADATA(IID "org.qt-project.Qt.QQmlExtensionInterface/1.0" FILE "metadata.json")

public:
    explicit Mdl_qt_plugin();
    virtual ~Mdl_qt_plugin() = default;
    void registerTypes(const char *uri);
    void initializeEngine(QQmlEngine *engine, const char *uri) override;
    bool set_context(
        QQmlApplicationEngine* engine,
        Mdl_qt_plugin_context* context) final override;
    void show_select_material_dialog(
        Mdl_qt_plugin_context* context,
        Mdl_qt_plguin_browser_handle* out_handle) final override;
    void unload() final override;

protected:
    bool initialize(QPluginLoader* loader) final override;

private:
    QQmlApplicationEngine* m_engine;
    View_model* m_view_model;
    QPluginLoader* m_loader;
};
#endif
