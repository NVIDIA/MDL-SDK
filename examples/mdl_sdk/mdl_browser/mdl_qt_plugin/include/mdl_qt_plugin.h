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
 /// \brief external interface of the plug-in used by applications.

#ifndef MDL_SDK_EXAMPLES_MDL_QTPLUGIN_H
#define MDL_SDK_EXAMPLES_MDL_QTPLUGIN_H

#include <QApplication>
#include <QDebug>
#include <QPluginLoader>
#include <QQmlExtensionPlugin>
#include <thread>
#include "../view_model/view_model.h"

#include <mi/base/handle.h>
#include <utils/io.h>
namespace mi
{
    namespace neuraylib
    {
        class INeuray;
        class ITransaction;
    }
}

class Mdl_qt_plugin_interface;
Q_DECLARE_INTERFACE(Mdl_qt_plugin_interface, "com.qt.mdl.plugin_interface")

class QQmlApplicationEngine;

struct Mdl_browser_callbacks
{
    std::function<void(const std::string&)> on_accepted;
    std::function<void()> on_rejected;
};

struct Mdl_qt_plugin_context
{
    // constructor.
    explicit Mdl_qt_plugin_context(
        mi::neuraylib::INeuray* neuray,
        mi::neuraylib::ITransaction* transaction)
        : m_neuray(neuray, mi::base::DUP_INTERFACE)
        , m_transaction(transaction, mi::base::DUP_INTERFACE)
    {
    }

    // destructor.
    ~Mdl_qt_plugin_context()
    {
        m_transaction = nullptr;
        m_neuray = nullptr;
    }

    // Force to cache to rebuild.
    bool rebuild_module_cache = false;

    // callbacks for mdl browser events.
    Mdl_browser_callbacks* get_mdl_browser_callbacks() { return &m_mdl_browser; }

    // top level interface of the MDL SDK.
    mi::neuraylib::INeuray* get_neuray() { return m_neuray.get(); }

    // transaction to use while generating the cache.
    mi::neuraylib::ITransaction* get_transaction() { return m_transaction.get(); }

private:
    Mdl_browser_callbacks m_mdl_browser;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};

// used with non-qt applications
struct Mdl_qt_plguin_browser_handle
{
    // qualified name of the selected material or empty
    // is available after joining the thread.
    std::string result = "";

    // true if a material was selected, false if the interaction was aborted
    // is available after joining the thread.
    bool accepted = false;

    // thread in which the dialog window lives.
    // call join to wait for completion of the interaction (accept or abort)
    std::thread thread;
};

// application interface to the plugin
class Mdl_qt_plugin_interface
{
public:
    virtual ~Mdl_qt_plugin_interface() = default;

    // To be called from the application to load and initialize the plug-in
    static Mdl_qt_plugin_interface* load(
        const char* plugin_path = nullptr)  // optional path to look for the module folder
    {
        Mdl_qt_plugin_interface* plugin_interface = nullptr;

        #if defined(Q_OS_WIN)
            const QString fileSuffix = ".dll";
        #elif defined(Q_OS_MACOS)
            const QString fileSuffix = ".dylib";
        #else
            const QString fileSuffix = ".so";
        #endif

        QString path;
        if (plugin_path)
        {
            // user provided path that contains the module folder, 
            // which contains the plugin dll and the qmldir
            std::string plugin_path_s = mi::examples::io::normalize(std::string(plugin_path));
            if (plugin_path_s.back() == '/')
                plugin_path_s = plugin_path_s.substr(0, plugin_path_s.length() - 1);

            path = QString(plugin_path_s.c_str()) + "/MdlQtPlugin/mdl_qt_plugin" + fileSuffix;
        }
        else
        {
            // assuming the working directory contains the module folder
            path = QString("MdlQtPlugin/mdl_qt_plugin") + fileSuffix;
        }

        QPluginLoader* loader = new QPluginLoader(path);

        try
        {
            if (loader->load() && loader->isLoaded())
            {
                plugin_interface = qobject_cast<Mdl_qt_plugin_interface*>(loader->instance());
                if (!plugin_interface->initialize(loader))
                {
                    qDebug() << "[error] Failed to init the Mdl_qt_plugin.";
                    qDebug() << "Location: " << path << "\n";
                    loader->unload();
                    return nullptr;
                }

                qDebug() << "Plugin: loaded MdlQtPlugin v.1.0\n";
                qDebug() << "Location: " << loader->fileName() << "\n";
            }
            else
            {
                qDebug() << "[error] while opening plugin: " << loader->errorString() << "\n";
                qDebug() << "Location: " << path << "\n";
            }
        }
        catch (std::exception & ex)
        {
            qDebug() << "[error] exception while opening plugin: " << ex.what() << "\n";
            qDebug() << "Location: " << path << "\n";
        }

        return plugin_interface;
    }

    // connect the plugin with the MDL SDK instances of the application.
    // meant to be used with Qt-based applications.
    virtual bool set_context(
        QQmlApplicationEngine* engine,
        Mdl_qt_plugin_context* context) = 0;

    // in case the application is not based on qt, the browser can be used as standalone window.
    // meant to be used for non-Qt-based applications.
    virtual void show_select_material_dialog(
        Mdl_qt_plugin_context* context,
        Mdl_qt_plguin_browser_handle* out_handle) = 0;

    // to be called from the application to unload the plugin and free its resources.
    virtual void unload() = 0;

protected:
    // internal function that takes ownership of the loader in order to unload the plugin.
    virtual bool initialize(QPluginLoader* loader) = 0;

};

#endif
