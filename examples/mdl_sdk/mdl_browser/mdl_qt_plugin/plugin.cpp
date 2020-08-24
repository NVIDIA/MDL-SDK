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

#include "plugin.h"
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QtQuick>
#include <QtQuickControls2/QQuickStyle>
#include <iostream>
#include <thread>
#include <chrono>

#include "view_model/view_model.h"
#include "view_model/navigation/vm_nav_stack.h"
#include "view_model/navigation/vm_nav_stack_level_model.h"
#include "view_model/navigation/vm_nav_stack_level_proxy_model.h"
#include "view_model/navigation/vm_nav_package.h"

#include "view_model/selection/vm_sel_model.h"
#include "view_model/selection/vm_sel_proxy_model.h"
#include "view_model/selection/vm_sel_element.h"
#include "mdl_browser_settings.h"
#include "utilities/qt/mdl_archive_image_provider.h"
#include "utilities/platform_helper.h"

Mdl_qt_plugin::Mdl_qt_plugin()
    : m_engine(nullptr)
    , m_view_model(nullptr)
    , m_loader(nullptr)
{
}

void Mdl_qt_plugin::registerTypes(const char *uri)
{
    Q_ASSERT(uri == QLatin1String("MdlQtPlugin"));

    qmlRegisterType(QUrl("qrc:/mdlqtplugin/BrowserApp.qml"), uri, 1, 0, "BrowserApp");
    qmlRegisterType(QUrl("qrc:/mdlqtplugin/BrowserDialog.qml"), uri, 1, 0, "BrowserDialog");
    qmlRegisterType(QUrl("qrc:/mdlqtplugin/BrowserMain.qml"), uri, 1, 0, "BrowserMain");

    qmlRegisterType<View_model>(uri, 1, 0, "ViewModel");
    qmlRegisterType<VM_nav_stack>(uri, 1, 0, "NavStack");
    qmlRegisterType<VM_nav_stack_level_model>(uri, 1, 0, "NavStackLevelModel");
    qmlRegisterType<VM_nav_stack_level_proxy_model>(uri, 1, 0, "NavStackLevelProxyModel");
    qmlRegisterType<VM_nav_package>(uri, 1, 0, "NavPackage");
    qmlRegisterType<VM_sel_model>(uri, 1, 0, "SelModel");
    qmlRegisterType<VM_sel_proxy_model>(uri, 1, 0, "SelProxyModel");
    qmlRegisterType<VM_sel_element>(uri, 1, 0, "SelElement");
    qmlRegisterType<Mdl_browser_settings>(uri, 1, 0, "MdlBrowserSettings");
}

void Mdl_qt_plugin::initializeEngine(QQmlEngine* engine, const char * uri)
{
    QQmlExtensionPlugin::initializeEngine(engine, uri);
}

bool Mdl_qt_plugin::set_context(
    QQmlApplicationEngine* engine,
    Mdl_qt_plugin_context* context)
{
    m_engine = engine;

    // attach c++ back-end
    std::string app_dir = mi::examples::io::get_executable_folder();
    m_view_model = new View_model(
        context->get_neuray(),
        context->get_transaction(),
        context->get_mdl_browser_callbacks(),
        context->rebuild_module_cache,
        app_dir.c_str());

    m_engine->rootContext()->setContextProperty("vm_mdl_browser", m_view_model);

    // image provided for mdl archive thumbnails (takes ownership)
    m_engine->addImageProvider(QLatin1String("mdl_archive"),
                               new Mdl_archive_image_provider(context->get_neuray()));
    return true;
}

bool Mdl_qt_plugin::initialize(QPluginLoader* loader)
{
    m_loader = loader;
    return true;
}

void Mdl_qt_plugin::show_select_material_dialog(
    Mdl_qt_plugin_context* context,
    Mdl_qt_plguin_browser_handle* out_handle)
{
    // setup callbacks to get the result
    context->get_mdl_browser_callbacks()->on_accepted = [&](const std::string& s)
    {
        out_handle->result.append(s);
        out_handle->accepted = true;
    };

    out_handle->result = "";
    out_handle->result.reserve(4096);
    out_handle->accepted = false;
    out_handle->thread = std::thread([&, context]()
    {
        // global Qt Setting that is used for the example
        QQuickStyle::setStyle("Material");
        #if defined(Q_OS_WIN)
            QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
            // QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL); // hangs in remote desktop mode
            QCoreApplication::setAttribute(Qt::AA_UseOpenGLES);
        #endif

        // view model as connection between C++ and QML world
        std::string app_dir = mi::examples::io::get_executable_folder();
        View_model* view_model = new View_model(
            context->get_neuray(),
            context->get_transaction(),
            context->get_mdl_browser_callbacks(),
            context->rebuild_module_cache,
            app_dir.c_str());

        // create and run an internal application
        int argc = 0;
        QGuiApplication app(argc, nullptr);
        QQmlApplicationEngine engine;
        engine.rootContext()->setContextProperty("vm_mdl_browser", view_model);

        // image provided for mdl archive thumbnails (takes ownership)
        engine.addImageProvider(QLatin1String("mdl_archive"),
            new Mdl_archive_image_provider(context->get_neuray()));

        // Run the Application
        app.setWindowIcon(QIcon(":/mdlqtplugin/graphics/mdl_icon.svg"));
        engine.load(":/mdlqtplugin/BrowserApp.qml");
        int exit_code = app.exec();
        if (exit_code != 0)
        {
            qDebug() << "[error] Qt application crashed.\n";
        }
        engine.removeImageProvider(QLatin1String("mdl_archive"));
        delete view_model;
    });
}

void Mdl_qt_plugin::unload()
{
    if (m_view_model)
        delete m_view_model;

    m_loader->unload();
}
