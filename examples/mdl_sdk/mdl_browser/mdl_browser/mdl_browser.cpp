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

//Defines the entry point for the console application.

// example helper
#include "mdl_sdk_helper.h"
#include "mdl_browser_command_line_options.h"
#include "example_shared.h"

// system
#include <iostream>

// Qt
#include <QtQuick>
#include <QtQuickControls2/QQuickStyle>
#include <QQmlApplicationEngine>
#include <QtSvg>

// MDL QtPlugin
#include <mdl_qt_plugin.h>

// illustrates the use of the mdl qt module in a Qt based application
int run_demo_qt_application(Mdl_qt_plugin_interface* plugin,
                            mi::neuraylib::INeuray* neuray,
                            mi::neuraylib::ITransaction* transaction,
                            Mdl_browser_command_line_options& options)
{
    // global Qt Setting that is used for the example
    QQuickStyle::setStyle("Material");
    #if defined(Q_OS_WIN)
        QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
        // QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL); // hangs in remote desktop mode
        QCoreApplication::setAttribute(Qt::AA_UseOpenGLES);
    #endif

    // create a Qt application
    int argc = 0;
    QGuiApplication app(argc, nullptr);
    QQmlApplicationEngine engine;

    // create and pass the context
    Mdl_qt_plugin_context plugin_context(neuray, transaction);
    plugin_context.rebuild_module_cache = options.cache_rebuild;
    plugin->set_context(&engine, &plugin_context);

    // setup callbacks to get the result (not required)
    plugin_context.get_mdl_browser_callbacks()->on_accepted = [&](const std::string& s)
    {
        std::cout << "[info] Selection: " << s.c_str() << "\n";
    };
    plugin_context.get_mdl_browser_callbacks()->on_rejected = [&]()
    {
        std::cout << "[info] Selection: nothing\n";
    };

    // load main window
    app.setWindowIcon(QIcon(QString(":/mdlqtplugin/graphics/mdl_icon.svg")));
    engine.load(":/Main.qml");

    // Run the Application
    int exit_code = app.exec();
    if (exit_code != 0)
        std::cerr << "[error] Qt application crashed.\n";

    return exit_code;
}

// illustrates the standalone usage of the mdl browser in a non-Qt-based application
int run_non_qt_application(Mdl_qt_plugin_interface* plugin,
                           mi::neuraylib::INeuray* neuray,
                           mi::neuraylib::ITransaction* transaction,
                           Mdl_browser_command_line_options& options)
{
    do
    {
        // create and pass the context for the (next) invocation of the browser
        Mdl_qt_plugin_context* plugin_context = new Mdl_qt_plugin_context(neuray, transaction);
        plugin_context->rebuild_module_cache = options.cache_rebuild;

        Mdl_qt_plguin_browser_handle selection_handle;
        plugin->show_select_material_dialog(plugin_context, &selection_handle);

        // ...
        // do something else in your application
        // ...

        selection_handle.thread.join();
        if (selection_handle.accepted)
        {
            // (only) output consumed by an application that uses this browser
            // instead of implementing an own one
            std::cout << selection_handle.result;
        }
        else
        {
            std::cout << "[info] nothing selected\n";
        }
        std::cout << std::flush;
        delete plugin_context;

    } while (options.keep_open);

    exit_success();
}


int MAIN_UTF8(int argc, char* argv[])
{
    // parse command line options of the example
    Mdl_browser_command_line_options options(argc, argv);

    // Access the MDL SDK
    // every application that used the MDL SDK does this in some way
    mi::base::Handle<mi::neuraylib::INeuray> neuray(load_mdl_sdk(options));
    if (!neuray)
    {
        std::cout << "[MDL Browser] Failed to load the MDL SDK libraries.\n";
        return -1;
    }

    // create a transaction that used for loading modules and building the cache
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(create_transaction(neuray.get()));
    if (!transaction)
    {
        std::cout << "[MDL Browser] Failed to create a DB transaction.\n";
        return -1;
    }

    // load and initialize the MDL Qt Plugin
    Mdl_qt_plugin_interface* plugin = Mdl_qt_plugin_interface::load(
        mi::examples::io::get_executable_folder().c_str());
    if (!plugin)
    {
        std::cout << "[MDL Browser] Failed to load the MdlQtPlugin.\n";
        return -1;
    }

    // depending on the use-case, the browser can be used from the as standalone windows in case
    // of an application that is not using Qt so far. If the application is based on Qt, the
    // browser can be included from the QML module.
    int exit_code = -1;
    if (options.no_qt_mode)
        exit_code = run_non_qt_application(plugin, neuray.get(), transaction.get(), options);
    else
        exit_code = run_demo_qt_application(plugin, neuray.get(), transaction.get(), options);

    // unload the plugin
    plugin->unload();

    // close the MDL SDK
    transaction->commit();
    transaction = nullptr;

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        std::cerr << "Failed to shutdown the SDK.\n";

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        std::cerr << "Failed to unload the SDK.\n";

    return exit_code;
}

COMMANDLINE_TO_UTF8
