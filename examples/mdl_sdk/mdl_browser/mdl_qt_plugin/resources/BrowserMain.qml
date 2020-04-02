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

import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Controls.Material 2.3
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick.Dialogs 1.3

import "navigation" as Navigation
import "selection" as Selection
import "side_panels" as SidePanels
import "search" as Search



Rectangle {
    id: id_control;

    // contains the selected material, function, .. or an empty string
    property string result: ""

    signal accepted()
    signal rejected()

    // colors can be found here: 
    // https://material.io/guidelines/style/color.html#color-color-palette
    Material.theme: Material.Dark
    Material.foreground: Material.color(Material.Grey, Material.Shade300)
    Material.primary: Material.color(Material.Grey, Material.Shade800)
    Material.accent: "#76b900" // nvidia green

    function reset() {
        id_mainContent.clearSelection();
    }

    function accept()
    {
        result = vm_mdl_browser.get_result();
        vm_mdl_browser.accept();
        accepted();
    }

    function reject()
    {
        result = "";
        vm_mdl_browser.reject();
        rejected();
    }

    // background
    Rectangle {
        anchors.fill: parent
        color: Material.primary
    }

    Component.onCompleted: {
        id_searchBar.set_focus();
    }

    Connections { // only confirmations
        target: vm_mdl_browser
        onClose_window: {
            accept();
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // top bar of the window, currently only used for searching and sorting controls
        Search.SearchBar {
            id: id_searchBar;
            height: 80
            Layout.fillWidth: true
            z: 2

            Connections {
                target: id_searchBar
                onSortModeChanged: vm_mdl_browser.set_sort_mode(sortBy, sortAscending)
                onViewModeChanged: id_mainContent.setViewMode(viewAs)
            }
        }

        // tabbed drawer on the left of the window
        SidePanels.SidePanelView {
            Layout.fillHeight: true
            Layout.fillWidth: true
            z: 1

            // currently the only tab for package navigation
            SidePanels.SidePanelTab {
                id: id_packagesTab
                labelText: "Packages"
                titleText: "Package Explorer"

                Navigation.NavStack {
                    Layout.fillHeight: true
                    Layout.minimumWidth: 300
                    Layout.maximumWidth: 300

                    vm_navigation: vm_mdl_browser.navigation
                    scrollBarWidth: 7.5
                }
            }

            // tabs for future features can be added here

            // list of panels to add
            SidePanels.SidePanel {
                id: id_leftPanel
                width: 300
                opened: true
                currentTabIndex: 0
                mininumWidth: 200
                maximumWidth: id_control.width - 100
                tabs: [ id_packagesTab ]
            }

            leftPanel: id_leftPanel

            // main selection panel with different view modes (list and grid)
            main: StackLayout {
                id: id_mainContent
                anchors.fill: parent
                property var model: vm_mdl_browser.selectionModel

                function setViewMode(viewAs) {
                    switch(viewAs)
                    {
                        case "List":    currentIndex = 0; break;
                        case "Grid":    currentIndex = 1; break;
                    }

                    id_mainContent.children[currentIndex].clearSelection();
                }

                function clearSelection() {
                    id_mainContent.children[currentIndex].clearSelection();
                }

                Component.onCompleted: setViewMode(vm_mdl_browser.settings.last_view_mode)

                currentIndex: 1

                // child index: 0
                Selection.SelList {
                    model: parent.model
                }

                // child index: 1
                Selection.SelGrid {
                    model: parent.model
                }
            }
        }
    }
}
