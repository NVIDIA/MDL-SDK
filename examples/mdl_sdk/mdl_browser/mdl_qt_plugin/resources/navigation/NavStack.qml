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

Item {
    id: id_control
    property bool showBackbar: true  /* can be hidden when at root */
    property real scrollBarWidth: 7.5
    width: parent.width
    height: parent.height

    property int topMargin: 10
    property int bottomMargin: 10

    property var vm_navigation: nav_stack_mockup

    function push(nav_package)
    {
        // get the model of the next level
        var newLevel = id_control.vm_navigation.expand_package(nav_package);
        id_breadcrumbs.push(newLevel.currentPackage)
        id_stack.push(id_navigation_stack_level, {"model": newLevel});

        set_current_level()
    }

    function pop()
    {
        id_stack.pop()
        id_breadcrumbs.pop()
         
        // set_current_level() is set by the caller 
        // (breadcrumbs may pop multiple times)    // update c++ model
    }

    // updates the c++ side of the navigation stack
    function set_current_level() 
    {
        vm_navigation.set_current_level(id_stack.currentItem.model);    // update c++ model
        vm_navigation.update_presentation_counters();       // updates presentation counters
        id_stack.currentItem.currentIndex = -1; // deselected module
    }

    Component.onCompleted: {
        set_current_level()
    }

    Timer {
        id: id_timer

        // helper function to schedule a certain callback multiple times
        function delayIterated(duration, iterations, callback, finished) {
            id_timer.interval = duration;
            id_timer.repeat = true;

            var triggerFunction = function(){
                if(iterations > 0) {
                    callback();
                    iterations--;
                }
                if(iterations <= 0) {
                    id_timer.repeat = false;
                    id_timer.triggered.disconnect(triggerFunction);
                    finished();
                }
            }

            id_timer.triggered.connect(triggerFunction);
            id_timer.start();
        }
    }

    onWidthChanged: {
        id_breadcrumbs.updateShortModes()
    }

    Item {
        id: id_header
        anchors.left: parent.left
        anchors.leftMargin: 5
        anchors.right: parent.right
        anchors.rightMargin: 5
        anchors.top: parent.top
        height: 35

        NavBreadcrumbs {
            id: id_breadcrumbs
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            height: 25

            onPressed: {
                id_timer.delayIterated(
                    25,                                 // delay between iterations
                    elementsToPop,                      // iteration count
                    function() { id_control.pop(); },   // each iteration  
                    function() { set_current_level(); } // when finished
                ); 
            }

            Component.onCompleted: {
                id_breadcrumbs.push(id_stack.currentItem.model.currentPackage)
            }
        }
    }

    Item {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: id_header.bottom

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: 20
            color: Material.background
            layer.enabled: true
            layer.effect: InnerShadow {
                visible: !id_control.isCurrent
                horizontalOffset: 0
                verticalOffset: 4
                radius: 12.0
                samples: 16
                color: "#40000000"
            }
        }


        // back button bar on the left
        ItemDelegate {
            id: id_backBar
            visible: showBackbar
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 30
            spacing: 0
            padding: 0

            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                anchors.topMargin: 22
                text: "<"
                font.bold: true
                color: id_backBar.enabled
                    ? id_backBar.hovered
                        ? Material.accent
                        : Material.foreground
                    : Material.primary
            }

            onClicked: {
                id_control.pop();
                set_current_level();
            }
            enabled: {id_stack.depth > 1}
        }

        StackView {
            id: id_stack
            anchors.left: id_backBar.right
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.top: parent.top

            initialItem: id_navigation_stack_level

            Component{
                id: id_navigation_stack_level

                ListView {
                    id: id_listView
                    spacing: 1
                    clip: true
                    focus: true
                    cacheBuffer: 128
                    currentIndex: -1

                    header: Item {
                        height: id_control.topMargin
                    }
                    footer: Item {
                        height: id_control.bottomMargin
                    }

                    // init with the root level
                    model: id_control.vm_navigation
                        ? id_control.vm_navigation.create_root_level()
                        : null

                    // display package or module
                    delegate: NavPackage {
                        id: id_delegate
                        width: {id_listView.width - id_control.scrollBarWidth * 1.5}
                        showSepBar: index > 0
                        height: 40
                        isSelected: id_listView.currentIndex == index

                        Connections {
                            target: id_delegate
                            onClicked: {
                                var clicked_item = id_listView.model.get_package(index);

                                // clicked item is a module
                                if(packageIsModule)
                                {
                                    // module is now deselected
                                    if(id_listView.currentIndex == index)
                                    {
                                        id_listView.currentIndex = -1;
                                        vm_navigation.set_selected_module(undefined);
                                    }
                                    // module is now selected
                                    else
                                    {
                                        id_listView.currentIndex = index;
                                        vm_navigation.set_selected_module(clicked_item); 
                                    }
                                }
                                // clicked item is a package
                                else
                                {
                                    // navigate to the next level
                                    id_control.push(clicked_item);
                                    id_listView.currentIndex = -1; // also deselect
                                }
                            }
                        }
                    }

                    ScrollBar.vertical: ScrollBar {
                        implicitWidth: id_control.scrollBarWidth
                    }
                }
            }
        }
    }
}
