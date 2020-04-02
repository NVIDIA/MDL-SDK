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

// inspired by drawer
// ATTENTION: this implementation works only for the left side
Item {
    id: id_control
    height: parent.height
    width: parent.width

    property list<SidePanelTab> tabs

    property real position: opened ? 1.0 : 0.0 // fully opened vs. closed
    property bool rightSide: true
    property bool opened: false
    property int currentTabIndex: 0

    property real mininumWidth: 200
    property real maximumWidth: 65000

    // internal properties for alignment
    property real topOffset: 20
    property real labelOffset: 75
    property real labelWidth: 25

    Component.onCompleted: updateTabStyle()

    Item {
        id: id_shadowCaster
        anchors.fill: parent
        anchors.topMargin: topOffset

        Pane {
            id: id_main
            anchors.fill: parent
            anchors.rightMargin: labelWidth
            padding: 0

            Item {
                id: id_header
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                height: 28

                Label {
                    id: id_headerTitle
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    anchors.top: parent.top
                    anchors.topMargin: 5

                    text: ((currentTabIndex < 0) ? "Titel" : id_control.tabs[currentTabIndex].titleText).toUpperCase()
                    horizontalAlignment: Text.AlignLeft
                    font.pointSize: 10
                }

                ItemDelegate {
                    anchors.right: parent.right
                    anchors.rightMargin: 0
                    anchors.top: parent.top
                    anchors.topMargin: 3

                    width: id_hideLabel.implicitWidth + 20
                    height: id_hideLabel.implicitHeight + 4

                    Label {
                        id: id_hideLabel
                        anchors.centerIn: parent
                        text: "x"
                    }

                    onClicked: id_control.close()
                }
            }

            StackLayout {
                id: id_tabContents
                anchors.top: id_header.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom

                currentIndex: Math.max(0, id_control.currentTabIndex)

                children: tabs
                z: -1
            }
        }

        // list of tabs
        ListView {
            id: id_tabList

            anchors.top: parent.top
            anchors.topMargin: labelOffset
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            width: labelWidth

            spacing: 4
            model: tabs

            delegate: SidePanelTabDelegate {
                id: id_delegate
                text: labelText
                width: labelWidth
                isCurrent: highlighted

                onClicked: {
                    id_control.currentTabIndex = index
                    if(!id_control.opened)
                        open()
                }
            }
        }

        Rectangle {
            id: id_dragHandle
            anchors.top: id_main.top
            anchors.bottom: parent.bottom

            Component.onCompleted: {
                x = id_main.width - 2
            }

            width: 4
            color: "transparent"

            onXChanged: {
                if(x > 0) {
                    var targetWidth = id_dragHandle.x + 2 + id_control.labelWidth
                    id_control.width = Math.min(id_control.maximumWidth, Math.max(targetWidth, id_control.mininumWidth))
                }
            }

            MouseArea {
                id: id_dragArea
                anchors.fill: parent

                hoverEnabled: true
                cursorShape: Qt.SizeHorCursor

                drag.target: id_dragHandle
                drag.axis: Drag.XAxis

                onReleased: {
                    id_dragHandle.x = id_main.width - 2
                }
            }
        }
    }


    function open() {
        opened = true
    }

    function close() {
        opened = false
        currentTabIndex = -1
    }

    function updateTabStyle() {
        for(var t=0; t<tabs.length; ++t)
            tabs[t].highlighted = (t == currentTabIndex)
    }

    onCurrentTabIndexChanged: updateTabStyle()


    Behavior on position {

        NumberAnimation {
            duration: 250
            easing.type: Easing.InOutQuad
        }
    }

    transform: Translate { x: position * (width - labelWidth) - (width - labelWidth) }


    DropShadow {
        anchors.fill: id_shadowCaster
        horizontalOffset: 0
        verticalOffset: 0
        radius: 12.0
        samples: 16
        color: "#BB000000"
        source: id_shadowCaster
    }
}
