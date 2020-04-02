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

import "../utils" as Utils

Rectangle {
    id: id_control
    width: parent.width
    height: 25
    color: "transparent"

    property var model: ListModel {} // nav_breadcrumbs_mockup /* for design in Qt Creator */

    signal pressed(int elementsToPop)

    Component.onCompleted: {
        // when using no mockup we want to start with an empty list
        if(id_listView.model === undefined)
            id_listView.model = id_emptyListModel;
    }

    function updateShortModes() {
        id_listView.adjustSizeOnAdd(undefined);
        id_listView.adjustSizeOnRemove(undefined);
    }

    ListModel {
        id: id_emptyListModel
    }

    function push(nav_package)
    {
        id_listView.model.append(nav_package)
    }

    function pop()
    {
        id_listView.model.remove(id_listView.model.count-1, 1)
    }


    ListView {
        id: id_listView
        anchors.fill: parent

        spacing: 0
        orientation: ListView.Horizontal
        clip: true
        focus: true

        model: id_control.model

        delegate: RowLayout {
            id: id_delegate

            anchors.top: parent.top
            anchors.bottom: parent.bottom
            spacing: 0

            ItemDelegate {
                id: id_item
                padding: 0
                Layout.fillHeight: true
                implicitWidth: id_label.implicitWidth + ((index == 0) // the following defines the spacing
                    ? id_textMetricsFull.boundingRect.height * 0.75 
                    : id_textMetricsFull.boundingRect.height * 0.25) 

                onClicked: {
                    var elementsToPop = id_listView.count - index - 1;
                    if(elementsToPop >= 0)
                        id_control.pressed(elementsToPop)
                }

                hoverEnabled: true

                Utils.Icon {
                    visible: index == 0
                    anchors.centerIn: id_item
                    width: id_textMetricsFull.boundingRect.height * 1.125
                    height: id_textMetricsFull.boundingRect.height * 1.125
                    file_path: "../graphics/home_icon"

                    // highlight if root is the only item
                    color: (id_listView.count == 1)
                        ? Material.accent
                        : Material.foreground

                    onClicked: parent.onClicked(mouse)

                    onHoveredChanged: {
                        if(hovered) id_toolTip_root.show();
                        else id_toolTip_root.hide();
                    }

                    Utils.ToolTipExtended {
                        id: id_toolTip_root
                        delay: 500
                        text: "The <b>root package</b> that contains packages of all search paths"
                        maximumWidth: 500
                        font.pointSize: 10
                    }
                }

                Label {
                    id: id_label
                    visible: index > 0
                    anchors.centerIn: id_item
                    horizontalAlignment: Text.AlignHCenter
                    font.pointSize: 10
                    text: packageShortMode 
                        ? packageName.charAt(0) 
                        : packageName

                    // highlight current (last in the list)
                    color: (index == (id_listView.count - 1)) 
                        ? Material.accent 
                        : Material.foreground
                }

                onHoveredChanged: {
                    if(hovered && packageShortMode) id_toolTip_short.show();
                    else id_toolTip_short.hide();
                }

                Utils.ToolTipExtended {
                    id: id_toolTip_short
                    delay: 500
                    text: packageName
                    maximumWidth: 500
                    font.pointSize: 10
                }
            }

            Label {
                id: id_sep
                visible: {index < (id_listView.count - 1)}
                width: visible ? implicitWidth : 0
                Layout.alignment: Qt.AlignHCenter
                font.pointSize: 10
                text: "::"
            }

            TextMetrics {
                id: id_textMetricsFull
                font: id_label.font
                elide: id_label.elide
                text: packageName
            }

            TextMetrics {
                id: id_textMetricsFirstChar
                font: id_label.font
                elide: id_label.elide
                text: packageName.charAt(0)
            }

            ListView.onAdd: {
                id_listView.model.get(index).packageShortModeSaving = id_textMetricsFull.advanceWidth - id_textMetricsFirstChar.advanceWidth;
                id_listView.adjustSizeOnAdd(id_delegate);
            }

            ListView.onRemove: id_listView.adjustSizeOnRemove(id_delegate)
        }


        function adjustSizeOnAdd(addedItem)
        {
            // check the width
            var spaceLeft = id_control.width - (id_listView.childrenRect.width + (addedItem == undefined ? 0 : addedItem.width));
            if(spaceLeft < 0)
            {
                for(var i=1; i<id_listView.count-2; ++i) // iterate from second to second last
                {
                    var item = id_listView.model.get(i);
                    if(!item.packageShortMode) // enable short mode
                    {
                        item.packageShortMode = true;
                        spaceLeft += item.packageShortModeSaving;
                        if(spaceLeft > 0)
                            break;
                    }
                }
            }
        }

        function adjustSizeOnRemove(removedItem)
        {
            // check the width
            var spaceLeft = id_control.width - (id_listView.childrenRect.width - (removedItem == undefined ? 0 : removedItem.width));
            if(spaceLeft > 0)
            {
                for(var i2=id_listView.count-1; i2>0; --i2) // iterate from last to second
                {
                    var item2 = id_listView.model.get(i2);
                    if(item2.packageShortMode) // enable short mode
                    {
                        if(spaceLeft < item2.packageShortModeSaving)
                            break;

                        item2.packageShortMode = false;
                        spaceLeft -= item2.packageShortModeSaving;
                    }
                }
            }
        }
    }
}
