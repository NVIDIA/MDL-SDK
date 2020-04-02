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

Item {
    id: id_control
    width: parent.width
    height: parent.height

    property var sortByColor: Material.color(Material.Grey, Material.Shade500)
    property int sortingCriterionSpacing: 5
    property int searchFieldWidth: 500
    property int presentationBarHeight: 20

    property string sortBy: "Relevance"
    signal sortModeChanged(string sortBy, bool sortAscending)

    property string viewAs: vm_mdl_browser.settings.last_view_mode
    signal viewModeChanged(string viewAs)

    function set_sort_mode(sortBy, sortAscending)
    {
        sortModeChanged(sortBy, sortAscending);
        apply_settings();
    }

    function setViewMode(viewAs)
    {
        vm_mdl_browser.settings.last_view_mode = viewAs;
        viewModeChanged(viewAs);
    }

    function set_search_query(query)
    {
        vm_mdl_browser.update_user_filter(query);
    }

    function set_focus()
    {
        id_searchField.forceActiveFocus()
    }

    function apply_settings()
    {
        // settings do not notify on change
        id_control.sortBy = vm_mdl_browser.settings.last_sort_critereon;
        id_sort_relevance.ascending = vm_mdl_browser.settings.sort_by_relevance_ascending;
        id_sort_name.ascending = vm_mdl_browser.settings.sort_by_name_ascending;
        id_sort_date.ascending = vm_mdl_browser.settings.sort_by_date_ascending;
    }

    Component.onCompleted: {
        apply_settings();
        if(id_control.sortBy == "Relevance")
            set_sort_mode(id_control.sortBy, id_sort_relevance.ascending)
        else if(id_control.sortBy == "Name")
            set_sort_mode(id_control.sortBy, id_sort_name.ascending)
        else if(id_control.sortBy == "Modification Date")
            set_sort_mode(id_control.sortBy, id_sort_date.ascending)
    }

    Rectangle {
        height: 100
        anchors.fill: parent
        color: Material.background

        layer.enabled: true
        layer.effect: DropShadow {
            horizontalOffset: 0
            verticalOffset: 0
            radius: 12.0
            samples: 16
            color: "#BB000000"
        }
    }

    Label {
        anchors.verticalCenter: id_searchField.verticalCenter
        anchors.verticalCenterOffset: -implicitHeight / 4
        anchors.left: parent.left
        anchors.right: id_searchField.left
        anchors.rightMargin: sortingCriterionSpacing * 2

        text: "apply filter:"
        color: sortByColor
        horizontalAlignment: Text.AlignRight
        font.pointSize: 10
    }

    TextField {
        id: id_searchField;
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.rightMargin: 30
        anchors.topMargin: 10
        width: id_control.searchFieldWidth
        leftPadding: 10
        font.pointSize: 10

        placeholderText: "e.g.: metal brushed -steel"

        selectByMouse: true
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.IBeamCursor
            acceptedButtons: Qt.NoButton
        }

        onTextEdited: {
            id_control.set_search_query(id_searchField.text)
        }

        onAccepted: {
            id_control.set_sort_mode("Relevance", false);
        }

        Utils.Icon {
            id: id_clearSearchField
            visible: id_searchField.text !== ""

            property int padding: 12

            anchors.top: parent.top;
            anchors.topMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 10

            height: parent.height - padding * 2
            width: parent.height - padding * 2

            file_path: "../graphics/clear_icon"

            color: hovered
                ? Material.foreground
                : id_control.sortByColor

            onClicked: {
                id_searchField.text = ""
                id_searchField.forceActiveFocus()
                id_control.set_search_query(id_searchField.text)
            }
        }
    }

    RowLayout{
        id: id_row_layout
        anchors.left: parent.left
        anchors.leftMargin: 30
        anchors.right: parent.right
        anchors.rightMargin: 30

        anchors.bottom: parent.bottom
        anchors.bottomMargin: 5
        height: id_control.presentationBarHeight
        spacing: 0

        Item {
            Layout.alignment: Qt.AlignVCenter
            Layout.fillWidth: true
        }

        // -- SORTING ---------------------------------------------

        Label {
            Layout.alignment: Qt.AlignVCenter
            Layout.rightMargin: 2 * id_control.sortingCriterionSpacing
            text: "sort by:"
            color: sortByColor
            font.pointSize: 10
        }

        SortingCriterion {
            id: id_sort_relevance
            Layout.rightMargin: id_control.sortingCriterionSpacing
            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: id_control.presentationBarHeight

            name: "Relevance" // ATTENTION: if changed also change in 'View_model.cpp'
            ascending: false
            isSelected: (id_control.sortBy == name)
            onSelected: set_sort_mode(name, ascending)
        }

        SortingCriterion {
            id: id_sort_name
            Layout.rightMargin: id_control.sortingCriterionSpacing
            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: id_control.presentationBarHeight

            name: "Name" // ATTENTION: if changed also change in 'View_model.cpp'
            ascending: true
            isSelected: (id_control.sortBy == name)
            onSelected: set_sort_mode(name, ascending)
        }

        SortingCriterion {
            id: id_sort_date
            Layout.rightMargin: id_control.sortingCriterionSpacing
            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: id_control.presentationBarHeight

            name: "Modification Date" // ATTENTION: if changed also change in 'View_model.cpp'
            ascending: true
            isSelected: (id_control.sortBy == name)
            onSelected: set_sort_mode(name, ascending)
        }

        // -- VIEW MODE ---------------------------------------------

        Label {
            Layout.alignment: Qt.AlignVCenter
            Layout.leftMargin: 2 * id_control.sortingCriterionSpacing
            Layout.rightMargin: 2 * id_control.sortingCriterionSpacing
            text: "view as:"
            color: sortByColor
            font.pointSize: 10
        }

        SelectionViewOption {
            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: id_control.presentationBarHeight
            Layout.preferredWidth: 25

            name: "List"
            iconFilePath: "../graphics/list_icon"
            isSelected: id_control.viewAs === name
            onSelected: id_control.setViewMode(name)
        }

        SelectionViewOption {
            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: id_control.presentationBarHeight
            Layout.preferredWidth: 25

            name: "Grid"
            iconFilePath: "../graphics/grid_icon"
            isSelected: id_control.viewAs === name
            onSelected: id_control.setViewMode(name)
        }
    }
}
