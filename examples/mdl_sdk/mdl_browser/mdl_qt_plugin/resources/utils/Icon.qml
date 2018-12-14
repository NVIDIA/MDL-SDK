import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Controls.Material 2.3
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0


Item {
    id: id_control
    width: 32
    height: 32

    property string file_path: ""           // resource files, assuming svg
    property string full_file_path: ""      // full path file:///path/file.extension (overrides the file_path)
    property color color: Material.foreground
    property bool hovered: id_icon_mouse.containsMouse
    property bool clickable: true

    signal clicked(var mouse)

    Image {
        id: id_image
        anchors.fill: parent
        width: parent.width
        height: parent.height

        // general
        fillMode: Image.PreserveAspectFit
    
        // vector graphics
        sourceSize.width: width
        sourceSize.height: height
        source: (full_file_path != "") 
            ? full_file_path                // full_file_path overrides the composed one
            : file_path + ".svg"
        mipmap: true
        asynchronous: true
    }

    ColorOverlay {
        id: id_tint;
        anchors.fill: parent
        source: id_image
        color: id_control.color
    }

    MouseArea {
        id: id_icon_mouse
        enabled: clickable
        anchors.fill: parent
        hoverEnabled: true

        onClicked: id_control.clicked(mouse)
    }
}

