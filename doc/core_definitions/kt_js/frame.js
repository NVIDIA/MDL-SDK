
function frame_resize() {
    /*
      var top_header = header_in_window();
      if (top_header) {
      var start_scroll = $('#content').scrollTop();
      var header_position = top_header.position().top;
      }
    */

    var height = $(window).innerHeight();
    var header_height = $('#blackheader').outerHeight();
    var footer_height = $('#blackfooter').outerHeight();
    var content_height = height - header_height - footer_height;
    $('#content').innerHeight(content_height);

    /*
      var control_height = $('#navcontrol').outerHeight();
      var offset = 0; //-1;
      var folder_height = height - header_height - footer_height - offset;
      var folder_components_height = folder_height - control_height;
      $('#content').height(folder_components_height);
      $('#page_content').height(folder_components_height);
      $('#search_content').height(folder_components_height);
      $('#help_content').height(folder_components_height);
      $('#navigation').height(folder_components_height);
      navigation_height = $('#navigation').height();
      //$('#search_input').width(Math.min(200, $(window).width() - 350));
      resize_content_width();
      adjust_search_input_pane();
      adjust_images();

      if (top_header) {
      var new_header_position = top_header.position().top;
      $('#content').scrollTop(start_scroll + new_header_position - header_position);
      }
    */
}

$(window).on("load", function() {
    $('#content').css('background-color', 'white');
    $(window).resize(function() {
	frame_resize();
    });
    frame_resize();
});
