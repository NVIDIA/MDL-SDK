
function rightpos(name) {
    var elt = $('#'+name);
    var result = 0;
    if (elt.is(":visible")) {
	result = elt.position().left + elt.outerWidth();
    }
    return result;
}


function adjust_search_input_pane() {
    if ($("#navshow").is(":visible")) {
	var search_input_width = 200;
	var button_right = Math.max(
	    rightpos('navshow'), rightpos('focus'), rightpos('linkshow'), rightpos('clear'));
	var help_left = $('#help').position().left;
	var new_width = Math.min(search_input_width, help_left - button_right - 90);
	if (new_width < 10)
	    new_width = search_input_width;
	$('#search_input').width(new_width);
    }
}


function adjust_toc_slider_height() {
    $('div.ui-resizable-handle')
	.height($('#navigation').height() + $('#navigation').scrollTop());
}


function resize() {

    var top_header = header_in_window();
    var height = $(window).height();

    adjust_search_input_pane();

    var nc = $('#navcontrol');
    var control_height;
    if (nc.is(':visible'))
	control_height = $('#navcontrol').outerHeight();
    else
	control_height = 0;

    var header_height = $('#blackheader').outerHeight();
    var footer_height = $('#blackfooter').outerHeight();

    var folder_height = height - header_height - footer_height;
    var folder_components_height = folder_height - control_height;

    $('#content').height(folder_components_height);
    $('#page_content').height(folder_components_height);
    $('#search_content').height(folder_components_height);
    $('#help_content').height(folder_components_height);
    $('#navigation').height(folder_components_height);

    resize_content_width();
    set_annotate_max_width();
    set_code_comment_width();
    set_imagecode_width();
    adjust_footnotes();
    adjust_slanted_header_table();
    check_visible_link_boxes();
    adjust_toc_slider_height();
    set_code_caption_width();
    $('#bottomspacer').height($("#page_content").height() * 0.9);
    resize_images();
    adjust_widths();
    center_elements();
}


$(window).on("load", function() {
    $(window).on("resize", function(event) {
	event.preventDefault();
	maintain_content_position(resize);
    });

    $('#navigation').scroll( function() {
	adjust_toc_slider_height();	
    });

    $("#navigation").resizable({handles : 'e'});
});
