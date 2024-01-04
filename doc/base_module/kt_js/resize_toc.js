/* Resizing the table of contents to fit into the pane: */

var navigation_width_history = [];

function current_navigation_width() {
    var width = 0;
    var right_margin = 16;
    $('span.navlink:visible').map( function() {
	var element_width = $(this).outerWidth() + $(this).offset().left;
	width = Math.max(width, element_width);
    });
    return width + right_margin;
}


function resize_toc() {
    var navigation = $('#navigation');
    if ($('#navigation').is(":visible")) {
	var current_width = current_navigation_width();
	var width = Math.min($("html").width() - minimum_content_width, current_width);
	$('#navigation').width(width);
	resize_content_width();
	resize_images();
	adjust_widths();
	center_elements();
	set_annotate_max_width();
	set_code_comment_width();
	set_imagecode_width();
    }
}

$(window).on("load", function() {
    $('#navfit').on('click', function(event) {
	maintain_content_position(resize_toc);
    });
    keypress("R", "#navfit");
});
