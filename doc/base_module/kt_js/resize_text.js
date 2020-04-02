/* Toggling between text with navigation and text without: */

function resize_content_width() {
    var left = rightpos("navigation");
    var maximum_width = $('#content').innerWidth;
    var content_width = Math.max(minimum_content_width, maximum_width - left);
    $("#content").width(content_width);
    $("#navigation").width(maximum_width - content_width);
}

//DBG
function draw(x, y, width, height, color) {
    //var width = x2 - x1 + 1;
    //var height = y2 - y1 + 1;
    var css = 'position:absolute; width: ' + width + 'px; height: ' + height + 'px; ';
    css += 'left: ' + x + 'px; top: ' + y + 'px; ';
    css += 'z-index: 999; background-color: ' + color + ';';
    var div = '<div id="draw" style=' + css + '"></div>';
    $("html #draw").remove();
    $("html").append(div);
}
//DBG

function header_in_window(tags) {
    var content_top = $('#content').offset().top;
    var content_bottom = content_top + $('#content').height();
    var result = '';
    var last;
    var header_tags = '#page_content > div.part, #page_content > h1,' +
	'#page_content > h2, #page_content > h3, #page_content > h4, #page_content > h5';
    if (typeof tags !== 'undefined')
	header_tags += "," + tags;
    $(header_tags).map( function() {
	if (result == '') {
	    var ths = $(this);
	    var midpoint = ths.offset().top + (ths.height()/2); // - ths.outerHeight() / 2;
	    if ((midpoint > content_top) && (midpoint < content_bottom)) {
		result = ths;
	    } else if (midpoint < content_top) {
		last = ths;
	    }
	}
    });
    if (result == '')
	result = last;
    return result;
}


function maintain_content_position(func) {
    var top_header =
	header_in_window("#page_content > p, #page_content > div");
    var original_top_pos = top_header.position().top;  
    func();
    var new_top_pos = top_header.position().top;
    var offset = original_top_pos - new_top_pos;
    if (offset != 0) {
	$('#content').scrollTop($('#content').scrollTop() - offset);
    }
}

var last_width;

function navshow() {
    $('#navigation').toggle();
    $('#navfit').toggle();

    if ($('#navigation').is(":visible")) {
	$('#navigation').width(last_width);
	$('#content').css('border-width', '0 0 0 1px');
	$('#navshow').html('Text only');
	$('.depthchoice').show();
	$('#next').show();
	$('#last').show();
	$('#focus').show();
    }
    else {
	last_width = $('#navigation').width();
	$('#content').css('border-width', '0 0 0 0');	
	$('#navshow').html('Table of contents');
	$('.depthchoice').hide();
	$('#next').hide();
	$('#last').hide();
	$('#focus').hide();
    }
    resize_content_width();
    resize_images();
    adjust_widths();
    center_elements();
    set_code_comment_width();
    set_imagecode_width();
    adjust_footnotes();
}


$(window).on("load", function() {
    $('#navshow').on('click', function(event) {
	maintain_content_position(navshow);
    });
    keypress("T", "#navshow");
});
