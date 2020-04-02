
var slide_time = 0;
var closed_symbol = '&#9656;';
var open_symbol = '&#9662;';
var toc_debug = false;
var scroll_timer;

var SCROLLING = false;


function open_if_parts() {
    var document_has_parts = false;
    $("#navigation").find(".navlink").each(function() {
	//if (!document_has_parts)
	//    console.log($(this).text());

	if ($(this).text().substring(0, 6) == 'Part 1') {
	    //console.log('FOUND');
	    document_has_parts = true;
	}
    });
    if (document_has_parts) {
	$(".depthchoice[data-depth='2']").trigger('click');
	resize_toc();
	/*

	var button = $(".depthchoice[data-depth='2']");
	console.log("button: ");
	console.log(button);
	$(button).trigger('click');
	*/
	/*
		    $('.depthchoice[data-depth="' + depth + '"]').trigger('click');
	var e = jQuery.Event("keypress");
	e.keyCode = "2";
	$("navcontrol").trigger(e);
	*/
    }
}	

function initialize_nodes() {
    var nodes = $('#navigation').find('li ul').parent();
    nodes.addClass('closed');
    nodes.children().slideUp(slide_time);
    nodes.children('div').slideDown(slide_time);
    nodes.children('.navlink').slideDown(0);
    if (toc_debug)
	nodes.css('background-color', 'pink');
    $('#navigation')
	.find('li.closed > span.nodestatus').html(closed_symbol)
	.find('.level0').html(closed_symbol);
    $('.nodestatus').slideDown(0);

}


function toc_load(basename, target, push_state) {
    var current_basename = window.location.hash.toString().split('#')[1];
    load_page(
	basename, target,
	function() { 
	    var initialize = false;
	    if (current_basename == loaded_basename)
		current_page_callback();
	    else
		new_page_callback();
	},
	push_state);
}


function toggle(selected) {
    selected.toggleClass('closed');
    selected.toggleClass('open');
}

function show_navlink_in_toc(navlink) {
    var nav = $("#navigation");
    var nav_top = nav.offset().top;
    var nav_bottom = nav_top + nav.height();
    var link_top = navlink.offset().top;
    if (link_top > nav_bottom) {
	nav.scrollTop(nav.scrollTop() - link_top - nav_bottom);
    }
}


function open_to_navlink_level(target) {
    $('#navcontrol span.depthchoice[data-depth="1"]').trigger('click');
    var navlink = $('.navlink[data-target="' + target + '"]');  
    navlink.parent().parents("#navigation li").each( function() {
	var ns = $(this).find(".nodestatus:first");
	if (ns.parent().hasClass("closed")) {
	    ns.trigger("click");
	}
    });
    $('#navfit').trigger('click');
}


function toggle_current_navlink() {
    $('html').keypress(function(event) {
	if (search_not_active()) {
	    var uppercase = "D".charCodeAt();
	    var lowercase = uppercase + 32;
	    if (event.which == uppercase || event.which == lowercase) {
		$("#navigation .highlight").siblings(".nodestatus").trigger('click');
	    }
	}
    });
}

$(window).on("load", function() {
    toggle_current_navlink();

    $("#focus").on("click", function () {
	var navlink = $(".highlight");
	var basename = navlink.attr('data-basename');
	var target = navlink.attr('data-target');
	load_page(basename, target, new_page_callback, false);
	open_to_navlink_level(target);
	navlink.trigger('click');
	loaded_basename = basename;
    });
    keypress("F", "#focus");


    $('#navigation .nodestatus').on('click', function(event) {
	event.stopPropagation();
	var level = $(this).parent();
	var container = level;
	var contents = level.children('li ul,.nodestatus');

	if (container.hasClass('open')) {  // close
	    toggle(container);
	    contents.slideUp(slide_time);
	    level.children('.nodestatus').slideDown(0);
	    level.children('li > span.nodestatus').html(closed_symbol); // >
	    if (toc_debug)
		container.css('background-color', 'pink');
	}
	else if (container.hasClass('closed')) { // open
	    toggle(container);
	    container.slideDown(slide_time);
	    container.children('ul').slideDown(slide_time);
	    contents.slideDown(slide_time);
	    level.children('li > span.nodestatus').html(open_symbol); // >
	    if (toc_debug)
		container.css('background-color', 'yellow');
	}
    });

    $(document).on("click", ".panelink", function(event) {
	event.preventDefault();
	update_page_title($(this));
	toc_load($(this).attr('data-basename'), $(this).attr('data-target'), true);
	return false;
    });
    

    $('.navlink').on('click', function(event) {
	event.preventDefault();
	update_page_title($(this));

	var basename = $(this).attr('data-basename');
	var target = '#' + $(this).attr('data-target'); 

	toc_load(basename, target, event.originalEvent);
	show_navlink_in_toc($(this));

	if (event.originalEvent) {
	    var node = $(this).parent().find('.nodestatus').first();
	    var is_closed = node.parent().hasClass('closed');
	    if (is_closed) {
		node.trigger('click');
		changed = true;
	    }
	}
	/*
	if (event.metaKey | event.ctrlKey) { // Both display and open all lower sections
	    $(this).parent().find('#navigation .nodestatus').each(function() {
		var never_this_deep = 61;
		var display_depth = never_this_deep;
		var node = $(this);

		var is_open = node.parent().hasClass('open');
		var node_depth = parseInt($(this).attr('data-depth'));
		if ((is_closed && display_depth > node_depth) ||
		    (is_open && display_depth <= node_depth)) {
		    $(this).trigger('click');
		    changed = true;
		}
	    });
	}
	*/
	return false;
    });

    $('#content').scroll( function() {
	SCROLLING = true;
	var header = header_in_window();
	var tag = header.get(0).tagName;
	if (tag.charAt(0) == "H") {
	    highlight_visible_toc_sections();
	}
	/*
	  clearTimeout(scroll_timer);
	  scroll_timer = setTimeout(function() {
	  if (tag.charAt(0) == "H") {
	  var parts = window.location.toString().split('#');
	  window.location = parts[0] + '#' + parts[1] + '#' + header.attr('id');
	  }
	  }, 250);
	*/
	SCROLLING = false;
	return false;
    });
    
});
