
function first_basename() {
    return $(".navlink:first").attr("data-basename");
}

function scroll_to_target(target_name) {
    var content = $('#content');
    var folder_top = content.position().top;
    var target = $(target_name);
    var target_position = target.position();
    if (target_position) {
	var scroll_position = content.scrollTop() +
	    target.position().top - folder_top;
	$('#content').scrollTop(scroll_position);
    }
    highlight_visible_toc_sections();
}

function modify_links() {
    $("#page_content a").not("a.linkd").not("a:has(img)").replaceWith(function() {
	var href = $(this).attr('href');
	if ((href.indexOf("http://") != -1) ||
	    (href.indexOf("https://") != -1) ||
	    (href.indexOf(".pdf") != -1) ||
	    (href.indexOf("mailto:") != -1) ||
	    (href.indexOf("/") == 0) ||
//	    (href.indexOf("#") == 0) ||
	    (href.indexOf("#") != -1) ||
	    (href.indexOf('..') == 0))
	    return '<a href="' + href + '">' + $(this).html() + '</a>';
	var parts = href.split('#');
        //var page = 'pages/' + parts[0];
	var len = parts[0].length;
	var page = parts[0].substring(0,len-5);
	var target = '#' + parts[1];
	var result = '<span class="panelink" ';
	result += 'data-basename="' + page + '" ';
	result += 'data-target="' + target + '" ';
	result += 'data-title="' + $(this).text() + '">';
	result += $(this).text() + '</span>';
	return result;
    });
}


//var target_name = "";

function new_page_callback() {
    show_page_content();
    modify_links();
    resize();
    /*
      if (initialize) {
      initialize_nodes();
      resize_toc();
      }
    */
    current_page_callback();
}

var image_timer;
var last_basename;

function current_page_callback() {
    if (target_name == "#")
	return;
    var selector = '.navlink[data-target="' + (target_name.split("#")[1] || "") + '"]';
    update_page_title($(selector));
    resize();
    scroll_to_target(target_name);
    if (typeof MathJax !== 'undefined')
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, "page_content"]);
}


function load_page_content(basename, new_page, callback) {
    if (new_page || (basename != loaded_basename)) {
	var page_selector = 'pages/' + basename + '.html' + ' #page>';
	$("#page_content").load(page_selector, callback);
	loaded_basename = basename;
    }
    else
	callback();
}


function load_page(basename, target, callback, push_state) {
    var init_complete = target != '#';
    var push = typeof push_state !== "undefined";
    var state, state_target;

    if (init_complete) {
	var parts = window.location.hash.toString().split('#');
	state = {"basename" : parts[1], "target" : '#' + parts[2]};
	state_target = '#' + basename + target;
	if (push)
	    history.pushState(state, "", state_target);
	else
	    history.replaceState(state, "", state_target);

	load_page_content(
	    basename, basename != loaded_basename,
	    function() { 
		target_name = target; 
		initialize = false;
		callback(); } );

    } else {
	basename = first_basename();
	state = {"basename" : basename, "target" : "#"};
	state_target = "#" + basename + "#";

	init_complete = true;

	load_page_content(
	    basename, basename != loaded_basename,
	    function() { 
		target_name = target; 
		initialize = true;
		callback(); } );

	history.replaceState(state, "", state_target);
    }
    loaded_basename = basename;

}

$(window).on("load", function() {
    $(window).on("popstate", function(event) {
	if (SCROLLING)
	    return;
	var parts = window.location.hash.split("#");
	var basename = parts[1];
	var target = '#' + parts[2];

	load_page(basename, target, new_page_callback);
	loaded_basename = basename;

	return false;
    });
});
