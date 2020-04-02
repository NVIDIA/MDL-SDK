function last_highlighted_navlink(visible_navlinks) {
    var visible = $('.navlink:visible');
    var i;
    for (i = visible.length - 1; i >= 0; i--) {
	if ($(visible[i]).hasClass('highlight') ||
	    $(visible[i]).hasClass('highlight_parent'))
	    break;
    }
    return i;
}

function show_button_status(id, active_p) {
    var inactive_color = 'rgb(60%,60%,60%)';
    if (active_p)
	$(id).css('color', '').css('text-decoration', '');
    else
	$(id).css('color', inactive_color).css('text-decoration', 'none');	    
}

function set_next_last_button_status(current_index, visible_length) {
    var index = current_index;
    var length = visible_length;
    if (typeof index === "undefined") {
	var visible_navlinks = $('.navlink:visible');
	index = last_highlighted_navlink(visible_navlinks);
	length = visible_navlinks.length;
    }
    show_button_status('#next', index < length - 1);
    show_button_status('#last', index > 0);
}


$(window).on("load", function() {
    all_navlink_items = $("#navigation li");
    $('#next').on('click', function(event) {
	hide_help();
	var visible_navlinks = $('.navlink:visible');
	var i = last_highlighted_navlink(visible_navlinks);
	i += 1;
	if (i < visible_navlinks.length)
	    $(visible_navlinks[i]).trigger('click');
    });
    keypress("N", "#next");
    //arrowpress($.ui.keyCode.DOWN, "#next");

    $('#last').on('click', function(event) {    
	hide_help();
	var visible_navlinks = $('.navlink:visible');
	var i = last_highlighted_navlink(visible_navlinks);
	i -= 1;
	if (i >= 0) 
	    $(visible_navlinks[i]).trigger('click');
	set_next_last_button_status(i, visible_navlinks.length);
    });
    keypress("P", "#last");
    //arrowpress($.ui.keyCode.UP, "#last");

});
