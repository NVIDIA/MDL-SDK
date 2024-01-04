/* Open up the table of contents to a specified level in the hierarchy: */

$(window).on("load", function() {
    $('.depthchoice').on('click', function(event) {
	hide_help();
	var parts_exists = $(".level0").length > 0;
	var display_depth = parseInt($(this).attr('data-depth')) - (1 ? parts_exists : 0);
	var changed = false;
	$('#navigation .nodestatus').each( function() {
	    var node = $(this);
	    var is_closed = node.parent().hasClass('closed');
	    var is_open = node.parent().hasClass('open');
	    var node_depth = parseInt($(this).attr('data-depth'));
 	    if ((is_closed && display_depth > node_depth) ||
		(is_open && display_depth <= node_depth)) {
		$(this).trigger('click');
		changed = true;
	    }
	});
    });
    
    $('html').keypress(function(event) {
	if (search_not_active()) {
	    var zero = 48;
	    for (var depth = 1; depth <= 7; depth ++) {
		if (event.which == zero + depth) {
		    $('.depthchoice[data-depth="' + depth + '"]').trigger('click');
		}
	    }
	}
    });
});
