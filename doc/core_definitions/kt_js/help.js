
var search_displayed;
var content_position;

function hide_help() {
    var help_label = $('#help');
    help_label.text("Help");
    $('#help_content').hide();	    
    if (search_displayed) {
	show_search_content();
	$('#content').scrollTop(content_position);
	$('#search_content').show();		
	$('#page_content').hide();
    } else {
	$('#search_content').hide();
	$('#page_content').show();
	$('#content').scrollTop(content_position);
	highlight_visible_toc_sections();
    }
}

$(window).on("load", function() {
    $('#help').on('click', function() {
	$('#content').css('background', 'white');
	var text = $(this).text();
	if (text == "Help") {
	    search_displayed = $('#search_content').is(':visible');
	    $(this).text("Back");
	    content_position = $('#content').scrollTop();
	    $('#page_content').hide();
	    $('#search_content').hide();
	    $('#help_content').load('help.html').scrollTop(0).show();
	} else {
	    hide_help();
	}
    });
    keypress("H", "#help");
});
