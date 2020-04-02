//DBG
var show_debugging_statements = false; //true;

function mrk(s) {
    if (show_debugging_statements) {
	console.log('----------');
	console.log(s);
    }
}

function dbg(s) {
    if (show_debugging_statements) {
	console.log(s);
    }
}
//DBG

function search_not_active() {
    return $(document.activeElement).attr('id') != 'search_input';
}

function keypress(letter, id) {
    $('html').keypress(function(event) {
	if (search_not_active()) {
	    var uppercase = letter.charCodeAt();
	    var lowercase = uppercase + 32;
	    if (event.which == uppercase || event.which == lowercase) {
		$(id).trigger('click');
	    }
	}
    });
}

function arrowpress(arrow, id) {
    $('html').keydown(function(event) {
	if (search_not_active()) {
	    if (event.which == arrow) {
		$(id).trigger('click');
	    }
	}
    });
}


function update_page_title(element) {
    var element_text = $(element).text();
    if (element_text) {
	var regex = /((?:Part )?[0-9.]*)(.*)/;
	var parts = regex.exec(element_text);
	var title;
	if (!parts[1])
	    title = parts[2].trim();
	else
	    title = parts[1] + " - " + parts[2];
	$("title").text(title);
    }
}


function highlight_visible_toc_sections(activate) {
    if (typeof activate === "undefined")
	activate = true;
    var heading = header_in_window();
    var navlink = $('.navlink[data-target="' + heading.attr("id") + '"]');
    $(".navlink").removeClass("highlight").removeClass("highlight_parent");
    navlink.addClass('highlight');
    var parents = $(navlink).parents().siblings('.navlink');
    parents.addClass('highlight_parent');
}


function show_page_content() {
    $('#search_content').hide();
    $('#help_content').hide();
    $('#page_content').show();
    $('#content').css('background-color', 'white');
    $('#help').text('Help');
    highlight_visible_toc_sections();
}
