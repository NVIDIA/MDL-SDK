var search_scroll_position = 0;

/*
  function regex_debug(s) {
  if (false) {
  console.log(s);
  }
  }
*/

function searchhit_element(value) {
    return '<span class="searchhit">' + value + '</span>';
}

function clear_search_hits(html) {
    $("#search_clear").hide();
    var hit_regex = new RegExp(searchhit_element("([^<]*)"), "igm");
    return html.replace(hit_regex, "\$1");
}

function mark_search_targets(text, status) {
    var search_target = $('#search_input').val();
    //regex_debug("search target to mark: " + search_target);
    var html = $('#page_content').html();
    html = clear_search_hits(html);
    // Protect strings inside tags:
    var protect_pattern = new RegExp('(<[^>]*?)' + search_target, "igm");
    var stub = '____KLMMRTXT____\0';
    var last_html = null;
    var i = 1;
    while (html !== last_html){
	last_html = html;
	html = html.replace(protect_pattern, '$1'+stub);
	//regex_debug("protect " + i);
	i += 1;
	//regex_debug(html);
    }
    //regex_debug("PROTECTED");
    //regex_debug(html);
    // Replace search strings with marking span:
    var pattern = new RegExp('(' + search_target + ')', "igm");
    //var marked_html = html.replace(pattern, '<span class="searchhit">$1</span>');
    var marked_html = html.replace(pattern, searchhit_element("$1"));
    //regex_debug("REPLACED");
    //regex_debug(marked_html);
    var stub_regex = new RegExp(stub, "igm");
    // Restore protected strings inside tags:
    marked_html = marked_html.replace(stub_regex, search_target);
    //regex_debug("RESTORED");
    //regex_debug(marked_html);
    $('#search_content').hide();
    $('#page_content').html(marked_html).show();
    $("#search_clear").show();
    new_page_callback();
    // I didn't say I was proud of this, except perversely.
}


function show_search_content() {
    if ($('#page_content').is(":visible")) {
	$('#page_content').hide();
    }
    if ($('#help_content').is(":visible"))
	$('#help_content').hide();

    // This color should be part of the parameterization.
    $('#content').css('background-color', 'rgb(220,225,220)'); 

    $('#help').text('Help');

    if (!$('#search_content').is(":visible"))
	$('#search_content').show();

    $('.searchcontext, .searchsection').on('click', function(event) {
	var ths = $(this);
	search_scroll_position = $('#content').scrollTop();
	var filename = 'pages/' + ths.attr('data-filename') + '.html';

	target_name = '#' + ths.attr('data-sid');
	//$('#page_content').load(filename + " #page>", mark_search_targets);
	load_page(ths.attr('data-filename'), target_name, mark_search_targets);
	$('#content').css('background-color', 'white');
    });

    $('.searchcontext').mouseenter( function() {
	$(this).css('background-color', 'rgb(90%,95%,100%)');
    });

    $('.searchcontext').mouseleave( function() {
	$(this).css('background-color', 'white');
    });

    $('.searchsection').mouseenter( function() {
	$(this).css('color', 'rgb(10%,20%,50%)').css('text-decoration', 'underline');
    });

    $('.searchsection').mouseleave( function() {
	$(this).css('color', 'black').css('text-decoration', 'none');
    });

}


function search_title(target, count, section_count) {
    var title = '';
    if (count == 0) {
	title = 'No matches for "' + target + '"';
    } else {
	var suffix = count > 1 ? 'es' : '';
	title = "Search results: " + count + ' match' + suffix + ' for "' + target + '"';
	if (count != 1 && section_count != 1) 
	    title += " in " + section_count + " section" + (section_count > 1 ? "s" : "");
    }
    return '<div class="searchtitle">' + title + '</div>\n\n';
}

function search_section_title(title, marked_title) {
    var location = section_headers[title]; // from kt_js/search_targets.js
    return '<div class="searchsection" data-filename="'	+
	location[0] + '" data-sid="' + location[1] + '">' +
	marked_title + '</div>\n\n';
}

function search_item(filename, id, text) {
    return '<div class="searchcontext" ' +
	'data-filename="' + filename + '" data-sid="' + id + '">\n' +
	text + '\n' +
	'</div>\n\n';
}


$(window).on("load", function() {
    $('#search_input').keyup(function(event) {
	var enter_key = 13;
	if (event.which == enter_key) {
	    var search_target = $('#search_input').val().trim();
	    var pattern = new RegExp('('+search_target+')', "ig");
	    var html = '';
	    var count = 0;
	    var section_count = 0;
	    for (var i = 0; i < search_targets.length; ++i) {
		var title = search_targets[i][0];
		var marked_title = title;
		var title_match = false;
		if (pattern.test(title)) {
		    title_match = true;
		    count += (title.match(pattern) || []).length;
		    marked_title = title.replace(
			pattern, '<span class="searchhit">$1</span>');
		}
		var matches = '';
		for (var j = 0; j < search_targets[i][1].length; j++) {
		    var section = search_targets[i][1][j];
		    var text = section[2];
		    if (pattern.test(text)) {
			count += (text.match(pattern) || []).length;
			var marked_text = text.replace(
			    pattern, '<span class="searchhit">$1</span>');
			matches += search_item(section[0], section[1], marked_text);
		    }
		}
		if (matches != '') {
		    html += search_section_title(title, marked_title) + matches;
		    section_count += 1;
		} else if (title_match) {
		    html += search_section_title(title, marked_title);
		    section_count += 1;
		}
	    }
	    html = search_title(search_target, count, section_count) + html;
	    $('#search_content').html(html);
	    show_search_content();
	    $('#content').scrollTop(0);
	}
    });

    $('#search_label').on('click', function(event) {
	if (!$('#search_content').is(':visible')) {
	    show_search_content();
	    $('#content').scrollTop(search_scroll_position);
	} else {
	    search_scroll_position = $('#content').scrollTop();
	    show_page_content();
	}
    });
    keypress("S", "#search_label");

    $('#search_clear').on('click', function(event) {
	var content = $("#page_content");
	content.html(clear_search_hits(content.html()));
    });

    keypress("C", "#search_clear");
});
