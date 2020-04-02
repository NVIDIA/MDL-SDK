
var max_box_width_without_scaling = 400;
var max_box_width_proportional_to_code = 2.2;

function describe(element, comment) {
    console.log("child: " + $(element).prop("tagName") +
		", id: " + $(element).attr("id") + ", " + comment +
		" [" + $(element).text() + "]");
}

function annotate_debug() {
    var annotate = $(".annotatebox");
    annotate.css("background-color", "lightblue");
    annotate.find(".codecomment")
	.css("background-color", "rgb(100%,100%,90%)");
    annotate.find(".uncommented")
	.css("background-color", "rgb(95%,100%,90%)");
}

function right_pos(element) {
    return $(element).offset().left + $(element).outerWidth();
    //    - $(".annotatebox").offset().left;
	//- 0;
}

function code_width(box) {
    var result = 0;
    $(box).find(".codeblock").each( function(index) {
	//result = Math.max(result, $(this).outerWidth());
	result = Math.max(result, right_pos($(this)));
    });
    $(box).find(".uncommented").each( function(index) {
	result = Math.max(result, $(this).outerWidth());
	//?result = Math.max(result, right_pos($(this)));
    });
    
    return result;
}

function set_annotate_max_width() {
    $(".annotatebox").each(function(index) {
	var max_x = 200;
	var box = $(this);
	$(box).width(2000);
	$(box).find(".annotatedcode").each(function(index) {
	    var code = $(this);
	    var width = 0;
	    $(code).children().each( function() {
		width += $(this).outerWidth();
		//width += $(this).innerWidth();		
	    });
	    max_x = Math.max(max_x, width);
	    max_x += 2;
	});
	$(box).find(".uncommented").each(function(index) {
	    max_x = Math.max(max_x, $(this).outerWidth());
	    max_x += 2;
	});
	//console.log("max_x: " + max_x);

	max_x += 20; // annotate box padding

	var scale = parseFloat($(this).attr("data-scale"));
	max_x *= scale;
	//console.log("data-scale: " + scale);
	/*
	  if (1 && max_x > max_box_width_without_scaling) {
	  //console.log("max_x: " + max_x);
	  //console.log("proportional: " + code_width(box) 
	  //	     * max_box_width_proportional_to_code);
	  max_x = Math.min(max_x, 
	  code_width(box) * max_box_width_proportional_to_code);
	  }
	*/
	var current_max_width = $(box).attr("data-maxwidth");
	if (typeof current_max_width === "undefined") {
	    $(box).attr("data-maxwidth", max_x);
	} else if (current_max_width > max_x) {
	    max_x = current_max_width;
	    $(box).attr("data-maxwidth", max_x);
	}
	$(box).innerWidth(max_x);

    });

}

function set_code_caption_width() {
    $(".annotatebox").each(function(index) {
	    var box = $(this);
        $(box).find(".codecaption").innerWidth(
            $(box).find(".annotateline").width())
    });
}


function find_parent_inner_width(element) {
    var result = element;
    var in_table = false;
    var width;
    var box_offset = parseFloat($(".annotatebox").css("margin-left")) +
	parseFloat($(".annotatebox").css("margin-right")) +
	parseFloat($(".annotatepad").css("padding-left")) +
	parseFloat($(".annotatepad").css("padding-right"));

    while (true) {
	result = $(result).parent();
	var tag = $(result).prop("tagName");
	if (tag == "TD" || tag == "DD") {
	    //console.log("parent: " + tag);
	    in_table = true;
	    break; }
	if ($(result).attr("id") == "page_content") {
	    //console.log("parent: #page_content");
	    break; }
	if (tag == "BODY") {
	    //console.log("parent: <body>");
	    break; }
    }
    if (in_table) {
	var other_cell = $(result).closest("tr").children("td:first");
	//console.log("other_cell:");
	//console.log(other_cell);
	var cell_width = $(other_cell).outerWidth() +
	    parseFloat($(other_cell).css("padding-left")) +
	    parseFloat($(other_cell).css("padding-right")) +
	    parseFloat($(other_cell).css("margin-left")) +
	    parseFloat($(other_cell).css("margin-right"));
	//console.log("other_cell width: " + cell_width);
	width = $("#page_content").innerWidth() - cell_width - box_offset;
    } else {
	width = result.width() -
	    parseFloat(result.css("margin-left")) -
            parseFloat(result.css("margin-right")) -
	    parseFloat(result.css("padding-left")) -
	    parseFloat(result.css("padding-right")) -
	    box_offset;
    }
    //console.log("parent inner width: " + width);
    return width;
}


function set_code_comment_width() {
    $(".codecomment").each( function(index) {
        var box = $(this).closest(".annotatebox");
	if (box.attr("data-fixed") === "true") {
	    $(this).css("display", "inline").css("float", "none");
	} else {
	    var scale = parseFloat($(box).attr("data-scale"));
	    var parent_inner_width = find_parent_inner_width($(box));
	    var new_box_width = 
		Math.min(parseFloat($(box).attr("data-maxwidth")), 
			 parent_inner_width);
	    //console.log("parent inner: " + parent_inner_width + " max: " + $(box).attr("data-maxwidth"));
	    $(box).width(new_box_width);
	    var available_width = $(box).innerWidth();
	    var indent_width = $(this).siblings(".codeindent").outerWidth();
	    var code_width = $(this).siblings(".codeblock").outerWidth() + 1;
	    var box_pad = 2 * parseFloat(
		$(this).closest(".annotatepad").css("padding-left"));

	    /*
            var fudge = 1;
	    box_pad += fudge;
	    */

	    var comment_width = 
		available_width - indent_width - code_width - box_pad;
	    
	    comment_width = Math.trunc(comment_width);

	    comment_width -= 2;

	    if (comment_width < 8) { // Extremely narrow browser.
		$(this).hide();
		$(this).siblings(".codeindent").hide();
	    } else {
		$(this).siblings(".codeindent").show();
		//$(this).show().outerWidth(comment_width);
		//$(this).show().innerWidth(comment_width);
		$(this).show();
		$(this).innerWidth(comment_width);		

	    }
	}
    });
}

//DBG
function show_width() {
    $("*").each( function() {
	$(this).append("[" + $(this).width() + "]");
    });
}
//DBG
