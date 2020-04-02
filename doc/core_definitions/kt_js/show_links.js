
function show_link_boxes(show) {
    $("#page_content a, .panelink").each( function() {	
	if (show) {
	    $(this).css("background-color", "rgb(85%,90%,95%)").
		//css("color", "black").
		css("box-shadow", "0em 0em 0em .15em rgb(85%,90%,95%)");
	} else {
	    $(this).css("background-color", "transparent"). //"rgb(100%,100%,100%)").
		css("color", "#1A2680").
		css("box-shadow", "none");
	}
    });
}

function check_visible_link_boxes() {
    if ($("#page_content a, .panelink").length == 0) {
	$("#linkshow").hide();
    } else {
	$("#linkshow").show();
	show_link_boxes($("#linkshow").attr("data-show") == 1);
    }
}


$(window).on("load", function() {
    $('#linkshow').on('click', function(event) {
	var linkshow = $("#linkshow");
	if (linkshow.attr("data-show") == "0") {
	    //linkshow.html("Link&#160;text&#160;in&#160;color");
	    linkshow.html("Links&#160;in&#160;color");
	    linkshow.attr("data-show", "1");
	    show_link_boxes(true);
	} else {
	    //linkshow.html("Show&#160;links");
	    linkshow.html("Links&#160;in&#160;boxes");
	    linkshow.attr("data-show", "0");
	    show_link_boxes(false);
	}
    });
    keypress("L", "#linkshow");
});
