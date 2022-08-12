
function adjust_footnotes() {
    $(".footnote .footnotetext").each( function(index) {
	var left = $(this).parent().position().left;
	var c = $("#page_content");
	var half_width = $(c).width() / 2.0;
	var mid = $(c).position().left + half_width;
	var max_width = 400;
	if (left > mid) {
	    $(this).css("right", "0em");
	    $(this).css("left", "auto");
	} else {
	    $(this).css("left", "0em");
	    $(this).css("right", "auto");
	}
	$(this).width(Math.min(max_width, half_width));
    });
}


$(window).on("load", function() {
    // This means "no animation", not "don't show the tooltip."
    $(document).tooltip({ hide: false, show: false });
});
