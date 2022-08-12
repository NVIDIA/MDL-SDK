
function set_imagecode_width() {
    $(".tdimage").each( function(index) {
	//$(this).css("background-color", "lightblue");
	var code = $(this).parent().parent().find(".tdcode");
	//console.log("code: " + code);
	//$(code).css("background-color", "yellow");
	var available_width = $("#page_content").outerWidth();
	//console.log("available_width: " + available_width);
	var code_width = $(code).outerWidth();
	//console.log("code_width: " + code_width);
	var aspect = parseFloat($(this).children(".rimg-box").attr("data-aspect"));
	//console.log("aspect: " + aspect);
	var new_image_width = available_width - code_width;

	//console.log("new_image_width: " + new_image_width);
	//new_image_width = Math.min(new_image_width,
	//0.67 * parseFloat($("#page_content").width()));
	
	new_image_width = Math.min(new_image_width,
				   parseFloat($(this).find("img").parent().attr("data-width")));

	$(this).find("img").innerWidth(new_image_width);
	$(this).find("img").innerHeight(new_image_width / aspect);
    });
}

/*
  $(document).ready(function() {
  $(window).resize(function() {
  set_imagecode_width();
  });
  set_imagecode_width();
  });
*/
