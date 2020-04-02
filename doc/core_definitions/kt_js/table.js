
function set_imagetext_width() {
    $(".imagetext").each( function(index) {
	$(this).css("background-color", "lightblue");
	var image = $(this).find(".tdimage");
	var text = $(this).find(".tdtext");

	//var scaleto = $(this).attr("data-scaleto");
	//var textwidth = $(this).attr("data-textwidth");
	/*
	  if (textwidth != "0") {
	  console.log("textwidth:" + textwidth + " " + text.text());
	  //$(text).width(textwidth);
	  $(".annotatebox").width(textwidth);
	  $(this).find(".uncommented").width(textwidth);
	  $(this).find(".annotatedcode").width(textwidth);
	  $(this).find(".codecomment").width(textwidth);
	  } 
	*/
	var available_width = $("#page_content").innerWidth();
	var new_image_width = available_width - $(text).outerWidth();
	var aspect = parseFloat(
            $(this).children(".rimg-box").attr("data-aspect"));
	$(image).innerWidth(new_image_width);
	$(image).innerHeight(new_image_width / aspect);

	/*
	  var code = $(this).parent().parent().find(".tdcode");
	  //console.log("code: " + code);
	  //$(code).css("background-color", "yellow");
	  var available_width = $("#page_content").innerWidth();
	  //console.log("available_width: " + available_width);
	  var code_width = $(code).outerWidth();
	  //console.log("code_width: " + code_width);

	  //console.log("aspect: " + aspect);
	  var new_image_width = available_width - code_width;
	  //console.log("new_image_width: " + new_image_width);
	  $(this).find("img").innerWidth(new_image_width);
	  $(this).find("img").innerHeight(new_image_width / aspect);
	  shrinkwrap($(".annotatebox"));
	*/
    });
}

function adjust_slanted_header_table() {
    var angle = Math.PI / 3;
    $("table").each( function() {  
	var slant_select = "td[data-slant='1'] div span";
	if ($(slant_select).length == 0)
	    return;
	var max_slant_height = 0;
	$(this).find(slant_select).each( function() {
	    max_slant_height = Math.max
	    (max_slant_height, 
	     $(this).outerWidth() * Math.sin(angle) - 
	     $(this).closest("td").outerHeight());
	});
	$(this).find(".slant_spacer").height(max_slant_height + 10);
    });
}
