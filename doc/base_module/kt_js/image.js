
function adjust_image(imgbox) {
    var parent = $(imgbox).parent();
    if (parent.is('td'))
	parent = $('#page_content');
    var wscale = $(imgbox).attr('data-rwscale');
    var max_width = parseFloat($(imgbox).attr('data-width'));
    var max_height = parseFloat($(imgbox).attr('data-height'));	
    var new_width = Math.round(Math.min(max_width, $(parent).width() * wscale));
    var new_height = Math.round(Math.min(max_height, new_width / parseFloat($(imgbox).attr('data-aspect'))));

    /*
      $(imgbox).children('img').width(new_width);
      $(imgbox).children('img').attr("width", new_width);
      $(imgbox).children('img').height(new_height);
      $(imgbox).children('img').attr("height", new_height);
      if ($(imgbox).attr('data-center') == "0") {
      $(imgbox).width(new_width);
      $(imgbox).attr("width", new_width);
      }
    */
    $(imgbox).children('img')
	.width(new_width)
	.attr("width", new_width)
	.height(new_height)
	.attr("height", new_height);
    if ($(imgbox).attr('data-center') == "0") {
	$(imgbox).width(new_width);
	$(imgbox).attr("width", new_width);
    }

}

function adjust_grid_caption_size(caption) {
    var parent = $(caption).parent();
    var row = $(parent).children('#page_content .rimg-row').first();
    if ($(row).css('text-align') == 'left') {
	var images = $(row).children('.rimg-inline');
	var width = 0;
	$(images).each( function(index) {
	    width += $(this).outerWidth(true);
	});
	$(caption).width(width);
    }
}

function adjust_images() {
    $('#page_content .rimg-dynamic').each( function(index) {
	adjust_image($(this));
    });
    $('#page_content .rimg-table-caption').each( function(index) {
	adjust_grid_caption_size($(this));
    });
}
