function find_parent(element) {
    var result;
    var li_parents = $(element).parents('li');
    if (li_parents.length) {
	//console.log("parent is <li>");
	result = li_parents.first();
    } else {
	var dd_parents = $(element).parents('dd');
	if (dd_parents.length) {
	    //console.log("parent is <dd>");
	    result = dd_parents.first();
	}
	else {
	    //console.log("parent is #page_content");
	    result = $("#page_content");
	}
    }
    //console.log(result);
    return result;
}

function resize_image(img) {
    dbg("---------------------");
    dbg("function resize_image");
    //var parent = $("#page_content");
    var parent = find_parent(img);
    var parent_width = $(parent).innerWidth();
    var parent_height = $(parent).innerHeight();
    var max_width = parseFloat($(img).attr("data-wmax"));
    var max_height = parseFloat($(img).attr("data-hmax"));
    var aspect = parseFloat($(img).attr("data-aspect"));
    dbg("Parent size: " + parent_width + " " + parent_height);
    dbg("Image: " + max_width + " " + max_height + " " + aspect);

    var width_scale = $(img)[0].hasAttribute("data-width") ? parseFloat($(img).attr("data-width")) : 0;
    var height_scale = $(img)[0].hasAttribute("data-height") ? parseFloat($(img).attr("data-height")) : 0;

    var new_width = 0;
    var new_height = 0;
    if (width_scale == 0 && height_scale == 0) {
        dbg("Image resize default");
        new_width = parent_width;
        new_height = new_width / aspect;
    } else if (height_scale == 0) {
        dbg("Image resize width: " + width_scale);
        new_width = Math.min(max_width, parent_width * width_scale)
        new_height = new_width / aspect;
    } else if (width_scale == 0) {
        dbg("Image resize height: " + height_scale);
        new_height = Math.min(max_height, parent_height * height_scale)
        new_width = new_height * aspect;
    } else {
        dbg("Image resize: " + width_scale + " by " + height_scale);        
        new_width = Math.min(max_width, parent_width * width_scale)        
        //new_height = Math.min(max_height, parent_height * height_scale)
	new_height = new_width / aspect;
    }
    //$(img).attr("width", Math.round(new_width));
    //$(img).attr("height", Math.round(new_height));
    $(img).outerWidth(Math.round(new_width));
    $(img).outerHeight(Math.round(new_height));
    //dbg("New: " + new_width + " " + new_height);
}


function resize_images() {
    $('img[data-resize="true"]').each( function(index) {
	resize_image($(this));
    });
}

function adjust_width(elt) {
    var parent = find_parent(elt);
    var parent_width = $(parent).innerWidth();
    var width_scale = $(elt)[0].hasAttribute("data-width") ? parseFloat($(elt).attr("data-width")) : 0;
    if (width_scale) {
	var new_width = Math.round(parent_width * width_scale);
	$(elt).css("width", new_width);
    }
}


function adjust_widths() {
    $("[data-adjustwidth]").each( function(index) {
	adjust_width($(this));
    });
}


function center_element(element) {
    var parent = find_parent(element);
    var parent_width = parseFloat($(parent).innerWidth());
    var width = parseFloat($(element).outerWidth());
    var margin = (parent_width - width) / 2.0;
    $(element).css("margin-left", margin);
    $(element).css("margin-right", margin);
}

function center_elements() {
    $("[data-centered='true']").each( function(index) {
	center_element($(this));
    });
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

function adjust_grid_captions() {
    $('#page_content .rimg-table-caption').each( function(index) {
	adjust_grid_caption_size($(this));
    });
}

