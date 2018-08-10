function resize_image(id) {
    var fractional_size = 0.5;
    var window_width = $("html").width();
    var max_width = 500;
    var new_width = Math.min(max_width, window_width * fractional_size);
    $(id).width(new_width);
};

$(window).on("load", function() {
    $(window).resize(function() {
	resize_image("#mdl_local_coordinates");
    });
    resize_image("#mdl_local_coordinates");
});

