var loaded_basename;
var minimum_content_width = 300;
var chrome_browser_p = false;

function restore_page_visibility() {
    $("#content").css("color", "black");
    $("#content a, #content a:link").css("color", "rgb(10%,15%,50%)");
}

$("#content, #content a, #content a:link").css("color", "white");

$(window).on("load", function() {
    chrome_browser_p = typeof window.chrome === "object";
    //console.log(window);
    //history.scrollRestoration = 'manual';
    $("#search_clear").hide();
    var location = window.location.toString();

    if (location.indexOf("#") == -1) {
	var new_basename = first_basename();
	//load_page(new_basename, '#', function() {});
	load_page(new_basename, '#', 
		  function () {
		      initialize = true; new_page_callback();
		      initialize_nodes();
		      resize_toc();
		      restore_page_visibility();
		  });
	//load_page(new_basename, "#"+$(".navlink").attr("data-target"),
	//function() {});
	//new_page_callback);
	loaded_basename = new_basename;
	navigation_width_history = [];
    } else {
	var parts = location.split('#');
	var basename = parts[1];
	var target = (parts.length == 3) ? '#'+parts[2] : "#";
	var cb = function() 
	{ target_name = target;  initialize = true;
	  //loaded_basename != basename ? new_page_callback() : current_page_callback();
	  new_page_callback();
	  initialize_nodes();
	  open_to_navlink_level(parts[2]);
	  restore_page_visibility();

	  open_if_parts();
	};
	//initialize_nodes();
	//resize_toc();
	load_page(basename, target, cb);
	loaded_basename = basename;
    }


    return false;
});

