 function frame_resize(){var height=$(window).innerHeight();var header_height=$('#blackheader').outerHeight();
 var footer_height=$('#blackfooter').outerHeight();var content_height=height - header_height - footer_height;
 $('#content').innerHeight(content_height);}function set_imagecode_width(){$(".tdimage").each(function(index){
 var code=$(this).parent().parent().find(".tdcode");var available_width=$("#page_content").outerWidth();
 var code_width=$(code).outerWidth();var aspect=parseFloat($(this).children(".rimg-box").attr("data-aspect"));
 var new_image_width=available_width - code_width;new_image_width=Math.min(new_image_width,
 parseFloat($(this).find("img").parent().attr("data-width")));$(this).find("img").innerWidth(new_image_width);
 $(this).find("img").innerHeight(new_image_width/aspect);});}function set_imagetext_width(){
 $(".imagetext").each(function(index){$(this).css("background-color","lightblue");var image=$(this).find(".tdimage");
 var text=$(this).find(".tdtext");var available_width=$("#page_content").innerWidth();var new_image_width=available_width - $(text).outerWidth();
 var aspect=parseFloat($(this).children(".rimg-box").attr("data-aspect"));$(image).innerWidth(new_image_width);
 $(image).innerHeight(new_image_width/aspect);});}function adjust_slanted_header_table(){var angle=Math.PI/3;
 $("table").each(function(){var slant_select="td[data-slant='1'] div span";if($(slant_select).length==0)
 return;var max_slant_height=0;$(this).find(slant_select).each(function(){max_slant_height=Math.max
(max_slant_height,$(this).outerWidth()* Math.sin(angle)-  $(this).closest("td").outerHeight());});
 $(this).find(".slant_spacer").height(max_slant_height+10);});}function adjust_footnotes(){$(".footnote .footnotetext").each(function(index){
 var left=$(this).parent().position().left;var c=$("#page_content");var half_width=$(c).width()/2.0;
 var mid=$(c).position().left+half_width;var max_width=400;if(left>mid){$(this).css("right","0em");
 $(this).css("left","auto");}else{$(this).css("left","0em");$(this).css("right","auto");}
 $(this).width(Math.min(max_width,half_width));});}function search_not_active(){return $(document.activeElement).attr('id')!='search_input';}
 function keypress(letter,id){$('html').keypress(function(event){if(search_not_active()){
 var uppercase=letter.charCodeAt();var lowercase=uppercase+32;if(event.which==uppercase || event.which==lowercase){
 $(id).trigger('click');}}});}function arrowpress(arrow,id){$('html').keydown(function(event){
 if(search_not_active()){if(event.which==arrow){$(id).trigger('click');}}});}function update_page_title(element){
 var element_text=$(element).text();if(element_text){var regex=/((?:Part)?[0-9.]*)(.*)/;
 var parts=regex.exec(element_text);var title;if(!parts[1])title=parts[2].trim();else title=parts[1]+" - "+parts[2];
 $("title").text(title);}}function highlight_visible_toc_sections(activate){if(typeof activate==="undefined")
 activate=true;var heading=header_in_window();var navlink=$('.navlink[data-target="'+heading.attr("id")+'"]');
 $(".navlink").removeClass("highlight").removeClass("highlight_parent");navlink.addClass('highlight');
 var parents=$(navlink).parents().siblings('.navlink');parents.addClass('highlight_parent');}
 function show_page_content(){$('#search_content').hide();$('#help_content').hide();$('#page_content').show();
 $('#content').css('background-color','white');$('#help').text('Help');highlight_visible_toc_sections();}
 function last_highlighted_navlink(visible_navlinks){var visible=$('.navlink:visible');var i;
 for(i=visible.length - 1;i>=0;i--){if($(visible[i]).hasClass('highlight')|| $(visible[i]).hasClass('highlight_parent'))
 break;}return i;}function show_button_status(id,active_p){var inactive_color='rgb(60%,60%,60%)';
 if(active_p)$(id).css('color','').css('text-decoration','');else $(id).css('color',inactive_color).css('text-decoration','none');}
 function set_next_last_button_status(current_index,visible_length){var index=current_index;
 var length=visible_length;if(typeof index==="undefined"){var visible_navlinks=$('.navlink:visible');
 index=last_highlighted_navlink(visible_navlinks);length=visible_navlinks.length;}show_button_status('#next',index<length - 1);
 show_button_status('#last',index>0);}var search_displayed;var content_position;function hide_help(){
 var help_label=$('#help');help_label.text("Help");$('#help_content').hide();if(search_displayed){
 show_search_content();$('#content').scrollTop(content_position);$('#search_content').show();
 $('#page_content').hide();}else{$('#search_content').hide();$('#page_content').show();$('#content').scrollTop(content_position);
 highlight_visible_toc_sections();}}function adjust_image(imgbox){var parent=$(imgbox).parent();
 if(parent.is('td'))parent=$('#page_content');var wscale=$(imgbox).attr('data-rwscale');var max_width=parseFloat($(imgbox).attr('data-width'));
 var max_height=parseFloat($(imgbox).attr('data-height'));var new_width=Math.round(Math.min(max_width,$(parent).width()* wscale));
 var new_height=Math.round(Math.min(max_height,new_width/parseFloat($(imgbox).attr('data-aspect'))));
 $(imgbox).children('img').width(new_width).attr("width",new_width).height(new_height).attr("height",new_height);
 if($(imgbox).attr('data-center')=="0"){$(imgbox).width(new_width);$(imgbox).attr("width",new_width);}}
 function adjust_grid_caption_size(caption){var parent=$(caption).parent();var row=$(parent).children('#page_content .rimg-row').first();
 if($(row).css('text-align')=='left'){var images=$(row).children('.rimg-inline');var width=0;
 $(images).each(function(index){width+=$(this).outerWidth(true);});$(caption).width(width);}}
 function adjust_images(){$('#page_content .rimg-dynamic').each(function(index){adjust_image($(this));});
 $('#page_content .rimg-table-caption').each(function(index){adjust_grid_caption_size($(this));});}
 function resize_content_width(){var left=rightpos("navigation");var maximum_width=$('#content').innerWidth;
 var content_width=Math.max(minimum_content_width,maximum_width - left);$("#content").width(content_width);
 $("#navigation").width(maximum_width - content_width);}function header_in_window(tags){var content_top=$('#content').offset().top;
 var content_bottom=content_top+$('#content').height();var result='';var last;var header_tags='#page_content>div.part,#page_content>h1,'+
 '#page_content>h2,#page_content>h3,#page_content>h4,#page_content>h5';if(typeof tags !=='undefined')
 header_tags+=","+tags;$(header_tags).map(function(){if(result==''){var ths=$(this);
 var midpoint=ths.offset().top+(ths.height()/2);if((midpoint>content_top)&&(midpoint<content_bottom)){
 result=ths;}else if(midpoint<content_top){last=ths;}}});if(result=='')result=last;
 return result;}function maintain_content_position(func){var top_header=header_in_window("#page_content>p,#page_content>div");
 var original_top_pos=top_header.position().top;func();var new_top_pos=top_header.position().top;
 var offset=original_top_pos - new_top_pos;if(offset !=0){$('#content').scrollTop($('#content').scrollTop()- offset);}}
 var last_width;function navshow(){$('#navigation').toggle();$('#navfit').toggle();if($('#navigation').is(":visible")){
 $('#navigation').width(last_width);$('#content').css('border-width','0 0 0 1px');$('#navshow').html('Text only');
 $('.depthchoice').show();$('#next').show();$('#last').show();$('#focus').show();}else{last_width=$('#navigation').width();
 $('#content').css('border-width','0 0 0 0');$('#navshow').html('Table of contents');$('.depthchoice').hide();
 $('#next').hide();$('#last').hide();$('#focus').hide();}resize_content_width();adjust_images();
 set_code_comment_width();set_imagecode_width();adjust_footnotes();}var navigation_width_history=[];
 function current_navigation_width(){var width=0;var right_margin=16;$('span.navlink:visible').map(function(){
 var element_width=$(this).outerWidth()+$(this).offset().left;width=Math.max(width,element_width);});
 return width+right_margin;}function resize_toc(){var navigation=$('#navigation');if($('#navigation').is(":visible")){
 var current_width=current_navigation_width();var width=Math.min($("html").width()- minimum_content_width,current_width);
 $('#navigation').width(width);resize_content_width();set_annotate_max_width();set_code_comment_width();
 set_imagecode_width();}}function show_link_boxes(show){$("#page_content a,.panelink").each(function(){
 if(show){$(this).css("background-color","rgb(85%,90%,95%)"). css("box-shadow","0em 0em 0em .15em rgb(85%,90%,95%)");}else{
 $(this).css("background-color","transparent").//"rgb(100%,100%,100%)"). css("color","#1A2680").
 css("box-shadow","none");}});}function check_visible_link_boxes(){if($("#page_content a,.panelink").length==0){
 $("#linkshow").hide();}else{$("#linkshow").show();show_link_boxes($("#linkshow").attr("data-show")==1);}}
 function rightpos(name){var elt=$('#'+name);var result=0;if(elt.is(":visible")){result=elt.position().left+elt.outerWidth();}
 return result;}function adjust_search_input_pane(){if($("#navshow").is(":visible")){var search_input_width=200;
 var button_right=Math.max(rightpos('navshow'),rightpos('focus'),rightpos('linkshow'),rightpos('clear'));
 var help_left=$('#help').position().left;var new_width=Math.min(search_input_width,help_left - button_right - 90);
 if(new_width<10)new_width=search_input_width;$('#search_input').width(new_width);}}function adjust_toc_slider_height(){
 $('div.ui-resizable-handle').height($('#navigation').height()+$('#navigation').scrollTop());}
 function resize(){var top_header=header_in_window();var height=$(window).height();adjust_search_input_pane();
 var nc=$('#navcontrol');var control_height;if(nc.is(':visible'))control_height=$('#navcontrol').outerHeight();
 else control_height=0;var header_height=$('#blackheader').outerHeight();var footer_height=$('#blackfooter').outerHeight();
 var folder_height=height - header_height - footer_height;var folder_components_height=folder_height - control_height;
 $('#content').height(folder_components_height);$('#page_content').height(folder_components_height);
 $('#search_content').height(folder_components_height);$('#help_content').height(folder_components_height);
 $('#navigation').height(folder_components_height);resize_content_width();adjust_images();set_annotate_max_width();
 set_code_comment_width();set_imagecode_width();adjust_footnotes();adjust_slanted_header_table();
 check_visible_link_boxes();adjust_toc_slider_height();set_code_caption_width();$('#bottomspacer').height($("#page_content").height()* 0.9);}
 function first_basename(){return $(".navlink:first").attr("data-basename");}function scroll_to_target(target_name){
 var content=$('#content');var folder_top=content.position().top;var target=$(target_name);
 var target_position=target.position();if(target_position){var scroll_position=content.scrollTop()+
 target.position().top - folder_top;$('#content').scrollTop(scroll_position);}highlight_visible_toc_sections();}
 function modify_links(){$("#page_content a").not("a.linkd").not("a:has(img)").replaceWith(function(){
 var href=$(this).attr('href');if((href.indexOf("http://")!=-1)||(href.indexOf("https://")!=-1)||
(href.indexOf(".pdf")!=-1)||(href.indexOf("mailto:")!=-1)||(href.indexOf("/")==0)||(href.indexOf("#")==0)||
(href.indexOf('..')==0))return '<a href="'+href+'">'+$(this).html()+'</a>';var parts=href.split('#');
 var len=parts[0].length;var page=parts[0].substring(0,len-5);var target='#'+parts[1];
 var result='<span class="panelink" ';result+='data-basename="'+page+'" ';result+='data-target="'+target+'" ';
 result+='data-title="'+$(this).text()+'">';result+=$(this).text()+'</span>';return result;});}
 function new_page_callback(){show_page_content();modify_links();resize();current_page_callback();}
 var image_timer;var last_basename;function current_page_callback(){if(target_name=="#")return;
 var selector='.navlink[data-target="'+(target_name.split("#")[1] || "")+'"]';update_page_title($(selector));
 resize();scroll_to_target(target_name);if(typeof MathJax !=='undefined')MathJax.Hub.Queue(["Typeset",MathJax.Hub,"page_content"]);}
 function load_page_content(basename,new_page,callback){if(new_page ||(basename !=loaded_basename)){
 var page_selector='pages/'+basename+'.html'+' #page>';$("#page_content").load(page_selector,callback);
 loaded_basename=basename;}else callback();}function load_page(basename,target,callback,push_state){
 var init_complete=target !='#';var push=typeof push_state !=="undefined";var state,state_target;
 if(init_complete){var parts=window.location.hash.toString().split('#');state={"basename" : parts[1],"target" : '#'+parts[2]};
 state_target='#'+basename+target;if(push)history.pushState(state,"",state_target);else
 history.replaceState(state,"",state_target);load_page_content(basename,basename !=loaded_basename,
 function(){target_name=target;initialize=false;callback();});}else{basename=first_basename();
 state={"basename" : basename,"target" : "#"};state_target="#"+basename+"#";init_complete=true;
 load_page_content(basename,basename !=loaded_basename,function(){target_name=target;initialize=true;
 callback();});history.replaceState(state,"",state_target);}loaded_basename=basename;}var search_scroll_position=0;
 function searchhit_element(value){return '<span class="searchhit">'+value+'</span>';}function clear_search_hits(html){
 $("#search_clear").hide();var hit_regex=new RegExp(searchhit_element("([^<]*)"),"igm");return html.replace(hit_regex,"\$1");}
 function mark_search_targets(text,status){var search_target=$('#search_input').val();var html=$('#page_content').html();
 html=clear_search_hits(html);var protect_pattern=new RegExp('(<[^>]*?)'+search_target,"igm");
 var stub='____KLMMRTXT____\0';var last_html=null;var i=1;while(html !==last_html){last_html=html;
 html=html.replace(protect_pattern,'$1'+stub);i+=1;}var pattern=new RegExp('('+search_target+')',"igm");
 var marked_html=html.replace(pattern,searchhit_element("$1"));var stub_regex=new RegExp(stub,"igm");
 marked_html=marked_html.replace(stub_regex,search_target);$('#search_content').hide();$('#page_content').html(marked_html).show();
 $("#search_clear").show();new_page_callback();}function show_search_content(){if($('#page_content').is(":visible")){
 $('#page_content').hide();}if($('#help_content').is(":visible"))$('#help_content').hide();$('#content').css('background-color','rgb(220,225,220)');
 $('#help').text('Help');if(!$('#search_content').is(":visible"))$('#search_content').show();
 $('.searchcontext,.searchsection').on('click',function(event){var ths=$(this);search_scroll_position=$('#content').scrollTop();
 var filename='pages/'+ths.attr('data-filename')+'.html';target_name='#'+ths.attr('data-sid');
 load_page(ths.attr('data-filename'),target_name,mark_search_targets);$('#content').css('background-color','white');});
 $('.searchcontext').mouseenter(function(){$(this).css('background-color','rgb(90%,95%,100%)');});
 $('.searchcontext').mouseleave(function(){$(this).css('background-color','white');});$('.searchsection').mouseenter(function(){
 $(this).css('color','rgb(10%,20%,50%)').css('text-decoration','underline');});$('.searchsection').mouseleave(function(){
 $(this).css('color','black').css('text-decoration','none');});}function search_title(target,count,section_count){
 var title='';if(count==0){title='No matches for "'+target+'"';}else{var suffix=count>1 ? 'es' : '';
 title="Search results: "+count+' match'+suffix+' for "'+target+'"';if(count !=1 && section_count !=1)
 title+=" in "+section_count+" section"+(section_count>1 ? "s" : "");}return '<div class="searchtitle">'+title+'</div>\n\n';}
 function search_section_title(title,marked_title){var location=section_headers[title];return '<div class="searchsection" data-filename="'	+
 location[0]+'" data-sid="'+location[1]+'">'+marked_title+'</div>\n\n';}function search_item(filename,id,text){
 return '<div class="searchcontext" '+'data-filename="'+filename+'" data-sid="'+id+'">\n'+
 text+'\n'+'</div>\n\n';}var slide_time=0;var closed_symbol='&#9656;';var open_symbol='&#9662;';
 var toc_debug=false;var scroll_timer;var SCROLLING=false;function open_if_parts(){var document_has_parts=false;
 $("#navigation").find(".navlink").each(function(){if(!document_has_parts)if($(this).text().substring(0,6)=='Part 1'){
  document_has_parts=true;}});if(document_has_parts){$(".depthchoice[data-depth='2']").trigger('click');
 resize_toc();}}function initialize_nodes(){var nodes=$('#navigation').find('li ul').parent();
 nodes.addClass('closed');nodes.children().slideUp(slide_time);nodes.children('div').slideDown(slide_time);
 nodes.children('.navlink').slideDown(0);if(toc_debug)nodes.css('background-color','pink');$('#navigation')
 .find('li.closed>span.nodestatus').html(closed_symbol).find('.level0').html(closed_symbol);$('.nodestatus').slideDown(0);}
 function toc_load(basename,target,push_state){var current_basename=window.location.hash.toString().split('#')[1];
 load_page(basename,target,function(){var initialize=false;if(current_basename==loaded_basename)
 current_page_callback();else new_page_callback();},push_state);}function toggle(selected){selected.toggleClass('closed');
 selected.toggleClass('open');}function show_navlink_in_toc(navlink){var nav=$("#navigation");
 var nav_top=nav.offset().top;var nav_bottom=nav_top+nav.height();var link_top=navlink.offset().top;
 if(link_top>nav_bottom){nav.scrollTop(nav.scrollTop()- link_top - nav_bottom);}}function open_to_navlink_level(target){
 $('#navcontrol span.depthchoice[data-depth="1"]').trigger('click');var navlink=$('.navlink[data-target="'+target+'"]');
 navlink.parent().parents("#navigation li").each(function(){var ns=$(this).find(".nodestatus:first");
 if(ns.parent().hasClass("closed")){ns.trigger("click");}});$('#navfit').trigger('click');}function toggle_current_navlink(){
 $('html').keypress(function(event){if(search_not_active()){var uppercase="D".charCodeAt();
 var lowercase=uppercase+32;if(event.which==uppercase || event.which==lowercase){$("#navigation .highlight").siblings(".nodestatus").trigger('click');}}});}
 var max_box_width_without_scaling=400;var max_box_width_proportional_to_code=2.2;function describe(element,comment){
}function annotate_debug(){var annotate=$(".annotatebox");annotate.css("background-color","lightblue");
 annotate.find(".codecomment").css("background-color","rgb(100%,100%,90%)");annotate.find(".uncommented")
 .css("background-color","rgb(95%,100%,90%)");}function right_pos(element){return $(element).offset().left+$(element).outerWidth();}
 function code_width(box){var result=0;$(box).find(".codeblock").each(function(index){result=Math.max(result,right_pos($(this)));});
 $(box).find(".uncommented").each(function(index){result=Math.max(result,$(this).outerWidth());});
 return result;}function set_annotate_max_width(){$(".annotatebox").each(function(index){var max_x=0;
 var box=$(this);$(box).width(2000);$(box).find(".annotatedcode").each(function(index){var code=$(this);
 var width=0;$(code).children().each(function(){width+=$(this).outerWidth();});max_x=Math.max(max_x,width);
 max_x+=2;});$(box).find(".uncommented").each(function(index){max_x=Math.max(max_x,$(this).outerWidth());
 max_x+=2;});max_x+=20;var scale=parseFloat($(this).attr("data-scale"));max_x *=scale;
 var current_max_width=$(box).attr("data-maxwidth");if(typeof current_max_width==="undefined"){
 $(box).attr("data-maxwidth",max_x);}else if(current_max_width>max_x){max_x=current_max_width;
 $(box).attr("data-maxwidth",max_x);}$(box).innerWidth(max_x);});}function set_code_caption_width(){
 $(".annotatebox").each(function(index){var box=$(this);$(box).find(".codecaption").innerWidth(
 $(box).find(".annotateline").width())});}function find_parent_inner_width(element){var result=element;
 var in_table=false;var width;var box_offset=parseFloat($(".annotatebox").css("margin-left"))+
 parseFloat($(".annotatebox").css("margin-right"))+parseFloat($(".annotatepad").css("padding-left"))+
 parseFloat($(".annotatepad").css("padding-right"));while(true){result=$(result).parent();
 var tag=$(result).prop("tagName");if(tag=="TD" || tag=="DD"){in_table=true;break;}
 if($(result).attr("id")=="page_content"){break;}if(tag=="BODY"){break;}}if(in_table){
 var other_cell=$(result).closest("tr").children("td:first");var cell_width=$(other_cell).outerWidth()+
 parseFloat($(other_cell).css("padding-left"))+parseFloat($(other_cell).css("padding-right"))+
 parseFloat($(other_cell).css("margin-left"))+parseFloat($(other_cell).css("margin-right"));width=$("#page_content").innerWidth()- cell_width - box_offset;}else{
 width=result.width()- parseFloat(result.css("margin-left"))- parseFloat(result.css("margin-right"))-
 parseFloat(result.css("padding-left"))- parseFloat(result.css("padding-right"))- box_offset;}
 return width;}function set_code_comment_width(){$(".codecomment").each(function(index){var box=$(this).closest(".annotatebox");
 if(box.attr("data-fixed")==="true"){$(this).css("display","inline").css("float","none");}else{
 var scale=parseFloat($(box).attr("data-scale"));var parent_inner_width=find_parent_inner_width($(box));
 var new_box_width=Math.min(parseFloat($(box).attr("data-maxwidth")),parent_inner_width);$(box).width(new_box_width);
 var available_width=$(box).innerWidth();var indent_width=$(this).siblings(".codeindent").outerWidth();
 var code_width=$(this).siblings(".codeblock").outerWidth()+1;var box_pad=2 * parseFloat(
 $(this).closest(".annotatepad").css("padding-left"));var comment_width=available_width - indent_width - code_width - box_pad;
 comment_width=Math.trunc(comment_width);comment_width -=2;if(comment_width<8){$(this).hide();
 $(this).siblings(".codeindent").hide();}else{$(this).siblings(".codeindent").show();$(this).show();
 $(this).innerWidth(comment_width);}}});}var loaded_basename;var minimum_content_width=300;
 var chrome_browser_p=false;function restore_page_visibility(){$("#content").css("color","black");
 $("#content a,#content a:link").css("color","rgb(10%,15%,50%)");}$("#content,#content a,#content a:link").css("color","white");
 $(window).on("load",function(){$('#content').css('background-color','white');$(window).resize(function(){
 frame_resize();});frame_resize();$(document).tooltip({hide: false,show: false});all_navlink_items=$("#navigation li");
 $('#next').on('click',function(event){hide_help();var visible_navlinks=$('.navlink:visible');
 var i=last_highlighted_navlink(visible_navlinks);i+=1;if(i<visible_navlinks.length)$(visible_navlinks[i]).trigger('click');});
 keypress("N","#next");$('#last').on('click',function(event){hide_help();var visible_navlinks=$('.navlink:visible');
 var i=last_highlighted_navlink(visible_navlinks);i -=1;if(i>=0)$(visible_navlinks[i]).trigger('click');
 set_next_last_button_status(i,visible_navlinks.length);});keypress("P","#last");$('#help').on('click',function(){
 $('#content').css('background','white');var text=$(this).text();if(text=="Help"){search_displayed=$('#search_content').is(':visible');
 $(this).text("Back");content_position=$('#content').scrollTop();$('#page_content').hide();$('#search_content').hide();
 $('#help_content').load('help.html').scrollTop(0).show();}else{hide_help();}});keypress("H","#help");
 $('.depthchoice').on('click',function(event){hide_help();var parts_exists=$(".level0").length>0;
 var display_depth=parseInt($(this).attr('data-depth'))-(1 ? parts_exists : 0);var changed=false;
 $('#navigation .nodestatus').each(function(){var node=$(this);var is_closed=node.parent().hasClass('closed');
 var is_open=node.parent().hasClass('open');var node_depth=parseInt($(this).attr('data-depth'));
 if((is_closed && display_depth>node_depth)||(is_open && display_depth<=node_depth)){$(this).trigger('click');
 changed=true;}});});$('html').keypress(function(event){if(search_not_active()){var zero=48;
 for(var depth=1;depth<=7;depth++){if(event.which==zero+depth){$('.depthchoice[data-depth="'+depth+'"]').trigger('click');}}}});
 $('#navshow').on('click',function(event){maintain_content_position(navshow);});keypress("T","#navshow");
 $('#navfit').on('click',function(event){maintain_content_position(resize_toc);});keypress("R","#navfit");
 $('#linkshow').on('click',function(event){var linkshow=$("#linkshow");if(linkshow.attr("data-show")=="0"){
 linkshow.html("Links&#160;in&#160;color");linkshow.attr("data-show","1");show_link_boxes(true);}else{
 linkshow.html("Links&#160;in&#160;boxes");linkshow.attr("data-show","0");show_link_boxes(false);}});
 keypress("L","#linkshow");$(window).on("resize",function(event){event.preventDefault();maintain_content_position(resize);});
 $('#navigation').scroll(function(){adjust_toc_slider_height();});$("#navigation").resizable({handles : 'e'});
 $(window).on("popstate",function(event){if(SCROLLING)return;var parts=window.location.hash.split("#");
 var basename=parts[1];var target='#'+parts[2];load_page(basename,target,new_page_callback);
 loaded_basename=basename;return false;});$('#search_input').keyup(function(event){var enter_key=13;
 if(event.which==enter_key){var search_target=$('#search_input').val().trim();var pattern=new RegExp('('+search_target+')',"ig");
 var html='';var count=0;var section_count=0;for(var i=0;i<search_targets.length;++i){
 var title=search_targets[i][0];var marked_title=title;var title_match=false;if(pattern.test(title)){
 title_match=true;count+=(title.match(pattern)|| []).length;marked_title=title.replace(
 pattern,'<span class="searchhit">$1</span>');}var matches='';for(var j=0;j<search_targets[i][1].length;j++){
 var section=search_targets[i][1][j];var text=section[2];if(pattern.test(text)){count+=(text.match(pattern)|| []).length;
 var marked_text=text.replace(pattern,'<span class="searchhit">$1</span>');matches+=search_item(section[0],section[1],marked_text);}}
 if(matches !=''){html+=search_section_title(title,marked_title)+matches;section_count+=1;}else if(title_match){
 html+=search_section_title(title,marked_title);section_count+=1;}}html=search_title(search_target,count,section_count)+html;
 $('#search_content').html(html);show_search_content();$('#content').scrollTop(0);}});$('#search_label').on('click',function(event){
 if(!$('#search_content').is(':visible')){show_search_content();$('#content').scrollTop(search_scroll_position);}else{
 search_scroll_position=$('#content').scrollTop();show_page_content();}});keypress("S","#search_label");
 $('#search_clear').on('click',function(event){var content=$("#page_content");content.html(clear_search_hits(content.html()));});
 keypress("C","#search_clear");toggle_current_navlink();$("#focus").on("click",function(){
 var navlink=$(".highlight");var basename=navlink.attr('data-basename');var target=navlink.attr('data-target');
 load_page(basename,target,new_page_callback,false);open_to_navlink_level(target);navlink.trigger('click');
 loaded_basename=basename;});keypress("F","#focus");$('#navigation .nodestatus').on('click',function(event){
 event.stopPropagation();var level=$(this).parent();var container=level;var contents=level.children('li ul,.nodestatus');
 if(container.hasClass('open')){toggle(container);contents.slideUp(slide_time);level.children('.nodestatus').slideDown(0);
 level.children('li>span.nodestatus').html(closed_symbol);if(toc_debug)container.css('background-color','pink');}
 else if(container.hasClass('closed')){toggle(container);container.slideDown(slide_time);container.children('ul').slideDown(slide_time);
 contents.slideDown(slide_time);level.children('li>span.nodestatus').html(open_symbol);if(toc_debug)
 container.css('background-color','yellow');}});$(document).on("click",".panelink",function(event){
 event.preventDefault();update_page_title($(this));toc_load($(this).attr('data-basename'),$(this).attr('data-target'),true);
 return false;});$('.navlink').on('click',function(event){event.preventDefault();update_page_title($(this));
 var basename=$(this).attr('data-basename');var target='#'+$(this).attr('data-target');toc_load(basename,target,event.originalEvent);
 show_navlink_in_toc($(this));if(event.originalEvent){var node=$(this).parent().find('.nodestatus').first();
 var is_closed=node.parent().hasClass('closed');if(is_closed){node.trigger('click');changed=true;}}
 return false;});$('#content').scroll(function(){SCROLLING=true;var header=header_in_window();
 var tag=header.get(0).tagName;if(tag.charAt(0)=="H"){highlight_visible_toc_sections();}
 SCROLLING=false;return false;});chrome_browser_p=typeof window.chrome==="object";$("#search_clear").hide();
 var location=window.location.toString();if(location.indexOf("#")==-1){var new_basename=first_basename();
 load_page(new_basename,'#',function(){initialize=true;new_page_callback();initialize_nodes();
 resize_toc();restore_page_visibility();});loaded_basename=new_basename;navigation_width_history=[];}else{
 var parts=location.split('#');var basename=parts[1];var target=(parts.length==3)? '#'+parts[2] : "#";
 var cb=function(){target_name=target;initialize=true;new_page_callback();initialize_nodes();
 open_to_navlink_level(parts[2]);restore_page_visibility();open_if_parts();};load_page(basename,target,cb);
 loaded_basename=basename;}return false;});
